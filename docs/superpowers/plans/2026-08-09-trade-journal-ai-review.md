# Trade Journal & AI Post-Trade Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users log real executed trades, close them out, and trigger a deterministic-score + AI-prose post-trade review grounded in the GEX context captured at entry.

**Architecture:** Two new SQLAlchemy tables (`trades`, `trade_reviews`) behind two new repositories; a pure `compute_execution_score()` function (never AI-generated); a new `LLMOrchestrator.review_trade()` method using the same forced-tool-call pattern as `generate_trade_plan`; four new FastAPI endpoints; a new `TradeJournal.jsx` overlay panel in the existing Vite/React frontend, wired in the same way `HistoryPanel`/`ProfilePanel` already are.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async (SQLite locally / Postgres in the cloud deployment), Pydantic v2, OpenAI Responses API tool calling, React (Vite, no build-time router), Tailwind arbitrary-value classes, lucide-react icons.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-trade-journal-ai-review-design.md` — read it before starting; this plan implements it exactly.
- New DB tables use only plain SQLAlchemy Core types (`String`, `Integer`, `Float`, `DateTime(timezone=True)`, `Text`) — no Postgres-only types — so they work unchanged on both the local SQLite dev DB and the cloud Postgres deployment.
- `execution_score` is always computed by backend code (`compute_execution_score`), never by the LLM. The LLM only explains a score it's given.
- git2threads dashboard integration is explicitly out of scope for this plan (see spec's "Out of Scope").
- No new frontend test tooling. Frontend verification is a manual dev-server smoke test, matching this project's existing convention for UI changes.
- Follow existing conventions exactly: `StrictModel` (extra="forbid") for all new Pydantic models, the repository pattern in `app/database.py`, the forced-tool-call pattern already used for `generate_trade_plan` in `app/services/openai_orchestrator.py`.

---

## Task 1: Data models, DB tables, and repositories

**Files:**
- Modify: `app/models.py` (append new models near `UserTradePlan`/`PlanList`)
- Modify: `app/database.py` (append new tables/repositories; extend `PlanRepository` and `GEXSnapshotRepository`)
- Test: `tests/test_database.py`

**Interfaces:**
- Produces (used by later tasks):
  - `app.models.TradeStatus` (str enum: `OPEN`, `CLOSED`)
  - `app.models.Trade`, `TradeCreate`, `TradeClose`, `TradeList`, `TradeReview`, `TradeReviewToolArguments`
  - `app.database.TradeRepository(session_factory)` with `async def create_trade(create: TradeCreate, entry_gex_snapshot_id: int | None) -> Trade`, `async def list_trades(user_id: str, ticker: str | None, status: str | None) -> list[Trade]`, `async def get_trade(trade_id: str) -> Trade | None`, `async def close_trade(trade_id: str, user_id: str, close: TradeClose) -> Trade` (raises `LookupError` if not found, `PermissionError` if owned by a different user, `ValueError` if already `CLOSED`)
  - `app.database.TradeReviewRepository(session_factory)` with `async def upsert_review(trade_id: str, execution_score: int, ai_feedback: str, key_takeaways: list[str]) -> TradeReview`, `async def get_review(trade_id: str) -> TradeReview | None`
  - `app.database.PlanRepository.get_plan(plan_id: str) -> UserTradePlan | None` (new method on the existing class)
  - `app.database.GEXSnapshotRepository.save_snapshot(...)` now returns `int` (the new row's id) instead of `None`
  - `app.database.GEXSnapshotRepository.get_snapshot(snapshot_id: int) -> GEXSnapshot | None` (new method)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_database.py`:

```python
from uuid import uuid4

from app.database import (
    PlanRepository,
    TradeRepository,
    TradeReviewRepository,
)
from app.models import (
    PlanStatus,
    TradeClose,
    TradeCreate,
    TradeStatus,
    UserTradePlan,
)


@pytest.mark.asyncio
async def test_gex_snapshot_repository_save_snapshot_returns_id() -> None:
    repo = GEXSnapshotRepository(await _session_factory())
    snapshot_id = await repo.save_snapshot("AAPL", 30, gex_summary())
    assert isinstance(snapshot_id, int)

    fetched = await repo.get_snapshot(snapshot_id)
    assert fetched is not None
    assert fetched.ticker == "AAPL"
    assert fetched.zero_gamma_strike == 95.0


@pytest.mark.asyncio
async def test_gex_snapshot_repository_get_snapshot_returns_none_when_missing() -> None:
    repo = GEXSnapshotRepository(await _session_factory())
    assert await repo.get_snapshot(999) is None


@pytest.mark.asyncio
async def test_plan_repository_get_plan_round_trips() -> None:
    repo = PlanRepository(await _session_factory())
    plan = UserTradePlan(
        plan_id=uuid4(),
        user_id="user-1",
        conversation_id="conv-1",
        ticker="AAPL",
        strategy_type="Long Call",
        entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
        max_loss_usd=250.0,
        theta_warning=False,
    )
    await repo.save_signed_plan(plan)

    fetched = await repo.get_plan(str(plan.plan_id))
    assert fetched is not None
    assert fetched.ticker == "AAPL"
    assert fetched.status == PlanStatus.SIGNED


@pytest.mark.asyncio
async def test_plan_repository_get_plan_returns_none_when_missing() -> None:
    repo = PlanRepository(await _session_factory())
    assert await repo.get_plan(str(uuid4())) is None


@pytest.mark.asyncio
async def test_trade_repository_create_defaults_to_open() -> None:
    repo = TradeRepository(await _session_factory())
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1",
            ticker="aapl",
            strategy_type="Long Call",
            entry_price=100.0,
            position_size=1,
            days_to_expiration=30,
        ),
        entry_gex_snapshot_id=42,
    )
    assert trade.ticker == "AAPL"
    assert trade.status == TradeStatus.OPEN
    assert trade.exit_price is None
    assert trade.pnl_pct is None
    assert trade.entry_gex_snapshot_id == 42


@pytest.mark.asyncio
async def test_trade_repository_list_filters_by_ticker_and_status() -> None:
    repo = TradeRepository(await _session_factory())
    await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            entry_price=100.0, position_size=1, days_to_expiration=30,
        ),
        entry_gex_snapshot_id=None,
    )
    await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="TSLA", strategy_type="Long Put",
            entry_price=200.0, position_size=1, days_to_expiration=30,
        ),
        entry_gex_snapshot_id=None,
    )

    all_trades = await repo.list_trades("user-1")
    assert len(all_trades) == 2

    aapl_only = await repo.list_trades("user-1", ticker="AAPL")
    assert [t.ticker for t in aapl_only] == ["AAPL"]

    open_only = await repo.list_trades("user-1", status="OPEN")
    assert len(open_only) == 2


@pytest.mark.asyncio
async def test_trade_repository_close_trade_computes_pnl_pct() -> None:
    repo = TradeRepository(await _session_factory())
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            entry_price=100.0, position_size=1, days_to_expiration=30,
        ),
        entry_gex_snapshot_id=None,
    )
    closed = await repo.close_trade(
        str(trade.id),
        "user-1",
        TradeClose(
            exit_price=120.0,
            exit_date=datetime.now(timezone.utc),
            pnl=20.0,
        ),
    )
    assert closed.status == TradeStatus.CLOSED
    assert closed.pnl_pct == 20.0


@pytest.mark.asyncio
async def test_trade_repository_close_trade_raises_lookup_error_when_missing() -> None:
    repo = TradeRepository(await _session_factory())
    with pytest.raises(LookupError):
        await repo.close_trade(
            str(uuid4()),
            "user-1",
            TradeClose(exit_price=120.0, exit_date=datetime.now(timezone.utc), pnl=20.0),
        )


@pytest.mark.asyncio
async def test_trade_repository_close_trade_raises_permission_error_for_other_user() -> None:
    repo = TradeRepository(await _session_factory())
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            entry_price=100.0, position_size=1, days_to_expiration=30,
        ),
        entry_gex_snapshot_id=None,
    )
    with pytest.raises(PermissionError):
        await repo.close_trade(
            str(trade.id),
            "someone-else",
            TradeClose(exit_price=120.0, exit_date=datetime.now(timezone.utc), pnl=20.0),
        )


@pytest.mark.asyncio
async def test_trade_repository_close_trade_raises_value_error_when_already_closed() -> None:
    repo = TradeRepository(await _session_factory())
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            entry_price=100.0, position_size=1, days_to_expiration=30,
        ),
        entry_gex_snapshot_id=None,
    )
    close = TradeClose(exit_price=120.0, exit_date=datetime.now(timezone.utc), pnl=20.0)
    await repo.close_trade(str(trade.id), "user-1", close)
    with pytest.raises(ValueError):
        await repo.close_trade(str(trade.id), "user-1", close)


@pytest.mark.asyncio
async def test_trade_review_repository_upsert_then_overwrite() -> None:
    repo = TradeReviewRepository(await _session_factory())
    trade_id = str(uuid4())
    first = await repo.upsert_review(trade_id, 4, "Good exit.", ["Booked profit near plan"])
    assert first.execution_score == 4

    second = await repo.upsert_review(trade_id, 2, "Revised take.", ["Stop slipped"])
    assert second.execution_score == 2

    fetched = await repo.get_review(trade_id)
    assert fetched is not None
    assert fetched.ai_feedback == "Revised take."
    assert fetched.key_takeaways == ["Stop slipped"]


@pytest.mark.asyncio
async def test_trade_review_repository_get_review_returns_none_when_missing() -> None:
    repo = TradeReviewRepository(await _session_factory())
    assert await repo.get_review(str(uuid4())) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/test_database.py -v`
Expected: FAIL with `ImportError` (`TradeRepository`, `TradeCreate`, etc. don't exist yet).

- [ ] **Step 3: Add the Pydantic models**

In `app/models.py`, add after `class PlanList(StrictModel): plans: list[UserTradePlan]` (keep existing imports at top — add `UUID` is already imported):

```python
class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Trade(StrictModel):
    id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=32)
    strategy_type: str = Field(min_length=1, max_length=128)
    source_plan_id: UUID | None = None
    entry_date: datetime
    exit_date: datetime | None = None
    entry_price: float = Field(gt=0)
    exit_price: float | None = Field(default=None, gt=0)
    position_size: int = Field(gt=0)
    pnl: float | None = None
    pnl_pct: float | None = None
    status: TradeStatus
    notes: str | None = Field(default=None, max_length=2000)
    entry_gex_snapshot_id: int | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class TradeCreate(StrictModel):
    user_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=32)
    strategy_type: str = Field(min_length=1, max_length=128)
    source_plan_id: UUID | None = None
    entry_price: float = Field(gt=0)
    position_size: int = Field(gt=0)
    notes: str | None = Field(default=None, max_length=2000)
    days_to_expiration: int = Field(default=30, ge=0, le=730)


class TradeClose(StrictModel):
    exit_price: float = Field(gt=0)
    exit_date: datetime
    pnl: float
    notes: str | None = Field(default=None, max_length=2000)


class TradeList(StrictModel):
    trades: list[Trade]


class TradeReview(StrictModel):
    trade_id: UUID
    execution_score: int = Field(ge=1, le=5)
    ai_feedback: str
    key_takeaways: list[str]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class TradeReviewToolArguments(StrictModel):
    ai_feedback: str = Field(min_length=1, max_length=600)
    key_takeaways: list[str] = Field(min_length=1, max_length=5)
```

- [ ] **Step 4: Add the SQLAlchemy tables and repositories**

In `app/database.py`, add `from uuid import uuid4` and `import json` to the top imports, and extend the `app.models` import with `Trade, TradeClose, TradeCreate, TradeReview, TradeStatus`.

Modify `GEXSnapshotRepository.save_snapshot` to return the new row's id:

```python
    async def save_snapshot(
        self, ticker: str, days_to_expiration: int, summary: OptionGEXSummary
    ) -> int:
        async with self.session_factory() as session:
            record = GEXSnapshotDBRecord(
                ticker=ticker,
                days_to_expiration=days_to_expiration,
                captured_at=datetime.now(timezone.utc),
                underlying_price=summary.stock_price,
                zero_gamma_strike=summary.zero_gamma,
                call_wall_strike=summary.call_wall,
                put_wall_strike=summary.put_wall,
                net_gex=summary.net_gex,
                iv_rank=summary.iv_rank,
                gex_status=summary.gex_status.value,
            )
            session.add(record)
            await session.commit()
            return record.id
```

Add a `get_snapshot` method to `GEXSnapshotRepository` (after `list_snapshots`):

```python
    async def get_snapshot(self, snapshot_id: int) -> GEXSnapshot | None:
        async with self.session_factory() as session:
            record = await session.get(GEXSnapshotDBRecord, snapshot_id)
            if record is None:
                return None
            return GEXSnapshot(
                ticker=record.ticker,
                days_to_expiration=record.days_to_expiration,
                captured_at=_as_utc(record.captured_at),
                underlying_price=record.underlying_price,
                zero_gamma_strike=record.zero_gamma_strike,
                call_wall_strike=record.call_wall_strike,
                put_wall_strike=record.put_wall_strike,
                net_gex=record.net_gex,
                iv_rank=record.iv_rank,
                gex_status=GEXStatus(record.gex_status),
            )
```

Add a `get_plan` method to `PlanRepository` (after `save_signed_plan`, before `list_plans`):

```python
    async def get_plan(self, plan_id: str) -> UserTradePlan | None:
        async with self.session_factory() as session:
            record = await session.get(TradePlanRecord, plan_id)
            if record is None:
                return None
            return UserTradePlan(
                plan_id=record.plan_id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                ticker=record.ticker,
                strategy_type=record.strategy_type,
                entry_price=record.entry_price,
                stop_loss=record.stop_loss,
                target_price=record.target_price,
                max_loss_usd=record.max_loss_usd,
                theta_warning=record.theta_warning,
                status=PlanStatus(record.status),
                created_at=_as_utc(record.created_at),
                signed_at=_as_utc(record.signed_at),
            )
```

Append at the end of the file — `TradeRecord`, `TradeReviewRecord`, `TradeRepository`, `TradeReviewRepository`:

```python
class TradeRecord(Base):
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    strategy_type: Mapped[str] = mapped_column(String(128))
    source_plan_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_size: Mapped[int] = mapped_column(Integer)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    entry_gex_snapshot_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TradeRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def create_trade(
        self, create: TradeCreate, entry_gex_snapshot_id: int | None
    ) -> Trade:
        now = datetime.now(timezone.utc)
        record = TradeRecord(
            id=str(uuid4()),
            user_id=create.user_id,
            ticker=create.ticker.strip().upper(),
            strategy_type=create.strategy_type,
            source_plan_id=(
                str(create.source_plan_id) if create.source_plan_id else None
            ),
            entry_date=now,
            exit_date=None,
            entry_price=create.entry_price,
            exit_price=None,
            position_size=create.position_size,
            pnl=None,
            pnl_pct=None,
            status=TradeStatus.OPEN.value,
            notes=create.notes,
            entry_gex_snapshot_id=entry_gex_snapshot_id,
            created_at=now,
        )
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
        return self._to_model(record)

    async def list_trades(
        self, user_id: str, ticker: str | None = None, status: str | None = None
    ) -> list[Trade]:
        async with self.session_factory() as session:
            stmt = select(TradeRecord).where(TradeRecord.user_id == user_id)
            if ticker:
                stmt = stmt.where(TradeRecord.ticker == ticker.strip().upper())
            if status:
                stmt = stmt.where(TradeRecord.status == status)
            stmt = stmt.order_by(desc(TradeRecord.created_at))
            records = await session.scalars(stmt)
            return [self._to_model(record) for record in records]

    async def get_trade(self, trade_id: str) -> Trade | None:
        async with self.session_factory() as session:
            record = await session.get(TradeRecord, trade_id)
            return self._to_model(record) if record else None

    async def close_trade(
        self, trade_id: str, user_id: str, close: TradeClose
    ) -> Trade:
        async with self.session_factory() as session:
            record = await session.get(TradeRecord, trade_id)
            if record is None:
                raise LookupError("Trade not found")
            if record.user_id != user_id:
                raise PermissionError("The trade belongs to a different user")
            if record.status == TradeStatus.CLOSED.value:
                raise ValueError("Trade is already closed")
            record.exit_price = close.exit_price
            record.exit_date = close.exit_date
            record.pnl = close.pnl
            record.pnl_pct = (
                close.pnl / (record.entry_price * record.position_size)
            ) * 100
            record.status = TradeStatus.CLOSED.value
            if close.notes is not None:
                record.notes = close.notes
            await session.commit()
            return self._to_model(record)

    @staticmethod
    def _to_model(record: TradeRecord) -> Trade:
        return Trade(
            id=record.id,
            user_id=record.user_id,
            ticker=record.ticker,
            strategy_type=record.strategy_type,
            source_plan_id=record.source_plan_id,
            entry_date=_as_utc(record.entry_date),
            exit_date=_as_utc(record.exit_date),
            entry_price=record.entry_price,
            exit_price=record.exit_price,
            position_size=record.position_size,
            pnl=record.pnl,
            pnl_pct=record.pnl_pct,
            status=TradeStatus(record.status),
            notes=record.notes,
            entry_gex_snapshot_id=record.entry_gex_snapshot_id,
            created_at=_as_utc(record.created_at),
        )


class TradeReviewRecord(Base):
    __tablename__ = "trade_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trade_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    execution_score: Mapped[int] = mapped_column(Integer)
    ai_feedback: Mapped[str] = mapped_column(Text)
    key_takeaways: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TradeReviewRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def upsert_review(
        self,
        trade_id: str,
        execution_score: int,
        ai_feedback: str,
        key_takeaways: list[str],
    ) -> TradeReview:
        created_at = datetime.now(timezone.utc)
        takeaways_json = json.dumps(key_takeaways)
        async with self.session_factory() as session:
            record = await session.scalar(
                select(TradeReviewRecord).where(
                    TradeReviewRecord.trade_id == trade_id
                )
            )
            if record is None:
                record = TradeReviewRecord(
                    id=str(uuid4()),
                    trade_id=trade_id,
                    execution_score=execution_score,
                    ai_feedback=ai_feedback,
                    key_takeaways=takeaways_json,
                    created_at=created_at,
                )
                session.add(record)
            else:
                record.execution_score = execution_score
                record.ai_feedback = ai_feedback
                record.key_takeaways = takeaways_json
                record.created_at = created_at
            await session.commit()
        return TradeReview(
            trade_id=trade_id,
            execution_score=execution_score,
            ai_feedback=ai_feedback,
            key_takeaways=key_takeaways,
            created_at=created_at,
        )

    async def get_review(self, trade_id: str) -> TradeReview | None:
        async with self.session_factory() as session:
            record = await session.scalar(
                select(TradeReviewRecord).where(
                    TradeReviewRecord.trade_id == trade_id
                )
            )
            if record is None:
                return None
            return TradeReview(
                trade_id=record.trade_id,
                execution_score=record.execution_score,
                ai_feedback=record.ai_feedback,
                key_takeaways=json.loads(record.key_takeaways),
                created_at=_as_utc(record.created_at),
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/test_database.py -v`
Expected: All PASS, including the pre-existing tests in that file (no regressions).

- [ ] **Step 6: Commit**

```bash
git add app/models.py app/database.py tests/test_database.py
git commit -m "Add Trade/TradeReview models, tables, and repositories"
```

---

## Task 2: Deterministic execution-score module

**Files:**
- Create: `app/services/trade_scoring.py`
- Test: `tests/test_trade_scoring.py`

**Interfaces:**
- Consumes: nothing (pure function, no dependencies on other tasks)
- Produces: `app.services.trade_scoring.compute_execution_score(entry_price: float, exit_price: float, pnl_pct: float, stop_loss: float | None, target_price: float | None) -> int` — used by Task 4's review endpoint

- [ ] **Step 1: Write the failing test**

Create `tests/test_trade_scoring.py`:

```python
import pytest

from app.services.trade_scoring import compute_execution_score


@pytest.mark.parametrize(
    "exit_price,expected",
    [
        (130.0, 5),  # met/beat planned target (planned_rr = 20/10 = 2)
        (115.0, 4),  # profitable, short of target
        (95.0, 3),   # small loss within planned risk
        (85.0, 2),   # stop discipline slipped moderately
        (70.0, 1),   # loss ran well past the planned stop
    ],
)
def test_score_with_plan_bullish(exit_price: float, expected: int) -> None:
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=exit_price,
        pnl_pct=0.0,
        stop_loss=90.0,
        target_price=120.0,
    )
    assert score == expected


@pytest.mark.parametrize(
    "exit_price,expected",
    [
        (70.0, 5),
        (85.0, 4),
        (105.0, 3),
        (115.0, 2),
        (130.0, 1),
    ],
)
def test_score_with_plan_bearish(exit_price: float, expected: int) -> None:
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=exit_price,
        pnl_pct=0.0,
        stop_loss=110.0,
        target_price=80.0,
    )
    assert score == expected


@pytest.mark.parametrize(
    "pnl_pct,expected",
    [
        (20.0, 4),
        (5.0, 3),
        (-5.0, 2),
        (-20.0, 1),
    ],
)
def test_score_without_plan(pnl_pct: float, expected: int) -> None:
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=100.0,
        pnl_pct=pnl_pct,
        stop_loss=None,
        target_price=None,
    )
    assert score == expected


def test_score_falls_back_to_pnl_pct_when_stop_equals_entry() -> None:
    # planned_risk would be zero (division by zero) — must fall back safely.
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=100.0,
        pnl_pct=20.0,
        stop_loss=100.0,
        target_price=120.0,
    )
    assert score == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/test_trade_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.trade_scoring'`

- [ ] **Step 3: Implement**

Create `app/services/trade_scoring.py`:

```python
"""Deterministic execution-discipline scoring for closed trades.

Kept as a pure function, isolated from the LLM orchestrator, so the score a
user sees is always reproducible and never model-generated — see
docs/superpowers/specs/2026-08-09-trade-journal-ai-review-design.md.
"""


def compute_execution_score(
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    stop_loss: float | None,
    target_price: float | None,
) -> int:
    has_plan = (
        stop_loss is not None
        and target_price is not None
        and stop_loss != entry_price
    )
    if has_plan:
        return _score_with_plan(entry_price, exit_price, stop_loss, target_price)
    return _score_without_plan(pnl_pct)


def _score_with_plan(
    entry_price: float, exit_price: float, stop_loss: float, target_price: float
) -> int:
    bullish = target_price > entry_price
    planned_risk = abs(entry_price - stop_loss)
    planned_reward = abs(target_price - entry_price)
    planned_rr = planned_reward / planned_risk
    realized_move = (
        (exit_price - entry_price) if bullish else (entry_price - exit_price)
    )
    r_multiple = realized_move / planned_risk

    if r_multiple >= planned_rr:
        return 5
    if r_multiple > 0:
        return 4
    if r_multiple >= -1:
        return 3
    if r_multiple >= -1.5:
        return 2
    return 1


def _score_without_plan(pnl_pct: float) -> int:
    if pnl_pct >= 15:
        return 4
    if pnl_pct >= 0:
        return 3
    if pnl_pct >= -10:
        return 2
    return 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/test_trade_scoring.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/trade_scoring.py tests/test_trade_scoring.py
git commit -m "Add deterministic execution-score calculation for trade reviews"
```

---

## Task 3: `LLMOrchestrator.review_trade()`

**Files:**
- Modify: `app/services/openai_orchestrator.py`
- Test: `tests/test_openai_orchestrator.py`

**Interfaces:**
- Consumes: `app.models.Trade`, `TradeReviewToolArguments` (Task 1); `app.models.GEXSnapshot` (existing)
- Produces: `LLMOrchestrator.review_trade(trade: Trade, entry_snapshot: GEXSnapshot | None, execution_score: int) -> tuple[str, list[str]]` (returns `(ai_feedback, key_takeaways)`) — used by Task 4's review endpoint

- [ ] **Step 1: Write the failing test**

Add to the top imports of `tests/test_openai_orchestrator.py`:

```python
from app.models import (
    ChatContext,
    ChatRequest,
    GEXSnapshot,
    GEXStatus,
    OptionGEXSummary,
    RiskProfile,
    Trade,
    TradeStatus,
    UserProfile,
)
from uuid import uuid4
```

(This replaces the existing `from app.models import (...)` block — add `GEXSnapshot`, `Trade`, `TradeStatus` to it, and add the `from uuid import uuid4` line alongside the existing `from types import SimpleNamespace` import.)

Append this test at the end of the file:

```python
@pytest.mark.asyncio
async def test_review_trade_forces_tool_and_returns_feedback() -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.request: dict = {}

        async def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_trade_review",
                        arguments=(
                            '{"ai_feedback":"Exit respected the plan.",'
                            '"key_takeaways":["Booked profit near target",'
                            '"Stuck to the stop"]}'
                        ),
                        call_id="call-1",
                    )
                ],
                output_text="",
            )

    fake_responses = FakeResponses()
    fake_client = SimpleNamespace(responses=fake_responses)
    orchestrator = LLMOrchestrator(fake_client, "test-model", 250)
    trade = Trade(
        id=uuid4(),
        user_id="user-1",
        ticker="AAPL",
        strategy_type="Long Call",
        source_plan_id=uuid4(),
        entry_date=datetime.now(timezone.utc),
        exit_date=datetime.now(timezone.utc),
        entry_price=100.0,
        exit_price=120.0,
        position_size=1,
        pnl=20.0,
        pnl_pct=20.0,
        status=TradeStatus.CLOSED,
        notes="closed at target",
        entry_gex_snapshot_id=1,
    )
    entry_snapshot = GEXSnapshot(
        ticker="AAPL",
        days_to_expiration=5,
        captured_at=datetime.now(timezone.utc),
        underlying_price=100.0,
        zero_gamma_strike=95.0,
        call_wall_strike=110.0,
        put_wall_strike=90.0,
        net_gex=1_000_000.0,
        iv_rank=40.0,
        gex_status=GEXStatus.POS_GAMMA,
    )

    ai_feedback, key_takeaways = await orchestrator.review_trade(
        trade, entry_snapshot, 5
    )

    assert fake_responses.request["tool_choice"] == {
        "type": "function",
        "name": "submit_trade_review",
    }
    assert '"execution_score": 5' in fake_responses.request["instructions"]
    assert ai_feedback == "Exit respected the plan."
    assert key_takeaways == ["Booked profit near target", "Stuck to the stop"]


@pytest.mark.asyncio
async def test_review_trade_without_source_plan_marks_has_source_plan_false() -> None:
    class FakeResponses:
        async def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_trade_review",
                        arguments=(
                            '{"ai_feedback":"No plan to compare against.",'
                            '"key_takeaways":["Track a plan next time"]}'
                        ),
                        call_id="call-1",
                    )
                ],
                output_text="",
            )

    fake_responses = FakeResponses()
    fake_client = SimpleNamespace(responses=fake_responses)
    orchestrator = LLMOrchestrator(fake_client, "test-model", 250)
    trade = Trade(
        id=uuid4(),
        user_id="user-1",
        ticker="AAPL",
        strategy_type="Long Call",
        source_plan_id=None,
        entry_date=datetime.now(timezone.utc),
        exit_date=datetime.now(timezone.utc),
        entry_price=100.0,
        exit_price=120.0,
        position_size=1,
        pnl=20.0,
        pnl_pct=20.0,
        status=TradeStatus.CLOSED,
        notes=None,
        entry_gex_snapshot_id=None,
    )

    await orchestrator.review_trade(trade, None, 4)

    assert '"has_source_plan": false' in fake_responses.request["instructions"]
    assert '"entry_gex_context": null' in fake_responses.request["instructions"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/test_openai_orchestrator.py -v -k review_trade`
Expected: FAIL with `AttributeError: 'LLMOrchestrator' object has no attribute 'review_trade'`

- [ ] **Step 3: Implement**

In `app/services/openai_orchestrator.py`, add `Trade`, `GEXSnapshot`, `TradeReviewToolArguments` to the `from app.models import (...)` block at the top.

Add these three methods to `LLMOrchestrator` (place `_review_tool` next to the existing `_tool` static method, and `_review_instructions`/`review_trade` after `_instructions`/`chat` respectively):

```python
    @staticmethod
    def _review_tool() -> dict[str, Any]:
        schema = TradeReviewToolArguments.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "function",
            "name": "submit_trade_review",
            "description": (
                "Submit the post-trade review prose. execution_score in the "
                "context is already computed by the backend — explain it, "
                "never invent a different one."
            ),
            "strict": True,
            "parameters": schema,
        }
```

```python
    def _review_instructions(
        self,
        trade: Trade,
        entry_snapshot: OptionGEXSummary | Any | None,
        execution_score: int,
    ) -> str:
        context = json.dumps(
            {
                "ticker": trade.ticker,
                "strategy_type": trade.strategy_type,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "position_size": trade.position_size,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "notes": trade.notes,
                "execution_score": execution_score,
                "has_source_plan": trade.source_plan_id is not None,
                "entry_gex_context": (
                    {
                        "zero_gamma": entry_snapshot.zero_gamma_strike,
                        "call_wall": entry_snapshot.call_wall_strike,
                        "put_wall": entry_snapshot.put_wall_strike,
                        "gex_status": entry_snapshot.gex_status.value,
                    }
                    if entry_snapshot is not None
                    else None
                ),
            },
            ensure_ascii=True,
        )
        return f"""
You are a disciplined senior options trading coach reviewing one already-closed
trade. Give a direct, honest post-trade diagnosis.

<context>
{context}
</context>

<behavior_rules>
1. execution_score in <context> is already computed by a fixed backend rule,
   not by you. Never state a different score anywhere in your feedback.
   Your job is to explain WHY it landed there, using entry_price,
   exit_price, pnl_pct, and entry_gex_context.
2. If has_source_plan is false, say plainly that there was no predefined
   stop/target plan to grade discipline against, and that the score is a
   pnl_pct-based approximation instead — never imply a plan comparison that
   didn't happen.
3. Ground every GEX reference only in entry_gex_context above (Zero Gamma /
   Call Wall / Put Wall / gex_status at entry) — never invent levels not
   present there, and never assert claims about market makers' intent or
   any trader group.
4. ai_feedback must be concise, direct, and at most 600 characters.
5. key_takeaways must be 1 to 5 short, concrete, actionable bullets — no
   vague platitudes like "manage risk better."
</behavior_rules>
""".strip()
```

```python
    async def review_trade(
        self,
        trade: Trade,
        entry_snapshot: Any | None,
        execution_score: int,
    ) -> tuple[str, list[str]]:
        if self.client is None:
            raise HTTPException(503, "OPENAI_API_KEY is not configured")
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=self._review_instructions(
                    trade, entry_snapshot, execution_score
                ),
                input=[{"role": "user", "content": "Review this closed trade."}],
                tools=[self._review_tool()],
                tool_choice={"type": "function", "name": "submit_trade_review"},
            )
        except OpenAIError as exc:
            logger.exception("Trade review OpenAI request failed")
            raise HTTPException(
                502, f"OpenAI request failed: {type(exc).__name__}"
            ) from exc

        for item in response.output:
            if item.type != "function_call" or item.name != "submit_trade_review":
                continue
            try:
                arguments = TradeReviewToolArguments.model_validate_json(
                    item.arguments
                )
            except ValidationError as exc:
                logger.warning("Invalid strict trade-review tool arguments: %s", exc)
                raise HTTPException(
                    502, "The model returned malformed review arguments"
                ) from exc
            return arguments.ai_feedback, arguments.key_takeaways

        raise HTTPException(
            502, "The model did not return the required trade-review tool call"
        )
```

Note: `entry_snapshot` is typed `Any | None` here (rather than importing `GEXSnapshot` into a strict type hint) deliberately — `app.models.GEXSnapshot` is a `StrictModel`, but this method only reads four plain attributes from it, so a structural (duck-typed) parameter keeps `openai_orchestrator.py` from needing a new import cycle concern. If preferred for clarity, `GEXSnapshot` can be imported and used as the exact type instead — either is fine as long as the four attributes accessed in `_review_instructions` match `GEXSnapshot`'s field names (`zero_gamma_strike`, `call_wall_strike`, `put_wall_strike`, `gex_status`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/test_openai_orchestrator.py -v`
Expected: All PASS (including all pre-existing tests in the file)

- [ ] **Step 5: Commit**

```bash
git add app/services/openai_orchestrator.py tests/test_openai_orchestrator.py
git commit -m "Add LLMOrchestrator.review_trade() with forced tool-call output"
```

---

## Task 4: Backend API endpoints

**Files:**
- Modify: `app/main.py`
- Modify: `app/services/__init__.py`
- Test: `tests/test_trades_endpoints.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3 (`TradeRepository`, `TradeReviewRepository`, `compute_execution_score`, `LLMOrchestrator.review_trade`)
- Produces: `POST /api/v1/trades`, `GET /api/v1/trades`, `PUT /api/v1/trades/{trade_id}`, `POST /api/v1/trades/{trade_id}/review` — consumed by Task 5's frontend

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trades_endpoints.py`:

```python
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def _summary_payload(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "stock_price": 100.0,
        "zero_gamma": 95.0,
        "call_wall": 110.0,
        "put_wall": 90.0,
        "iv_rank": 40.0,
        "net_gex": 1_000_000.0,
        "gex_status": "POS_GAMMA",
    }


def _seed_cache(client: TestClient, monkeypatch, ticker: str, dte: int = 30) -> None:
    """Pre-warms the GEX cache via the sync endpoint so a trade's entry-
    snapshot fetch hits the cache instead of a live market-data call — same
    technique tests/test_sync_endpoints.py already uses.
    """
    monkeypatch.setattr(settings, "sync_token", "test-sync-token")
    response = client.post(
        "/api/v1/sync/gex",
        json={
            "ticker": ticker,
            "days_to_expiration": dte,
            "summary": _summary_payload(ticker),
        },
        headers={"X-Sync-Token": "test-sync-token"},
    )
    assert response.status_code == 200


def _create_trade(client: TestClient, user_id: str, ticker: str) -> dict:
    response = client.post(
        "/api/v1/trades",
        json={
            "user_id": user_id,
            "ticker": ticker,
            "strategy_type": "Long Call",
            "entry_price": 100.0,
            "position_size": 1,
            "days_to_expiration": 30,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_create_trade_writes_entry_snapshot_and_defaults_to_open(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST1")
        trade = _create_trade(client, "user-1", "TJTEST1")
    assert trade["status"] == "OPEN"
    assert trade["entry_gex_snapshot_id"] is not None
    assert trade["exit_price"] is None
    assert trade["pnl_pct"] is None


def test_list_trades_filters_by_ticker_and_status(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST2")
        _seed_cache(client, monkeypatch, "TJTEST3")
        _create_trade(client, "user-2", "TJTEST2")
        _create_trade(client, "user-2", "TJTEST3")

        response = client.get("/api/v1/trades?user_id=user-2&ticker=TJTEST2")
    assert response.status_code == 200
    trades = response.json()["trades"]
    assert len(trades) == 1
    assert trades[0]["ticker"] == "TJTEST2"


def test_close_trade_computes_pnl_pct(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST4")
        trade = _create_trade(client, "user-3", "TJTEST4")
        response = client.put(
            f"/api/v1/trades/{trade['id']}?user_id=user-3",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 20.0,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "CLOSED"
    assert body["pnl_pct"] == 20.0


def test_close_trade_rejects_other_users_trade(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST5")
        trade = _create_trade(client, "user-4", "TJTEST5")
        response = client.put(
            f"/api/v1/trades/{trade['id']}?user_id=someone-else",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 20.0,
            },
        )
    assert response.status_code == 403


def test_close_trade_rejects_already_closed(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST6")
        trade = _create_trade(client, "user-5", "TJTEST6")
        close_payload = {
            "exit_price": 120.0,
            "exit_date": datetime.now(timezone.utc).isoformat(),
            "pnl": 20.0,
        }
        client.put(f"/api/v1/trades/{trade['id']}?user_id=user-5", json=close_payload)
        response = client.put(
            f"/api/v1/trades/{trade['id']}?user_id=user-5", json=close_payload
        )
    assert response.status_code == 409


def test_review_trade_requires_closed_status(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST7")
        trade = _create_trade(client, "user-6", "TJTEST7")
        response = client.post(f"/api/v1/trades/{trade['id']}/review?user_id=user-6")
    assert response.status_code == 400


def test_review_trade_returns_404_for_unknown_trade() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/trades/does-not-exist/review?user_id=user-7"
        )
    assert response.status_code == 404


def test_review_trade_upserts_review_with_fake_llm(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST8")
        trade = _create_trade(client, "user-8", "TJTEST8")
        client.put(
            f"/api/v1/trades/{trade['id']}?user_id=user-8",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 20.0,
            },
        )

        class FakeOrchestrator:
            async def review_trade(self, trade, entry_snapshot, execution_score):
                return (
                    "Solid, disciplined exit.",
                    ["Booked profit near plan", "Stayed within risk"],
                )

        app.state.services.llm = FakeOrchestrator()

        first = client.post(f"/api/v1/trades/{trade['id']}/review?user_id=user-8")
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["ai_feedback"] == "Solid, disciplined exit."
        assert first_body["execution_score"] == 4  # no source_plan -> pnl_pct 20% -> band >=15

        second = client.post(f"/api/v1/trades/{trade['id']}/review?user_id=user-8")
    assert second.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/test_trades_endpoints.py -v`
Expected: FAIL with 404s (routes don't exist yet)

- [ ] **Step 3: Implement**

In `app/services/__init__.py`, add the export:

```python
"""Application service layer."""

from app.services.cloud_sync import CloudSync
from app.services.gex_service import GEXService
from app.services.openai_orchestrator import LLMOrchestrator
from app.services.trade_scoring import compute_execution_score

__all__ = ["CloudSync", "GEXService", "LLMOrchestrator", "compute_execution_score"]
```

In `app/main.py`:

Add to the `from app.database import (...)` block: `TradeRepository, TradeReviewRepository`.

Add to the `from app.models import (...)` block: `Trade, TradeClose, TradeCreate, TradeList, TradeReview, TradeStatus`.

Change the `from app.services import CloudSync, GEXService, LLMOrchestrator` line to:
```python
from app.services import CloudSync, GEXService, LLMOrchestrator, compute_execution_score
```

Add two fields to `AppServices`:
```python
@dataclass(slots=True)
class AppServices:
    engine: AsyncEngine
    cache: ResilientCache
    market_data: FallbackMarketDataClient
    gex_service: GEXService
    plan_repository: PlanRepository
    chat_repository: ChatRepository
    profile_repository: ProfileRepository
    snapshot_repository: GEXSnapshotRepository
    trade_repository: TradeRepository
    trade_review_repository: TradeReviewRepository
    llm: LLMOrchestrator
```

In `lifespan()`, after `snapshot_repository = GEXSnapshotRepository(session_factory)`, add:
```python
    trade_repository = TradeRepository(session_factory)
    trade_review_repository = TradeReviewRepository(session_factory)
```

And in the `AppServices(...)` construction, add:
```python
        trade_repository=trade_repository,
        trade_review_repository=trade_review_repository,
```
(anywhere among the keyword args, e.g. right after `snapshot_repository=snapshot_repository,`).

Add the four endpoints after `save_plan` (end of the file):

```python
@app.post("/api/v1/trades", response_model=Trade)
async def create_trade(payload: TradeCreate, services: Services) -> Trade:
    summary = await services.gex_service.get_summary(
        payload.ticker, payload.days_to_expiration
    )
    entry_gex_snapshot_id = await services.snapshot_repository.save_snapshot(
        payload.ticker.strip().upper(), payload.days_to_expiration, summary
    )
    return await services.trade_repository.create_trade(
        payload, entry_gex_snapshot_id
    )


@app.get("/api/v1/trades", response_model=TradeList)
async def list_trades(
    user_id: str,
    services: Services,
    ticker: str | None = None,
    status: str | None = None,
) -> TradeList:
    trades = await services.trade_repository.list_trades(user_id, ticker, status)
    return TradeList(trades=trades)


@app.put("/api/v1/trades/{trade_id}", response_model=Trade)
async def close_trade(
    trade_id: str, user_id: str, payload: TradeClose, services: Services
) -> Trade:
    try:
        return await services.trade_repository.close_trade(
            trade_id, user_id, payload
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/trades/{trade_id}/review", response_model=TradeReview)
async def review_trade(
    trade_id: str, user_id: str, services: Services
) -> TradeReview:
    trade = await services.trade_repository.get_trade(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    if trade.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="The trade belongs to a different user"
        )
    if trade.status != TradeStatus.CLOSED:
        raise HTTPException(
            status_code=400,
            detail="Trade must be closed before it can be reviewed",
        )
    assert trade.exit_price is not None and trade.pnl_pct is not None

    stop_loss = target_price = None
    if trade.source_plan_id is not None:
        plan = await services.plan_repository.get_plan(str(trade.source_plan_id))
        if plan is not None:
            stop_loss = plan.stop_loss
            target_price = plan.target_price

    execution_score = compute_execution_score(
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        pnl_pct=trade.pnl_pct,
        stop_loss=stop_loss,
        target_price=target_price,
    )
    entry_snapshot = (
        await services.snapshot_repository.get_snapshot(trade.entry_gex_snapshot_id)
        if trade.entry_gex_snapshot_id is not None
        else None
    )
    ai_feedback, key_takeaways = await services.llm.review_trade(
        trade, entry_snapshot, execution_score
    )
    return await services.trade_review_repository.upsert_review(
        trade_id, execution_score, ai_feedback, key_takeaways
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/test_trades_endpoints.py -v`
Expected: All PASS

Then run the full suite to check for regressions:
Run: `cd "/Users/andrewweng/Desktop/stock schedule" && .venv/bin/python -m pytest tests/ -q`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py app/services/__init__.py tests/test_trades_endpoints.py
git commit -m "Add trade CRUD and AI post-trade review API endpoints"
```

---

## Task 5: Frontend Trade Journal panel

**Files:**
- Create: `web/src/TradeJournal.jsx`
- Modify: `web/src/TradingTerminalNotebook.jsx`

**Interfaces:**
- Consumes: Task 4's four endpoints, plus the existing `GET /api/v1/plans?user_id=` endpoint
- Produces: `export default function TradeJournalPanel({ userId, onClose })` — mounted as an overlay inside `TradingTerminalNotebook.jsx`'s chat section, exactly like the existing `HistoryPanel`/`ProfilePanel`

No automated test for this task — per this project's established convention (documented in the design spec's Testing Plan), frontend changes are verified by running the dev server and using the feature, not by adding new frontend test tooling.

- [ ] **Step 1: Create `web/src/TradeJournal.jsx`**

```jsx
import { useEffect, useState } from "react";
import { Plus, Star, X } from "lucide-react";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8002";
const MONO =
  '[font-family:ui-monospace,"SF_Mono","JetBrains_Mono","IBM_Plex_Mono",Menlo,Consolas,monospace]';

async function parseErrorDetail(res) {
  try {
    const body = await res.json();
    return body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

function fmtDollar(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return "$" + n.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function fmtPct(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function fmtDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-TW", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export default function TradeJournalPanel({ userId, onClose }) {
  const [trades, setTrades] = useState([]);
  const [plans, setPlans] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [draft, setDraft] = useState({
    ticker: "",
    strategyType: "",
    entryPrice: "",
    positionSize: "1",
    notes: "",
    sourcePlanId: "",
  });
  const [creating, setCreating] = useState(false);

  const [closingId, setClosingId] = useState(null);
  const [closeDraft, setCloseDraft] = useState({ exitPrice: "", pnl: "" });
  const [closing, setClosing] = useState(false);

  const [reviews, setReviews] = useState({});
  const [reviewingId, setReviewingId] = useState(null);

  async function loadTrades() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BASE_URL}/api/v1/trades?user_id=${userId}`);
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const data = await res.json();
      setTrades(data.trades || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadPlans() {
    try {
      const res = await fetch(`${BASE_URL}/api/v1/plans?user_id=${userId}`);
      if (!res.ok) return;
      const data = await res.json();
      setPlans(data.plans || []);
    } catch {
      // Optional dropdown data — a failed fetch just leaves it empty.
    }
  }

  useEffect(() => {
    loadTrades();
    loadPlans();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  async function createTrade() {
    if (!draft.ticker.trim() || !draft.strategyType.trim() || !draft.entryPrice) return;
    setCreating(true);
    setError(null);
    try {
      const res = await fetch(`${BASE_URL}/api/v1/trades`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          ticker: draft.ticker.trim().toUpperCase(),
          strategy_type: draft.strategyType.trim(),
          entry_price: Number(draft.entryPrice),
          position_size: Number(draft.positionSize) || 1,
          notes: draft.notes.trim() || null,
          source_plan_id: draft.sourcePlanId || null,
          days_to_expiration: 30,
        }),
      });
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const trade = await res.json();
      setTrades((t) => [trade, ...t]);
      setDraft({
        ticker: "",
        strategyType: "",
        entryPrice: "",
        positionSize: "1",
        notes: "",
        sourcePlanId: "",
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setCreating(false);
    }
  }

  async function closeTrade(tradeId) {
    if (!closeDraft.exitPrice || closeDraft.pnl === "") return;
    setClosing(true);
    setError(null);
    try {
      const res = await fetch(
        `${BASE_URL}/api/v1/trades/${tradeId}?user_id=${userId}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            exit_price: Number(closeDraft.exitPrice),
            exit_date: new Date().toISOString(),
            pnl: Number(closeDraft.pnl),
          }),
        }
      );
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const updated = await res.json();
      setTrades((ts) => ts.map((t) => (t.id === updated.id ? updated : t)));
      setClosingId(null);
      setCloseDraft({ exitPrice: "", pnl: "" });
    } catch (err) {
      setError(err.message);
    } finally {
      setClosing(false);
    }
  }

  async function triggerReview(tradeId) {
    setReviewingId(tradeId);
    setError(null);
    try {
      const res = await fetch(
        `${BASE_URL}/api/v1/trades/${tradeId}/review?user_id=${userId}`,
        { method: "POST" }
      );
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const review = await res.json();
      setReviews((r) => ({ ...r, [tradeId]: review }));
    } catch (err) {
      setError(err.message);
    } finally {
      setReviewingId(null);
    }
  }

  const openTrades = trades.filter((t) => t.status === "OPEN");
  const closedTrades = trades.filter((t) => t.status === "CLOSED");

  return (
    <div className="absolute inset-x-0 top-11 bottom-0 z-30 bg-[#121214] border-t border-[rgba(240,237,229,.09)] flex flex-col">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[rgba(240,237,229,.09)]">
        <span className="text-[10px] tracking-wider uppercase text-[#8d8d93] font-semibold">
          交易日誌
        </span>
        <button type="button" onClick={onClose} className="text-[#8d8d93] hover:text-[#f0ede5]">
          <X size={14} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4">
        {error && <div className="text-[11px] text-[#d8622b]">{error}</div>}

        {/* Quick-add card */}
        <div className="border border-[rgba(240,237,229,.09)] rounded-md p-3 flex flex-col gap-2">
          <div className="text-[10px] tracking-wider uppercase text-[#57575c]">新增交易</div>
          <div className="flex gap-2">
            <input
              value={draft.ticker}
              onChange={(e) => setDraft((d) => ({ ...d, ticker: e.target.value }))}
              placeholder="代號"
              className={`w-20 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
            <input
              value={draft.strategyType}
              onChange={(e) => setDraft((d) => ({ ...d, strategyType: e.target.value }))}
              placeholder="策略類型（例如 Long Call）"
              className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
          </div>
          <div className="flex gap-2">
            <input
              value={draft.entryPrice}
              onChange={(e) => setDraft((d) => ({ ...d, entryPrice: e.target.value }))}
              placeholder="進場價"
              inputMode="decimal"
              className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
            <input
              value={draft.positionSize}
              onChange={(e) => setDraft((d) => ({ ...d, positionSize: e.target.value }))}
              placeholder="口數"
              inputMode="numeric"
              className={`w-16 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
          </div>
          {plans.length > 0 && (
            <select
              value={draft.sourcePlanId}
              onChange={(e) => setDraft((d) => ({ ...d, sourcePlanId: e.target.value }))}
              className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            >
              <option value="">（可選）連結已儲存的交易計畫</option>
              {plans.map((p) => (
                <option key={p.plan_id} value={p.plan_id}>
                  {p.ticker} {p.strategy_type} E{p.entry_price}
                </option>
              ))}
            </select>
          )}
          <textarea
            value={draft.notes}
            onChange={(e) => setDraft((d) => ({ ...d, notes: e.target.value }))}
            placeholder="備註（選填）"
            rows={2}
            className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] resize-none ${MONO}`}
          />
          <button
            type="button"
            onClick={createTrade}
            disabled={creating}
            className="flex items-center justify-center gap-1.5 py-2 rounded-md bg-[#c9a15c] text-[#1a1408] text-[11.5px] font-bold uppercase tracking-wide hover:bg-[#d8b06c] disabled:opacity-50 transition-colors"
          >
            <Plus size={13} />
            {creating ? "建立中…" : "新增交易"}
          </button>
        </div>

        {loading && <div className="text-[11px] text-[#57575c] text-center">載入中…</div>}

        {/* Open trades */}
        <div>
          <div className="text-[10px] tracking-wider uppercase text-[#57575c] mb-2">
            持倉中（{openTrades.length}）
          </div>
          <div className="flex flex-col gap-2">
            {openTrades.length === 0 && (
              <div className="text-[11px] text-[#57575c]">尚無持倉中交易</div>
            )}
            {openTrades.map((t) => (
              <div
                key={t.id}
                className="border border-[rgba(240,237,229,.09)] rounded-md bg-[#1b1b1e] p-3"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11.5px] font-bold text-[#f0ede5]">
                    {t.ticker} · {t.strategy_type}
                  </span>
                  <span className="text-[9.5px] text-[#57575c]">{fmtDateTime(t.entry_date)}</span>
                </div>
                <div className="text-[10.5px] text-[#8d8d93] mb-2">
                  進場 {fmtDollar(t.entry_price)} × {t.position_size}
                </div>
                {closingId === t.id ? (
                  <div className="flex flex-col gap-1.5">
                    <div className="flex gap-1.5">
                      <input
                        value={closeDraft.exitPrice}
                        onChange={(e) =>
                          setCloseDraft((d) => ({ ...d, exitPrice: e.target.value }))
                        }
                        placeholder="出場價"
                        inputMode="decimal"
                        className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                      />
                      <input
                        value={closeDraft.pnl}
                        onChange={(e) => setCloseDraft((d) => ({ ...d, pnl: e.target.value }))}
                        placeholder="損益（$）"
                        inputMode="decimal"
                        className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                      />
                    </div>
                    <div className="flex gap-1.5">
                      <button
                        type="button"
                        onClick={() => closeTrade(t.id)}
                        disabled={closing}
                        className="flex-1 py-1.5 rounded bg-[#c9a15c] text-[#1a1408] text-[10.5px] font-bold disabled:opacity-50"
                      >
                        {closing ? "平倉中…" : "確認平倉"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setClosingId(null)}
                        className="px-3 py-1.5 rounded border border-[rgba(240,237,229,.09)] text-[#8d8d93] text-[10.5px]"
                      >
                        取消
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      setClosingId(t.id);
                      setCloseDraft({ exitPrice: "", pnl: "" });
                    }}
                    className="w-full py-1.5 rounded border border-[rgba(240,237,229,.16)] text-[#8d8d93] text-[10.5px] font-semibold hover:text-[#f0ede5] hover:border-[rgba(240,237,229,.28)] transition-colors"
                  >
                    平倉
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Closed trades */}
        <div>
          <div className="text-[10px] tracking-wider uppercase text-[#57575c] mb-2">
            已平倉（{closedTrades.length}）
          </div>
          <div className="flex flex-col gap-2">
            {closedTrades.length === 0 && (
              <div className="text-[11px] text-[#57575c]">尚無已平倉交易</div>
            )}
            {closedTrades.map((t) => {
              const review = reviews[t.id];
              const isPositive = (t.pnl_pct ?? 0) >= 0;
              return (
                <div
                  key={t.id}
                  className="border border-[rgba(240,237,229,.09)] rounded-md bg-[#1b1b1e] p-3"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[11.5px] font-bold text-[#f0ede5]">
                      {t.ticker} · {t.strategy_type}
                    </span>
                    <span
                      className={`text-[11px] font-bold ${MONO}`}
                      style={{ color: isPositive ? "#2fa37a" : "#d8622b" }}
                    >
                      {fmtPct(t.pnl_pct)}
                    </span>
                  </div>
                  <div className="text-[10.5px] text-[#8d8d93] mb-2">
                    {fmtDollar(t.entry_price)} → {fmtDollar(t.exit_price)} · 損益{" "}
                    {fmtDollar(t.pnl)}
                  </div>

                  {review ? (
                    <div className="border-t border-[rgba(240,237,229,.09)] pt-2 mt-1 flex flex-col gap-1.5">
                      <div className="flex items-center gap-0.5">
                        {Array.from({ length: 5 }).map((_, i) => (
                          <Star
                            key={i}
                            size={12}
                            fill={i < review.execution_score ? "#c9a15c" : "none"}
                            color="#c9a15c"
                          />
                        ))}
                      </div>
                      <div className="text-[11px] text-[#f0ede5] leading-relaxed">
                        {review.ai_feedback}
                      </div>
                      <ul className="list-disc pl-4 flex flex-col gap-0.5">
                        {review.key_takeaways.map((k, i) => (
                          <li key={i} className="text-[10.5px] text-[#8d8d93]">
                            {k}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => triggerReview(t.id)}
                      disabled={reviewingId === t.id}
                      className="w-full py-1.5 rounded border border-[rgba(201,161,92,.35)] bg-[rgba(201,161,92,.08)] text-[#c9a15c] text-[10.5px] font-semibold hover:bg-[rgba(201,161,92,.18)] disabled:opacity-50 transition-colors"
                    >
                      {reviewingId === t.id ? "分析中…" : "🤖 觸發 AI 覆盤分析"}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire the panel into `TradingTerminalNotebook.jsx`**

Add `BookOpen` to the `lucide-react` import list at the top of the file:

```jsx
import {
  Activity,
  AlertTriangle,
  BookOpen,
  Calendar,
  CheckCircle2,
  History,
  Mic,
  Plus,
  PenTool,
  Send,
  Settings,
  ShieldAlert,
  Smile,
  Target,
  Wifi,
  X,
} from "lucide-react";
```

Add the import for the new component right after the other local imports at the top (there are none currently besides `lucide-react` and `react` — add this as a new line just below the `lucide-react` import block):

```jsx
import TradeJournalPanel from "./TradeJournal.jsx";
```

Add a new state variable next to `showProfile` (around where `const [showProfile, setShowProfile] = useState(false);` is):

```jsx
  const [showJournal, setShowJournal] = useState(false);
```

In the chat section's header `trailing` block, add a fourth icon button after the existing History/Settings buttons (inside the same `<div className="flex items-center gap-1">` that wraps them):

```jsx
                  <button
                    type="button"
                    onClick={() => {
                      setShowJournal((v) => !v);
                      setShowHistory(false);
                      setShowProfile(false);
                    }}
                    title="交易日誌"
                    className={`w-6 h-6 flex items-center justify-center rounded transition-colors ${
                      showJournal ? "text-[#c9a15c] bg-[rgba(201,161,92,.13)]" : "text-[#8d8d93] hover:text-[#f0ede5] hover:bg-[rgba(240,237,229,.08)]"
                    }`}
                  >
                    <BookOpen size={13} />
                  </button>
```

Also make the existing History and Settings button handlers close the journal panel too, so only one overlay shows at a time — change:

```jsx
                    onClick={() => {
                      setShowHistory((v) => !v);
                      setShowProfile(false);
                    }}
```
to:
```jsx
                    onClick={() => {
                      setShowHistory((v) => !v);
                      setShowProfile(false);
                      setShowJournal(false);
                    }}
```
and:
```jsx
                    onClick={() => {
                      setShowProfile((v) => !v);
                      setShowHistory(false);
                    }}
```
to:
```jsx
                    onClick={() => {
                      setShowProfile((v) => !v);
                      setShowHistory(false);
                      setShowJournal(false);
                    }}
```

Finally, render the panel right after the existing `{showProfile && (...)}` block:

```jsx
          {showJournal && (
            <TradeJournalPanel userId={userId} onClose={() => setShowJournal(false)} />
          )}
```

- [ ] **Step 3: Manual smoke test**

Run: `cd "/Users/andrewweng/Desktop/stock schedule" && lsof -ti:8002 | xargs -r kill -9; .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8002 &`
Run: `cd "/Users/andrewweng/Desktop/stock schedule/web" && npm run dev`

Open `http://127.0.0.1:5173`, click the AI Copilot pane's new 📔 (BookOpen) icon, and confirm:
1. The panel opens showing "尚無持倉中交易" / "尚無已平倉交易".
2. Fill in ticker/strategy/entry price/size, click "新增交易" — a new OPEN card appears.
3. Click "平倉" on that card, fill in exit price and P/L, confirm — the card moves to the CLOSED list showing the correct PnL%.
4. Click "🤖 觸發 AI 覆盤分析" on the closed card — after a short wait, a star rating, AI prose, and key takeaways appear.
5. Refresh the page and reopen the panel — the trade and its review persist (confirms the backend round-trip, not just local state).

Expected: All five behaviors work as described. If step 4 fails with a 503, check that `OPENAI_API_KEY` is set in `.env` and the backend was restarted after Task 4's changes.

- [ ] **Step 4: Commit**

```bash
git add web/src/TradeJournal.jsx web/src/TradingTerminalNotebook.jsx
git commit -m "Add Trade Journal panel with AI post-trade review UI"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the spec's Data Model section in full. Task 2 covers the Execution Score section's both formulas exactly (verified by hand against every parametrized test case). Task 3 covers the AI review-generation half of the spec's API section (the `submit_trade_review` tool contract). Task 4 covers all four endpoints and their exact status-code mapping (404/403/409/400) as specified. Task 5 covers the Frontend section's three UI requirements (quick-add, close-with-pnl_pct, review button + star/prose/takeaways display). git2threads is explicitly excluded, matching the spec's Out of Scope section.
- **Placeholder scan:** No TBDs; every step has runnable code. The one deliberately-flagged ambiguity (`entry_snapshot` typed as `Any | None` in Task 3 rather than importing `GEXSnapshot` as a strict type) is called out with a clear justification and an explicit note on what to preserve if changed, not left vague.
- **Type consistency:** `compute_execution_score`'s signature (Task 2) matches its call site in Task 4's `review_trade` endpoint exactly (`entry_price`, `exit_price`, `pnl_pct`, `stop_loss`, `target_price`, all keyword args). `LLMOrchestrator.review_trade`'s signature (Task 3) matches its call site in Task 4 exactly (`trade`, `entry_snapshot`, `execution_score`, positional). `TradeRepository`/`TradeReviewRepository` method names and return types (Task 1) match every call site in Task 4.

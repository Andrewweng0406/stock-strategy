from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import (
    Base,
    ChatRepository,
    GEXSnapshotRepository,
    PlanRepository,
    ProfileRepository,
    TradeRepository,
    TradeReviewRepository,
    relax_gex_snapshot_level_columns,
)
from app.models import (
    GEXStatus,
    OptionGEXSummary,
    PlanStatus,
    TradeClose,
    TradeCreate,
    TradeStatus,
    UserProfileUpdate,
    UserTradePlan,
)


async def _session_factory() -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def gex_summary(stock_price: float = 100.0) -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker="AAPL",
        stock_price=stock_price,
        zero_gamma=95.0,
        call_wall=110.0,
        put_wall=90.0,
        iv_rank=40.0,
        net_gex=1_000_000.0,
        gex_status=GEXStatus.POS_GAMMA,
    )


@pytest.mark.asyncio
async def test_chat_repository_groups_messages_into_conversations() -> None:
    repo = ChatRepository(await _session_factory())
    await repo.save_message("conv-1", "user-1", "AAPL", "user", "hi")
    await repo.save_message("conv-1", "user-1", "AAPL", "assistant", "hello")
    await repo.save_message("conv-2", "user-1", "TSLA", "user", "other convo")

    conversations = await repo.list_conversations("user-1")
    assert len(conversations) == 2
    by_id = {c.conversation_id: c for c in conversations}
    assert by_id["conv-1"].message_count == 2
    assert by_id["conv-1"].last_message == "hello"
    assert by_id["conv-2"].ticker == "TSLA"

    messages = await repo.get_messages("conv-1", "user-1")
    assert [m.role for m in messages.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_chat_repository_scopes_conversations_by_user() -> None:
    repo = ChatRepository(await _session_factory())
    await repo.save_message("conv-1", "user-1", "AAPL", "user", "hi")

    other_user_messages = await repo.get_messages("conv-1", "user-2")
    assert other_user_messages.messages == []


@pytest.mark.asyncio
async def test_profile_repository_returns_none_until_upserted() -> None:
    repo = ProfileRepository(await _session_factory())
    assert await repo.get_profile("user-1") is None

    saved = await repo.upsert_profile(
        "user-1",
        UserProfileUpdate(
            risk_tolerance="AGGRESSIVE",
            preferred_strategy_types=["Long Put", "Bear Put Spread"],
            notes="likes short-dated plays",
        ),
    )
    assert saved.risk_tolerance == "AGGRESSIVE"
    assert saved.preferred_strategy_types == ["Long Put", "Bear Put Spread"]

    fetched = await repo.get_profile("user-1")
    assert fetched == saved


@pytest.mark.asyncio
async def test_profile_repository_upsert_overwrites_existing() -> None:
    repo = ProfileRepository(await _session_factory())
    await repo.upsert_profile(
        "user-1", UserProfileUpdate(risk_tolerance="CONSERVATIVE")
    )
    updated = await repo.upsert_profile(
        "user-1", UserProfileUpdate(risk_tolerance="BALANCED")
    )
    assert updated.risk_tolerance == "BALANCED"
    assert await repo.get_profile("user-1") == updated


@pytest.mark.asyncio
async def test_gex_snapshot_repository_save_and_list() -> None:
    repo = GEXSnapshotRepository(await _session_factory())
    assert await repo.last_snapshot_time("AAPL") is None

    await repo.save_snapshot("AAPL", 30, gex_summary(stock_price=100.0))
    await repo.save_snapshot("AAPL", 30, gex_summary(stock_price=101.0))
    await repo.save_snapshot("TSLA", 30, gex_summary(stock_price=250.0))

    snapshots = await repo.list_snapshots("AAPL")
    assert len(snapshots) == 2
    # Most recent first.
    assert snapshots[0].underlying_price == 101.0
    assert snapshots[1].underlying_price == 100.0
    assert all(s.ticker == "AAPL" for s in snapshots)

    last_time = await repo.last_snapshot_time("AAPL")
    assert isinstance(last_time, datetime)
    assert last_time.tzinfo is not None


@pytest.mark.asyncio
async def test_gex_snapshot_repository_list_respects_limit() -> None:
    repo = GEXSnapshotRepository(await _session_factory())
    for price in (100.0, 101.0, 102.0):
        await repo.save_snapshot("AAPL", 30, gex_summary(stock_price=price))

    snapshots = await repo.list_snapshots("AAPL", limit=1)
    assert len(snapshots) == 1
    assert snapshots[0].underlying_price == 102.0


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

    fetched = await repo.get_plan(str(plan.plan_id), "user-1")
    assert fetched is not None
    assert fetched.ticker == "AAPL"
    assert fetched.status == PlanStatus.SIGNED


@pytest.mark.asyncio
async def test_plan_repository_get_plan_returns_none_when_missing() -> None:
    repo = PlanRepository(await _session_factory())
    assert await repo.get_plan(str(uuid4()), "user-1") is None


@pytest.mark.asyncio
async def test_plan_repository_get_plan_returns_none_for_wrong_user() -> None:
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

    fetched = await repo.get_plan(str(plan.plan_id), "someone-else")
    assert fetched is None


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
async def test_trade_repository_create_defaults_entry_date_to_now() -> None:
    repo = TradeRepository(await _session_factory())
    before = datetime.now(timezone.utc)
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            entry_price=100.0, position_size=1, days_to_expiration=30,
        ),
        entry_gex_snapshot_id=None,
    )
    after = datetime.now(timezone.utc)
    assert before <= trade.entry_date <= after


@pytest.mark.asyncio
async def test_trade_repository_create_honors_explicit_entry_date() -> None:
    repo = TradeRepository(await _session_factory())
    backdated = datetime(2026, 8, 1, 14, 30, tzinfo=timezone.utc)
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            entry_price=100.0, position_size=1, days_to_expiration=30,
            entry_date=backdated,
        ),
        entry_gex_snapshot_id=None,
    )
    assert trade.entry_date == backdated


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
            pnl=2000.0,
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


LEGACY_GEX_SNAPSHOTS_DDL = """
CREATE TABLE gex_snapshots (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(32) NOT NULL,
    days_to_expiration INTEGER NOT NULL,
    captured_at DATETIME NOT NULL,
    underlying_price FLOAT NOT NULL,
    zero_gamma_strike FLOAT NOT NULL,
    call_wall_strike FLOAT NOT NULL,
    put_wall_strike FLOAT NOT NULL,
    net_gex FLOAT NOT NULL,
    iv_rank FLOAT NOT NULL,
    gex_status VARCHAR(16) NOT NULL
)
"""


@pytest.mark.asyncio
async def test_migration_relaxes_legacy_not_null_gex_level_columns() -> None:
    """A database created before the GEX levels became nullable keeps its
    NOT NULL constraints (create_all never alters an existing table), which
    would reject a snapshot of a chain with no gamma crossing.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.exec_driver_sql(LEGACY_GEX_SNAPSHOTS_DDL)
        await connection.exec_driver_sql(
            "CREATE INDEX ix_gex_snapshots_ticker ON gex_snapshots (ticker)"
        )
        await connection.exec_driver_sql(
            "INSERT INTO gex_snapshots VALUES "
            "(1, 'AAPL', 30, '2026-08-01 00:00:00', 100.0, 95.0, 110.0, 90.0, "
            "1000000.0, 40.0, 'POS_GAMMA')"
        )
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    repo = GEXSnapshotRepository(session_factory)

    summary = gex_summary().model_copy(
        update={"zero_gamma": None, "call_wall": None, "put_wall": None}
    )
    with pytest.raises(Exception):
        await repo.save_snapshot("AAPL", 30, summary)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(relax_gex_snapshot_level_columns)
        # Idempotent: a second run on an already-relaxed table is a no-op.
        await connection.run_sync(relax_gex_snapshot_level_columns)

    snapshot_id = await repo.save_snapshot("AAPL", 30, summary)
    fetched = await repo.get_snapshot(snapshot_id)
    assert fetched is not None
    assert fetched.zero_gamma_strike is None
    assert fetched.call_wall_strike is None
    assert fetched.put_wall_strike is None

    # The pre-existing row survived the rebuild intact.
    preserved = await repo.get_snapshot(1)
    assert preserved is not None
    assert preserved.zero_gamma_strike == 95.0

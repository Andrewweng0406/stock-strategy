from datetime import date, datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app import database
from app.database import (
    Base,
    ChatRepository,
    GEXSnapshotRepository,
    PlanRepository,
    PnlMismatchError,
    ProfileRepository,
    TradeRepository,
    TradeReviewRepository,
    ensure_trade_metadata_columns,
    relax_gex_snapshot_level_columns,
)
from app.models import (
    GEXStatus,
    OptionGEXSummary,
    PlanStatus,
    TradeClose,
    TradeCreate,
    TradeCreditDebit,
    TradeDirection,
    TradeStatus,
    UserProfileUpdate,
    UserTradePlan,
)


_ENGINES = []


@pytest_asyncio.fixture(autouse=True)
async def _dispose_async_engines():
    try:
        yield
    finally:
        while _ENGINES:
            await _ENGINES.pop().dispose()


async def _session_factory() -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    _ENGINES.append(engine)
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
            expiration_date=date(2099, 8, 14),
            option_type="CALL",
            strike_price=100.0,
            entry_price=100.0,
            position_size=1,
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
            expiration_date=date(2099, 8, 14),
            option_type="CALL", strike_price=100.0,
            entry_price=100.0, position_size=1,
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
            expiration_date=date(2099, 8, 14),
            option_type="CALL", strike_price=100.0,
            entry_price=100.0, position_size=1,
            entry_date=backdated,
        ),
        entry_gex_snapshot_id=None,
    )
    assert trade.entry_date == backdated


@pytest.mark.asyncio
async def test_trade_repository_create_persists_direction_and_credit_debit() -> None:
    repo = TradeRepository(await _session_factory())
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1",
            ticker="AAPL",
            strategy_type="Ratio Spread",
            direction=TradeDirection.NEUTRAL,
            credit_debit=TradeCreditDebit.CREDIT,
            expiration_date=date(2099, 8, 14),
            option_type="MULTI_LEG",
            legs=[
                {
                    "side": "BUY",
                    "option_type": "CALL",
                    "strike_price": 100.0,
                    "expiration_date": date(2099, 8, 14),
                    "quantity": 1,
                    "price": 2.0,
                },
                {
                    "side": "SELL",
                    "option_type": "CALL",
                    "strike_price": 105.0,
                    "expiration_date": date(2099, 8, 14),
                    "quantity": 2,
                    "price": 1.0,
                },
            ],
            entry_price=100.0,
            position_size=1,
        ),
        entry_gex_snapshot_id=None,
    )
    assert trade.direction == TradeDirection.NEUTRAL
    assert trade.credit_debit == TradeCreditDebit.CREDIT
    assert trade.expiration_date == date(2099, 8, 14)
    assert len(trade.legs) == 2
    assert trade.legs[0].side == "BUY"
    assert trade.legs[0].strike_price == 100.0
    assert trade.legs[0].quantity == 1
    assert trade.legs[0].price == 2.0
    assert trade.legs[1].side == "SELL"
    assert trade.legs[1].strike_price == 105.0
    assert trade.legs[1].quantity == 2
    assert trade.legs[1].price == 1.0


@pytest.mark.asyncio
async def test_trade_repository_list_filters_by_ticker_and_status() -> None:
    repo = TradeRepository(await _session_factory())
    await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            expiration_date=date(2099, 8, 14),
            option_type="CALL", strike_price=100.0,
            entry_price=100.0, position_size=1,
        ),
        entry_gex_snapshot_id=None,
    )
    await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="TSLA", strategy_type="Long Put",
            expiration_date=date(2099, 8, 21),
            option_type="PUT", strike_price=200.0,
            entry_price=200.0, position_size=1,
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
            expiration_date=date(2099, 8, 14),
            option_type="CALL", strike_price=100.0,
            entry_price=100.0, position_size=1,
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
            expiration_date=date(2099, 8, 14),
            option_type="CALL", strike_price=100.0,
            entry_price=100.0, position_size=1,
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
            expiration_date=date(2099, 8, 14),
            option_type="CALL", strike_price=100.0,
            entry_price=100.0, position_size=1,
        ),
        entry_gex_snapshot_id=None,
    )
    close = TradeClose(exit_price=120.0, exit_date=datetime.now(timezone.utc), pnl=2000.0)
    await repo.close_trade(str(trade.id), "user-1", close)
    with pytest.raises(ValueError):
        await repo.close_trade(str(trade.id), "user-1", close)


@pytest.mark.asyncio
async def test_close_trade_rejects_sign_flipped_pnl() -> None:
    """A winning long call closed with pnl submitted negative — the exact
    failure mode a client bug or a non-browser caller can produce with
    nothing else to catch it. entry 100 -> exit 120 on a DEBIT trade implies
    +2000, not -2000.
    """
    repo = TradeRepository(await _session_factory())
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            expiration_date=date(2099, 8, 14),
            option_type="CALL", strike_price=100.0,
            entry_price=100.0, position_size=1,
        ),
        entry_gex_snapshot_id=None,
    )
    with pytest.raises(PnlMismatchError):
        await repo.close_trade(
            str(trade.id),
            "user-1",
            TradeClose(
                exit_price=120.0, exit_date=datetime.now(timezone.utc), pnl=-2000.0
            ),
        )


@pytest.mark.asyncio
async def test_close_trade_accepts_pnl_within_realistic_slippage() -> None:
    """The manual pnl field stays authoritative for real-world slippage and
    commissions — only gross sign/scale disagreements get rejected.
    """
    repo = TradeRepository(await _session_factory())
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Long Call",
            expiration_date=date(2099, 8, 14),
            option_type="CALL", strike_price=100.0,
            entry_price=100.0, position_size=1,
        ),
        entry_gex_snapshot_id=None,
    )
    # Expected 2000.0; 1900.0 is a plausible slippage/commission haircut.
    closed = await repo.close_trade(
        str(trade.id),
        "user-1",
        TradeClose(
            exit_price=120.0, exit_date=datetime.now(timezone.utc), pnl=1900.0
        ),
    )
    assert closed.pnl == 1900.0


@pytest.mark.asyncio
async def test_close_trade_flips_expected_sign_for_credit_strategies() -> None:
    """A short (credit) trade's pnl formula is inverted from a long/debit
    one — entry is a credit collected, exit is a debit paid to close.
    """
    repo = TradeRepository(await _session_factory())
    trade = await repo.create_trade(
        TradeCreate(
            user_id="user-1", ticker="AAPL", strategy_type="Bull Put Credit Spread",
            direction=TradeDirection.LONG, credit_debit=TradeCreditDebit.CREDIT,
            expiration_date=date(2099, 8, 14),
            option_type="PUT", strike_price=100.0,
            entry_price=2.50, position_size=2,
        ),
        entry_gex_snapshot_id=None,
    )
    # Collected $2.50, bought back at $0.50 -> +$400, not -$400.
    with pytest.raises(PnlMismatchError):
        await repo.close_trade(
            str(trade.id),
            "user-1",
            TradeClose(
                exit_price=0.50, exit_date=datetime.now(timezone.utc), pnl=-400.0
            ),
        )
    closed = await repo.close_trade(
        str(trade.id),
        "user-1",
        TradeClose(
            exit_price=0.50, exit_date=datetime.now(timezone.utc), pnl=400.0
        ),
    )
    assert closed.pnl == 400.0


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

LEGACY_TRADES_DDL = """
CREATE TABLE trades (
    id VARCHAR(36) NOT NULL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    ticker VARCHAR(32) NOT NULL,
    strategy_type VARCHAR(128) NOT NULL,
    source_plan_id VARCHAR(36),
    entry_date DATETIME NOT NULL,
    exit_date DATETIME,
    entry_price FLOAT NOT NULL,
    exit_price FLOAT,
    position_size INTEGER NOT NULL,
    pnl FLOAT,
    pnl_pct FLOAT,
    status VARCHAR(16) NOT NULL,
    notes TEXT,
    entry_gex_snapshot_id INTEGER,
    created_at DATETIME NOT NULL
)
"""


def test_migration_adds_trade_direction_columns_and_backfills_existing_rows() -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql(LEGACY_TRADES_DDL)
        connection.exec_driver_sql(
            "INSERT INTO trades VALUES "
            "('trade-1', 'user-1', 'AAPL', 'Bull Put Credit Spread', NULL, "
            "'2026-08-01 00:00:00', NULL, 1.0, NULL, 1, NULL, NULL, 'OPEN', "
            "NULL, NULL, '2026-08-01 00:00:00')"
        )
        connection.exec_driver_sql(
            "INSERT INTO trades VALUES "
            "('trade-2', 'user-1', 'AAPL', 'Defined-Risk Iron Condor', NULL, "
            "'2026-08-01 00:00:00', NULL, 1.0, NULL, 1, NULL, NULL, 'OPEN', "
            "NULL, NULL, '2026-08-01 00:00:00')"
        )

        ensure_trade_metadata_columns(connection)
        ensure_trade_metadata_columns(connection)

        columns = {column["name"]: column for column in inspect(connection).get_columns("trades")}
        assert columns["direction"]["nullable"] is False
        assert columns["credit_debit"]["nullable"] is False
        assert columns["expiration_date"]["nullable"] is True
        assert columns["option_type"]["nullable"] is True
        assert columns["strike_price"]["nullable"] is True
        assert columns["contract_symbol"]["nullable"] is True
        assert columns["legs_json"]["nullable"] is True
        rows = connection.exec_driver_sql(
            "SELECT id, direction, credit_debit, expiration_date FROM trades ORDER BY id"
        ).all()

    assert rows == [
        # Bull Put Credit Spread is bullish despite containing "put" — the
        # backfill must check "bull" before the generic put/bear match.
        ("trade-1", "LONG", "CREDIT", None),
        ("trade-2", "NEUTRAL", "CREDIT", None),
    ]


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


# ---------- migration crash-safety (SQLite has autocommitting DDL) ----------

def _legacy_sqlite_engine(path, rows: int = 252):
    """A file-backed SQLite DB holding the pre-migration table, populated
    the way a real deployment's would be. File-backed on purpose: the whole
    hazard is that DDL autocommits to disk outside the transaction.
    """
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql(LEGACY_GEX_SNAPSHOTS_DDL)
        connection.exec_driver_sql(
            f"CREATE INDEX ix_gex_snapshots_ticker ON gex_snapshots (ticker)"
        )
        for index in range(rows):
            connection.exec_driver_sql(
                "INSERT INTO gex_snapshots VALUES "
                f"({index + 1}, 'AAPL', 30, '2026-08-01 00:00:00', 100.0, 95.0, "
                "110.0, 90.0, 1000000.0, 40.0, 'POS_GAMMA')"
            )
    return engine


def _table_names(engine) -> set[str]:
    with engine.connect() as connection:
        return set(inspect(connection).get_table_names())


def _count(engine, table: str) -> int:
    with engine.connect() as connection:
        return int(connection.exec_driver_sql(f"SELECT count(*) FROM {table}").scalar())


def test_migration_survives_a_crash_after_the_row_copy(tmp_path, monkeypatch) -> None:
    """The reviewer's scenario: crash after INSERT...SELECT succeeded but
    before the migration finished. Under pysqlite the RENAME/CREATE have
    already autocommitted while the INSERT rolls back, so the naive version
    of this left an EMPTY gex_snapshots table and no way back.
    """
    engine = _legacy_sqlite_engine(tmp_path / "legacy.db")

    monkeypatch.setattr(
        database, "_verify_copy",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated crash")),
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        with engine.begin() as connection:
            database.relax_gex_snapshot_level_columns(connection)

    # The pre-migration data still exists under a backup name; nothing dropped.
    backups = [n for n in _table_names(engine) if n.startswith("gex_snapshots_backup_")]
    assert len(backups) == 1
    assert _count(engine, backups[0]) == 252

    # Next startup resumes and completes, losing nothing.
    monkeypatch.undo()
    with engine.begin() as connection:
        database.relax_gex_snapshot_level_columns(connection)

    assert _count(engine, "gex_snapshots") == 252
    with engine.connect() as connection:
        columns = {c["name"]: c for c in inspect(connection).get_columns("gex_snapshots")}
    assert all(
        columns[name]["nullable"] for name in database._GEX_SNAPSHOT_LEVEL_COLUMNS
    )
    assert [n for n in _table_names(engine) if n.startswith("gex_snapshots_backup_")]


def test_migration_recovers_when_the_rebuilt_table_was_never_created(
    tmp_path, monkeypatch
) -> None:
    """The worst interleaving: the RENAME autocommitted, then the process
    died before the new table existed at all. `gex_snapshots` is simply gone.
    """
    engine = _legacy_sqlite_engine(tmp_path / "legacy.db")

    monkeypatch.setattr(
        database, "_create_gex_snapshots_table",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated crash")),
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        with engine.begin() as connection:
            database.relax_gex_snapshot_level_columns(connection)
    assert "gex_snapshots" not in _table_names(engine)

    monkeypatch.undo()
    with engine.begin() as connection:
        database.relax_gex_snapshot_level_columns(connection)
    assert _count(engine, "gex_snapshots") == 252


def test_migration_raises_loudly_on_a_short_copy_instead_of_continuing(
    tmp_path, monkeypatch
) -> None:
    """A partial copy must never be accepted as a finished migration — that
    is the step that would make the loss permanent and silent.
    """
    engine = _legacy_sqlite_engine(tmp_path / "legacy.db")

    def _copy_only_half(connection, source, target):
        connection.exec_driver_sql(
            f"INSERT INTO {target} SELECT * FROM {source} WHERE id <= 100"
        )

    monkeypatch.setattr(database, "_copy_rows", _copy_only_half)
    with pytest.raises(RuntimeError, match="copied 100 of 252 rows"):
        with engine.begin() as connection:
            database.relax_gex_snapshot_level_columns(connection)

    monkeypatch.undo()
    with engine.begin() as connection:
        database.relax_gex_snapshot_level_columns(connection)
    assert _count(engine, "gex_snapshots") == 252


def test_migration_leaves_a_completed_database_untouched(tmp_path) -> None:
    """Idempotence, including "don't resurrect rows from an old backup".
    A snapshot written after the migration must survive later startups.
    """
    engine = _legacy_sqlite_engine(tmp_path / "legacy.db")
    for _ in range(2):
        with engine.begin() as connection:
            database.relax_gex_snapshot_level_columns(connection)
    assert _count(engine, "gex_snapshots") == 252

    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO gex_snapshots VALUES "
            "(9999, 'MSFT', 30, '2026-08-02 00:00:00', 100.0, NULL, NULL, NULL, "
            "-5.0, 40.0, 'NEG_GAMMA')"
        )
    with engine.begin() as connection:
        database.relax_gex_snapshot_level_columns(connection)
    assert _count(engine, "gex_snapshots") == 253


def test_migration_does_not_resurrect_pruned_rows_on_a_later_startup(tmp_path) -> None:
    """The bug a row-count-based resume trigger has no way to avoid: once
    the migration has genuinely finished, ANY later legitimate shrinkage of
    the live table (a future retention/pruning policy, not corruption)
    must never be read as "the migration must have failed, restore from
    backup" — that would silently destroy every row deleted on purpose.
    """
    engine = _legacy_sqlite_engine(tmp_path / "legacy.db")
    with engine.begin() as connection:
        database.relax_gex_snapshot_level_columns(connection)
    assert _count(engine, "gex_snapshots") == 252

    # Simulate a retention policy pruning most of the table — nothing to do
    # with the migration, just normal application behavior after the fact.
    with engine.begin() as connection:
        connection.exec_driver_sql("DELETE FROM gex_snapshots WHERE id > 10")
    assert _count(engine, "gex_snapshots") == 10

    # A naive "live_rows < backup_rows -> resume" trigger would see 10 < 252
    # here and restore the full pre-migration backup, undoing the prune.
    with engine.begin() as connection:
        database.relax_gex_snapshot_level_columns(connection)
    assert _count(engine, "gex_snapshots") == 10


def test_migration_marks_a_pre_existing_completed_backup_done_without_recopying(
    tmp_path,
) -> None:
    """A database migrated before this fix landed has a backup that was
    never renamed with the done marker, even though its migration finished
    cleanly. The first startup under the new code must recognize that (live
    rows already cover the backup) and just mark it done — not treat it as
    unfinished and re-copy over rows the application may have already
    changed since.
    """
    engine = _legacy_sqlite_engine(tmp_path / "legacy.db")
    with engine.begin() as connection:
        database.relax_gex_snapshot_level_columns(connection)
    assert _count(engine, "gex_snapshots") == 252

    # Roll back the done marker to simulate a pre-fix backup: fully copied,
    # verified, just never renamed.
    backup = next(n for n in _table_names(engine) if n.startswith("gex_snapshots_backup_"))
    assert backup.endswith(database._GEX_SNAPSHOT_BACKUP_DONE_SUFFIX)
    undone = backup.removesuffix(database._GEX_SNAPSHOT_BACKUP_DONE_SUFFIX)
    with engine.begin() as connection:
        connection.exec_driver_sql(f"ALTER TABLE {backup} RENAME TO {undone}")

    # Real post-migration activity: a row the pre-fix backup doesn't have.
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO gex_snapshots VALUES "
            "(9999, 'MSFT', 30, '2026-08-02 00:00:00', 100.0, NULL, NULL, NULL, "
            "-5.0, 40.0, 'NEG_GAMMA')"
        )
    assert _count(engine, "gex_snapshots") == 253

    with engine.begin() as connection:
        database.relax_gex_snapshot_level_columns(connection)

    # The extra row survived — a re-copy from the (253-row-short) backup
    # would have wiped it.
    assert _count(engine, "gex_snapshots") == 253
    backups = [n for n in _table_names(engine) if n.startswith("gex_snapshots_backup_")]
    assert all(n.endswith(database._GEX_SNAPSHOT_BACKUP_DONE_SUFFIX) for n in backups)

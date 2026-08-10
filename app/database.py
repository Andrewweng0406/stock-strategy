import json
import logging
import time
from datetime import date, datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    desc,
    inspect,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models import (
    ChatMessageRecord,
    ConversationMessages,
    ConversationSummary,
    GEXSnapshot,
    GEXStatus,
    OptionGEXSummary,
    PlanStatus,
    Trade,
    TradeClose,
    TradeCreate,
    TradeCreditDebit,
    TradeDirection,
    TradeLeg,
    TradeOptionType,
    TradeReview,
    TradeStatus,
    UserProfile,
    UserProfileUpdate,
    UserTradePlan,
)


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite drops tzinfo on read even though every datetime column is
    declared timezone-aware (Postgres/asyncpg preserves it fine); every
    value is written via datetime.now(timezone.utc), so a naive read-back
    is always UTC.
    """
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class TradePlanRecord(Base):
    __tablename__ = "trade_plans"

    plan_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    conversation_id: Mapped[str] = mapped_column(String(128), index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    strategy_type: Mapped[str] = mapped_column(String(128))
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    max_loss_usd: Mapped[float] = mapped_column(Float)
    theta_warning: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(16), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    signed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class PlanRepository:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        self.session_factory = session_factory

    async def save_signed_plan(self, plan: UserTradePlan) -> UserTradePlan:
        signed_at = datetime.now(timezone.utc)
        signed = plan.model_copy(
            update={"status": PlanStatus.SIGNED, "signed_at": signed_at}
        )
        async with self.session_factory() as session:
            record = await session.scalar(
                select(TradePlanRecord).where(
                    TradePlanRecord.plan_id == str(plan.plan_id)
                )
            )
            if record and (
                record.user_id != plan.user_id
                or record.conversation_id != plan.conversation_id
            ):
                raise PermissionError(
                    "The plan belongs to a different user or conversation"
                )
            if record is None:
                record = TradePlanRecord(
                    plan_id=str(plan.plan_id),
                    user_id=plan.user_id,
                    conversation_id=plan.conversation_id,
                    ticker=plan.ticker,
                    strategy_type=plan.strategy_type,
                    entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss,
                    target_price=plan.target_price,
                    max_loss_usd=plan.max_loss_usd,
                    theta_warning=plan.theta_warning,
                    status=PlanStatus.DRAFT.value,
                    created_at=plan.created_at,
                )
                session.add(record)
            record.ticker = signed.ticker
            record.strategy_type = signed.strategy_type
            record.entry_price = signed.entry_price
            record.stop_loss = signed.stop_loss
            record.target_price = signed.target_price
            record.max_loss_usd = signed.max_loss_usd
            record.theta_warning = signed.theta_warning
            record.status = PlanStatus.SIGNED.value
            record.signed_at = signed_at
            await session.commit()
        return signed

    async def get_plan(self, plan_id: str, user_id: str) -> UserTradePlan | None:
        async with self.session_factory() as session:
            record = await session.scalar(
                select(TradePlanRecord).where(
                    TradePlanRecord.plan_id == plan_id,
                    TradePlanRecord.user_id == user_id,
                )
            )
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

    async def list_plans(self, user_id: str) -> list[UserTradePlan]:
        async with self.session_factory() as session:
            records = await session.scalars(
                select(TradePlanRecord)
                .where(
                    TradePlanRecord.user_id == user_id,
                    TradePlanRecord.status == PlanStatus.SIGNED.value,
                )
                .order_by(TradePlanRecord.signed_at.desc())
            )
            return [
                UserTradePlan(
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
                for record in records
            ]


class ChatMessageDBRecord(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    ticker: Mapped[str] = mapped_column(String(32))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ChatRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def save_message(
        self,
        conversation_id: str,
        user_id: str,
        ticker: str,
        role: str,
        content: str,
    ) -> None:
        async with self.session_factory() as session:
            session.add(
                ChatMessageDBRecord(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    ticker=ticker,
                    role=role,
                    content=content,
                    created_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    async def list_conversations(self, user_id: str) -> list[ConversationSummary]:
        # Grouped in Python rather than a GROUP BY/window-function query — at
        # personal-instance scale (hundreds, not millions, of rows) this is
        # simpler to read and just as fast, and avoids a query that behaves
        # differently across SQLite and Postgres.
        async with self.session_factory() as session:
            records = await session.scalars(
                select(ChatMessageDBRecord)
                .where(ChatMessageDBRecord.user_id == user_id)
                .order_by(ChatMessageDBRecord.created_at.asc())
            )
            grouped: dict[str, list[ChatMessageDBRecord]] = {}
            for record in records:
                grouped.setdefault(record.conversation_id, []).append(record)

        summaries = [
            ConversationSummary(
                conversation_id=conversation_id,
                ticker=messages[-1].ticker,
                last_message=messages[-1].content[:200],
                last_message_at=_as_utc(messages[-1].created_at),
                message_count=len(messages),
            )
            for conversation_id, messages in grouped.items()
        ]
        summaries.sort(key=lambda s: s.last_message_at, reverse=True)
        return summaries

    async def get_messages(
        self, conversation_id: str, user_id: str
    ) -> ConversationMessages:
        async with self.session_factory() as session:
            records = await session.scalars(
                select(ChatMessageDBRecord)
                .where(
                    ChatMessageDBRecord.conversation_id == conversation_id,
                    ChatMessageDBRecord.user_id == user_id,
                )
                .order_by(ChatMessageDBRecord.created_at.asc())
            )
            messages = [
                ChatMessageRecord(
                    role=record.role,
                    content=record.content,
                    created_at=_as_utc(record.created_at),
                )
                for record in records
            ]
        return ConversationMessages(
            conversation_id=conversation_id, messages=messages
        )


class UserProfileRecord(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    risk_tolerance: Mapped[str | None] = mapped_column(String(16), nullable=True)
    preferred_strategy_types: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ProfileRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    @staticmethod
    def _to_model(record: UserProfileRecord) -> UserProfile:
        return UserProfile(
            user_id=record.user_id,
            risk_tolerance=record.risk_tolerance,
            preferred_strategy_types=(
                record.preferred_strategy_types.split(",")
                if record.preferred_strategy_types
                else []
            ),
            notes=record.notes,
            updated_at=_as_utc(record.updated_at),
        )

    async def get_profile(self, user_id: str) -> UserProfile | None:
        async with self.session_factory() as session:
            record = await session.get(UserProfileRecord, user_id)
            return self._to_model(record) if record else None

    async def upsert_profile(
        self, user_id: str, update: UserProfileUpdate
    ) -> UserProfile:
        async with self.session_factory() as session:
            record = await session.get(UserProfileRecord, user_id)
            if record is None:
                record = UserProfileRecord(user_id=user_id)
                session.add(record)
            record.risk_tolerance = update.risk_tolerance
            record.preferred_strategy_types = ",".join(update.preferred_strategy_types)
            record.notes = update.notes
            record.updated_at = datetime.now(timezone.utc)
            await session.commit()
            return self._to_model(record)


class GEXSnapshotDBRecord(Base):
    __tablename__ = "gex_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    days_to_expiration: Mapped[int] = mapped_column(Integer)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    underlying_price: Mapped[float] = mapped_column(Float)
    # Nullable: a chain whose gamma profile never crosses zero has no
    # zero-gamma level, and a chain with no strike on one side of spot has
    # no wall on that side. See relax_gex_snapshot_level_columns() for how
    # databases created before this change are brought forward.
    zero_gamma_strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    call_wall_strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    put_wall_strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_gex: Mapped[float] = mapped_column(Float)
    iv_rank: Mapped[float] = mapped_column(Float)
    gex_status: Mapped[str] = mapped_column(String(16))


_GEX_SNAPSHOT_LEVEL_COLUMNS = (
    "zero_gamma_strike",
    "call_wall_strike",
    "put_wall_strike",
)
_GEX_SNAPSHOT_COLUMNS = (
    "id",
    "ticker",
    "days_to_expiration",
    "captured_at",
    "underlying_price",
    *_GEX_SNAPSHOT_LEVEL_COLUMNS,
    "net_gex",
    "iv_rank",
    "gex_status",
)


_GEX_SNAPSHOT_BACKUP_PREFIX = f"{GEXSnapshotDBRecord.__tablename__}_backup_"


def _row_count(connection, table: str) -> int:
    return int(connection.exec_driver_sql(f"SELECT count(*) FROM {table}").scalar())


def _copy_rows(connection, source: str, target: str) -> None:
    column_list = ", ".join(_GEX_SNAPSHOT_COLUMNS)
    connection.exec_driver_sql(
        f"INSERT INTO {target} ({column_list}) SELECT {column_list} FROM {source}"
    )


def _verify_copy(connection, source: str, target: str) -> None:
    """Refuse to continue on a short copy.

    Raising here leaves the backup table in place and the migration
    unfinished, which the next run detects and resumes. Continuing would be
    the one genuinely unrecoverable outcome.
    """
    expected = _row_count(connection, source)
    actual = _row_count(connection, target)
    if actual != expected:
        raise RuntimeError(
            f"Refusing to finish the {GEXSnapshotDBRecord.__tablename__} "
            f"migration: copied {actual} of {expected} rows. The original "
            f"data is intact in {source}; resolve before restarting."
        )


def _create_gex_snapshots_table(connection) -> None:
    GEXSnapshotDBRecord.__table__.create(connection)


def _resume_interrupted_migration(connection, table: str, backups: list[str]) -> None:
    """Re-copy from the newest backup if the live table is missing rows.

    Under pysqlite every DDL statement autocommits, escaping the enclosing
    transaction — so a crash partway through the rebuild leaves the rename
    and the CREATE committed while the INSERT...SELECT rolls back. That is
    exactly how the live table can end up empty next to a full backup, and
    it is why this runs before the nullability check below: by then the
    columns already look correct, so a plain "is it stale?" test would
    return early and call the data loss permanent.
    """
    backup = backups[-1]
    backup_rows = _row_count(connection, backup)
    live_exists = table in set(inspect(connection).get_table_names())
    live_rows = _row_count(connection, table) if live_exists else 0
    if live_exists and live_rows >= backup_rows:
        return

    logger.warning(
        "Resuming an interrupted %s migration: %s has %d row(s), backup %s "
        "has %d. Re-copying from the backup.",
        table, table, live_rows, backup, backup_rows,
    )
    if not live_exists:
        _create_gex_snapshots_table(connection)
    else:
        connection.exec_driver_sql(f"DELETE FROM {table}")
    _copy_rows(connection, backup, table)
    _verify_copy(connection, backup, table)
    logger.warning("Recovered %d row(s) into %s from %s", backup_rows, table, backup)


def relax_gex_snapshot_level_columns(connection) -> None:
    """Make the three GEX level columns nullable on a pre-existing table.

    Base.metadata.create_all only creates tables that don't exist yet, so a
    database created before the levels became nullable keeps its NOT NULL
    constraints and would reject a snapshot of a chain with no gamma
    crossing. Idempotent: does nothing once the columns already allow NULL.

    The SQLite path rebuilds the table, and SQLite's autocommitting DDL
    means that rebuild is NOT atomic no matter what transaction wraps it.
    So it is built to be crash-safe by construction rather than by
    transaction: the pre-migration table is renamed to a timestamped backup
    and never dropped, the row count is verified before the migration is
    considered finished, and an interrupted run is detected and resumed on
    the next call. Cleaning up an old {table}_backup_* is a deliberate
    human decision, not something this function will do for you.
    """
    table = GEXSnapshotDBRecord.__tablename__
    names = set(inspect(connection).get_table_names())

    backups = sorted(name for name in names if name.startswith(_GEX_SNAPSHOT_BACKUP_PREFIX))
    if backups:
        _resume_interrupted_migration(connection, table, backups)
        names = set(inspect(connection).get_table_names())

    if table not in names:
        return
    columns = {
        column["name"]: column for column in inspect(connection).get_columns(table)
    }
    stale = [
        name
        for name in _GEX_SNAPSHOT_LEVEL_COLUMNS
        if name in columns and not columns[name]["nullable"]
    ]
    if not stale:
        return

    if connection.dialect.name == "sqlite":
        # SQLite has no "ALTER COLUMN ... DROP NOT NULL"; the supported
        # route is a table rebuild. Dropping the old table takes its indexes
        # with it, so they're recreated with the new table under the same
        # names SQLAlchemy's index=True would have used.
        backup = f"{_GEX_SNAPSHOT_BACKUP_PREFIX}{int(time.time())}"
        logger.warning(
            "Rebuilding %s to relax NOT NULL on %s; the pre-migration table "
            "is preserved as %s and is not dropped.",
            table, ", ".join(stale), backup,
        )
        connection.exec_driver_sql(f"ALTER TABLE {table} RENAME TO {backup}")
        connection.exec_driver_sql(f"DROP INDEX IF EXISTS ix_{table}_ticker")
        connection.exec_driver_sql(f"DROP INDEX IF EXISTS ix_{table}_captured_at")
        _create_gex_snapshots_table(connection)
        _copy_rows(connection, backup, table)
        _verify_copy(connection, backup, table)
        return

    # Postgres has real transactional DDL, so this one is genuinely atomic.
    for name in stale:
        connection.exec_driver_sql(
            f"ALTER TABLE {table} ALTER COLUMN {name} DROP NOT NULL"
        )


class GEXSnapshotRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

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

    async def last_snapshot_time(self, ticker: str) -> datetime | None:
        async with self.session_factory() as session:
            captured_at = await session.scalar(
                select(GEXSnapshotDBRecord.captured_at)
                .where(GEXSnapshotDBRecord.ticker == ticker)
                .order_by(desc(GEXSnapshotDBRecord.captured_at))
                .limit(1)
            )
            # SQLite drops tzinfo on read even though the column is declared
            # timezone-aware (Postgres/asyncpg preserves it fine); every row
            # is written in UTC, so a naive value read back is always UTC.
            return _as_utc(captured_at)

    async def list_snapshots(self, ticker: str, limit: int = 100) -> list[GEXSnapshot]:
        async with self.session_factory() as session:
            records = await session.scalars(
                select(GEXSnapshotDBRecord)
                .where(GEXSnapshotDBRecord.ticker == ticker)
                .order_by(desc(GEXSnapshotDBRecord.captured_at))
                .limit(limit)
            )
            return [
                GEXSnapshot(
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
                for record in records
            ]

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


class TradeRecord(Base):
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    strategy_type: Mapped[str] = mapped_column(String(128))
    direction: Mapped[str] = mapped_column(String(16), default=TradeDirection.LONG.value)
    credit_debit: Mapped[str] = mapped_column(
        String(16), default=TradeCreditDebit.DEBIT.value
    )
    expiration_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    option_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    strike_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    contract_symbol: Mapped[str | None] = mapped_column(String(128), nullable=True)
    legs_json: Mapped[str | None] = mapped_column(Text, nullable=True)
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


def infer_trade_direction(strategy_type: str) -> TradeDirection:
    text = strategy_type.lower()
    if any(token in text for token in ("short", "sell", "covered call", "cash-secured")):
        return TradeDirection.SHORT
    if "iron condor" in text or "butterfly" in text or "calendar" in text:
        return TradeDirection.NEUTRAL
    if "bear" in text or "put" in text:
        return TradeDirection.SHORT
    return TradeDirection.LONG


def infer_trade_credit_debit(strategy_type: str) -> TradeCreditDebit:
    text = strategy_type.lower()
    if any(
        token in text
        for token in ("credit", "covered call", "cash-secured", "iron condor")
    ):
        return TradeCreditDebit.CREDIT
    return TradeCreditDebit.DEBIT


def ensure_trade_metadata_columns(connection) -> None:
    table = TradeRecord.__tablename__
    if table not in set(inspect(connection).get_table_names()):
        return

    columns = {
        column["name"]: column for column in inspect(connection).get_columns(table)
    }
    added_column = False
    added_direction_or_credit = False
    if "direction" not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {table} ADD COLUMN direction VARCHAR(16) "
            f"NOT NULL DEFAULT '{TradeDirection.LONG.value}'"
        )
        added_column = True
        added_direction_or_credit = True
    if "credit_debit" not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {table} ADD COLUMN credit_debit VARCHAR(16) "
            f"NOT NULL DEFAULT '{TradeCreditDebit.DEBIT.value}'"
        )
        added_column = True
        added_direction_or_credit = True
    if "expiration_date" not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {table} ADD COLUMN expiration_date DATE"
        )
        added_column = True
    if "option_type" not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {table} ADD COLUMN option_type VARCHAR(16)"
        )
        added_column = True
    if "strike_price" not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {table} ADD COLUMN strike_price FLOAT"
        )
        added_column = True
    if "contract_symbol" not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {table} ADD COLUMN contract_symbol VARCHAR(128)"
        )
        added_column = True
    if "legs_json" not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {table} ADD COLUMN legs_json TEXT"
        )
        added_column = True
    if not added_column:
        return
    if not added_direction_or_credit:
        return

    connection.exec_driver_sql(
        f"""
        UPDATE {table}
        SET
            direction = CASE
                WHEN lower(strategy_type) LIKE '%iron condor%'
                  OR lower(strategy_type) LIKE '%butterfly%'
                  OR lower(strategy_type) LIKE '%calendar%' THEN '{TradeDirection.NEUTRAL.value}'
                WHEN lower(strategy_type) LIKE '%short%'
                  OR lower(strategy_type) LIKE '%sell%'
                  OR lower(strategy_type) LIKE '%covered call%'
                  OR lower(strategy_type) LIKE '%cash-secured%'
                  OR lower(strategy_type) LIKE '%bear%'
                  OR lower(strategy_type) LIKE '%put%' THEN '{TradeDirection.SHORT.value}'
                ELSE '{TradeDirection.LONG.value}'
            END
        WHERE direction IS NULL OR direction = '{TradeDirection.LONG.value}'
        """
    )
    connection.exec_driver_sql(
        f"""
        UPDATE {table}
        SET credit_debit = CASE
            WHEN lower(strategy_type) LIKE '%credit%'
              OR lower(strategy_type) LIKE '%covered call%'
              OR lower(strategy_type) LIKE '%cash-secured%'
              OR lower(strategy_type) LIKE '%iron condor%' THEN '{TradeCreditDebit.CREDIT.value}'
            ELSE '{TradeCreditDebit.DEBIT.value}'
        END
        WHERE credit_debit IS NULL
           OR credit_debit = '{TradeCreditDebit.DEBIT.value}'
        """
    )


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
            direction=create.direction.value,
            credit_debit=create.credit_debit.value,
            expiration_date=create.expiration_date,
            option_type=create.option_type.value,
            strike_price=create.strike_price,
            contract_symbol=create.contract_symbol,
            legs_json=(
                json.dumps(
                    [leg.model_dump(mode="json") for leg in create.legs],
                    separators=(",", ":"),
                )
                if create.legs
                else None
            ),
            source_plan_id=(
                str(create.source_plan_id) if create.source_plan_id else None
            ),
            entry_date=create.entry_date or now,
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
                close.pnl / (record.entry_price * 100 * record.position_size)
            ) * 100
            record.status = TradeStatus.CLOSED.value
            if close.notes is not None:
                record.notes = close.notes
            await session.commit()
            return self._to_model(record)

    @staticmethod
    def _to_model(record: TradeRecord) -> Trade:
        legs: list[TradeLeg] = []
        if record.legs_json:
            try:
                raw_legs = json.loads(record.legs_json)
                if isinstance(raw_legs, list):
                    legs = [TradeLeg.model_validate(leg) for leg in raw_legs]
            except (TypeError, ValueError):
                logger.warning("Invalid trade legs JSON on trade %s", record.id)
        return Trade(
            id=record.id,
            user_id=record.user_id,
            ticker=record.ticker,
            strategy_type=record.strategy_type,
            direction=TradeDirection(record.direction),
            credit_debit=TradeCreditDebit(record.credit_debit),
            expiration_date=record.expiration_date,
            option_type=TradeOptionType(record.option_type) if record.option_type else None,
            strike_price=record.strike_price,
            contract_symbol=record.contract_symbol,
            legs=legs,
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

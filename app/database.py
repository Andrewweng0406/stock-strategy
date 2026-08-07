from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, desc, select
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
    UserProfile,
    UserProfileUpdate,
    UserTradePlan,
)


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
    zero_gamma_strike: Mapped[float] = mapped_column(Float)
    call_wall_strike: Mapped[float] = mapped_column(Float)
    put_wall_strike: Mapped[float] = mapped_column(Float)
    net_gex: Mapped[float] = mapped_column(Float)
    iv_rank: Mapped[float] = mapped_column(Float)
    gex_status: Mapped[str] = mapped_column(String(16))


class GEXSnapshotRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def save_snapshot(
        self, ticker: str, days_to_expiration: int, summary: OptionGEXSummary
    ) -> None:
        async with self.session_factory() as session:
            session.add(
                GEXSnapshotDBRecord(
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
            )
            await session.commit()

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

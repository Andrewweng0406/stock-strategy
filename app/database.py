from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models import (
    ChatMessageRecord,
    ConversationMessages,
    ConversationSummary,
    PlanStatus,
    UserProfile,
    UserProfileUpdate,
    UserTradePlan,
)


class Base(DeclarativeBase):
    pass


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
                    created_at=record.created_at,
                    signed_at=record.signed_at,
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
                last_message_at=messages[-1].created_at,
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
                    created_at=record.created_at,
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
            updated_at=record.updated_at,
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

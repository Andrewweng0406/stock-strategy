from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models import PlanStatus, UserTradePlan


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

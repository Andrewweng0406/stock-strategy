from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class GEXStatus(str, Enum):
    POS_GAMMA = "POS_GAMMA"
    NEG_GAMMA = "NEG_GAMMA"


class ExpirationType(str, Enum):
    ZERO_DTE = "0DTE"
    ONE_DTE = "1DTE"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class PlanStatus(str, Enum):
    DRAFT = "DRAFT"
    SIGNED = "SIGNED"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptionGEXSummary(StrictModel):
    ticker: str
    stock_price: float = Field(gt=0)
    zero_gamma: float = Field(gt=0)
    call_wall: float = Field(gt=0)
    put_wall: float = Field(gt=0)
    iv_rank: float = Field(ge=0, le=100)
    net_gex: float
    gex_status: GEXStatus
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class RiskProfile(StrictModel):
    gex_status: GEXStatus
    volatility_regime: Literal["LOW_VOL_MEAN_REVERSION", "HIGH_VOL_TRENDING"]
    risk_level: Literal["NORMAL", "HIGH"]
    warnings: list[str]
    locked_warning: bool


class UserTradePlan(StrictModel):
    plan_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=32)
    strategy_type: str = Field(min_length=1, max_length=128)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    target_price: float = Field(gt=0)
    max_loss_usd: float = Field(gt=0)
    theta_warning: bool
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    signed_at: datetime | None = None


class TradePlanToolArguments(StrictModel):
    ticker: str = Field(min_length=1, max_length=32)
    strategy_type: str = Field(min_length=1, max_length=128)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    target_price: float = Field(gt=0)
    max_loss_usd: float = Field(gt=0)
    theta_warning: bool


class ChatMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatContext(StrictModel):
    user_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=32)
    days_to_expiration: int = Field(ge=0, le=730)
    history: list[ChatMessage] = Field(default_factory=list, max_length=30)


class ChatRequest(StrictModel):
    user_message: str = Field(min_length=1, max_length=20_000)
    context: ChatContext


class ChatResponse(StrictModel):
    assistant_message: str
    gex_summary: OptionGEXSummary
    risk_profile: RiskProfile
    trade_plan_card: UserTradePlan | None = None


class SavePlanRequest(StrictModel):
    plan: UserTradePlan


class ExpirationInfo(StrictModel):
    date: date
    days_to_expiration: int = Field(ge=0)
    expiration_type: ExpirationType


class ExpirationList(StrictModel):
    ticker: str
    expirations: list[ExpirationInfo]


class SyncGexRequest(StrictModel):
    ticker: str = Field(min_length=1, max_length=32)
    days_to_expiration: int = Field(ge=0, le=730)
    summary: OptionGEXSummary


class HealthResponse(StrictModel):
    status: str
    market_data_mode: str


class ChatMessageRecord(StrictModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ConversationSummary(StrictModel):
    conversation_id: str
    ticker: str | None = None
    last_message: str
    last_message_at: datetime
    message_count: int


class ConversationList(StrictModel):
    conversations: list[ConversationSummary]


class ConversationMessages(StrictModel):
    conversation_id: str
    messages: list[ChatMessageRecord]


class PlanList(StrictModel):
    plans: list[UserTradePlan]


RiskTolerance = Literal["CONSERVATIVE", "BALANCED", "AGGRESSIVE"]


class UserProfile(StrictModel):
    user_id: str
    risk_tolerance: RiskTolerance | None = None
    preferred_strategy_types: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=2000)
    updated_at: datetime | None = None


class UserProfileUpdate(StrictModel):
    risk_tolerance: RiskTolerance | None = None
    preferred_strategy_types: list[str] = Field(default_factory=list, max_length=10)
    notes: str = Field(default="", max_length=2000)

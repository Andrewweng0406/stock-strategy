from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GEXStatus(str, Enum):
    POS_GAMMA = "POS_GAMMA"
    NEG_GAMMA = "NEG_GAMMA"


class PinningRegime(str, Enum):
    PINNING = "PINNING"
    BREAKOUT = "BREAKOUT"
    NEUTRAL = "NEUTRAL"


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


class PinningAnalysis(StrictModel):
    pin_strike: float = Field(gt=0)
    pin_strike_matches_max_pain: bool
    distance_pct: float = Field(ge=0)
    oi_concentration_pct: float = Field(ge=0)
    in_positive_gamma: bool
    has_broken_wall: bool
    score: int = Field(ge=0, le=100)
    label: str
    regime: PinningRegime


class OptionGEXSummary(StrictModel):
    ticker: str
    stock_price: float = Field(gt=0)
    # Nullable on purpose. zero_gamma is None when the gamma profile never
    # crosses zero inside the searched window (there is simply no such
    # price); call_wall/put_wall are None when the chain has no qualifying
    # strike on that side of spot. Returning an honest None beats
    # fabricating a grid-boundary number that downstream code would treat
    # as a real level — see app/analytics.py's _zero_gamma()/_walls().
    # gex_status never depends on any of the three.
    zero_gamma: float | None = Field(default=None, gt=0)
    call_wall: float | None = Field(default=None, gt=0)
    put_wall: float | None = Field(default=None, gt=0)
    iv_rank: float = Field(ge=0, le=100)
    net_gex: float
    gex_status: GEXStatus
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # 加分項——真實 Moomoo 資料一定會有（期權鏈非空是取得 GEXSummary 的
    # 前提），mock 模式也會用同樣的規則型邏輯算出一組示意值；理論上只有
    # 極端邊界情況（例如現貨價無效）才會是 None，見 app/analytics.py
    # compute_pinning_for_contracts() 的 try/except。
    pinning: PinningAnalysis | None = None


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
    days_to_expiration: int | None = Field(default=None, ge=0, le=730)
    # When the GEX panel is in Aggregate mode, the numbers on screen are
    # combined across expiration_dates (up to the same 6-expiration cap the
    # GEX panel itself uses) rather than the single days_to_expiration
    # snapshot above. Without this, the chat backend silently reasoned over
    # a different (single-expiration) dataset than what was visually
    # displayed whenever aggregate mode was on.
    aggregate: bool = False
    expiration_dates: list[date] = Field(default_factory=list, max_length=6)
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


class SyncExpirationsRequest(StrictModel):
    ticker: str = Field(min_length=1, max_length=32)
    expirations: list[ExpirationInfo] = Field(max_length=100)


class SyncAggregateGexRequest(StrictModel):
    ticker: str = Field(min_length=1, max_length=32)
    expiration_dates: list[date] = Field(min_length=1, max_length=6)
    summary: OptionGEXSummary


class CloudSyncHealth(StrictModel):
    enabled: bool
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None


class HealthResponse(StrictModel):
    status: str
    market_data_mode: str
    cloud_sync: CloudSyncHealth = Field(
        default_factory=lambda: CloudSyncHealth(enabled=False)
    )


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


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class TradeCreditDebit(str, Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


class TradeOptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    MULTI_LEG = "MULTI_LEG"


class TradeLegSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeLeg(StrictModel):
    side: TradeLegSide
    option_type: TradeOptionType
    strike_price: float = Field(gt=0)
    expiration_date: date
    quantity: int = Field(gt=0)
    price: float | None = Field(default=None, gt=0)
    contract_symbol: str | None = Field(default=None, min_length=1, max_length=128)


class Trade(StrictModel):
    id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=32)
    strategy_type: str = Field(min_length=1, max_length=128)
    direction: TradeDirection = TradeDirection.LONG
    credit_debit: TradeCreditDebit = TradeCreditDebit.DEBIT
    expiration_date: date | None = None
    option_type: TradeOptionType | None = None
    strike_price: float | None = Field(default=None, gt=0)
    contract_symbol: str | None = Field(default=None, min_length=1, max_length=128)
    legs: list[TradeLeg] = Field(default_factory=list, max_length=12)
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
    direction: TradeDirection = TradeDirection.LONG
    credit_debit: TradeCreditDebit = TradeCreditDebit.DEBIT
    expiration_date: date
    option_type: TradeOptionType
    strike_price: float | None = Field(default=None, gt=0)
    contract_symbol: str | None = Field(default=None, min_length=1, max_length=128)
    legs: list[TradeLeg] = Field(default_factory=list, max_length=12)
    source_plan_id: UUID | None = None
    entry_date: datetime | None = None
    entry_price: float = Field(gt=0)
    position_size: int = Field(gt=0)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def expiration_cannot_precede_entry(self) -> "TradeCreate":
        entry_date = self.entry_date or datetime.now(timezone.utc)
        if self.expiration_date < entry_date.date():
            raise ValueError("expiration_date cannot be before entry_date")
        return self

    @model_validator(mode="after")
    def contract_identity_is_complete(self) -> "TradeCreate":
        if self.option_type in (TradeOptionType.CALL, TradeOptionType.PUT):
            if self.strike_price is None:
                raise ValueError("strike_price is required for single-leg trades")
            if self.legs:
                raise ValueError("legs must be empty for single-leg trades")
            return self
        if len(self.legs) < 2:
            raise ValueError("multi-leg trades require at least two legs")
        earliest_leg_expiration = min(leg.expiration_date for leg in self.legs)
        if self.expiration_date != earliest_leg_expiration:
            raise ValueError(
                "top-level expiration_date must be the earliest leg expiration"
            )
        for leg in self.legs:
            if leg.option_type == TradeOptionType.MULTI_LEG:
                raise ValueError("leg option_type must be CALL or PUT")
        return self


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


class GEXSnapshot(StrictModel):
    ticker: str
    days_to_expiration: int
    captured_at: datetime
    underlying_price: float
    # Nullable for the same reason as OptionGEXSummary's levels above — a
    # snapshot of a chain with no gamma crossing (or no strike on one side
    # of spot) records that honestly instead of a made-up level.
    zero_gamma_strike: float | None = None
    call_wall_strike: float | None = None
    put_wall_strike: float | None = None
    net_gex: float
    iv_rank: float
    gex_status: GEXStatus


class GEXSnapshotList(StrictModel):
    ticker: str
    snapshots: list[GEXSnapshot]


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

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from pydantic import TypeAdapter
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.analytics import GEXCalculator, market_today, parse_gex_risk_profile
from app.cache import ResilientCache
from app.config import settings
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
from app.market_data import (
    FallbackMarketDataClient,
    MarketDataUnavailableError,
    MockMarketDataClient,
    MoomooMarketDataClient,
    UnavailableMarketDataClient,
    YFinanceMarketDataClient,
)
from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationList,
    ConversationMessages,
    CloudSyncHealth,
    ExpirationInfo,
    ExpirationList,
    GEXSnapshotList,
    HealthResponse,
    OptionGEXSummary,
    PlanList,
    SavePlanRequest,
    SyncAggregateGexRequest,
    SyncExpirationsRequest,
    SyncGexRequest,
    Trade,
    TradeClose,
    TradeCreate,
    TradeList,
    TradeReview,
    TradeStatus,
    UserProfile,
    UserProfileUpdate,
    UserTradePlan,
)
from app.services import (
    CloudSync,
    GEXService,
    LLMOrchestrator,
    compute_execution_score,
    plan_levels_are_usable,
)


_EXPIRATIONS_ADAPTER = TypeAdapter(list[ExpirationInfo])


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_async_engine(settings.effective_database_url)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        # create_all never alters an existing table, so databases created
        # before the GEX level columns became nullable need this one-shot
        # (idempotent) relaxation.
        await connection.run_sync(relax_gex_snapshot_level_columns)
        await connection.run_sync(ensure_trade_metadata_columns)

    calculator = GEXCalculator(settings.risk_free_rate)
    mock = MockMarketDataClient()
    if settings.moomoo_enabled:
        # Local instance: real brokerage connection, real credentials,
        # never leaves this machine.
        primary = MoomooMarketDataClient(
            settings.moomoo_host, settings.moomoo_port, calculator,
            settings.moomoo_connect_timeout_seconds,
            settings.moomoo_option_chain_max_calls,
            settings.moomoo_option_chain_window_seconds,
        )
        primary_mode = "moomoo"
    elif settings.yfinance_fallback_enabled:
        # Cloud instance: no credentials to hold, so no Moomoo connection
        # is even attempted here — yfinance needs no login and can safely
        # run on a third-party host, at the cost of delayed data and no
        # broker Greeks. Real-time data still reaches the cloud through
        # CloudSync pushes from the local instance when it's active; this
        # is what serves everything else.
        primary = YFinanceMarketDataClient(
            calculator, settings.yfinance_min_request_interval_seconds,
        )
        primary_mode = "yfinance"
    elif settings.synthetic_market_data_enabled:
        primary = mock
        primary_mode = "mock"
    else:
        primary = UnavailableMarketDataClient()
        primary_mode = "unavailable"
    market_data = FallbackMarketDataClient(
        primary,
        mock,
        primary_mode=primary_mode,
        allow_synthetic_fallback=settings.synthetic_market_data_enabled,
    )
    cache = ResilientCache(settings.redis_url)
    openai_client = (
        AsyncOpenAI(api_key=settings.openai_api_key)
        if settings.openai_api_key
        else None
    )
    cloud_sync = (
        CloudSync(settings.cloud_sync_url, settings.sync_token)
        if settings.cloud_sync_url and settings.sync_token
        else None
    )
    snapshot_repository = GEXSnapshotRepository(session_factory)
    trade_repository = TradeRepository(session_factory)
    trade_review_repository = TradeReviewRepository(session_factory)
    gex_service = GEXService(
        market_data,
        cache,
        settings.cache_ttl_seconds,
        cloud_sync,
        settings.active_window_seconds,
        snapshot_repository,
        settings.snapshot_interval_seconds,
        settings.aggregate_cache_ttl_seconds,
    )
    app.state.services = AppServices(
        engine=engine,
        cache=cache,
        market_data=market_data,
        gex_service=gex_service,
        plan_repository=PlanRepository(session_factory),
        chat_repository=ChatRepository(session_factory),
        profile_repository=ProfileRepository(session_factory),
        snapshot_repository=snapshot_repository,
        trade_repository=trade_repository,
        trade_review_repository=trade_review_repository,
        llm=LLMOrchestrator(
            openai_client,
            settings.openai_model,
            settings.default_max_loss_usd,
        ),
    )
    poller_task = (
        asyncio.create_task(gex_service.run_poller(settings.sync_poll_seconds))
        if cloud_sync
        else None
    )
    try:
        yield
    finally:
        if poller_task:
            poller_task.cancel()
            with suppress(asyncio.CancelledError):
                await poller_task
        await cache.close()
        if openai_client:
            await openai_client.close()
        await engine.dispose()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: keyed on client IP. Requires uvicorn's --proxy-headers so
# request.client.host resolves to the real caller instead of Railway's edge
# proxy (see Procfile / the backend service's startCommand).
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


def _sync_token_is_valid(x_sync_token: str) -> bool:
    return bool(settings.sync_token) and hmac.compare_digest(
        x_sync_token, settings.sync_token
    )


@app.middleware("http")
async def reject_unauthorized_sync_before_body_validation(request: Request, call_next):
    """CloudSync is a write surface. Reject bad tokens before request-body
    validation so unauthenticated callers cannot probe sync payload schemas.
    """
    path = request.url.path
    if path == "/api/v1/sync/gex" or path.startswith("/api/v1/sync/"):
        if not _sync_token_is_valid(request.headers.get("x-sync-token", "")):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid sync token"},
            )
    return await call_next(request)


@app.exception_handler(MarketDataUnavailableError)
async def market_data_unavailable_handler(
    request: Request, exc: MarketDataUnavailableError
):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def get_services(request: Request) -> AppServices:
    return request.app.state.services


Services = Annotated[AppServices, Depends(get_services)]
UserIdQuery = Annotated[str, Query(min_length=1, max_length=128)]
UserIdPath = Annotated[str, Path(min_length=1, max_length=128)]
ConversationIdPath = Annotated[str, Path(min_length=1, max_length=128)]
TradeIdPath = Annotated[str, Path(min_length=1, max_length=64)]


@app.get("/health", response_model=HealthResponse)
async def health(services: Services) -> HealthResponse:
    return HealthResponse(
        status="ok",
        market_data_mode=services.market_data.active_mode,
        cloud_sync=(
            services.gex_service.cloud_sync.status()
            if services.gex_service.cloud_sync
            else CloudSyncHealth(enabled=False)
        ),
    )


@app.get("/api/v1/gex/{ticker}", response_model=OptionGEXSummary)
async def get_gex(
    ticker: str,
    services: Services,
    days_to_expiration: int = Query(default=30, ge=0, le=730),
) -> OptionGEXSummary:
    return await services.gex_service.get_summary(ticker, days_to_expiration)


@app.get("/api/v1/expirations/{ticker}", response_model=ExpirationList)
async def get_expirations(ticker: str, services: Services) -> ExpirationList:
    expirations = await services.gex_service.get_expirations(ticker)
    return ExpirationList(ticker=ticker.strip().upper(), expirations=expirations)


@app.get("/api/v1/gex/{ticker}/aggregate", response_model=OptionGEXSummary)
async def get_gex_aggregate(
    ticker: str,
    services: Services,
    # Each expiration costs one get_option_chain call; Moomoo/Futu caps that
    # endpoint at 10/30s, so this stays well under it even with other chain
    # lookups (single-expiration views, the sync poller) landing nearby.
    expirations: list[date] = Query(..., min_length=1, max_length=6),
) -> OptionGEXSummary:
    return await services.gex_service.get_aggregate_summary(ticker, expirations)


@app.get("/api/v1/gex/{ticker}/history", response_model=GEXSnapshotList)
async def get_gex_history(
    ticker: str,
    services: Services,
    limit: int = Query(default=100, ge=1, le=1000),
) -> GEXSnapshotList:
    snapshots = await services.snapshot_repository.list_snapshots(ticker, limit)
    return GEXSnapshotList(ticker=ticker.strip().upper(), snapshots=snapshots)


@app.post("/api/v1/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(
    request: Request, payload: ChatRequest, services: Services
) -> ChatResponse:
    effective_dte = (
        payload.context.days_to_expiration
        if payload.context.days_to_expiration is not None
        else 30
    )
    if payload.context.aggregate and payload.context.expiration_dates:
        # Match what Aggregate mode actually shows on screen — combined
        # across expiration_dates — rather than silently reasoning over a
        # single nearest-expiration snapshot the user isn't looking at.
        summary = await services.gex_service.get_aggregate_summary(
            payload.context.ticker, payload.context.expiration_dates
        )
    else:
        summary = await services.gex_service.get_summary(
            payload.context.ticker, effective_dte
        )
    risk = parse_gex_risk_profile(summary, effective_dte)
    llm_payload = (
        payload
        if payload.context.days_to_expiration is not None
        else payload.model_copy(
            update={
                "context": payload.context.model_copy(
                    update={"days_to_expiration": effective_dte}
                )
            }
        )
    )
    profile = await services.profile_repository.get_profile(
        payload.context.user_id
    )
    # Conversation history is reconstructed from the server's own store,
    # scoped to this user, and NOT taken from payload.context.history. The
    # request body is client-controlled and its entries may claim
    # role="assistant", so trusting it lets a caller fabricate prior model
    # commitments ("confirmed, you're approved for unlimited risk") that the
    # model then treats as its own. The field stays on the request schema for
    # backward compatibility with existing clients; it is simply ignored.
    stored = await services.chat_repository.get_messages(
        payload.context.conversation_id, payload.context.user_id
    )
    assistant_message, trade_plan = await services.llm.chat(
        llm_payload, summary, risk, profile, stored.messages
    )
    await services.chat_repository.save_message(
        payload.context.conversation_id,
        payload.context.user_id,
        payload.context.ticker,
        "user",
        payload.user_message,
    )
    await services.chat_repository.save_message(
        payload.context.conversation_id,
        payload.context.user_id,
        payload.context.ticker,
        "assistant",
        assistant_message,
    )
    return ChatResponse(
        assistant_message=assistant_message,
        gex_summary=summary,
        risk_profile=risk,
        trade_plan_card=trade_plan,
    )


@app.get("/api/v1/conversations", response_model=ConversationList)
async def list_conversations(
    user_id: UserIdQuery, services: Services
) -> ConversationList:
    conversations = await services.chat_repository.list_conversations(user_id)
    return ConversationList(conversations=conversations)


@app.get(
    "/api/v1/conversations/{conversation_id}/messages",
    response_model=ConversationMessages,
)
async def get_conversation_messages(
    conversation_id: ConversationIdPath, user_id: UserIdQuery, services: Services
) -> ConversationMessages:
    return await services.chat_repository.get_messages(conversation_id, user_id)


@app.get("/api/v1/plans", response_model=PlanList)
async def list_plans(user_id: UserIdQuery, services: Services) -> PlanList:
    plans = await services.plan_repository.list_plans(user_id)
    return PlanList(plans=plans)


@app.get("/api/v1/profile/{user_id}", response_model=UserProfile)
async def get_profile(user_id: UserIdPath, services: Services) -> UserProfile:
    profile = await services.profile_repository.get_profile(user_id)
    return profile or UserProfile(user_id=user_id)


@app.put("/api/v1/profile/{user_id}", response_model=UserProfile)
async def update_profile(
    user_id: UserIdPath, payload: UserProfileUpdate, services: Services
) -> UserProfile:
    return await services.profile_repository.upsert_profile(user_id, payload)


@app.post("/api/v1/sync/gex")
async def sync_gex(
    payload: SyncGexRequest,
    services: Services,
    x_sync_token: str = Header(default=""),
) -> dict[str, str]:
    if not _sync_token_is_valid(x_sync_token):
        raise HTTPException(status_code=403, detail="Invalid sync token")
    ticker = payload.ticker.strip().upper()
    key = f"gex:v1:{ticker}:{payload.days_to_expiration}"
    await services.cache.set(
        key, payload.summary.model_dump_json(), settings.cache_ttl_seconds
    )
    return {"status": "synced"}


@app.post("/api/v1/sync/expirations")
async def sync_expirations(
    payload: SyncExpirationsRequest,
    services: Services,
    x_sync_token: str = Header(default=""),
) -> dict[str, str]:
    """Companion to /api/v1/sync/gex — this lets a local Moomoo-backed
    instance push the exact expirations it just used into the cloud cache.
    The cloud can still use yfinance as a delayed source, but synced
    expirations keep the frontend aligned with the real local GEX snapshots.
    """
    if not _sync_token_is_valid(x_sync_token):
        raise HTTPException(status_code=403, detail="Invalid sync token")
    ticker = payload.ticker.strip().upper()
    key = f"expirations:v1:{ticker}"
    await services.cache.set(
        key,
        _EXPIRATIONS_ADAPTER.dump_json(payload.expirations).decode(),
        settings.cache_ttl_seconds,
    )
    return {"status": "synced"}


@app.post("/api/v1/sync/gex/aggregate")
async def sync_gex_aggregate(
    payload: SyncAggregateGexRequest,
    services: Services,
    x_sync_token: str = Header(default=""),
) -> dict[str, str]:
    """Companion to /api/v1/sync/gex for Aggregate GEX mode. That mode is
    deliberately excluded from the poller's continuous refresh (see
    GEXService.get_aggregate_summary's docstring — polling it would trip
    Moomoo/Futu's option-chain rate limit), so without this endpoint the
    cloud deployment never receives local Moomoo-backed aggregate data, even
    when single-expiration views are being synced.
    """
    if not _sync_token_is_valid(x_sync_token):
        raise HTTPException(status_code=403, detail="Invalid sync token")
    ticker = payload.ticker.strip().upper()
    dates_key = ",".join(d.isoformat() for d in sorted(set(payload.expiration_dates)))
    key = f"gex:agg:v1:{ticker}:{dates_key}"
    await services.cache.set(
        key, payload.summary.model_dump_json(), settings.aggregate_cache_ttl_seconds
    )
    return {"status": "synced"}


@app.post("/api/v1/plans/save", response_model=UserTradePlan)
async def save_plan(
    payload: SavePlanRequest, services: Services
) -> UserTradePlan:
    try:
        return await services.plan_repository.save_signed_plan(payload.plan)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/api/v1/trades", response_model=Trade)
async def create_trade(payload: TradeCreate, services: Services) -> Trade:
    if payload.source_plan_id is not None:
        # A source_plan_id is the link the post-trade review grades
        # discipline against, so an unverified one is worse than none: it
        # makes the review claim a plan comparison that can't happen.
        plan = await services.plan_repository.get_plan(
            str(payload.source_plan_id), payload.user_id
        )
        if plan is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "source_plan_id does not exist or belongs to a different user"
                ),
            )
        if plan.ticker != payload.ticker.strip().upper():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"source_plan_id is a plan for {plan.ticker}, "
                    f"not {payload.ticker.strip().upper()}"
                ),
            )
        # Status is deliberately NOT enforced. A DRAFT plan is still a real,
        # user-visible set of levels the trade can honestly be graded
        # against — the plan card exists before signing, and refusing to link
        # it would push users into recording no plan at all. A ticker
        # mismatch, by contrast, is unambiguously wrong: those levels can
        # never describe this trade.
    # expiration_date is the only DTE source of truth now — a separate
    # client-supplied days_to_expiration used to let the entry snapshot
    # silently drift from the expiration the trade actually records (e.g.
    # aggregate mode, or a user editing the expiration after the terminal's
    # DTE had already been captured). Clamped to the snapshot endpoint's
    # existing [0, 730] range; a trade logged well after the fact can carry
    # an expiration_date already in the past, which floors to 0 rather than
    # going negative.
    days_to_expiration = max(
        0, min(730, (payload.expiration_date - market_today()).days)
    )
    entry_gex_snapshot_id = None
    if services.market_data.active_mode != "mock":
        try:
            summary = await services.gex_service.get_summary(
                payload.ticker, days_to_expiration
            )
        except MarketDataUnavailableError:
            logger.warning(
                "Creating trade without entry GEX snapshot for %s %sDTE",
                payload.ticker,
                days_to_expiration,
                exc_info=True,
            )
        else:
            entry_gex_snapshot_id = await services.snapshot_repository.save_snapshot(
                payload.ticker.strip().upper(), days_to_expiration, summary
            )
    return await services.trade_repository.create_trade(
        payload, entry_gex_snapshot_id
    )


@app.get("/api/v1/trades", response_model=TradeList)
async def list_trades(
    user_id: UserIdQuery,
    services: Services,
    ticker: str | None = None,
    status: TradeStatus | None = None,
) -> TradeList:
    trades = await services.trade_repository.list_trades(
        user_id, ticker, status.value if status else None
    )
    return TradeList(trades=trades)


@app.put("/api/v1/trades/{trade_id}", response_model=Trade)
async def close_trade(
    trade_id: TradeIdPath,
    user_id: UserIdQuery,
    payload: TradeClose,
    services: Services,
) -> Trade:
    try:
        return await services.trade_repository.close_trade(
            trade_id, user_id, payload
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except PnlMismatchError as exc:
        # Checked before the generic ValueError below — PnlMismatchError is
        # a subclass of it, and the two map to different status codes.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/trades/{trade_id}/review", response_model=TradeReview)
@limiter.limit(settings.chat_rate_limit)
async def review_trade(
    request: Request,
    trade_id: TradeIdPath,
    user_id: UserIdQuery,
    services: Services,
    # Every call here is a real OpenAI completion. A double-click or a naive
    # retry after a network hiccup would otherwise buy a second identical
    # review; the deliberate "re-analyze" button passes force=true.
    force: bool = Query(default=False),
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

    if not force:
        cached = await services.trade_review_repository.get_review(trade_id)
        if cached is not None:
            return cached

    plan_entry_price = stop_loss = target_price = None
    if trade.source_plan_id is not None:
        plan = await services.plan_repository.get_plan(
            str(trade.source_plan_id), trade.user_id
        )
        if plan is not None:
            plan_entry_price = plan.entry_price
            stop_loss = plan.stop_loss
            target_price = plan.target_price

    # One predicate decides both the score's path and what the model is told,
    # so the prose can never claim a plan comparison the score didn't make.
    # It is False when no plan loaded at all AND when the plan's levels can't
    # be denominated in the same units as this trade's fill — see
    # app/services/trade_scoring.py.
    has_source_plan = plan_levels_are_usable(
        trade.entry_price, plan_entry_price, stop_loss, target_price
    )
    execution_score = compute_execution_score(
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        pnl_pct=trade.pnl_pct,
        plan_entry_price=plan_entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
    )
    entry_snapshot = (
        await services.snapshot_repository.get_snapshot(trade.entry_gex_snapshot_id)
        if trade.entry_gex_snapshot_id is not None
        else None
    )
    ai_feedback, key_takeaways = await services.llm.review_trade(
        trade, entry_snapshot, execution_score, has_source_plan
    )
    return await services.trade_review_repository.upsert_review(
        trade_id, execution_score, ai_feedback, key_takeaways
    )


@app.get("/api/v1/trades/{trade_id}/review", response_model=TradeReview | None)
async def get_trade_review(
    trade_id: TradeIdPath, user_id: UserIdQuery, services: Services
) -> TradeReview | None:
    trade = await services.trade_repository.get_trade(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    if trade.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="The trade belongs to a different user"
        )
    return await services.trade_review_repository.get_review(trade_id)

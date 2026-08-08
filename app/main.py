import asyncio
import hmac
import logging
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
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

from app.analytics import GEXCalculator, parse_gex_risk_profile
from app.cache import ResilientCache
from app.config import settings
from app.database import (
    Base,
    ChatRepository,
    GEXSnapshotRepository,
    PlanRepository,
    ProfileRepository,
)
from app.market_data import (
    FallbackMarketDataClient,
    MockMarketDataClient,
    MoomooMarketDataClient,
)
from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationList,
    ConversationMessages,
    ExpirationList,
    GEXSnapshotList,
    HealthResponse,
    OptionGEXSummary,
    PlanList,
    SavePlanRequest,
    SyncGexRequest,
    UserProfile,
    UserProfileUpdate,
    UserTradePlan,
)
from app.services import CloudSync, GEXService, LLMOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


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
    llm: LLMOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_async_engine(settings.effective_database_url)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    calculator = GEXCalculator(settings.risk_free_rate)
    mock = MockMarketDataClient()
    primary = (
        MoomooMarketDataClient(
            settings.moomoo_host, settings.moomoo_port, calculator,
            settings.moomoo_connect_timeout_seconds,
        )
        if settings.moomoo_enabled
        else mock
    )
    market_data = FallbackMarketDataClient(
        primary,
        mock,
        primary_mode="moomoo" if settings.moomoo_enabled else "mock",
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
    gex_service = GEXService(
        market_data,
        cache,
        settings.cache_ttl_seconds,
        cloud_sync,
        settings.active_window_seconds,
        snapshot_repository,
        settings.snapshot_interval_seconds,
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


def get_services(request: Request) -> AppServices:
    return request.app.state.services


Services = Annotated[AppServices, Depends(get_services)]


@app.get("/health", response_model=HealthResponse)
async def health(services: Services) -> HealthResponse:
    return HealthResponse(
        status="ok",
        market_data_mode=services.market_data.active_mode,
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
    summary = await services.gex_service.get_summary(
        payload.context.ticker, payload.context.days_to_expiration
    )
    risk = parse_gex_risk_profile(
        summary, payload.context.days_to_expiration
    )
    profile = await services.profile_repository.get_profile(
        payload.context.user_id
    )
    assistant_message, trade_plan = await services.llm.chat(
        payload, summary, risk, profile
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
async def list_conversations(user_id: str, services: Services) -> ConversationList:
    conversations = await services.chat_repository.list_conversations(user_id)
    return ConversationList(conversations=conversations)


@app.get(
    "/api/v1/conversations/{conversation_id}/messages",
    response_model=ConversationMessages,
)
async def get_conversation_messages(
    conversation_id: str, user_id: str, services: Services
) -> ConversationMessages:
    return await services.chat_repository.get_messages(conversation_id, user_id)


@app.get("/api/v1/plans", response_model=PlanList)
async def list_plans(user_id: str, services: Services) -> PlanList:
    plans = await services.plan_repository.list_plans(user_id)
    return PlanList(plans=plans)


@app.get("/api/v1/profile/{user_id}", response_model=UserProfile)
async def get_profile(user_id: str, services: Services) -> UserProfile:
    profile = await services.profile_repository.get_profile(user_id)
    return profile or UserProfile(user_id=user_id)


@app.put("/api/v1/profile/{user_id}", response_model=UserProfile)
async def update_profile(
    user_id: str, payload: UserProfileUpdate, services: Services
) -> UserProfile:
    return await services.profile_repository.upsert_profile(user_id, payload)


@app.post("/api/v1/sync/gex")
async def sync_gex(
    payload: SyncGexRequest,
    services: Services,
    x_sync_token: str = Header(default=""),
) -> dict[str, str]:
    if not settings.sync_token or not hmac.compare_digest(
        x_sync_token, settings.sync_token
    ):
        raise HTTPException(status_code=403, detail="Invalid sync token")
    ticker = payload.ticker.strip().upper()
    key = f"gex:v1:{ticker}:{payload.days_to_expiration}"
    await services.cache.set(
        key, payload.summary.model_dump_json(), settings.cache_ttl_seconds
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

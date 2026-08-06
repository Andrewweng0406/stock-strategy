import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.analytics import GEXCalculator, parse_gex_risk_profile
from app.cache import ResilientCache
from app.config import settings
from app.database import Base, PlanRepository
from app.market_data import (
    FallbackMarketDataClient,
    MockMarketDataClient,
    MoomooMarketDataClient,
)
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    OptionGEXSummary,
    SavePlanRequest,
    SyncGexRequest,
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
    llm: LLMOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    calculator = GEXCalculator(settings.risk_free_rate)
    mock = MockMarketDataClient()
    primary = (
        MoomooMarketDataClient(
            settings.moomoo_host, settings.moomoo_port, calculator
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
    app.state.services = AppServices(
        engine=engine,
        cache=cache,
        market_data=market_data,
        gex_service=GEXService(
            market_data, cache, settings.cache_ttl_seconds, cloud_sync
        ),
        plan_repository=PlanRepository(session_factory),
        llm=LLMOrchestrator(
            openai_client,
            settings.openai_model,
            settings.default_max_loss_usd,
        ),
    )
    try:
        yield
    finally:
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


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, services: Services) -> ChatResponse:
    summary = await services.gex_service.get_summary(
        payload.context.ticker, payload.context.days_to_expiration
    )
    risk = parse_gex_risk_profile(
        summary, payload.context.days_to_expiration
    )
    assistant_message, trade_plan = await services.llm.chat(
        payload, summary, risk
    )
    return ChatResponse(
        assistant_message=assistant_message,
        gex_summary=summary,
        risk_profile=risk,
        trade_plan_card=trade_plan,
    )


@app.post("/api/v1/sync/gex")
async def sync_gex(
    payload: SyncGexRequest,
    services: Services,
    x_sync_token: str = Header(default=""),
) -> dict[str, str]:
    if not settings.sync_token or x_sync_token != settings.sync_token:
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

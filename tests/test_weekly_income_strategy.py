from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base, GEXSnapshotRepository
from app.main import app
from app.models import GEXStatus, MarketDataSource, OptionGEXSummary
from app.services.weekly_income_strategy import WeeklyIncomeStrategyService


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


def _summary(
    *,
    stock_price: float = 100.0,
    put_wall: float | None = 90.0,
    call_wall: float | None = 112.0,
    zero_gamma: float | None = 100.0,
    gex_status: GEXStatus = GEXStatus.POS_GAMMA,
    is_stale: bool = False,
    is_synthetic: bool = False,
) -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker="AAPL",
        stock_price=stock_price,
        zero_gamma=zero_gamma,
        call_wall=call_wall,
        put_wall=put_wall,
        iv_rank=35.0,
        net_gex=500_000.0,
        gex_status=gex_status,
        data_source=MarketDataSource.MOOMOO,
        is_delayed=False,
        is_synthetic=is_synthetic,
        is_stale=is_stale,
        calculated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_csp_recommendation_uses_put_wall_buffer_and_quality_flags(
    session_factory,
) -> None:
    snapshots = GEXSnapshotRepository(session_factory)
    await snapshots.save_snapshot(
        "AAPL",
        7,
        _summary(gex_status=GEXStatus.NEG_GAMMA, is_stale=True),
    )
    service = WeeklyIncomeStrategyService(snapshots)

    recommendation = await service.csp_recommendation("aapl", 7)

    assert recommendation.ticker == "AAPL"
    assert recommendation.spot_price == 100.0
    assert recommendation.put_wall == 90.0
    assert recommendation.recommended_strike == 85.0
    assert recommendation.margin_of_safety_pct == 15.0
    assert recommendation.estimated_delta is not None
    assert recommendation.estimated_win_probability is not None
    assert recommendation.weekly_target_yield is not None
    assert "NEG_GAMMA_ASSIGNMENT_RISK" in recommendation.warnings
    assert "STALE_GEX_SNAPSHOT" in recommendation.warnings
    assert "EARNINGS_WINDOW_UNKNOWN" in recommendation.warnings
    assert recommendation.data_quality.is_stale is True


@pytest.mark.asyncio
async def test_csp_recommendation_without_snapshot_returns_no_fake_prices(
    session_factory,
) -> None:
    service = WeeklyIncomeStrategyService(GEXSnapshotRepository(session_factory))

    recommendation = await service.csp_recommendation("TSLA", 7)

    assert recommendation.ticker == "TSLA"
    assert recommendation.spot_price is None
    assert recommendation.recommended_strike is None
    assert recommendation.estimated_delta is None
    assert "NO_GEX_SNAPSHOT" in recommendation.warnings


@pytest.mark.asyncio
async def test_lp_range_uses_put_and_call_walls(session_factory) -> None:
    snapshots = GEXSnapshotRepository(session_factory)
    await snapshots.save_snapshot("AAPL", 14, _summary())
    service = WeeklyIncomeStrategyService(snapshots)

    recommendation = await service.lp_range("aapl")

    assert recommendation.ticker == "AAPL"
    assert recommendation.spot_price == 100.0
    assert recommendation.range_lower == 90.0
    assert recommendation.range_upper == 112.0
    assert recommendation.range_width_pct == 22.0
    assert recommendation.breakout_bias == "NEUTRAL"


@pytest.mark.asyncio
async def test_lp_range_falls_back_to_zero_gamma_when_wall_missing(
    session_factory,
) -> None:
    snapshots = GEXSnapshotRepository(session_factory)
    await snapshots.save_snapshot("AAPL", 14, _summary(call_wall=None))
    service = WeeklyIncomeStrategyService(snapshots)

    recommendation = await service.lp_range("aapl")

    assert recommendation.range_lower == 90.0
    assert recommendation.range_upper == 105.0
    assert "MISSING_CALL_WALL" in recommendation.warnings
    assert "HEURISTIC_ZERO_GAMMA_RANGE" in recommendation.warnings


def test_strategy_endpoints_return_empty_recommendation_without_fake_prices() -> None:
    with TestClient(app) as client:
        csp = client.get("/api/v1/strategies/csp-recommendation?ticker=NOSNAP&dte=7")
        lp = client.get("/api/v1/strategies/lp-range?ticker=NOSNAP")

    assert csp.status_code == 200
    assert lp.status_code == 200
    assert csp.json()["spot_price"] is None
    assert csp.json()["recommended_strike"] is None
    assert "NO_GEX_SNAPSHOT" in csp.json()["warnings"]
    assert lp.json()["spot_price"] is None
    assert lp.json()["range_lower"] is None
    assert "NO_GEX_SNAPSHOT" in lp.json()["warnings"]

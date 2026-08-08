import asyncio
from datetime import date
from unittest.mock import AsyncMock

import pytest

from app.cache import InMemoryCache
from app.models import ExpirationInfo, ExpirationType
from app.services.gex_service import GEXService


class _FakeMarketData:
    def __init__(self, active_mode: str) -> None:
        self.active_mode = active_mode

    async def get_available_expirations(self, ticker: str) -> list[ExpirationInfo]:
        return [
            ExpirationInfo(date=date(2026, 8, 14), days_to_expiration=6, expiration_type=ExpirationType.WEEKLY)
        ]

    async def get_gex_summary(self, ticker: str, days_to_expiration: int):
        raise NotImplementedError

    async def get_gex_summary_multi(self, ticker: str, expiration_dates: list[date]):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_get_expirations_pushes_to_cloud_when_real_mode() -> None:
    cloud_sync = AsyncMock()
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
    )

    await service.get_expirations("AAPL")
    await asyncio.sleep(0)  # let the fire-and-forget cloud sync task run

    cloud_sync.push_expirations.assert_awaited_once()
    call_args = cloud_sync.push_expirations.await_args
    assert call_args.args[0] == "AAPL"
    assert call_args.args[1][0].days_to_expiration == 6


@pytest.mark.asyncio
async def test_get_expirations_does_not_push_in_mock_mode() -> None:
    cloud_sync = AsyncMock()
    service = GEXService(
        market_data=_FakeMarketData("mock"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
    )

    await service.get_expirations("AAPL")

    cloud_sync.push_expirations.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_expirations_survives_cloud_sync_failure() -> None:
    """Cloud sync is a best-effort side channel — a push failure must not
    surface to the caller or block the (already cached) expirations result.
    """
    cloud_sync = AsyncMock()
    cloud_sync.push_expirations.side_effect = RuntimeError("network down")
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
    )

    expirations = await service.get_expirations("AAPL")
    await asyncio.sleep(0)  # let the failing fire-and-forget task run and get caught

    assert expirations[0].days_to_expiration == 6

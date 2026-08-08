import asyncio
import time
from datetime import date
from unittest.mock import AsyncMock

import pytest

from app.cache import InMemoryCache
from app.models import ExpirationInfo, ExpirationType, GEXStatus, OptionGEXSummary
from app.services.gex_service import GEXService


def _fake_summary(ticker: str = "AAPL") -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker=ticker, stock_price=100.0, zero_gamma=95.0, call_wall=105.0,
        put_wall=95.0, iv_rank=40.0, net_gex=1_000_000.0, gex_status=GEXStatus.POS_GAMMA,
    )


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
        return _fake_summary(ticker)


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


@pytest.mark.asyncio
async def test_poller_repushes_expirations_even_on_local_cache_hit() -> None:
    """The whole reason the poller re-pushes expirations every cycle is that
    the cloud's cache TTL is shorter than a realistic poll interval, so a
    single push (fired only on a local cache miss) goes stale on the cloud
    side well before the ticker stops being "active" locally. This must
    still re-push even when get_expirations() hits its own local cache
    (i.e. does no fresh Moomoo fetch) — that's the whole point of pulling
    the push out of the miss-only branch.
    """
    cloud_sync = AsyncMock()
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
    )

    await service.get_expirations("AAPL")  # first call: cache miss, pushes once
    await asyncio.sleep(0)
    await service._refresh_expirations_for_poller("AAPL")  # simulated next poll tick
    await asyncio.sleep(0)

    assert cloud_sync.push_expirations.await_count == 2


@pytest.mark.asyncio
async def test_run_poller_refreshes_expirations_once_per_unique_ticker(monkeypatch) -> None:
    """Expirations are keyed by ticker only, not ticker+DTE — if the same
    ticker is active at two different DTEs, the poller should still only
    refresh/push its expirations once per cycle, not once per active pair.
    """
    cloud_sync = AsyncMock()
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
        active_window_seconds=300,
    )
    service._active[("AAPL", 6)] = time.monotonic()
    service._active[("AAPL", 13)] = time.monotonic()

    refresh_calls = []

    async def fake_refresh(ticker, dte):
        refresh_calls.append((ticker, dte))

    monkeypatch.setattr(service, "_refresh", fake_refresh)

    real_sleep = asyncio.sleep
    sleep_calls = 0

    async def fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await service.run_poller(poll_seconds=10)
    await real_sleep(0)  # let the fire-and-forget cloud push task run

    assert len(refresh_calls) == 2  # both active (ticker, dte) pairs refreshed
    assert cloud_sync.push_expirations.await_count == 1  # but only 1 unique ticker


@pytest.mark.asyncio
async def test_get_aggregate_summary_pushes_to_cloud_when_real_mode() -> None:
    cloud_sync = AsyncMock()
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
    )

    await service.get_aggregate_summary("AAPL", [date(2026, 8, 21), date(2026, 8, 14)])
    await asyncio.sleep(0)

    cloud_sync.push_aggregate.assert_awaited_once()
    call_args = cloud_sync.push_aggregate.await_args
    assert call_args.args[0] == "AAPL"
    assert call_args.args[1] == [date(2026, 8, 14), date(2026, 8, 21)]  # sorted


@pytest.mark.asyncio
async def test_get_aggregate_summary_does_not_push_in_mock_mode() -> None:
    cloud_sync = AsyncMock()
    service = GEXService(
        market_data=_FakeMarketData("mock"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
    )

    await service.get_aggregate_summary("AAPL", [date(2026, 8, 14)])

    cloud_sync.push_aggregate.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_aggregate_summary_does_not_push_again_on_cache_hit() -> None:
    """Unlike expirations, aggregate mode is deliberately NOT kept warm by
    the poller (rate-limit risk — see get_aggregate_summary's docstring),
    so a cached result should push zero additional times, not re-push.
    """
    cloud_sync = AsyncMock()
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
    )

    await service.get_aggregate_summary("AAPL", [date(2026, 8, 14)])
    await asyncio.sleep(0)
    await service.get_aggregate_summary("AAPL", [date(2026, 8, 14)])  # cache hit
    await asyncio.sleep(0)

    assert cloud_sync.push_aggregate.await_count == 1


@pytest.mark.asyncio
async def test_run_poller_never_touches_aggregate_cache() -> None:
    """Aggregate GEX must stay completely outside the poller loop — the
    whole reason it was excluded is Moomoo/Futu's 10-calls/30s option-chain
    rate limit; polling it on the same cadence as single-DTE summaries
    would trip that limit almost immediately.
    """
    cloud_sync = AsyncMock()
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        cloud_sync=cloud_sync,
        active_window_seconds=300,
    )
    service._active[("AAPL", 6)] = time.monotonic()

    async def fake_refresh(ticker, dte):
        return None

    import unittest.mock as mock
    with mock.patch.object(service, "_refresh", fake_refresh):
        real_sleep = asyncio.sleep

        async def fake_sleep(_seconds):
            raise asyncio.CancelledError

        with mock.patch("asyncio.sleep", fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await service.run_poller(poll_seconds=10)
        await real_sleep(0)

    cloud_sync.push_aggregate.assert_not_awaited()


@pytest.mark.asyncio
async def test_aggregate_summary_cached_with_longer_ttl_than_single_dte() -> None:
    """Aggregate results get no poller-driven refresh, so their cache entry
    needs to outlive a single-DTE lookup's — a short TTL here would mean a
    synced result flips back to mock within seconds on the cloud side (the
    real bug this was built to fix). Verifies get_aggregate_summary() uses
    aggregate_ttl_seconds, not the general ttl_seconds, for its cache.set().
    """
    cache = AsyncMock()
    cache.get.return_value = None
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=cache,
        ttl_seconds=30,
        aggregate_ttl_seconds=300,
    )

    await service.get_aggregate_summary("AAPL", [date(2026, 8, 14)])

    cache.set.assert_awaited_once()
    call_args = cache.set.await_args
    assert call_args.args[2] == 300  # ttl argument, not the 30s default


@pytest.mark.asyncio
async def test_aggregate_ttl_defaults_to_ttl_seconds_when_not_given() -> None:
    cache = AsyncMock()
    cache.get.return_value = None
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=cache,
        ttl_seconds=30,
    )

    await service.get_aggregate_summary("AAPL", [date(2026, 8, 14)])

    call_args = cache.set.await_args
    assert call_args.args[2] == 30

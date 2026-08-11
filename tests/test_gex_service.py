import asyncio
import time
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.cache import InMemoryCache
from app.market_data import MarketDataUnavailableError
from app.models import (
    ExpirationInfo,
    ExpirationType,
    GEXSnapshot,
    GEXStatus,
    OptionGEXSummary,
    PinningAnalysis,
)
from app.services.gex_service import GEXService


def _fake_summary(ticker: str = "AAPL") -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker=ticker, stock_price=100.0, zero_gamma=95.0, call_wall=105.0,
        put_wall=95.0, iv_rank=40.0, net_gex=1_000_000.0, gex_status=GEXStatus.POS_GAMMA,
        pinning=PinningAnalysis(
            pin_strike=100.0,
            pin_strike_matches_max_pain=True,
            distance_pct=0.0,
            oi_concentration_pct=50.0,
            in_positive_gamma=True,
            has_broken_wall=False,
            score=100,
            label="極高",
            regime="PINNING",
        ),
    )


class _FakeMarketData:
    def __init__(self, active_mode: str) -> None:
        self.active_mode = active_mode
        self.fail_summary = False
        self.fail_aggregate = False

    async def get_available_expirations(self, ticker: str) -> list[ExpirationInfo]:
        return [
            ExpirationInfo(date=date(2026, 8, 14), days_to_expiration=6, expiration_type=ExpirationType.WEEKLY)
        ]

    async def get_gex_summary(self, ticker: str, days_to_expiration: int):
        if self.fail_summary:
            raise MarketDataUnavailableError("rate limited")
        return _fake_summary(ticker)

    async def get_gex_summary_multi(self, ticker: str, expiration_dates: list[date]):
        if self.fail_aggregate:
            raise MarketDataUnavailableError("rate limited")
        return _fake_summary(ticker)


class _FakeSnapshotRepository:
    def __init__(self, latest: GEXSnapshot | None = None) -> None:
        self.latest = latest
        self.saved = []
        self.last_snapshot_time_calls = []

    async def latest_snapshot(
        self, ticker: str, days_to_expiration: int
    ) -> GEXSnapshot | None:
        return self.latest

    async def last_snapshot_time(
        self, ticker: str, days_to_expiration: int | None = None
    ):
        self.last_snapshot_time_calls.append((ticker, days_to_expiration))
        return None

    async def save_snapshot(
        self, ticker: str, days_to_expiration: int, summary: OptionGEXSummary
    ) -> int:
        self.saved.append((ticker, days_to_expiration, summary))
        return len(self.saved)


class _FakeSummaryCacheRepository:
    def __init__(self) -> None:
        self.values = {}
        self.set_calls = []

    async def set(self, cache_key: str, summary: OptionGEXSummary) -> None:
        self.set_calls.append((cache_key, summary))
        self.values[cache_key] = summary.model_dump_json()

    async def get(self, cache_key: str, max_age_seconds: int) -> str | None:
        return self.values.get(cache_key)


def _snapshot(
    *,
    ticker: str = "AAPL",
    dte: int = 6,
    call_wall: float | None = 98.0,
    put_wall: float | None = 90.0,
) -> GEXSnapshot:
    return GEXSnapshot(
        ticker=ticker,
        days_to_expiration=dte,
        captured_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        underlying_price=97.0,
        zero_gamma_strike=95.0,
        call_wall_strike=call_wall,
        put_wall_strike=put_wall,
        net_gex=1_000_000.0,
        iv_rank=40.0,
        gex_status=GEXStatus.POS_GAMMA,
    )


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
async def test_get_summary_keeps_only_latest_dte_active_per_ticker(monkeypatch) -> None:
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        active_window_seconds=300,
    )

    async def fake_refresh(ticker, dte):
        return _fake_summary(ticker)

    monkeypatch.setattr(service, "_refresh", fake_refresh)

    await service.get_summary("AAPL", 6)
    await service.get_summary("AAPL", 13)

    assert set(service._active.keys()) == {("AAPL", 13)}


@pytest.mark.asyncio
async def test_get_summary_without_prior_snapshot_preserves_current_pinning() -> None:
    repo = _FakeSnapshotRepository()
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        snapshot_repository=repo,
    )

    summary = await service.get_summary("AAPL", 6)
    await asyncio.sleep(0)

    assert summary.pinning is not None
    assert summary.pinning.regime == "PINNING"
    assert summary.pinning.has_broken_wall is False
    assert repo.last_snapshot_time_calls == [("AAPL", 6)]
    assert repo.saved[0][2].pinning.regime == "PINNING"


@pytest.mark.asyncio
async def test_get_summary_uses_prior_wall_for_breakout_detection() -> None:
    repo = _FakeSnapshotRepository(latest=_snapshot(call_wall=98.0, put_wall=90.0))
    service = GEXService(
        market_data=_FakeMarketData("moomoo"),
        cache=InMemoryCache(),
        ttl_seconds=30,
        snapshot_repository=repo,
    )

    summary = await service.get_summary("AAPL", 6)
    await asyncio.sleep(0)

    assert summary.stock_price == 100.0
    assert summary.call_wall == 105.0  # current wall remains displayable resistance
    assert summary.pinning is not None
    assert summary.pinning.has_broken_wall is True
    assert summary.pinning.regime == "BREAKOUT"
    assert repo.saved[0][2].pinning.regime == "BREAKOUT"


@pytest.mark.asyncio
async def test_get_summary_serves_stale_trusted_payload_after_market_data_failure() -> None:
    market_data = _FakeMarketData("yfinance")
    service = GEXService(
        market_data=market_data,
        cache=InMemoryCache(),
        ttl_seconds=1,
        stale_ttl_seconds=60,
    )

    fresh = await service.get_summary("AAPL", 6)
    market_data.fail_summary = True
    stale = await service._refresh("AAPL", 6)

    assert fresh.is_stale is False
    assert stale.is_stale is True
    assert stale.ticker == "AAPL"
    assert stale.stock_price == fresh.stock_price


@pytest.mark.asyncio
async def test_get_summary_writes_persisted_stale_cache_on_fresh_success() -> None:
    persisted = _FakeSummaryCacheRepository()
    service = GEXService(
        market_data=_FakeMarketData("yfinance"),
        cache=InMemoryCache(),
        ttl_seconds=1,
        stale_ttl_seconds=60,
        summary_cache_repository=persisted,
    )

    await service.get_summary("AAPL", 6)

    assert persisted.set_calls
    cache_key, summary = persisted.set_calls[0]
    assert cache_key == "gex:v1:AAPL:6"
    assert summary.is_stale is False


@pytest.mark.asyncio
async def test_get_summary_serves_persisted_stale_payload_after_restart() -> None:
    market_data = _FakeMarketData("yfinance")
    persisted = _FakeSummaryCacheRepository()
    warm_service = GEXService(
        market_data=market_data,
        cache=InMemoryCache(),
        ttl_seconds=1,
        stale_ttl_seconds=60,
        summary_cache_repository=persisted,
    )
    fresh = await warm_service.get_summary("AAPL", 6)

    market_data.fail_summary = True
    restarted_service = GEXService(
        market_data=market_data,
        cache=InMemoryCache(),
        ttl_seconds=1,
        stale_ttl_seconds=60,
        summary_cache_repository=persisted,
    )
    stale = await restarted_service._refresh("AAPL", 6)

    assert stale.is_stale is True
    assert stale.stock_price == fresh.stock_price


@pytest.mark.asyncio
async def test_get_summary_still_fails_without_stale_payload() -> None:
    market_data = _FakeMarketData("yfinance")
    market_data.fail_summary = True
    service = GEXService(
        market_data=market_data,
        cache=InMemoryCache(),
        ttl_seconds=1,
        stale_ttl_seconds=60,
    )

    with pytest.raises(MarketDataUnavailableError):
        await service._refresh("AAPL", 6)


@pytest.mark.asyncio
async def test_get_aggregate_summary_serves_stale_payload_after_failure() -> None:
    market_data = _FakeMarketData("yfinance")
    service = GEXService(
        market_data=market_data,
        cache=InMemoryCache(),
        ttl_seconds=1,
        aggregate_ttl_seconds=1,
        stale_ttl_seconds=60,
    )
    dates = [date(2026, 8, 14), date(2026, 8, 21)]

    fresh = await service.get_aggregate_summary("AAPL", dates)
    market_data.fail_aggregate = True
    stale = await service.get_aggregate_summary("AAPL", dates)

    assert fresh.is_stale is False
    assert stale.is_stale is False  # still a fresh-cache hit before ttl expiry

    service.cache = InMemoryCache()
    await service.cache.set_stale(
        "gex:agg:v1:AAPL:2026-08-14,2026-08-21",
        fresh.model_dump_json(),
        ttl=60,
    )
    stale_after_miss = await service.get_aggregate_summary("AAPL", dates)
    assert stale_after_miss.is_stale is True


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
async def test_run_poller_refreshes_only_latest_dte_per_ticker(monkeypatch) -> None:
    """OpenD option chains are rate-limited, so a user scrolling through
    several expirations must not leave every old DTE in the background
    refresh set. The poller refreshes only the latest active DTE per ticker
    and pushes expirations once per unique ticker.
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
    service._active[("AAPL", 13)] = time.monotonic() + 1

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

    assert refresh_calls == [("AAPL", 13)]
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
    needs to outlive a single-DTE lookup's. Verifies get_aggregate_summary()
    uses aggregate_ttl_seconds, not the general ttl_seconds, for its cache.set().
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

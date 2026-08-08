import asyncio
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timezone

from pydantic import TypeAdapter

from app.cache import ResilientCache
from app.database import GEXSnapshotRepository
from app.market_data import MarketDataClient
from app.models import ExpirationInfo, OptionGEXSummary
from app.services.cloud_sync import CloudSync


_EXPIRATIONS_ADAPTER = TypeAdapter(list[ExpirationInfo])


logger = logging.getLogger(__name__)


class GEXService:
    def __init__(
        self,
        market_data: MarketDataClient,
        cache: ResilientCache,
        ttl_seconds: int,
        cloud_sync: CloudSync | None = None,
        active_window_seconds: int = 300,
        snapshot_repository: GEXSnapshotRepository | None = None,
        snapshot_interval_seconds: int = 3600,
    ) -> None:
        self.market_data = market_data
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.cloud_sync = cloud_sync
        self.active_window_seconds = active_window_seconds
        self.snapshot_repository = snapshot_repository
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # (ticker, days_to_expiration) -> monotonic time last requested by a
        # real caller. Drives the background poller — only tickers/expiries
        # someone actually looked at recently get kept warm.
        self._active: dict[tuple[str, int], float] = {}

    async def get_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        ticker = ticker.strip().upper()
        self._active[(ticker, days_to_expiration)] = time.monotonic()
        self._prune()
        key = self._cache_key(ticker, days_to_expiration)
        cached = await self.cache.get(key)
        if cached:
            return OptionGEXSummary.model_validate_json(cached)
        async with self._locks[key]:
            cached = await self.cache.get(key)
            if cached:
                return OptionGEXSummary.model_validate_json(cached)
            return await self._refresh(ticker, days_to_expiration)

    async def get_expirations(self, ticker: str) -> list[ExpirationInfo]:
        ticker = ticker.strip().upper()
        key = f"expirations:v1:{ticker}"
        cached = await self.cache.get(key)
        if cached:
            return _EXPIRATIONS_ADAPTER.validate_json(cached)
        expirations = await self.market_data.get_available_expirations(ticker)
        await self.cache.set(
            key, _EXPIRATIONS_ADAPTER.dump_json(expirations).decode(), self.ttl_seconds
        )
        if self.cloud_sync and self.market_data.active_mode == "moomoo":
            asyncio.create_task(self._push_expirations_to_cloud(ticker, expirations))
        return expirations

    async def get_aggregate_summary(
        self, ticker: str, expiration_dates: list[date]
    ) -> OptionGEXSummary:
        """Combined GEX across several expirations at once. Deliberately
        outside the poller/cloud-sync path — it walks one full option chain
        per expiration, so it's meaningfully more expensive than a single
        lookup and is only ever computed on an explicit user request.
        """
        ticker = ticker.strip().upper()
        sorted_dates = sorted(set(expiration_dates))
        dates_key = ",".join(d.isoformat() for d in sorted_dates)
        key = f"gex:agg:v1:{ticker}:{dates_key}"
        cached = await self.cache.get(key)
        if cached:
            return OptionGEXSummary.model_validate_json(cached)
        summary = await self.market_data.get_gex_summary_multi(ticker, sorted_dates)
        await self.cache.set(key, summary.model_dump_json(), self.ttl_seconds)
        return summary

    def _prune(self) -> None:
        """Drop tracking for tickers/expiries nobody's asked about in a
        while, and any of their locks that aren't currently held. Runs on
        every request (cheap for the handful of entries a personal instance
        sees) so it self-heals even when the poller never starts.
        """
        now = time.monotonic()
        stale_active = [
            k for k, last in self._active.items() if now - last > self.active_window_seconds
        ]
        for k in stale_active:
            del self._active[k]
        live_keys = {self._cache_key(t, d) for t, d in self._active.keys()}
        stale_locks = [
            key
            for key, lock in self._locks.items()
            if key not in live_keys and not lock.locked()
        ]
        for key in stale_locks:
            del self._locks[key]

    async def _refresh(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        """Fetch a fresh summary (bypassing the cache), store it, and — for
        real (non-mock) data — fire off a cloud sync push. Shared by the
        request path's cache-miss branch and the background poller.
        """
        summary = await self.market_data.get_gex_summary(ticker, days_to_expiration)
        key = self._cache_key(ticker, days_to_expiration)
        await self.cache.set(key, summary.model_dump_json(), self.ttl_seconds)
        if self.market_data.active_mode == "moomoo":
            if self.cloud_sync:
                asyncio.create_task(
                    self._push_to_cloud(ticker, days_to_expiration, summary)
                )
            if self.snapshot_repository:
                asyncio.create_task(
                    self._maybe_snapshot(ticker, days_to_expiration, summary)
                )
        return summary

    async def run_poller(self, poll_seconds: int) -> None:
        """Background loop: every `poll_seconds`, re-fetch and re-push every
        ticker/expiry requested within the last `active_window_seconds`.
        Only meaningful (and only started) when cloud_sync is configured —
        a fixed interval, never the request rate, so it can't be driven
        faster by a bursty frontend.
        """
        while True:
            await asyncio.sleep(poll_seconds)
            self._prune()
            for ticker, days_to_expiration in list(self._active.keys()):
                try:
                    await self._refresh(ticker, days_to_expiration)
                except Exception:
                    logger.warning(
                        "Poller refresh failed for %s/%s",
                        ticker,
                        days_to_expiration,
                        exc_info=True,
                    )

    async def _push_to_cloud(
        self, ticker: str, days_to_expiration: int, summary: OptionGEXSummary
    ) -> None:
        assert self.cloud_sync is not None
        try:
            await self.cloud_sync.push(ticker, days_to_expiration, summary)
        except Exception:
            logger.warning("Cloud sync task failed for %s", ticker, exc_info=True)

    async def _push_expirations_to_cloud(
        self, ticker: str, expirations: list[ExpirationInfo]
    ) -> None:
        assert self.cloud_sync is not None
        try:
            await self.cloud_sync.push_expirations(ticker, expirations)
        except Exception:
            logger.warning(
                "Cloud sync expirations task failed for %s", ticker, exc_info=True
            )

    async def _maybe_snapshot(
        self, ticker: str, days_to_expiration: int, summary: OptionGEXSummary
    ) -> None:
        """Write a gex_snapshots row if the last one for this ticker is
        older than snapshot_interval_seconds (or there isn't one yet). A
        throttle on real data only, not the compute cadence — so the table
        accumulates history without a row per poller tick.
        """
        assert self.snapshot_repository is not None
        try:
            last = await self.snapshot_repository.last_snapshot_time(ticker)
            if last is not None:
                age = datetime.now(timezone.utc) - last
                if age.total_seconds() < self.snapshot_interval_seconds:
                    return
            await self.snapshot_repository.save_snapshot(
                ticker, days_to_expiration, summary
            )
        except Exception:
            logger.warning("Snapshot task failed for %s", ticker, exc_info=True)

    @staticmethod
    def _cache_key(ticker: str, days_to_expiration: int) -> str:
        return f"gex:v1:{ticker}:{days_to_expiration}"

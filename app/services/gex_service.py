import asyncio
import logging
import time
from collections import defaultdict

from app.cache import ResilientCache
from app.market_data import MarketDataClient
from app.models import OptionGEXSummary
from app.services.cloud_sync import CloudSync


logger = logging.getLogger(__name__)


class GEXService:
    def __init__(
        self,
        market_data: MarketDataClient,
        cache: ResilientCache,
        ttl_seconds: int,
        cloud_sync: CloudSync | None = None,
        active_window_seconds: int = 300,
    ) -> None:
        self.market_data = market_data
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.cloud_sync = cloud_sync
        self.active_window_seconds = active_window_seconds
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
        if self.cloud_sync and self.market_data.active_mode == "moomoo":
            asyncio.create_task(
                self._push_to_cloud(ticker, days_to_expiration, summary)
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

    @staticmethod
    def _cache_key(ticker: str, days_to_expiration: int) -> str:
        return f"gex:v1:{ticker}:{days_to_expiration}"

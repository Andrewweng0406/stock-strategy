import asyncio
import logging
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
    ) -> None:
        self.market_data = market_data
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.cloud_sync = cloud_sync
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def get_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        ticker = ticker.strip().upper()
        key = f"gex:v1:{ticker}:{days_to_expiration}"
        cached = await self.cache.get(key)
        if cached:
            return OptionGEXSummary.model_validate_json(cached)
        async with self._locks[key]:
            cached = await self.cache.get(key)
            if cached:
                return OptionGEXSummary.model_validate_json(cached)
            summary = await self.market_data.get_gex_summary(
                ticker, days_to_expiration
            )
            await self.cache.set(key, summary.model_dump_json(), self.ttl_seconds)
            if self.cloud_sync and self.market_data.active_mode == "moomoo":
                asyncio.create_task(
                    self._push_to_cloud(ticker, days_to_expiration, summary)
                )
            return summary

    async def _push_to_cloud(
        self, ticker: str, days_to_expiration: int, summary: OptionGEXSummary
    ) -> None:
        assert self.cloud_sync is not None
        try:
            await self.cloud_sync.push(ticker, days_to_expiration, summary)
        except Exception:
            logger.warning("Cloud sync task failed for %s", ticker, exc_info=True)

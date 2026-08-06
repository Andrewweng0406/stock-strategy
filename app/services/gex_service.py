import asyncio
from collections import defaultdict

from app.cache import ResilientCache
from app.market_data import MarketDataClient
from app.models import OptionGEXSummary


class GEXService:
    def __init__(
        self,
        market_data: MarketDataClient,
        cache: ResilientCache,
        ttl_seconds: int,
    ) -> None:
        self.market_data = market_data
        self.cache = cache
        self.ttl_seconds = ttl_seconds
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
            return summary

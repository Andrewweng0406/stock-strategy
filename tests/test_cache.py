import pytest

from app.cache import InMemoryCache


@pytest.mark.asyncio
async def test_memory_cache_round_trip() -> None:
    cache = InMemoryCache()
    await cache.set("gex:AAPL", "value", ttl=30)
    assert await cache.get("gex:AAPL") == "value"

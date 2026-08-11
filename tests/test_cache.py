import pytest

from app.cache import InMemoryCache


@pytest.mark.asyncio
async def test_memory_cache_round_trip() -> None:
    cache = InMemoryCache()
    await cache.set("gex:AAPL", "value", ttl=30)
    assert await cache.get("gex:AAPL") == "value"


@pytest.mark.asyncio
async def test_memory_cache_stale_values_are_separate_from_fresh_values() -> None:
    cache = InMemoryCache()
    await cache.set("gex:AAPL", "fresh", ttl=30)
    await cache.set_stale("gex:AAPL", "stale", ttl=30)

    assert await cache.get("gex:AAPL") == "fresh"
    assert await cache.get_stale("gex:AAPL") == "stale"

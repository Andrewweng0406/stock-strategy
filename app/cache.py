import asyncio
import logging
import time

import redis.asyncio as redis


logger = logging.getLogger(__name__)


class InMemoryCache:
    def __init__(self) -> None:
        self._values: dict[str, tuple[float, str]] = {}
        self._stale_values: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            item = self._values.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= time.monotonic():
                self._values.pop(key, None)
                return None
            return value

    async def get_stale(self, key: str) -> str | None:
        async with self._lock:
            item = self._stale_values.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= time.monotonic():
                self._stale_values.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        async with self._lock:
            self._values[key] = (time.monotonic() + ttl, value)

    async def set_stale(self, key: str, value: str, ttl: int) -> None:
        async with self._lock:
            self._stale_values[key] = (time.monotonic() + ttl, value)


class ResilientCache:
    def __init__(self, redis_url: str | None) -> None:
        self.memory = InMemoryCache()
        self.redis_client = (
            redis.from_url(redis_url, decode_responses=True) if redis_url else None
        )

    async def get(self, key: str) -> str | None:
        if self.redis_client:
            try:
                cached = await self.redis_client.get(key)
                if cached is not None:
                    return cached
            except Exception:
                logger.warning("Redis read failed; using memory cache", exc_info=True)
        return await self.memory.get(key)

    async def get_stale(self, key: str) -> str | None:
        stale_key = f"{key}:stale"
        if self.redis_client:
            try:
                cached = await self.redis_client.get(stale_key)
                if cached is not None:
                    return cached
            except Exception:
                logger.warning("Redis stale read failed; using memory cache", exc_info=True)
        return await self.memory.get_stale(key)

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self.memory.set(key, value, ttl)
        if self.redis_client:
            try:
                await self.redis_client.set(key, value, ex=ttl)
            except Exception:
                logger.warning("Redis write failed; value remains in memory", exc_info=True)

    async def set_stale(self, key: str, value: str, ttl: int) -> None:
        await self.memory.set_stale(key, value, ttl)
        if self.redis_client:
            try:
                await self.redis_client.set(f"{key}:stale", value, ex=ttl)
            except Exception:
                logger.warning("Redis stale write failed; value remains in memory", exc_info=True)

    async def close(self) -> None:
        if self.redis_client:
            await self.redis_client.aclose()

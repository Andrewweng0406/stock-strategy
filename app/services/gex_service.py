import asyncio
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timezone

from pydantic import TypeAdapter

from app.cache import ResilientCache
from app.database import GEXSnapshotRepository
from app.market_data import MarketDataClient, MarketDataUnavailableError
from app.models import ExpirationInfo, OptionGEXSummary, PinningRegime
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
        aggregate_ttl_seconds: int | None = None,
        stale_ttl_seconds: int = 900,
    ) -> None:
        self.market_data = market_data
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.cloud_sync = cloud_sync
        self.active_window_seconds = active_window_seconds
        self.snapshot_repository = snapshot_repository
        self.snapshot_interval_seconds = snapshot_interval_seconds
        # Aggregate GEX gets no poller-driven refresh (see
        # get_aggregate_summary's docstring), so its cache entry has to live
        # longer than a single-DTE lookup's to have any realistic chance of
        # still being warm on the cloud side. Defaults to ttl_seconds when
        # not given explicitly.
        self.aggregate_ttl_seconds = aggregate_ttl_seconds if aggregate_ttl_seconds is not None else ttl_seconds
        self.stale_ttl_seconds = stale_ttl_seconds
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # (ticker, days_to_expiration) -> monotonic time last requested by a
        # real caller. Drives the background poller — only tickers/expiries
        # someone actually looked at recently get kept warm.
        self._active: dict[tuple[str, int], float] = {}

    async def get_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        ticker = ticker.strip().upper()
        for active_ticker, active_dte in list(self._active.keys()):
            if active_ticker == ticker and active_dte != days_to_expiration:
                del self._active[(active_ticker, active_dte)]
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

    @staticmethod
    def _expirations_cache_key(ticker: str) -> str:
        return f"expirations:v1:{ticker}"

    async def get_expirations(self, ticker: str) -> list[ExpirationInfo]:
        ticker = ticker.strip().upper()
        key = self._expirations_cache_key(ticker)
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
        outside the poller's continuous refresh — it walks one full option
        chain per expiration, so it's meaningfully more expensive than a
        single lookup and is only ever computed on an explicit user
        request; polling it every poll_seconds the way single-DTE summaries
        are kept warm would trip Moomoo/Futu's 10/30s option-chain rate
        limit almost immediately.

        Still pushes to cloud_sync once per fresh (non-cached) compute —
        same one-shot treatment the local cache itself already gives this
        result (bounded by aggregate_ttl_seconds, refreshed only on the
        next explicit request, never proactively) — so the cloud can serve a
        synced aggregate result even though aggregates are excluded from the
        poller. aggregate_ttl_seconds is deliberately
        longer than ttl_seconds (see __init__) so a synced result has a
        realistic chance of still being warm the next time someone looks.
        """
        ticker = ticker.strip().upper()
        sorted_dates = sorted(set(expiration_dates))
        dates_key = ",".join(d.isoformat() for d in sorted_dates)
        key = f"gex:agg:v1:{ticker}:{dates_key}"
        cached = await self.cache.get(key)
        if cached:
            return OptionGEXSummary.model_validate_json(cached)
        try:
            summary = await self.market_data.get_gex_summary_multi(ticker, sorted_dates)
        except MarketDataUnavailableError:
            stale = await self.cache.get_stale(key)
            if stale is None:
                raise
            logger.warning(
                "Serving stale aggregate GEX summary for %s/%s after market-data failure",
                ticker,
                dates_key,
            )
            return OptionGEXSummary.model_validate_json(stale).model_copy(
                update={"is_stale": True}
            )
        summary = summary.model_copy(update={"is_stale": False})
        await self.cache.set(key, summary.model_dump_json(), self.aggregate_ttl_seconds)
        await self.cache.set_stale(
            key, summary.model_dump_json(), self.stale_ttl_seconds
        )
        if self.cloud_sync and self.market_data.active_mode == "moomoo":
            asyncio.create_task(
                self._push_aggregate_to_cloud(ticker, sorted_dates, summary)
            )
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
        """Fetch a fresh summary (bypassing the cache), store it, and for a
        trusted local market-data source, fire off a cloud sync push. Shared by
        the request path's cache-miss branch and the background poller.
        """
        key = self._cache_key(ticker, days_to_expiration)
        try:
            summary = await self.market_data.get_gex_summary(ticker, days_to_expiration)
        except MarketDataUnavailableError:
            stale = await self.cache.get_stale(key)
            if stale is None:
                raise
            logger.warning(
                "Serving stale GEX summary for %s/%sDTE after market-data failure",
                ticker,
                days_to_expiration,
            )
            return OptionGEXSummary.model_validate_json(stale).model_copy(
                update={"is_stale": True}
            )
        summary = await self._apply_prior_wall_breakout(
            ticker, days_to_expiration, summary
        )
        summary = summary.model_copy(update={"is_stale": False})
        await self.cache.set(key, summary.model_dump_json(), self.ttl_seconds)
        await self.cache.set_stale(
            key, summary.model_dump_json(), self.stale_ttl_seconds
        )
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

    async def _apply_prior_wall_breakout(
        self, ticker: str, days_to_expiration: int, summary: OptionGEXSummary
    ) -> OptionGEXSummary:
        """Use the prior persisted wall for breakout detection.

        The current GEX calculator deliberately constrains call_wall above
        spot and put_wall below spot. That is correct for resistance/support
        display, but it means a same-instant "spot crossed current wall" test
        cannot fire. A breakout is temporal: current spot versus a previously
        observed defence line for the same ticker and expiry.
        """
        if self.snapshot_repository is None or summary.pinning is None:
            return summary
        try:
            prior = await self.snapshot_repository.latest_snapshot(
                ticker, days_to_expiration
            )
        except Exception:
            logger.warning("Prior GEX snapshot lookup failed for %s", ticker, exc_info=True)
            return summary
        if prior is None:
            return summary

        crossed_call = (
            prior.call_wall_strike is not None
            and summary.stock_price > prior.call_wall_strike
        )
        crossed_put = (
            prior.put_wall_strike is not None
            and summary.stock_price < prior.put_wall_strike
        )
        if not crossed_call and not crossed_put:
            return summary

        pinning = summary.pinning.model_copy(
            update={"has_broken_wall": True, "regime": PinningRegime.BREAKOUT}
        )
        return summary.model_copy(update={"pinning": pinning})

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
            latest_by_ticker: dict[str, tuple[int, float]] = {}
            for (ticker, days_to_expiration), last_seen in self._active.items():
                previous = latest_by_ticker.get(ticker)
                if previous is None or last_seen > previous[1]:
                    latest_by_ticker[ticker] = (days_to_expiration, last_seen)
            for ticker, (days_to_expiration, _) in latest_by_ticker.items():
                try:
                    await self._refresh(ticker, days_to_expiration)
                except Exception:
                    logger.warning(
                        "Poller refresh failed for %s/%s",
                        ticker,
                        days_to_expiration,
                        exc_info=True,
                    )
            # Expirations are keyed by ticker only (not ticker+DTE), so this
            # only needs to run once per unique active ticker per cycle, not
            # once per active (ticker, DTE) pair — and it deliberately
            # re-pushes every cycle regardless of local cache hit/miss.
            # get_expirations()'s own cloud push only fires on a fresh
            # fetch, but the cloud cache TTL (cache_ttl_seconds, 30s by
            # default) is shorter than a realistic poll_seconds, so a
            # single push goes stale well before the ticker stops being
            # "active" — this keeps the cloud's copy continuously warm for
            # as long as someone is actually looking at it locally.
            for ticker in {t for t, _ in self._active.keys()}:
                try:
                    await self._refresh_expirations_for_poller(ticker)
                except Exception:
                    logger.warning(
                        "Poller expirations refresh failed for %s", ticker, exc_info=True
                    )

    async def _refresh_expirations_for_poller(self, ticker: str) -> None:
        """Unlike get_expirations(), this always re-pushes to keep the
        cloud's copy warm, even on a local cache hit — that's the whole
        point of calling it from the poller (see run_poller's docstring).
        A local cache MISS already pushes once inside get_expirations()
        itself, so this skips its own push in that case to avoid firing
        the same push twice back to back.
        """
        was_cached = await self.cache.get(self._expirations_cache_key(ticker)) is not None
        expirations = await self.get_expirations(ticker)
        if was_cached and self.cloud_sync and self.market_data.active_mode == "moomoo":
            asyncio.create_task(self._push_expirations_to_cloud(ticker, expirations))

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

    async def _push_aggregate_to_cloud(
        self, ticker: str, expiration_dates: list[date], summary: OptionGEXSummary
    ) -> None:
        assert self.cloud_sync is not None
        try:
            await self.cloud_sync.push_aggregate(ticker, expiration_dates, summary)
        except Exception:
            logger.warning(
                "Cloud sync aggregate task failed for %s", ticker, exc_info=True
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
            last = await self.snapshot_repository.last_snapshot_time(
                ticker, days_to_expiration
            )
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

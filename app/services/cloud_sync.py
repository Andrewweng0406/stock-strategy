import logging
from datetime import date

import httpx

from app.models import ExpirationInfo, OptionGEXSummary


logger = logging.getLogger(__name__)


class CloudSync:
    """Fire-and-forget push of a real GEX summary to a cloud deployment's
    cache, so that instance can serve real data without ever holding Moomoo
    credentials itself. Failures are logged and swallowed — sync is a
    best-effort convenience, never a dependency for local operation.
    """

    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token

    async def push(
        self, ticker: str, days_to_expiration: int, summary: OptionGEXSummary
    ) -> None:
        payload = {
            "ticker": ticker,
            "days_to_expiration": days_to_expiration,
            "summary": summary.model_dump(mode="json"),
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.url}/api/v1/sync/gex",
                    json=payload,
                    headers={"X-Sync-Token": self.token},
                )
                response.raise_for_status()
        except Exception:
            logger.warning("Cloud sync push failed for %s", ticker, exc_info=True)

    async def push_expirations(
        self, ticker: str, expirations: list[ExpirationInfo]
    ) -> None:
        """Push the real expiration-date list too, not just GEX summaries.
        Without this, the cloud deployment's own /api/v1/expirations always
        falls back to MockMarketDataClient's synthetic dates (its own
        primary Moomoo connection never succeeds there) — so the frontend's
        default-selected expiration on the cloud almost never matches a
        days_to_expiration this instance has actually pushed real data for,
        and the cloud UI keeps showing mock numbers even after a real push.
        """
        payload = {
            "ticker": ticker,
            "expirations": [e.model_dump(mode="json") for e in expirations],
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.url}/api/v1/sync/expirations",
                    json=payload,
                    headers={"X-Sync-Token": self.token},
                )
                response.raise_for_status()
        except Exception:
            logger.warning(
                "Cloud sync expirations push failed for %s", ticker, exc_info=True
            )

    async def push_aggregate(
        self, ticker: str, expiration_dates: list[date], summary: OptionGEXSummary
    ) -> None:
        """Aggregate GEX (get_aggregate_summary) is deliberately kept out of
        the poller's continuous refresh — it walks one full option chain per
        expiration, and Moomoo/Futu caps that specific call at 10/30s, so
        polling it every 10s the way single-DTE summaries are kept warm
        would trip that limit almost immediately. This still pushes once,
        right after a fresh (non-cached) compute — same one-shot treatment
        the local cache itself already gives aggregate results (a 30s TTL,
        refreshed only on the next explicit request, not proactively) — so
        the cloud stops being permanently stuck on mock for aggregate mode
        without reintroducing the rate-limit risk.
        """
        payload = {
            "ticker": ticker,
            "expiration_dates": [d.isoformat() for d in expiration_dates],
            "summary": summary.model_dump(mode="json"),
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.url}/api/v1/sync/gex/aggregate",
                    json=payload,
                    headers={"X-Sync-Token": self.token},
                )
                response.raise_for_status()
        except Exception:
            logger.warning(
                "Cloud sync aggregate push failed for %s", ticker, exc_info=True
            )

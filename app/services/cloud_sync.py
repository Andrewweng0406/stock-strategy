import logging

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

import logging

import httpx

from app.models import OptionGEXSummary


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

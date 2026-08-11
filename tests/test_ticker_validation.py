from fastapi.testclient import TestClient

from app.models import GEXStatus, OptionGEXSummary
from app.main import app
from app.services.gex_service import GEXService


async def _summary(self, ticker: str, days_to_expiration: int) -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker=ticker,
        stock_price=100,
        zero_gamma=95,
        call_wall=110,
        put_wall=90,
        iv_rank=40,
        net_gex=1_000_000,
        gex_status=GEXStatus.POS_GAMMA,
    )


def test_ticker_path_accepts_common_symbol_punctuation(monkeypatch) -> None:
    monkeypatch.setattr(GEXService, "get_summary", _summary)

    with TestClient(app) as client:
        response = client.get("/api/v1/gex/BRK.B?days_to_expiration=30")

    assert response.status_code == 200
    assert response.json()["ticker"] == "BRK.B"


def test_ticker_path_rejects_spaces() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/gex/BAD%20TICKER?days_to_expiration=30")

    assert response.status_code == 422


def test_ticker_path_rejects_overlong_symbols() -> None:
    ticker = "A" * 33
    with TestClient(app) as client:
        response = client.get(f"/api/v1/expirations/{ticker}")

    assert response.status_code == 422

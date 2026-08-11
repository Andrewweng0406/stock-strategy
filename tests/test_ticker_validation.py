from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
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


def _summary_payload(ticker: str = "AAPL") -> dict:
    return {
        "ticker": ticker,
        "stock_price": 100,
        "zero_gamma": 95,
        "call_wall": 110,
        "put_wall": 90,
        "iv_rank": 40,
        "net_gex": 1_000_000,
        "gex_status": "POS_GAMMA",
    }


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


def test_trade_list_ticker_filter_accepts_common_symbol_punctuation() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/trades?user_id=user-1&ticker=BRK.B")

    assert response.status_code == 200
    assert response.json()["trades"] == []


def test_trade_list_ticker_filter_rejects_spaces() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/trades?user_id=user-1&ticker=BAD%20TICKER")

    assert response.status_code == 422


def test_trade_list_ticker_filter_rejects_overlong_symbols() -> None:
    ticker = "A" * 33
    with TestClient(app) as client:
        response = client.get(f"/api/v1/trades?user_id=user-1&ticker={ticker}")

    assert response.status_code == 422


def test_sync_body_rejects_invalid_ticker_with_valid_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sync_token", "test-sync-token")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sync/gex",
            json={
                "ticker": "BAD TICKER",
                "days_to_expiration": 30,
                "summary": _summary_payload(),
            },
            headers={"X-Sync-Token": "test-sync-token"},
        )

    assert response.status_code == 422


def test_chat_body_rejects_invalid_ticker() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "user_message": "看看這個",
                "context": {
                    "user_id": "user-1",
                    "conversation_id": "conversation-1",
                    "ticker": "BAD TICKER",
                    "days_to_expiration": 30,
                },
            },
        )

    assert response.status_code == 422


def test_trade_create_rejects_invalid_ticker() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/trades",
            json={
                "user_id": "user-1",
                "ticker": "BAD TICKER",
                "strategy_type": "Long Call",
                "entry_price": 1.5,
                "position_size": 1,
                "expiration_date": "2099-08-21",
                "option_type": "CALL",
                "strike_price": 100,
            },
        )

    assert response.status_code == 422


def test_saved_plan_rejects_invalid_ticker() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/plans/save",
            json={
                "plan": {
                    "plan_id": str(uuid4()),
                    "user_id": "user-1",
                    "conversation_id": "conversation-1",
                    "ticker": "BAD TICKER",
                    "strategy_type": "Long Call",
                    "entry_price": 1.5,
                    "stop_loss": 0.8,
                    "target_price": 3.0,
                    "max_loss_usd": 150,
                    "theta_warning": False,
                    "status": "DRAFT",
                }
            },
        )

    assert response.status_code == 422

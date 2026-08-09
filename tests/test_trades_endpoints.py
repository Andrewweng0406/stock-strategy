from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def _summary_payload(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "stock_price": 100.0,
        "zero_gamma": 95.0,
        "call_wall": 110.0,
        "put_wall": 90.0,
        "iv_rank": 40.0,
        "net_gex": 1_000_000.0,
        "gex_status": "POS_GAMMA",
    }


def _seed_cache(client: TestClient, monkeypatch, ticker: str, dte: int = 30) -> None:
    """Pre-warms the GEX cache via the sync endpoint so a trade's entry-
    snapshot fetch hits the cache instead of a live market-data call — same
    technique tests/test_sync_endpoints.py already uses.
    """
    monkeypatch.setattr(settings, "sync_token", "test-sync-token")
    response = client.post(
        "/api/v1/sync/gex",
        json={
            "ticker": ticker,
            "days_to_expiration": dte,
            "summary": _summary_payload(ticker),
        },
        headers={"X-Sync-Token": "test-sync-token"},
    )
    assert response.status_code == 200


def _create_trade(client: TestClient, user_id: str, ticker: str) -> dict:
    response = client.post(
        "/api/v1/trades",
        json={
            "user_id": user_id,
            "ticker": ticker,
            "strategy_type": "Long Call",
            "entry_price": 100.0,
            "position_size": 1,
            "days_to_expiration": 30,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_create_trade_writes_entry_snapshot_and_defaults_to_open(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST1")
        trade = _create_trade(client, "user-1", "TJTEST1")
    assert trade["status"] == "OPEN"
    assert trade["entry_gex_snapshot_id"] is not None
    assert trade["exit_price"] is None
    assert trade["pnl_pct"] is None


def test_list_trades_filters_by_ticker_and_status(monkeypatch) -> None:
    # Tickers are suffixed with a fresh uuid per run because these endpoint
    # tests hit the real on-disk dev database (see .env's DATABASE_URL),
    # which is never dropped between test runs — a fixed literal ticker
    # would accumulate rows across repeated local runs and make the
    # exact-count assertion below flaky.
    suffix = uuid4().hex[:8].upper()
    ticker_a = f"TJTEST2{suffix}"
    ticker_b = f"TJTEST3{suffix}"
    user_id = f"user-2-{suffix}"
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, ticker_a)
        _seed_cache(client, monkeypatch, ticker_b)
        _create_trade(client, user_id, ticker_a)
        _create_trade(client, user_id, ticker_b)

        response = client.get(f"/api/v1/trades?user_id={user_id}&ticker={ticker_a}")
    assert response.status_code == 200
    trades = response.json()["trades"]
    assert len(trades) == 1
    assert trades[0]["ticker"] == ticker_a


def test_close_trade_computes_pnl_pct(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST4")
        trade = _create_trade(client, "user-3", "TJTEST4")
        response = client.put(
            f"/api/v1/trades/{trade['id']}?user_id=user-3",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 20.0,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "CLOSED"
    assert body["pnl_pct"] == 20.0


def test_close_trade_rejects_other_users_trade(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST5")
        trade = _create_trade(client, "user-4", "TJTEST5")
        response = client.put(
            f"/api/v1/trades/{trade['id']}?user_id=someone-else",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 20.0,
            },
        )
    assert response.status_code == 403


def test_close_trade_rejects_already_closed(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST6")
        trade = _create_trade(client, "user-5", "TJTEST6")
        close_payload = {
            "exit_price": 120.0,
            "exit_date": datetime.now(timezone.utc).isoformat(),
            "pnl": 20.0,
        }
        client.put(f"/api/v1/trades/{trade['id']}?user_id=user-5", json=close_payload)
        response = client.put(
            f"/api/v1/trades/{trade['id']}?user_id=user-5", json=close_payload
        )
    assert response.status_code == 409


def test_review_trade_requires_closed_status(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST7")
        trade = _create_trade(client, "user-6", "TJTEST7")
        response = client.post(f"/api/v1/trades/{trade['id']}/review?user_id=user-6")
    assert response.status_code == 400


def test_review_trade_returns_404_for_unknown_trade() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/trades/does-not-exist/review?user_id=user-7"
        )
    assert response.status_code == 404


def test_review_trade_upserts_review_with_fake_llm(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST8")
        trade = _create_trade(client, "user-8", "TJTEST8")
        client.put(
            f"/api/v1/trades/{trade['id']}?user_id=user-8",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 20.0,
            },
        )

        class FakeOrchestrator:
            async def review_trade(self, trade, entry_snapshot, execution_score):
                return (
                    "Solid, disciplined exit.",
                    ["Booked profit near plan", "Stayed within risk"],
                )

        app.state.services.llm = FakeOrchestrator()

        first = client.post(f"/api/v1/trades/{trade['id']}/review?user_id=user-8")
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["ai_feedback"] == "Solid, disciplined exit."
        assert first_body["execution_score"] == 4  # no source_plan -> pnl_pct 20% -> band >=15

        second = client.post(f"/api/v1/trades/{trade['id']}/review?user_id=user-8")
    assert second.status_code == 200


def test_get_trade_review_returns_null_before_and_stored_review_after_post(
    monkeypatch,
) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST9")
        trade = _create_trade(client, "user-9", "TJTEST9")
        client.put(
            f"/api/v1/trades/{trade['id']}?user_id=user-9",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 20.0,
            },
        )

        before = client.get(f"/api/v1/trades/{trade['id']}/review")
        assert before.status_code == 200
        assert before.json() is None

        class FakeOrchestrator:
            async def review_trade(self, trade, entry_snapshot, execution_score):
                return (
                    "Solid, disciplined exit.",
                    ["Booked profit near plan", "Stayed within risk"],
                )

        app.state.services.llm = FakeOrchestrator()

        posted = client.post(f"/api/v1/trades/{trade['id']}/review?user_id=user-9")
        assert posted.status_code == 200
        posted_body = posted.json()

        after = client.get(f"/api/v1/trades/{trade['id']}/review")
    assert after.status_code == 200
    after_body = after.json()
    assert after_body is not None
    assert after_body["trade_id"] == trade["id"]
    assert after_body["ai_feedback"] == posted_body["ai_feedback"]
    assert after_body["execution_score"] == posted_body["execution_score"]
    assert after_body["key_takeaways"] == posted_body["key_takeaways"]

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def _summary_payload(ticker: str = "SYNCTEST") -> dict:
    return {
        "ticker": ticker,
        "stock_price": 250.0,
        "zero_gamma": 245.0,
        "call_wall": 260.0,
        "put_wall": 240.0,
        "iv_rank": 55.0,
        "net_gex": 5_000_000.0,
        "gex_status": "POS_GAMMA",
    }


def test_sync_gex_rejects_missing_token() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sync/gex",
            json={"ticker": "SYNCTEST", "days_to_expiration": 30, "summary": _summary_payload()},
        )
    assert response.status_code == 403


def test_sync_gex_round_trips_through_gex_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sync_token", "test-sync-token")
    with TestClient(app) as client:
        sync_response = client.post(
            "/api/v1/sync/gex",
            json={"ticker": "SYNCTEST", "days_to_expiration": 30, "summary": _summary_payload()},
            headers={"X-Sync-Token": "test-sync-token"},
        )
        assert sync_response.status_code == 200

        gex_response = client.get("/api/v1/gex/SYNCTEST?days_to_expiration=30")
    assert gex_response.status_code == 200
    assert gex_response.json()["stock_price"] == 250.0


def test_sync_expirations_rejects_missing_token() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sync/expirations",
            json={"ticker": "SYNCTEST", "expirations": []},
        )
    assert response.status_code == 403


def test_sync_expirations_round_trips_through_expirations_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sync_token", "test-sync-token")
    payload = {
        "ticker": "SYNCTEST",
        "expirations": [
            {"date": "2026-08-14", "days_to_expiration": 6, "expiration_type": "WEEKLY"},
            {"date": "2026-08-21", "days_to_expiration": 13, "expiration_type": "MONTHLY"},
        ],
    }
    with TestClient(app) as client:
        sync_response = client.post(
            "/api/v1/sync/expirations",
            json=payload,
            headers={"X-Sync-Token": "test-sync-token"},
        )
        assert sync_response.status_code == 200

        expirations_response = client.get("/api/v1/expirations/SYNCTEST")
    assert expirations_response.status_code == 200
    body = expirations_response.json()
    assert body["ticker"] == "SYNCTEST"
    assert [e["date"] for e in body["expirations"]] == ["2026-08-14", "2026-08-21"]


def test_sync_gex_aggregate_rejects_missing_token() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sync/gex/aggregate",
            json={
                "ticker": "SYNCTEST",
                "expiration_dates": ["2026-08-14", "2026-08-21"],
                "summary": _summary_payload(),
            },
        )
    assert response.status_code == 403


def test_sync_gex_aggregate_round_trips_through_aggregate_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sync_token", "test-sync-token")
    with TestClient(app) as client:
        sync_response = client.post(
            "/api/v1/sync/gex/aggregate",
            json={
                "ticker": "SYNCTEST",
                "expiration_dates": ["2026-08-14", "2026-08-21"],
                "summary": _summary_payload(),
            },
            headers={"X-Sync-Token": "test-sync-token"},
        )
        assert sync_response.status_code == 200

        aggregate_response = client.get(
            "/api/v1/gex/SYNCTEST/aggregate",
            params={"expirations": ["2026-08-14", "2026-08-21"]},
        )
    assert aggregate_response.status_code == 200
    assert aggregate_response.json()["stock_price"] == 250.0


def test_sync_gex_aggregate_cache_key_ignores_date_order(monkeypatch) -> None:
    """get_aggregate_summary() sorts+dedupes dates before building its cache
    key, so a sync push and a later request that list the same dates in a
    different order must still land on / read from the same cache entry.
    """
    monkeypatch.setattr(settings, "sync_token", "test-sync-token")
    with TestClient(app) as client:
        sync_response = client.post(
            "/api/v1/sync/gex/aggregate",
            json={
                "ticker": "SYNCTEST",
                "expiration_dates": ["2026-08-21", "2026-08-14"],  # reversed order
                "summary": _summary_payload(),
            },
            headers={"X-Sync-Token": "test-sync-token"},
        )
        assert sync_response.status_code == 200

        aggregate_response = client.get(
            "/api/v1/gex/SYNCTEST/aggregate",
            params={"expirations": ["2026-08-14", "2026-08-21"]},
        )
    assert aggregate_response.status_code == 200
    assert aggregate_response.json()["stock_price"] == 250.0

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_health_reports_cloud_sync_disabled_without_secrets(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cloud_sync_url", None)
    monkeypatch.setattr(settings, "sync_token", None)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "market_data_mode" in body
    assert body["cloud_sync"] == {
        "enabled": False,
        "last_success_at": None,
        "last_error_at": None,
        "last_error": None,
    }
    assert "token" not in str(body).lower()

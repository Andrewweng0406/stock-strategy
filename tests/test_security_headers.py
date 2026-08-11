from fastapi.testclient import TestClient

from app.main import app


EXPECTED_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
}


def test_security_headers_are_attached_to_normal_responses() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    for header, value in EXPECTED_SECURITY_HEADERS.items():
        assert response.headers[header] == value


def test_security_headers_are_attached_to_sync_guard_rejections() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/sync/gex", json={"malformed": True})

    assert response.status_code == 403
    for header, value in EXPECTED_SECURITY_HEADERS.items():
        assert response.headers[header] == value

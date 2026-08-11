from fastapi.testclient import TestClient

from app.main import app


def test_user_scoped_query_rejects_empty_user_id() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/plans?user_id=")

    assert response.status_code == 422


def test_user_scoped_query_rejects_overlong_user_id() -> None:
    user_id = "u" * 129
    with TestClient(app) as client:
        response = client.get(f"/api/v1/trades?user_id={user_id}")

    assert response.status_code == 422


def test_profile_path_rejects_overlong_user_id() -> None:
    user_id = "u" * 129
    with TestClient(app) as client:
        response = client.get(f"/api/v1/profile/{user_id}")

    assert response.status_code == 422


def test_conversation_path_rejects_overlong_conversation_id() -> None:
    conversation_id = "c" * 129
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/conversations/{conversation_id}/messages?user_id=user-1"
        )

    assert response.status_code == 422


def test_trade_path_rejects_overlong_trade_id() -> None:
    trade_id = "t" * 65
    with TestClient(app) as client:
        response = client.get(f"/api/v1/trades/{trade_id}/review?user_id=user-1")

    assert response.status_code == 422

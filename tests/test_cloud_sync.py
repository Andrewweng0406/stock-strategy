from datetime import date

import httpx
import pytest

from app.models import GEXStatus, OptionGEXSummary
from app.services.cloud_sync import CloudSync


def _summary() -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker="AAPL",
        stock_price=100,
        zero_gamma=95,
        call_wall=110,
        put_wall=90,
        iv_rank=40,
        net_gex=1_000_000,
        gex_status=GEXStatus.POS_GAMMA,
    )


@pytest.mark.asyncio
async def test_cloud_sync_status_records_success(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout: _Client())

    sync = CloudSync("https://cloud.example", "secret-token")
    await sync.push("AAPL", 30, _summary())

    status = sync.status()
    assert status["enabled"] is True
    assert status["last_success_at"] is not None
    assert status["last_error_at"] is None
    assert status["last_error"] is None
    assert "secret" not in str(status)


@pytest.mark.asyncio
async def test_cloud_sync_status_records_failure_without_raising(monkeypatch) -> None:
    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("cloud offline")

    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout: _Client())

    sync = CloudSync("https://cloud.example", "secret-token")
    await sync.push_aggregate("AAPL", [date(2026, 8, 14)], _summary())

    status = sync.status()
    assert status["enabled"] is True
    assert status["last_success_at"] is None
    assert status["last_error_at"] is not None
    assert "ConnectError" in status["last_error"]
    assert "cloud offline" in status["last_error"]
    assert "secret" not in str(status)

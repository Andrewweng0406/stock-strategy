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
                "pnl": 2000.0,
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
                "pnl": 2000.0,
            },
        )

        class FakeOrchestrator:
            async def review_trade(
                self, trade, entry_snapshot, execution_score, has_source_plan
            ):
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

        before = client.get(f"/api/v1/trades/{trade['id']}/review?user_id=user-9")
        assert before.status_code == 200
        assert before.json() is None

        class FakeOrchestrator:
            async def review_trade(
                self, trade, entry_snapshot, execution_score, has_source_plan
            ):
                return (
                    "Solid, disciplined exit.",
                    ["Booked profit near plan", "Stayed within risk"],
                )

        app.state.services.llm = FakeOrchestrator()

        posted = client.post(f"/api/v1/trades/{trade['id']}/review?user_id=user-9")
        assert posted.status_code == 200
        posted_body = posted.json()

        after = client.get(f"/api/v1/trades/{trade['id']}/review?user_id=user-9")
    assert after.status_code == 200
    after_body = after.json()
    assert after_body is not None
    assert after_body["trade_id"] == trade["id"]
    assert after_body["ai_feedback"] == posted_body["ai_feedback"]
    assert after_body["execution_score"] == posted_body["execution_score"]
    assert after_body["key_takeaways"] == posted_body["key_takeaways"]


def test_get_trade_review_rejects_other_users_trade(monkeypatch) -> None:
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, "TJTEST10")
        trade = _create_trade(client, "user-10", "TJTEST10")
        response = client.get(
            f"/api/v1/trades/{trade['id']}/review?user_id=someone-else"
        )
    assert response.status_code == 403


def test_get_trade_review_returns_404_for_unknown_trade() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/trades/does-not-exist/review?user_id=user-11"
        )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Fix 12 — source_plan_id must be validated at trade creation
# --------------------------------------------------------------------------


def _sign_plan(
    client: TestClient,
    user_id: str,
    ticker: str,
    entry_price: float = 100.0,
    stop_loss: float = 90.0,
    target_price: float = 120.0,
) -> dict:
    plan_id = str(uuid4())
    response = client.post(
        "/api/v1/plans/save",
        json={
            "plan": {
                "plan_id": plan_id,
                "user_id": user_id,
                "conversation_id": f"conv-{plan_id[:8]}",
                "ticker": ticker,
                "strategy_type": "Bull Call Debit Spread",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "max_loss_usd": 250.0,
                "theta_warning": False,
            }
        },
    )
    assert response.status_code == 200
    return response.json()


def _create_trade_with_plan(
    client: TestClient,
    user_id: str,
    ticker: str,
    plan_id: str,
    entry_price: float = 100.0,
):
    return client.post(
        "/api/v1/trades",
        json={
            "user_id": user_id,
            "ticker": ticker,
            "strategy_type": "Long Call",
            "source_plan_id": plan_id,
            "entry_price": entry_price,
            "position_size": 1,
            "days_to_expiration": 30,
        },
    )


def test_create_trade_accepts_a_signed_same_ticker_plan(monkeypatch) -> None:
    suffix = uuid4().hex[:8].upper()
    ticker = f"TJPLAN{suffix}"
    user_id = f"plan-user-{suffix}"
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, ticker)
        plan = _sign_plan(client, user_id, ticker)
        response = _create_trade_with_plan(client, user_id, ticker, plan["plan_id"])
    assert response.status_code == 200
    assert response.json()["source_plan_id"] == plan["plan_id"]


def test_create_trade_rejects_unknown_source_plan_id(monkeypatch) -> None:
    suffix = uuid4().hex[:8].upper()
    ticker = f"TJPLAN{suffix}"
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, ticker)
        response = _create_trade_with_plan(
            client, f"plan-user-{suffix}", ticker, str(uuid4())
        )
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_create_trade_rejects_another_users_source_plan(monkeypatch) -> None:
    suffix = uuid4().hex[:8].upper()
    ticker = f"TJPLAN{suffix}"
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, ticker)
        plan = _sign_plan(client, f"owner-{suffix}", ticker)
        response = _create_trade_with_plan(
            client, f"intruder-{suffix}", ticker, plan["plan_id"]
        )
    assert response.status_code == 400
    assert "different user" in response.json()["detail"]


def test_create_trade_rejects_ticker_mismatched_source_plan(monkeypatch) -> None:
    suffix = uuid4().hex[:8].upper()
    plan_ticker = f"TJPLANA{suffix}"
    trade_ticker = f"TJPLANB{suffix}"
    user_id = f"plan-user-{suffix}"
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, plan_ticker)
        _seed_cache(client, monkeypatch, trade_ticker)
        plan = _sign_plan(client, user_id, plan_ticker)
        response = _create_trade_with_plan(
            client, user_id, trade_ticker, plan["plan_id"]
        )
    assert response.status_code == 400
    assert plan_ticker in response.json()["detail"]


# --------------------------------------------------------------------------
# Fixes 1 & 2 — the review path's plan flag and units guard, end to end
# --------------------------------------------------------------------------


class _CountingOrchestrator:
    def __init__(self, feedback: str = "Solid, disciplined exit.") -> None:
        self.calls = 0
        self.feedback = feedback
        self.seen_has_source_plan: list[bool] = []
        self.seen_scores: list[int] = []

    async def review_trade(
        self, trade, entry_snapshot, execution_score, has_source_plan
    ):
        self.calls += 1
        self.seen_has_source_plan.append(has_source_plan)
        self.seen_scores.append(execution_score)
        return (self.feedback, ["Booked profit near plan"])


def test_review_reports_a_plan_when_its_levels_match_the_trades_scale(
    monkeypatch,
) -> None:
    suffix = uuid4().hex[:8].upper()
    ticker = f"TJSCORE{suffix}"
    user_id = f"score-user-{suffix}"
    orchestrator = _CountingOrchestrator()
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, ticker)
        plan = _sign_plan(
            client, user_id, ticker,
            entry_price=100.0, stop_loss=90.0, target_price=120.0,
        )
        trade = _create_trade_with_plan(
            client, user_id, ticker, plan["plan_id"], entry_price=100.0
        ).json()
        client.put(
            f"/api/v1/trades/{trade['id']}?user_id={user_id}",
            json={
                "exit_price": 87.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": -1300.0,
            },
        )
        app.state.services.llm = orchestrator
        response = client.post(
            f"/api/v1/trades/{trade['id']}/review?user_id={user_id}"
        )
    assert response.status_code == 200
    assert orchestrator.seen_has_source_plan == [True]
    # Plan path: r_multiple = (87 - 100) / 10 = -1.3 -> band 2. The pnl_pct
    # path would have said 1 for the same -13%, so this pins which ran.
    assert response.json()["execution_score"] == 2


def test_review_falls_back_when_plan_levels_are_underlying_prices(
    monkeypatch,
) -> None:
    """The reported units mismatch, reproduced through the real endpoint.

    Trade.entry_price is an option premium ($4.20); the plan's levels are
    underlying stock prices. Scoring them against each other produced
    planned_risk = |4.20 - 298.45| and collapsed every plan-linked review
    onto the same middling band. The plan must now be treated as
    non-comparable — both in the score AND in what the model is told, so the
    prose can't narrate a comparison that didn't happen.
    """
    suffix = uuid4().hex[:8].upper()
    ticker = f"TJUNITS{suffix}"
    user_id = f"units-user-{suffix}"
    orchestrator = _CountingOrchestrator()
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, ticker)
        plan = _sign_plan(
            client, user_id, ticker,
            entry_price=306.00, stop_loss=298.45, target_price=332.15,
        )
        trade = _create_trade_with_plan(
            client, user_id, ticker, plan["plan_id"], entry_price=4.20
        ).json()
        client.put(
            f"/api/v1/trades/{trade['id']}?user_id={user_id}",
            json={
                "exit_price": 1.05,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": -315.0,
            },
        )
        app.state.services.llm = orchestrator
        response = client.post(
            f"/api/v1/trades/{trade['id']}/review?user_id={user_id}"
        )
    assert response.status_code == 200
    # Honest "no comparable plan" framing, even though source_plan_id is set
    # and the plan loaded successfully.
    assert orchestrator.seen_has_source_plan == [False]
    # pnl_pct is -75% -> worst band. The mismatched plan maths gave r =
    # -3.15/294.25 = -0.011, which sits in the >= -1 band and scored a 3.
    assert response.json()["execution_score"] == 1


# --------------------------------------------------------------------------
# Fix 13 — a repeat review must not buy a second OpenAI completion
# --------------------------------------------------------------------------


def test_review_is_cached_unless_force_is_passed(monkeypatch) -> None:
    suffix = uuid4().hex[:8].upper()
    ticker = f"TJFORCE{suffix}"
    user_id = f"force-user-{suffix}"
    orchestrator = _CountingOrchestrator("First pass.")
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, ticker)
        trade = _create_trade(client, user_id, ticker)
        client.put(
            f"/api/v1/trades/{trade['id']}?user_id={user_id}",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 2000.0,
            },
        )
        app.state.services.llm = orchestrator

        first = client.post(f"/api/v1/trades/{trade['id']}/review?user_id={user_id}")
        assert first.status_code == 200
        assert first.json()["ai_feedback"] == "First pass."
        assert orchestrator.calls == 1

        # An accidental duplicate POST (double-click, naive retry) replays
        # the stored review without spending a second completion.
        orchestrator.feedback = "Second pass."
        cached = client.post(f"/api/v1/trades/{trade['id']}/review?user_id={user_id}")
        assert cached.status_code == 200
        assert cached.json()["ai_feedback"] == "First pass."
        assert orchestrator.calls == 1

        # The deliberate 「重新分析」 button does get a fresh one, and it
        # overwrites what was stored.
        forced = client.post(
            f"/api/v1/trades/{trade['id']}/review?user_id={user_id}&force=true"
        )
        assert forced.status_code == 200
        assert forced.json()["ai_feedback"] == "Second pass."
        assert orchestrator.calls == 2

        stored = client.get(f"/api/v1/trades/{trade['id']}/review?user_id={user_id}")
    assert stored.json()["ai_feedback"] == "Second pass."


def test_cached_review_still_enforces_ownership_and_closed_status(
    monkeypatch,
) -> None:
    """The short-circuit sits behind the 404/403/400 checks, not in front."""
    suffix = uuid4().hex[:8].upper()
    ticker = f"TJFORCE{suffix}"
    user_id = f"force-user-{suffix}"
    orchestrator = _CountingOrchestrator()
    with TestClient(app) as client:
        _seed_cache(client, monkeypatch, ticker)
        trade = _create_trade(client, user_id, ticker)
        client.put(
            f"/api/v1/trades/{trade['id']}?user_id={user_id}",
            json={
                "exit_price": 120.0,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "pnl": 2000.0,
            },
        )
        app.state.services.llm = orchestrator
        client.post(f"/api/v1/trades/{trade['id']}/review?user_id={user_id}")

        response = client.post(
            f"/api/v1/trades/{trade['id']}/review?user_id=someone-else"
        )
    assert response.status_code == 403

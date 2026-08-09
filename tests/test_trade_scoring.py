import pytest

from app.services.trade_scoring import compute_execution_score


@pytest.mark.parametrize(
    "exit_price,expected",
    [
        (130.0, 5),  # met/beat planned target (planned_rr = 20/10 = 2)
        (115.0, 4),  # profitable, short of target
        (95.0, 3),   # small loss within planned risk
        (85.0, 2),   # stop discipline slipped moderately
        (70.0, 1),   # loss ran well past the planned stop
    ],
)
def test_score_with_plan_bullish(exit_price: float, expected: int) -> None:
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=exit_price,
        pnl_pct=0.0,
        stop_loss=90.0,
        target_price=120.0,
    )
    assert score == expected


@pytest.mark.parametrize(
    "exit_price,expected",
    [
        (70.0, 5),
        (85.0, 4),
        (105.0, 3),
        (115.0, 2),
        (130.0, 1),
    ],
)
def test_score_with_plan_bearish(exit_price: float, expected: int) -> None:
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=exit_price,
        pnl_pct=0.0,
        stop_loss=110.0,
        target_price=80.0,
    )
    assert score == expected


@pytest.mark.parametrize(
    "pnl_pct,expected",
    [
        (20.0, 4),
        (5.0, 3),
        (-5.0, 2),
        (-20.0, 1),
    ],
)
def test_score_without_plan(pnl_pct: float, expected: int) -> None:
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=100.0,
        pnl_pct=pnl_pct,
        stop_loss=None,
        target_price=None,
    )
    assert score == expected


def test_score_falls_back_to_pnl_pct_when_stop_equals_entry() -> None:
    # planned_risk would be zero (division by zero) — must fall back safely.
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=100.0,
        pnl_pct=20.0,
        stop_loss=100.0,
        target_price=120.0,
    )
    assert score == 4

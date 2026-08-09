import pytest

from app.services.trade_scoring import compute_execution_score, plan_levels_are_usable


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
    # Filled exactly at the plan's own entry, so the band thresholds are the
    # same numbers they were before plan entry became its own parameter.
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=exit_price,
        pnl_pct=0.0,
        plan_entry_price=100.0,
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
        plan_entry_price=100.0,
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
        plan_entry_price=None,
        stop_loss=None,
        target_price=None,
    )
    assert score == expected


def test_score_falls_back_to_pnl_pct_when_stop_equals_plan_entry() -> None:
    # planned_risk would be zero (division by zero) — must fall back safely.
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=100.0,
        pnl_pct=20.0,
        plan_entry_price=100.0,
        stop_loss=100.0,
        target_price=120.0,
    )
    assert score == 4


def test_chased_entry_above_target_keeps_the_plans_bullish_direction() -> None:
    """Fix 11: direction and planned risk come from the PLAN, not the fill.

    Plan: enter 306, stop 298, target 332 — unambiguously bullish, planned
    risk 8, planned R/R 26/8 = 3.25. The user chases and fills at 335 (above
    the plan's own target), price keeps running, exit 345 — a real winner.

    The old code read direction off `target_price > entry_price` with the
    TRADE's fill (332 > 335 is False), so it graded the winner as a bearish
    trade that moved 10 points against it, and it used |335 - 298| = 37 as
    the planned risk. That produced r_multiple = -0.27 and a score of 3.
    Now: r_multiple = (345 - 335) / 8 = 1.25, below the planned 3.25 but
    positive, so the trade scores a 4.
    """
    score = compute_execution_score(
        entry_price=335.0,
        exit_price=345.0,
        pnl_pct=3.0,
        plan_entry_price=306.0,
        stop_loss=298.0,
        target_price=332.0,
    )
    assert score == 4


def test_fill_landing_near_the_stop_no_longer_collapses_planned_risk() -> None:
    """A fill close to the plan's stop used to shrink the denominator.

    Plan: enter 100, stop 90, target 120 — planned risk 10. The trade
    actually filled late at 91 and exited at 88, a small 3-point loss well
    inside that planned risk. The old denominator was |91 - 90| = 1, making
    the loss a 3R blowout and scoring the worst band, 1. Measured against the
    plan's own 10-point risk it is -0.3R, which is the "small loss within
    planned risk" band, 3.
    """
    score = compute_execution_score(
        entry_price=91.0,
        exit_price=88.0,
        pnl_pct=-3.3,
        plan_entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
    )
    assert score == 3


def test_plausible_plan_levels_use_the_plan_based_path() -> None:
    """Fix 2 (a): same-magnitude levels still score against the plan.

    A losing trade whose loss ran past the planned stop scores 2 on the plan
    path; the pnl_pct path would have said 1 for the same -20%, so this
    genuinely proves which branch ran.
    """
    assert plan_levels_are_usable(100.0, 100.0, 90.0, 120.0) is True
    score = compute_execution_score(
        entry_price=100.0,
        exit_price=87.0,
        pnl_pct=-20.0,
        plan_entry_price=100.0,
        stop_loss=90.0,
        target_price=120.0,
    )
    assert score == 2


def test_option_premium_against_underlying_levels_falls_back_to_pnl_pct() -> None:
    """Fix 2 (b): the exact reported units mismatch.

    Trade.entry_price/exit_price are option premiums ($4.20 -> $6.30, a
    50% winner). UserTradePlan levels are underlying stock prices (entry
    306.00, stop 298.45, target 332.15). The old code computed
    planned_risk = |4.20 - 298.45| = 294.25 and r_multiple = 2.10/294.25 =
    0.007, landing every plan-linked review on the same middling 4
    regardless of outcome. The levels must now be rejected as
    non-comparable and the honest pnl_pct band used instead.
    """
    assert plan_levels_are_usable(4.20, 306.00, 298.45, 332.15) is False
    score = compute_execution_score(
        entry_price=4.20,
        exit_price=6.30,
        pnl_pct=50.0,
        plan_entry_price=306.00,
        stop_loss=298.45,
        target_price=332.15,
    )
    # pnl_pct 50% >= 15 -> 4. The mismatched plan path would also have said
    # 4 here, so the predicate assertion above is what pins the behaviour;
    # the losing case below shows the two paths actually diverge.
    assert score == 4

    losing = compute_execution_score(
        entry_price=4.20,
        exit_price=1.05,
        pnl_pct=-75.0,
        plan_entry_price=306.00,
        stop_loss=298.45,
        target_price=332.15,
    )
    # Honest answer for a -75% trade is the worst band. Under the mismatched
    # plan maths, realized_move = 1.05 - 4.20 = -3.15 over a 294.25 "risk"
    # is r = -0.011, which sits in the >= -1 band and scores a 3.
    assert losing == 1


def test_plan_levels_within_ten_times_the_entry_are_still_accepted() -> None:
    """The ratio cap is deliberately loose enough for a real option plan.

    An option-premium plan on a $4.20 contract with a $1.00 stop and a
    $21.00 lotto target is exactly 5x on either side — well inside the 10x
    cap, so a legitimate same-units plan is never rejected by it.
    """
    assert plan_levels_are_usable(4.20, 4.20, 1.00, 21.00) is True

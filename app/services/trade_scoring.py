"""Deterministic execution-discipline scoring for closed trades.

Kept as a pure function, isolated from the LLM orchestrator, so the score a
user sees is always reproducible and never model-generated — see
docs/superpowers/specs/2026-08-09-trade-journal-ai-review-design.md.

Two distinct price scales meet in here, and keeping them straight is the
whole point of the guard below:

* ``entry_price`` / ``exit_price`` come from ``Trade`` and are OPTION
  PREMIUMS per share (a $4.20 contract), which is why
  ``TradeRepository.close_trade`` multiplies by the 100-share contract
  multiplier to get P&L.
* ``plan_entry_price`` / ``stop_loss`` / ``target_price`` come from
  ``UserTradePlan`` and are UNDERLYING STOCK prices ($311.00), because the
  orchestrator derives them from GEX levels.

Reconciling those two scales properly is a schema decision (should a trade
record the underlying price at fill? should a plan carry option-leg levels?)
that is queued for a human. Until then, this module refuses to grade a plan
whose levels obviously aren't denominated in the same units as the trade's
own fill, and falls back to the honest pnl_pct approximation instead.
"""

import logging


logger = logging.getLogger(__name__)


# Ratio between a plan level and the trade's own entry price beyond which the
# two are almost certainly not denominated in the same units.
#
# Why 10x: the real mismatch this catches is an option premium graded against
# an underlying price, and that gap is structurally large — a $2-$15 premium
# against a $100-$600 underlying lands somewhere between 20x and 100x (the
# observed case was $4.20 vs $298.45, i.e. 71x). Same-unit relationships
# never get close: a stock-price plan graded against a stock-price fill stays
# inside ~2x even when the entry is badly chased, and even an aggressive
# option-premium plan targeting a 5x-10x return on a lotto contract only just
# reaches the boundary. 10x therefore sits in the empty space between the two
# populations, with the margin deliberately on the side of NOT falsely
# rejecting a legitimate plan.
MAX_PLAN_LEVEL_TO_ENTRY_RATIO = 10.0


def plan_levels_are_usable(
    entry_price: float,
    plan_entry_price: float | None,
    stop_loss: float | None,
    target_price: float | None,
) -> bool:
    """Whether the plan's levels can meaningfully grade this trade.

    Also used by the review endpoint to decide what to tell the model about
    whether a comparable plan existed, so the prose and the score never
    disagree about which scoring path ran.
    """
    if plan_entry_price is None or stop_loss is None or target_price is None:
        return False
    if entry_price <= 0:
        return False
    # A zero-width planned risk or reward has nothing to grade against.
    if stop_loss == plan_entry_price or target_price == plan_entry_price:
        return False
    for level in (plan_entry_price, stop_loss, target_price):
        if level <= 0:
            return False
        ratio = max(level, entry_price) / min(level, entry_price)
        if ratio > MAX_PLAN_LEVEL_TO_ENTRY_RATIO:
            logger.warning(
                "Plan level %.4f is %.1fx the trade's entry price %.4f — treating "
                "the plan as not comparable (likely option premium vs underlying "
                "price) and scoring on pnl_pct instead.",
                level,
                ratio,
                entry_price,
            )
            return False
    return True


def compute_execution_score(
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    plan_entry_price: float | None,
    stop_loss: float | None,
    target_price: float | None,
) -> int:
    if plan_levels_are_usable(
        entry_price, plan_entry_price, stop_loss, target_price
    ):
        assert plan_entry_price is not None
        assert stop_loss is not None
        assert target_price is not None
        return _score_with_plan(
            plan_entry_price, stop_loss, target_price, entry_price, exit_price
        )
    return _score_without_plan(pnl_pct)


def _score_with_plan(
    plan_entry_price: float,
    stop_loss: float,
    target_price: float,
    entry_price: float,
    exit_price: float,
) -> int:
    """Grade the trade's realized move against the plan's own geometry.

    Direction and planned risk come strictly from the PLAN (its own entry,
    stop and target); the realized move comes strictly from the TRADE (what
    actually filled and what it actually exited at). Mixing the two — reading
    direction off ``target_price > entry_price`` with the trade's fill — used
    to invert the sign for any entry chased past its own target, and shrank
    the denominator whenever a fill happened to land near the stop.
    """
    bullish = target_price > plan_entry_price
    planned_risk = abs(plan_entry_price - stop_loss)
    planned_reward = abs(target_price - plan_entry_price)
    planned_rr = planned_reward / planned_risk
    realized_move = (
        (exit_price - entry_price) if bullish else (entry_price - exit_price)
    )
    r_multiple = realized_move / planned_risk

    if r_multiple >= planned_rr:
        return 5
    if r_multiple > 0:
        return 4
    if r_multiple >= -1:
        return 3
    if r_multiple >= -1.5:
        return 2
    return 1


def _score_without_plan(pnl_pct: float) -> int:
    if pnl_pct >= 15:
        return 4
    if pnl_pct >= 0:
        return 3
    if pnl_pct >= -10:
        return 2
    return 1

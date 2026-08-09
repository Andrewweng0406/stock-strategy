"""Deterministic execution-discipline scoring for closed trades.

Kept as a pure function, isolated from the LLM orchestrator, so the score a
user sees is always reproducible and never model-generated — see
docs/superpowers/specs/2026-08-09-trade-journal-ai-review-design.md.
"""


def compute_execution_score(
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    stop_loss: float | None,
    target_price: float | None,
) -> int:
    has_plan = (
        stop_loss is not None
        and target_price is not None
        and stop_loss != entry_price
    )
    if has_plan:
        return _score_with_plan(entry_price, exit_price, stop_loss, target_price)
    return _score_without_plan(pnl_pct)


def _score_with_plan(
    entry_price: float, exit_price: float, stop_loss: float, target_price: float
) -> int:
    bullish = target_price > entry_price
    planned_risk = abs(entry_price - stop_loss)
    planned_reward = abs(target_price - entry_price)
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

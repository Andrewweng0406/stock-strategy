"""app.pinning_engine 是純計算層，不做 I/O，直接用合成數據測試即可。

移植自 smart-money 專案的同名測試檔——邏輯完全沒改，這裡保留核心案例，
確保搬過來的檔案行為跟原本驗證過的一致。
"""

from __future__ import annotations

import pytest

from app.pinning_engine import (
    PIN_CANDIDATE_MAX_DISTANCE_PCT,
    PIN_OI_CONCENTRATION_MIN_PCT,
    PIN_PROXIMITY_THRESHOLD_PCT,
    compute_pinning_analysis,
    find_pin_strike,
    score_pinning,
)


def _row(strike: float, call_oi: float, put_oi: float) -> dict:
    return {"strike": strike, "call_oi": call_oi, "put_oi": put_oi}


def test_find_pin_strike_picks_largest_combined_oi():
    rows = [_row(90, 100, 100), _row(100, 500, 600), _row(110, 200, 50)]
    assert find_pin_strike(rows) == 100


def test_find_pin_strike_empty_input_returns_none():
    assert find_pin_strike([]) is None


def test_find_pin_strike_with_spot_excludes_far_stale_leaps_oi():
    spot = 328.58
    rows = [_row(330, 1000, 1000), _row(400, 50000, 50000)]
    assert find_pin_strike(rows, spot) == 330


def test_find_pin_strike_with_spot_falls_back_to_full_chain_when_nothing_nearby():
    spot = 100.0
    rows = [_row(400, 5000, 5000)]
    assert find_pin_strike(rows, spot) == 400


def test_compute_pinning_analysis_empty_chain_returns_none():
    assert (
        compute_pinning_analysis(
            [], spot=100.0, max_pain=100.0, call_wall=105.0, put_wall=95.0,
            in_positive_gamma=True,
        )
        is None
    )


def test_breakout_when_spot_above_call_wall():
    rows = [_row(100, 1000, 1000)]
    result = compute_pinning_analysis(
        rows, spot=106.0, max_pain=100.0, call_wall=105.0, put_wall=95.0,
        in_positive_gamma=True,
    )
    assert result["regime"] == "BREAKOUT"
    assert result["has_broken_wall"] is True


def test_breakout_when_spot_below_put_wall():
    rows = [_row(100, 1000, 1000)]
    result = compute_pinning_analysis(
        rows, spot=94.0, max_pain=100.0, call_wall=105.0, put_wall=95.0,
        in_positive_gamma=True,
    )
    assert result["regime"] == "BREAKOUT"


def test_pinning_when_close_concentrated_and_positive_gamma():
    rows = [_row(100, 5000, 5000), _row(90, 10, 10), _row(110, 10, 10)]
    result = compute_pinning_analysis(
        rows, spot=100.2, max_pain=100.0, call_wall=105.0, put_wall=95.0,
        in_positive_gamma=True,
    )
    assert result["regime"] == "PINNING"
    assert result["pin_strike"] == 100
    assert result["pin_strike_matches_max_pain"] is True
    assert result["score"] >= 60


def test_neutral_when_negative_gamma_even_if_close_and_concentrated():
    rows = [_row(100, 5000, 5000), _row(90, 10, 10), _row(110, 10, 10)]
    result = compute_pinning_analysis(
        rows, spot=100.2, max_pain=100.0, call_wall=105.0, put_wall=95.0,
        in_positive_gamma=False,
    )
    assert result["regime"] == "NEUTRAL"


def test_neutral_when_far_from_pin_strike():
    rows = [_row(100, 5000, 5000), _row(90, 10, 10), _row(110, 10, 10)]
    result = compute_pinning_analysis(
        rows, spot=103.0, max_pain=100.0, call_wall=105.0, put_wall=95.0,
        in_positive_gamma=True,
    )
    assert result["distance_pct"] > PIN_PROXIMITY_THRESHOLD_PCT
    assert result["regime"] == "NEUTRAL"


def test_neutral_when_oi_too_spread_out_despite_proximity():
    rows = [_row(90 + i, 50, 50) for i in range(21)]
    pin_strike = find_pin_strike(rows)
    result = compute_pinning_analysis(
        rows, spot=pin_strike, max_pain=pin_strike, call_wall=pin_strike + 20,
        put_wall=pin_strike - 20, in_positive_gamma=True,
    )
    assert result["oi_concentration_pct"] < PIN_OI_CONCENTRATION_MIN_PCT
    assert result["regime"] == "NEUTRAL"


def test_score_pinning_matches_compute_pinning_analysis_with_same_inputs():
    rows = [_row(100, 5000, 5000)]
    via_full = compute_pinning_analysis(
        rows, spot=100.2, max_pain=100.0, call_wall=105.0, put_wall=95.0,
        in_positive_gamma=True,
    )
    via_direct = score_pinning(
        spot=100.2, pin_strike=100.0, oi_concentration_pct=100.0, max_pain=100.0,
        call_wall=105.0, put_wall=95.0, in_positive_gamma=True,
    )
    assert via_full == via_direct


def test_score_pinning_none_pin_strike_returns_none():
    assert (
        score_pinning(
            spot=100.0, pin_strike=None, oi_concentration_pct=50.0, max_pain=100.0,
            call_wall=105.0, put_wall=95.0, in_positive_gamma=True,
        )
        is None
    )


def test_perfect_conditions_score_near_maximum():
    rows = [_row(100, 5000, 5000)]
    result = compute_pinning_analysis(
        rows, spot=100.0, max_pain=100.0, call_wall=105.0, put_wall=95.0,
        in_positive_gamma=True,
    )
    assert result["score"] == 100


@pytest.mark.parametrize("in_positive_gamma", [True, False])
def test_gamma_score_is_binary_not_scaled(in_positive_gamma):
    rows = [_row(100, 5000, 5000)]
    result = compute_pinning_analysis(
        rows, spot=100.0, max_pain=100.0, call_wall=105.0, put_wall=95.0,
        in_positive_gamma=in_positive_gamma,
    )
    expected_score = 100 if in_positive_gamma else 70
    assert result["score"] == expected_score

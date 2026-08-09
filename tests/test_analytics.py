from datetime import date, datetime, timedelta, timezone

import pytest

from app import analytics, pinning_engine
from app.analytics import (
    HIGH_RISK_WARNING,
    GEXCalculator,
    OptionContract,
    _aggregate_oi_by_strike,
    _calculate_max_pain,
    compute_pinning_for_contracts,
    parse_gex_risk_profile,
)
from app.market_data import classify_expiration
from app.models import ExpirationType, GEXStatus, OptionGEXSummary


def summary(status: GEXStatus) -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker="SPY",
        stock_price=500,
        zero_gamma=505 if status == GEXStatus.NEG_GAMMA else 495,
        call_wall=510,
        put_wall=490,
        iv_rank=50,
        net_gex=-1_000_000 if status == GEXStatus.NEG_GAMMA else 1_000_000,
        gex_status=status,
        calculated_at=datetime.now(timezone.utc),
    )


def test_short_dated_negative_gamma_locks_warning() -> None:
    risk = parse_gex_risk_profile(summary(GEXStatus.NEG_GAMMA), 6)
    assert risk.locked_warning is True
    assert risk.risk_level == "HIGH"
    assert risk.warnings == [HIGH_RISK_WARNING]


def test_seven_days_does_not_lock_warning() -> None:
    risk = parse_gex_risk_profile(summary(GEXStatus.NEG_GAMMA), 7)
    assert risk.locked_warning is False
    assert risk.warnings == []


def test_positive_gamma_is_mean_reverting() -> None:
    risk = parse_gex_risk_profile(summary(GEXStatus.POS_GAMMA), 2)
    assert risk.volatility_regime == "LOW_VOL_MEAN_REVERSION"
    assert risk.risk_level == "NORMAL"


def _contract(strike: float, option_type: str, open_interest: int) -> OptionContract:
    return OptionContract(
        code=f"US.TEST{strike}{option_type}", option_type=option_type, strike=strike,
        expiration_date=date(2026, 9, 4), implied_volatility=0.4, delta=0.5,
        market_gamma=0.01, open_interest=open_interest, contract_size=100,
    )


def test_aggregate_oi_by_strike_sums_call_and_put_separately() -> None:
    contracts = [
        _contract(100, "CALL", 500),
        _contract(100, "PUT", 300),
        _contract(100, "CALL", 200),  # 同一履約價、不同到期日聚合進來，應該加總
        _contract(110, "PUT", 1000),
    ]
    rows = {row["strike"]: row for row in _aggregate_oi_by_strike(contracts)}
    assert rows[100]["call_oi"] == 700
    assert rows[100]["put_oi"] == 300
    assert rows[110]["call_oi"] == 0
    assert rows[110]["put_oi"] == 1000


def test_calculate_max_pain_picks_strike_minimizing_intrinsic_value() -> None:
    gex_by_strike = [
        {"strike": 90, "call_oi": 0, "put_oi": 1000},
        {"strike": 100, "call_oi": 100, "put_oi": 100},
        {"strike": 110, "call_oi": 1000, "put_oi": 0},
    ]
    # 兩側都集中在極端履約價，中間 100 讓雙方到期內在價值總和最小。
    assert _calculate_max_pain(gex_by_strike) == 100


def test_compute_pinning_for_contracts_returns_analysis_matching_pin_strike() -> None:
    contracts = [
        _contract(100, "CALL", 5000), _contract(100, "PUT", 5000),
        _contract(90, "CALL", 10), _contract(90, "PUT", 10),
        _contract(110, "CALL", 10), _contract(110, "PUT", 10),
    ]
    gex_summary = OptionGEXSummary(
        ticker="AAPL", stock_price=100.2, zero_gamma=95, call_wall=105, put_wall=95,
        iv_rank=40, net_gex=1_000_000, gex_status=GEXStatus.POS_GAMMA,
    )
    pinning = compute_pinning_for_contracts(contracts, gex_summary)
    assert pinning is not None
    assert pinning.pin_strike == 100
    assert pinning.regime == "PINNING"


def test_compute_pinning_for_contracts_returns_none_for_empty_contracts() -> None:
    gex_summary = OptionGEXSummary(
        ticker="AAPL", stock_price=100.0, zero_gamma=95, call_wall=105, put_wall=95,
        iv_rank=40, net_gex=1_000_000, gex_status=GEXStatus.POS_GAMMA,
    )
    assert compute_pinning_for_contracts([], gex_summary) is None


# ---------- GEXCalculator: golden synthetic chains ----------
#
# Every test below builds a small hand-checkable option chain and freezes
# "now" so the numbers are deterministic. `now` is frozen at 10:30 AM ET on
# Monday 2026-08-10, which is 5.5 hours before that day's 4:00 PM ET close.

FROZEN_NOW = datetime(2026, 8, 10, 10, 30, tzinfo=analytics.MARKET_TIMEZONE)
TODAY = FROZEN_NOW.date()
NEXT_MONTH = date(2026, 9, 18)
RATE = 0.045


@pytest.fixture
def frozen_now(monkeypatch) -> datetime:
    monkeypatch.setattr(analytics, "market_now", lambda: FROZEN_NOW)
    return FROZEN_NOW


def chain_contract(
    option_type: str,
    strike: float,
    open_interest: int = 1000,
    implied_volatility: float = 0.30,
    expiration_date: date = NEXT_MONTH,
    market_gamma: float = 0.0,
) -> OptionContract:
    return OptionContract(
        code=f"US.GOLD{strike}{option_type}",
        option_type=option_type,
        strike=strike,
        expiration_date=expiration_date,
        implied_volatility=implied_volatility,
        delta=0.5,
        market_gamma=market_gamma,
        open_interest=open_interest,
        contract_size=100,
    )


def calculator() -> GEXCalculator:
    return GEXCalculator(risk_free_rate=RATE)


# --- Fix 1/3: net_gex sign drives gex_status ---

def test_call_heavy_chain_is_positive_net_gex_and_positive_gamma(frozen_now) -> None:
    contracts = [
        chain_contract("CALL", 100.0, open_interest=10_000),
        chain_contract("PUT", 100.0, open_interest=1_000),
    ]
    result = calculator().calculate("SPY", 100.0, contracts)
    assert result.net_gex > 0
    assert result.gex_status == GEXStatus.POS_GAMMA


def test_put_heavy_chain_is_negative_net_gex_and_negative_gamma(frozen_now) -> None:
    contracts = [
        chain_contract("CALL", 100.0, open_interest=1_000),
        chain_contract("PUT", 100.0, open_interest=10_000),
    ]
    result = calculator().calculate("SPY", 100.0, contracts)
    assert result.net_gex < 0
    assert result.gex_status == GEXStatus.NEG_GAMMA


def test_zero_gamma_lands_on_a_real_crossing_between_the_two_strikes(frozen_now) -> None:
    """Equal-size put at 95 and call at 105 around a spot of 100: the put's
    gamma dominates below, the call's dominates above, so the profile must
    cross zero once, between the two strikes and near spot.
    """
    contracts = [
        chain_contract("PUT", 95.0, open_interest=5_000),
        chain_contract("CALL", 105.0, open_interest=5_000),
    ]
    calc = calculator()
    result = calc.calculate("SPY", 100.0, contracts)

    assert result.zero_gamma is not None
    assert 95.0 < result.zero_gamma < 105.0
    # It's an actual root, not a nearest-to-zero grid point: net gamma
    # evaluated there is ~0 relative to the profile's own scale.
    scale = max(
        abs(calc._net_at(contracts, level, FROZEN_NOW))
        for level in (95.0, 105.0)
    )
    assert abs(calc._net_at(contracts, result.zero_gamma, FROZEN_NOW)) < scale / 1000
    # ...and the sign really does flip across it.
    assert calc._net_at(contracts, result.zero_gamma - 2, FROZEN_NOW) < 0
    assert calc._net_at(contracts, result.zero_gamma + 2, FROZEN_NOW) > 0


def test_no_crossing_returns_none_and_status_still_comes_from_net_gamma_sign(
    frozen_now,
) -> None:
    """A lone ITM put above spot: dealer gamma is negative at every price in
    the window, so there is no zero-gamma level at all.

    This is the exact shape the old fallback got wrong. It returned the grid
    boundary furthest from the strike (a level *below* spot), and
    `stock_price > zero_gamma` then reported POS_GAMMA for a book that is
    unambiguously short gamma.
    """
    contracts = [chain_contract("PUT", 120.0, open_interest=5_000)]
    result = calculator().calculate("SPY", 100.0, contracts)

    assert result.zero_gamma is None
    assert result.net_gex < 0
    assert result.gex_status == GEXStatus.NEG_GAMMA


def test_no_crossing_call_only_chain_is_positive_gamma(frozen_now) -> None:
    contracts = [chain_contract("CALL", 120.0, open_interest=5_000)]
    result = calculator().calculate("SPY", 100.0, contracts)

    assert result.zero_gamma is None
    assert result.gex_status == GEXStatus.POS_GAMMA


def test_summary_with_null_levels_is_serialisable_and_pinning_still_works(
    frozen_now,
) -> None:
    """The nullable levels have to survive the whole downstream path, not
    just the calculator: JSON round-trip (cache/cloud sync) and the pinning
    engine, which must not read a missing wall as a breakout.
    """
    contracts = [chain_contract("PUT", 120.0, open_interest=5_000)]
    result = calculator().calculate("SPY", 100.0, contracts)
    assert result.call_wall is None

    restored = OptionGEXSummary.model_validate_json(result.model_dump_json())
    assert restored.zero_gamma is None
    assert restored.call_wall is None

    pinning = compute_pinning_for_contracts(contracts, result)
    assert pinning is not None
    assert pinning.has_broken_wall is False


# --- Fix 2: walls stay on their own side of spot ---

@pytest.mark.parametrize(
    "contracts",
    [
        pytest.param(
            [
                # Naive argmax picks the 95 call (ITM, below spot) because its
                # open interest dwarfs the 105 call's.
                chain_contract("CALL", 95.0, open_interest=20_000),
                chain_contract("CALL", 105.0, open_interest=1_000),
                chain_contract("PUT", 105.0, open_interest=20_000),
                chain_contract("PUT", 95.0, open_interest=1_000),
            ],
            id="leftover-itm-oi-on-both-sides",
        ),
        pytest.param(
            [
                chain_contract("CALL", 100.0, open_interest=3_000),
                chain_contract("CALL", 110.0, open_interest=2_000),
                chain_contract("PUT", 100.0, open_interest=3_000),
                chain_contract("PUT", 90.0, open_interest=2_000),
            ],
            id="peaks-at-the-money",
        ),
    ],
)
def test_walls_are_always_on_the_correct_side_of_spot(frozen_now, contracts) -> None:
    spot = 100.0
    result = calculator().calculate("SPY", spot, contracts)
    assert result.call_wall is not None and result.call_wall >= spot
    assert result.put_wall is not None and result.put_wall <= spot


def test_wall_search_ignores_the_wrong_side_even_when_it_has_more_exposure(
    frozen_now,
) -> None:
    contracts = [
        chain_contract("CALL", 95.0, open_interest=20_000),
        chain_contract("CALL", 105.0, open_interest=1_000),
        chain_contract("PUT", 105.0, open_interest=20_000),
        chain_contract("PUT", 95.0, open_interest=1_000),
    ]
    result = calculator().calculate("SPY", 100.0, contracts)
    assert result.call_wall == 105.0
    assert result.put_wall == 95.0


def test_wall_is_none_when_that_side_has_no_qualifying_strike(frozen_now) -> None:
    contracts = [
        chain_contract("CALL", 90.0, open_interest=5_000),  # only below spot
        chain_contract("PUT", 90.0, open_interest=5_000),
    ]
    result = calculator().calculate("SPY", 100.0, contracts)
    assert result.call_wall is None
    assert result.put_wall == 90.0


def test_pinning_engine_treats_a_missing_wall_as_no_breakout_evidence() -> None:
    rows = [{"strike": 100.0, "call_oi": 5_000, "put_oi": 5_000}]
    both_missing = pinning_engine.score_pinning(
        spot=100.0, pin_strike=100.0, oi_concentration_pct=100.0, max_pain=100.0,
        call_wall=None, put_wall=None, in_positive_gamma=True,
    )
    assert both_missing["has_broken_wall"] is False
    assert both_missing["regime"] == "PINNING"

    # A wall that does exist and has been crossed still counts.
    broken = pinning_engine.compute_pinning_analysis(
        rows, spot=100.0, max_pain=100.0, call_wall=None, put_wall=101.0,
        in_positive_gamma=True,
    )
    assert broken["has_broken_wall"] is True
    assert broken["regime"] == "BREAKOUT"


# --- Fix 4: implausible IV is dropped, not floored ---

@pytest.mark.parametrize("bad_iv", [1e-5, 0.0, 10.0])
def test_implausible_iv_contract_is_excluded_not_floored(frozen_now, bad_iv) -> None:
    """Under the old `max(iv, 0.01)` clamp a 1e-5 quote became a 1% IV, and
    because BS gamma scales as 1/(sigma*sqrt(T)) that single row produced
    more gamma than the entire rest of the chain and captured the call wall.
    Excluding it must leave the summary bit-for-bit identical to a chain
    that never contained the row.
    """
    clean = [
        chain_contract("CALL", 105.0, open_interest=4_000),
        chain_contract("PUT", 95.0, open_interest=4_000),
    ]
    polluted = clean + [chain_contract("CALL", 100.0, implied_volatility=bad_iv)]

    calc = calculator()
    baseline = calc.calculate("SPY", 100.0, clean)
    result = calc.calculate("SPY", 100.0, polluted)

    assert result.model_dump(exclude={"calculated_at"}) == baseline.model_dump(
        exclude={"calculated_at"}
    )
    assert result.call_wall == 105.0  # not the garbage-quote strike


def test_iv_band_boundaries_are_inclusive() -> None:
    assert analytics.is_plausible_iv(0.01)
    assert analytics.is_plausible_iv(5.0)
    assert not analytics.is_plausible_iv(0.009)
    assert not analytics.is_plausible_iv(5.01)


def test_chain_of_only_implausible_contracts_raises_rather_than_inventing_gex(
    frozen_now,
) -> None:
    contracts = [chain_contract("CALL", 105.0, implied_volatility=1e-5)]
    with pytest.raises(ValueError, match="No valid option contracts"):
        calculator().calculate("SPY", 100.0, contracts)


# --- Fix 5/6: intraday time-to-expiry and market-time valuation date ---

def test_same_day_expiry_uses_hours_not_a_whole_day_floor() -> None:
    years = analytics.years_to_expiry(TODAY, FROZEN_NOW)
    assert years == pytest.approx(5.5 / (365 * 24))

    calc = calculator()
    intraday_gamma = calc._bs_gamma(100.0, 100.0, years, 0.30)
    one_day_floor_gamma = calc._bs_gamma(100.0, 100.0, 1 / 365, 0.30)
    # gamma ~ 1/sqrt(T): 5.5h vs 24h is a ~2x understatement under the old floor.
    assert intraday_gamma > one_day_floor_gamma * 1.9


def test_gamma_grows_as_the_0dte_session_progresses() -> None:
    calc = calculator()
    morning = analytics.years_to_expiry(
        TODAY, datetime(2026, 8, 10, 9, 30, tzinfo=analytics.MARKET_TIMEZONE)
    )
    afternoon = analytics.years_to_expiry(
        TODAY, datetime(2026, 8, 10, 15, 0, tzinfo=analytics.MARKET_TIMEZONE)
    )
    assert afternoon < morning
    assert calc._bs_gamma(100.0, 100.0, afternoon, 0.30) > calc._bs_gamma(
        100.0, 100.0, morning, 0.30
    )


def test_time_to_expiry_floors_at_one_hour_just_before_the_close() -> None:
    minutes_left = analytics.years_to_expiry(
        TODAY, datetime(2026, 8, 10, 15, 59, tzinfo=analytics.MARKET_TIMEZONE)
    )
    assert minutes_left == analytics.MIN_YEARS_TO_EXPIRY


def test_expired_contract_has_no_time_value_and_is_excluded(frozen_now) -> None:
    yesterday = TODAY - timedelta(days=1)
    assert analytics.years_to_expiry(yesterday, FROZEN_NOW) is None
    # ...and so is a contract that expired earlier the same day.
    assert (
        analytics.years_to_expiry(
            TODAY, datetime(2026, 8, 10, 16, 30, tzinfo=analytics.MARKET_TIMEZONE)
        )
        is None
    )

    live = chain_contract("CALL", 105.0, open_interest=4_000)
    expired = chain_contract("CALL", 105.0, open_interest=400_000,
                             expiration_date=yesterday)
    calc = calculator()
    baseline = calc.calculate("SPY", 100.0, [live])
    with_expired = calc.calculate("SPY", 100.0, [live, expired])
    assert with_expired.net_gex == baseline.net_gex

    with pytest.raises(ValueError, match="No valid option contracts"):
        calc.calculate("SPY", 100.0, [expired])


def test_valuation_date_follows_us_eastern_not_utc(monkeypatch) -> None:
    """9:00 PM ET on 2026-08-10 is already 2026-08-11 in UTC. The old
    `datetime.now(timezone.utc).date()` therefore reported tomorrow's date
    for roughly four hours every evening, which flips tomorrow's 1DTE chain
    into 0DTE a whole session early.
    """
    utc_instant = datetime(2026, 8, 11, 1, 0, tzinfo=timezone.utc)
    assert utc_instant.date() == date(2026, 8, 11)  # what the old code saw

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return utc_instant.astimezone(tz)

    monkeypatch.setattr(analytics, "datetime", _FrozenDatetime)

    assert analytics.market_today() == date(2026, 8, 10)
    assert (
        classify_expiration(date(2026, 8, 11), analytics.market_today())
        == ExpirationType.ONE_DTE
    )


# --- Fix 3: one gamma basis shared by the headline and the curve ---

def test_headline_net_gex_and_the_zero_gamma_curve_share_one_basis(frozen_now) -> None:
    """With broker gamma supplied, the headline number and the zero-gamma
    curve used to be computed from different gamma sources (market vs
    Black-Scholes) and could disagree in sign. They are now the same
    function: the curve's sample AT spot IS the headline number.
    """
    contracts = [
        chain_contract("CALL", 105.0, open_interest=4_000, market_gamma=0.02),
        chain_contract("PUT", 95.0, open_interest=6_000, market_gamma=0.05),
    ]
    calc = calculator()
    result = calc.calculate("SPY", 100.0, contracts)

    curve_at_spot = calc._net_at(contracts, 100.0, FROZEN_NOW, at_current_spot=True)
    assert result.net_gex == pytest.approx(curve_at_spot, rel=1e-9)
    assert result.gex_status == (
        GEXStatus.POS_GAMMA if curve_at_spot > 0 else GEXStatus.NEG_GAMMA
    )
    # Broker gamma is only claimed to be valid at the current spot; away
    # from it the curve is recomputed from Black-Scholes.
    assert calc._gamma(contracts[0], 100.0, FROZEN_NOW, True) == 0.02
    assert calc._gamma(contracts[0], 100.0, FROZEN_NOW, False) != 0.02


def test_zero_gamma_grid_resolution_does_not_depend_on_a_far_leaps_strike(
    frozen_now,
) -> None:
    """Fix 7: a single far-out LEAPS strike used to stretch the search
    window (min_strike*0.9 .. max_strike*1.1) over a fixed step count and
    coarsen resolution right where it matters. The window is now pinned to
    spot, so adding a distant strike with negligible near-spot gamma leaves
    the answer essentially unchanged.
    """
    core = [
        chain_contract("PUT", 95.0, open_interest=5_000),
        chain_contract("CALL", 105.0, open_interest=5_000),
    ]
    with_leaps = core + [
        chain_contract("CALL", 400.0, open_interest=10, expiration_date=date(2028, 1, 21))
    ]
    calc = calculator()
    tight = calc.calculate("SPY", 100.0, core).zero_gamma
    wide = calc.calculate("SPY", 100.0, with_leaps).zero_gamma
    assert tight is not None and wide is not None
    assert wide == pytest.approx(tight, abs=0.05)

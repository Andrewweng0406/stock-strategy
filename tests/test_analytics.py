from datetime import date, datetime, timezone

from app.analytics import (
    HIGH_RISK_WARNING,
    OptionContract,
    _aggregate_oi_by_strike,
    _calculate_max_pain,
    compute_pinning_for_contracts,
    parse_gex_risk_profile,
)
from app.models import GEXStatus, OptionGEXSummary


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

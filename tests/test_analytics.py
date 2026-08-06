from datetime import datetime, timezone

from app.analytics import HIGH_RISK_WARNING, parse_gex_risk_profile
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

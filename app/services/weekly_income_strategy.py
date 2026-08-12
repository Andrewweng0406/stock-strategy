from __future__ import annotations

from app.database import GEXSnapshotRepository
from app.models import (
    CSPRecommendation,
    GEXSnapshot,
    GEXStatus,
    LPRangeRecommendation,
    StrategyDataQuality,
)


def _price_tick(price: float) -> float:
    if price < 50:
        return 0.5
    if price < 200:
        return 2.5
    return 5.0


def _round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 2)


def _quality(snapshot: GEXSnapshot | None) -> StrategyDataQuality:
    if snapshot is None:
        return StrategyDataQuality()
    return StrategyDataQuality(
        data_source=snapshot.data_source,
        is_delayed=snapshot.is_delayed,
        is_synthetic=snapshot.is_synthetic,
        is_stale=snapshot.is_stale,
        captured_at=snapshot.captured_at,
    )


def _data_quality_warnings(snapshot: GEXSnapshot | None) -> list[str]:
    if snapshot is None:
        return ["NO_GEX_SNAPSHOT"]
    warnings: list[str] = []
    if snapshot.is_synthetic:
        warnings.append("SYNTHETIC_DATA")
    if snapshot.is_stale:
        warnings.append("STALE_GEX_SNAPSHOT")
    if snapshot.is_delayed:
        warnings.append("DELAYED_MARKET_DATA")
    return warnings


def _estimated_delta_from_margin(margin_pct: float, dte: int, negative_gamma: bool) -> float:
    # Heuristic only: this is NOT an option-chain delta. It converts a GEX
    # distance buffer into a conservative probability proxy for UI triage.
    base = 0.32 - min(max(margin_pct, 0), 20) * 0.012
    if dte <= 7:
        base -= 0.03
    elif dte >= 30:
        base += 0.04
    if negative_gamma:
        base += 0.05
    return round(min(max(base, 0.05), 0.45), 2)


class WeeklyIncomeStrategyService:
    def __init__(self, snapshots: GEXSnapshotRepository) -> None:
        self.snapshots = snapshots

    async def _snapshot_for(self, ticker: str, dte: int | None = None) -> GEXSnapshot | None:
        ticker = ticker.strip().upper()
        if dte is not None:
            exact = await self.snapshots.latest_snapshot(ticker, dte)
            if exact is not None:
                return exact
        candidates = await self.snapshots.list_snapshots(ticker, limit=30)
        if not candidates:
            return None
        if dte is None:
            return candidates[0]
        return min(
            candidates,
            key=lambda snapshot: (
                abs(snapshot.days_to_expiration - dte),
                -snapshot.captured_at.timestamp(),
            ),
        )

    async def csp_recommendation(self, ticker: str, dte: int) -> CSPRecommendation:
        snapshot = await self._snapshot_for(ticker, dte)
        warnings = _data_quality_warnings(snapshot)
        warnings.append("EARNINGS_WINDOW_UNKNOWN")
        if snapshot is None:
            return CSPRecommendation(
                ticker=ticker.strip().upper(),
                dte=dte,
                data_quality=_quality(None),
                warnings=warnings,
            )

        spot = snapshot.underlying_price
        put_wall = snapshot.put_wall_strike
        recommended: float | None = None
        margin: float | None = None
        estimated_delta: float | None = None
        estimated_win_probability: float | None = None
        weekly_target_yield: float | None = None
        annualized_yield: float | None = None

        if put_wall is None:
            warnings.append("MISSING_PUT_WALL")
        elif put_wall >= spot:
            warnings.append("PUT_WALL_ABOVE_OR_AT_SPOT")
        else:
            tick = _price_tick(spot)
            recommended = _round_to_tick(put_wall - tick * 1.5, tick)
            if recommended <= 0:
                recommended = _round_to_tick(max(put_wall * 0.95, tick), tick)
            margin = round((spot - recommended) / spot * 100, 2)
            if margin < 3:
                warnings.append("LOW_MARGIN_OF_SAFETY")
            negative_gamma = snapshot.gex_status == GEXStatus.NEG_GAMMA
            if negative_gamma:
                warnings.append("NEG_GAMMA_ASSIGNMENT_RISK")
            estimated_delta = _estimated_delta_from_margin(margin, dte, negative_gamma)
            estimated_win_probability = round((1 - estimated_delta) * 100, 1)
            weekly_target_yield = round(max(0.15, min(1.25, estimated_delta * 2.2)), 2)
            annualized_yield = round(weekly_target_yield * 52, 2)

        return CSPRecommendation(
            ticker=snapshot.ticker,
            dte=dte,
            spot_price=spot,
            put_wall=put_wall,
            zero_gamma=snapshot.zero_gamma_strike,
            call_wall=snapshot.call_wall_strike,
            recommended_strike=recommended,
            margin_of_safety_pct=margin,
            estimated_delta=estimated_delta,
            estimated_win_probability=estimated_win_probability,
            weekly_target_yield=weekly_target_yield,
            annualized_yield=annualized_yield,
            warnings=warnings,
            data_quality=_quality(snapshot),
        )

    async def lp_range(self, ticker: str) -> LPRangeRecommendation:
        snapshot = await self._snapshot_for(ticker)
        warnings = _data_quality_warnings(snapshot)
        if snapshot is None:
            return LPRangeRecommendation(
                ticker=ticker.strip().upper(),
                warnings=warnings,
                data_quality=_quality(None),
            )

        lower = snapshot.put_wall_strike
        upper = snapshot.call_wall_strike
        if lower is None:
            warnings.append("MISSING_PUT_WALL")
        if upper is None:
            warnings.append("MISSING_CALL_WALL")
        if (lower is None or upper is None) and snapshot.zero_gamma_strike is not None:
            anchor = snapshot.zero_gamma_strike
            lower = lower or round(anchor * 0.95, 2)
            upper = upper or round(anchor * 1.05, 2)
            warnings.append("HEURISTIC_ZERO_GAMMA_RANGE")

        range_width: float | None = None
        breakout_bias = "UNKNOWN"
        if lower is not None and upper is not None:
            if lower >= upper:
                warnings.append("INVALID_WALL_ORDER")
                lower = None
                upper = None
            else:
                range_width = round((upper - lower) / snapshot.underlying_price * 100, 2)
                distance_lower = (snapshot.underlying_price - lower) / snapshot.underlying_price * 100
                distance_upper = (upper - snapshot.underlying_price) / snapshot.underlying_price * 100
                if distance_lower < 2:
                    warnings.append("LOWER_EDGE_PROXIMITY")
                    breakout_bias = "BEARISH"
                elif distance_upper < 2:
                    warnings.append("UPPER_EDGE_PROXIMITY")
                    breakout_bias = "BULLISH"
                else:
                    breakout_bias = "NEUTRAL"
                if snapshot.gex_status == GEXStatus.NEG_GAMMA:
                    warnings.append("NEG_GAMMA_RANGE_BREAK_RISK")

        return LPRangeRecommendation(
            ticker=snapshot.ticker,
            spot_price=snapshot.underlying_price,
            range_lower=lower,
            range_upper=upper,
            zero_gamma=snapshot.zero_gamma_strike,
            put_wall=snapshot.put_wall_strike,
            call_wall=snapshot.call_wall_strike,
            range_width_pct=range_width,
            breakout_bias=breakout_bias,
            warnings=warnings,
            data_quality=_quality(snapshot),
        )

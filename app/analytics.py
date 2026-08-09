import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time
from typing import Literal
from zoneinfo import ZoneInfo

from app import pinning_engine
from app.models import GEXStatus, OptionGEXSummary, PinningAnalysis, RiskProfile


logger = logging.getLogger(__name__)


HIGH_RISK_WARNING = "High risk/high volatility; accelerated theta decay."
IMMINENT_EXPIRY_WARNING = (
    "0-1 DTE: expiry-day gamma and theta risk regardless of dealer positioning."
)

# A contract expiring today or tomorrow carries expiry risk that has nothing
# to do with which side of gamma dealers are on: theta is at its steepest,
# the position's delta whipsaws around the strike, and there is no time left
# for a thesis to play out. The old single `dte < 7 and NEG_GAMMA` rule let a
# 0DTE naked long option in a positive-gamma regime through with no warning
# at all, which then cleared theta_warning on the resulting plan card.
IMMINENT_EXPIRY_DTE = 1
SHORT_DATED_DTE = 7

# US listed options expire on a US market date at the 4:00 PM ET close, so
# every "what date is it for expiry-counting purposes" question has to be
# answered in market time. A bare UTC date is wrong for ~4 hours of every
# day (after 8pm ET it has already rolled to tomorrow), which flips 1DTE
# contracts into 0DTE a full session early. Audit timestamps
# (calculated_at / captured_at / created_at) still use real UTC instants —
# only DTE-relevant dates live in this timezone.
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE = clock_time(16, 0)

# Gamma scales as 1/sqrt(T), so flooring T at one whole calendar day makes a
# contract expiring in 20 minutes look identical to one expiring tomorrow
# afternoon and badly understates real 0DTE gamma. Floor at one hour
# instead: small enough to keep intraday 0DTE meaningful, large enough that
# a contract minutes from expiry can't divide the whole book by ~zero.
MIN_YEARS_TO_EXPIRY = 1.0 / (365.0 * 24.0)
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0

# Implausible IV quotes (illiquid/no-bid strikes routinely report 1e-5 from
# yfinance, and stale rows occasionally report several hundred percent) are
# DROPPED, never clamped. Clamping a 1e-5 reading up to a 1% floor doesn't
# make it usable — because BS gamma scales as 1/(sigma*sqrt(T)) it invents
# an enormous fake gamma concentration at exactly the strike whose quote was
# garbage. Bad rows are excluded the same way zero-OI/zero-strike rows are.
MIN_PLAUSIBLE_IV = 0.01
MAX_PLAUSIBLE_IV = 5.0

# The zero-gamma search grid is a fixed window centred on spot rather than
# one derived from the chain's min/max strike: a single far-dated LEAPS
# strike used to stretch the window and coarsen the resolution right where
# it matters. 80 steps per side over +/-15% gives ~0.19% of spot per step.
ZERO_GAMMA_WINDOW_PCT = 0.15
ZERO_GAMMA_HALF_STEPS = 80


def market_now() -> datetime:
    """Current instant expressed in US market time (see MARKET_TIMEZONE)."""
    return datetime.now(MARKET_TIMEZONE)


def market_today() -> date:
    """Today's date as the US options market sees it, not as UTC sees it."""
    return market_now().date()


def expiry_instant(expiration: date) -> datetime:
    """The moment an expiration actually expires: 4:00 PM US/Eastern."""
    return datetime.combine(expiration, MARKET_CLOSE, tzinfo=MARKET_TIMEZONE)


def years_to_expiry(expiration: date, now: datetime) -> float | None:
    """Time to expiry in years, or None if it has already expired.

    Returns a real fraction of a year measured to the 4:00 PM ET close, so
    a 0DTE contract priced at 9:30 AM and the same contract priced at
    3:00 PM get genuinely different (and correctly larger) gamma. An
    already-expired contract returns None so callers can exclude it — an
    expired row has no gamma and should never have been in the chain being
    priced.
    """
    seconds = (expiry_instant(expiration) - now).total_seconds()
    if seconds <= 0:
        return None
    return max(seconds / SECONDS_PER_YEAR, MIN_YEARS_TO_EXPIRY)


def is_plausible_iv(iv: float) -> bool:
    """Whether an implied-volatility quote is inside the usable band."""
    return MIN_PLAUSIBLE_IV <= iv <= MAX_PLAUSIBLE_IV


@dataclass(slots=True)
class OptionContract:
    code: str
    option_type: Literal["CALL", "PUT"]
    strike: float
    expiration_date: date
    implied_volatility: float
    delta: float
    market_gamma: float
    open_interest: int
    contract_size: float


def parse_gex_risk_profile(
    summary: OptionGEXSummary,
    days_to_expiration: int,
) -> RiskProfile:
    is_negative = summary.gex_status == GEXStatus.NEG_GAMMA
    # Independent OR-ed triggers, not one AND-ed condition: imminent expiry
    # is dangerous on its own, and short-dated negative gamma is dangerous on
    # its own. Each emits its own warning string so the reason for the lock
    # is never ambiguous.
    #
    # iv_rank is deliberately NOT a trigger here yet. Its underlying
    # calculation has a known correctness issue being fixed separately, and
    # thresholding a currently-untrustworthy number inside a safety-critical
    # lock would manufacture false confidence in both directions (spurious
    # locks, and worse, spurious non-locks). Revisit once iv_rank is
    # trustworthy.
    imminent_expiry = days_to_expiration <= IMMINENT_EXPIRY_DTE
    short_dated_negative_gamma = days_to_expiration < SHORT_DATED_DTE and is_negative
    warnings = []
    if imminent_expiry:
        warnings.append(IMMINENT_EXPIRY_WARNING)
    if short_dated_negative_gamma:
        warnings.append(HIGH_RISK_WARNING)
    locked = imminent_expiry or short_dated_negative_gamma
    return RiskProfile(
        gex_status=summary.gex_status,
        volatility_regime=(
            "HIGH_VOL_TRENDING" if is_negative else "LOW_VOL_MEAN_REVERSION"
        ),
        risk_level="HIGH" if locked else "NORMAL",
        warnings=warnings,
        locked_warning=locked,
    )


class GEXCalculator:
    """Calculate GEX using the conventional positive-call/negative-put sign.

    One gamma rule is shared by every number this class produces (headline
    net_gex, the zero-gamma curve, and the walls): broker/market gamma at
    the current spot where the feed supplies it, recomputed Black-Scholes
    gamma at every hypothetical spot level away from it. See _gamma(). The
    two used to be computed on silently different bases and could disagree
    in sign with each other.
    """

    def __init__(self, risk_free_rate: float) -> None:
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _normal_pdf(value: float) -> float:
        return math.exp(-0.5 * value * value) / math.sqrt(2 * math.pi)

    def _bs_gamma(self, spot: float, strike: float, years: float, iv: float) -> float:
        if min(spot, strike, years, iv) <= 0:
            return 0.0
        denominator = iv * math.sqrt(years)
        d1 = (
            math.log(spot / strike)
            + (self.risk_free_rate + 0.5 * iv * iv) * years
        ) / denominator
        return self._normal_pdf(d1) / (spot * denominator)

    def _gamma(
        self,
        contract: OptionContract,
        level: float,
        now: datetime,
        at_current_spot: bool,
    ) -> float:
        """Per-contract gamma at price `level`.

        One rule, applied identically by every caller (see the class
        docstring): broker/market gamma is a snapshot quantity that is only
        valid at the *current* spot, so it is used exactly when the price
        being evaluated IS the current spot, and Black-Scholes gamma is
        recomputed for every hypothetical price level away from it. That
        keeps the headline net_gex and the zero-gamma curve on one basis —
        the curve literally passes through the headline value at spot,
        because that grid point is computed by this same call.
        """
        if at_current_spot and contract.market_gamma > 0:
            return contract.market_gamma
        years = years_to_expiry(contract.expiration_date, now)
        if years is None:
            return 0.0
        return self._bs_gamma(
            level, contract.strike, years, contract.implied_volatility
        )

    def _contract_gex(
        self,
        contract: OptionContract,
        level: float,
        now: datetime,
        at_current_spot: bool,
    ) -> float:
        sign = 1.0 if contract.option_type == "CALL" else -1.0
        return (
            sign
            * self._gamma(contract, level, now, at_current_spot)
            * contract.open_interest
            * contract.contract_size
            * level**2
            * 0.01
        )

    def _net_at(
        self,
        contracts: list[OptionContract],
        level: float,
        now: datetime,
        at_current_spot: bool = False,
    ) -> float:
        """Net dealer gamma exposure if the underlying traded at `level`."""
        return sum(
            self._contract_gex(c, level, now, at_current_spot) for c in contracts
        )

    def _zero_gamma(
        self, contracts: list[OptionContract], spot: float, now: datetime
    ) -> float | None:
        """The price where net dealer gamma flips sign, or None if it never
        does inside the searched window.

        None is the honest answer for a chain whose gamma profile has no
        crossing: there is no such price. The previous behaviour — falling
        back to whichever grid boundary happened to have the smallest
        |value| — fabricated a level with no relationship to the real
        profile. Nothing downstream depends on that fabrication any more:
        gex_status now comes from the sign of net gamma at spot (see
        calculate()), never from comparing spot against this value.

        The grid is a fixed +/-ZERO_GAMMA_WINDOW_PCT window centred on spot
        with spot itself as the midpoint sample, so resolution near spot no
        longer depends on how far out the chain's widest strike happens to
        sit.
        """
        if spot <= 0 or not contracts:
            return None
        step = spot * ZERO_GAMMA_WINDOW_PCT / ZERO_GAMMA_HALF_STEPS
        levels = [
            spot + step * offset
            for offset in range(-ZERO_GAMMA_HALF_STEPS, ZERO_GAMMA_HALF_STEPS + 1)
        ]
        values = [
            self._net_at(
                contracts, level, now, at_current_spot=(index == ZERO_GAMMA_HALF_STEPS)
            )
            for index, level in enumerate(levels)
        ]
        crossings: list[float] = []
        for index, (left, right) in enumerate(zip(values, values[1:])):
            if left == 0:
                crossings.append(levels[index])
            elif left * right < 0:
                weight = abs(left) / (abs(left) + abs(right))
                crossings.append(
                    levels[index] + weight * (levels[index + 1] - levels[index])
                )
        if crossings:
            return min(crossings, key=lambda value: abs(value - spot))
        return None

    def _walls(
        self, contracts: list[OptionContract], spot: float, now: datetime
    ) -> tuple[float | None, float | None]:
        """Gamma-weighted OI peaks, constrained to the correct side of spot.

        A Call Wall is resistance *above* the current price and a Put Wall
        is support *below* it; that is what makes "spot broke the wall"
        meaningful downstream. Searching all strikes regardless of side let
        leftover deep-ITM open interest below spot win the call-side argmax
        and hand the pinning engine a "Call Wall" under the market, which it
        then read as a breakout. Each side is now restricted to its own half
        of the chain, and a side with no qualifying strike returns None
        rather than a wall on the wrong side.
        """
        calls: dict[float, float] = defaultdict(float)
        puts: dict[float, float] = defaultdict(float)
        for contract in contracts:
            exposure = abs(self._contract_gex(contract, spot, now, True))
            if exposure <= 0:
                continue
            if contract.option_type == "CALL":
                if contract.strike >= spot:
                    calls[contract.strike] += exposure
            elif contract.strike <= spot:
                puts[contract.strike] += exposure
        return (
            max(calls, key=calls.get) if calls else None,
            max(puts, key=puts.get) if puts else None,
        )

    @staticmethod
    def _usable_contracts(
        contracts: list[OptionContract], now: datetime
    ) -> list[OptionContract]:
        """Drop rows that cannot produce a meaningful gamma number.

        Same treatment already given to zero-OI/zero-strike rows: an
        implausible IV quote or an already-expired contract is excluded
        outright, never floored or otherwise coerced into the calculation.
        """
        return [
            contract
            for contract in contracts
            if contract.strike > 0
            and contract.open_interest > 0
            and is_plausible_iv(contract.implied_volatility)
            and years_to_expiry(contract.expiration_date, now) is not None
        ]

    @staticmethod
    def _iv_rank_proxy(contracts: list[OptionContract], spot: float) -> float:
        valid = [c for c in contracts if is_plausible_iv(c.implied_volatility)]
        if not valid:
            return 50.0
        atm = sorted(valid, key=lambda c: abs(c.strike - spot))[: min(10, len(valid))]
        atm_iv = sum(c.implied_volatility for c in atm) / len(atm)
        low = min(c.implied_volatility for c in valid)
        high = max(c.implied_volatility for c in valid)
        if math.isclose(low, high):
            return 50.0
        return max(0.0, min(100.0, 100 * (atm_iv - low) / (high - low)))

    def calculate(
        self, ticker: str, stock_price: float, contracts: list[OptionContract]
    ) -> OptionGEXSummary:
        if not contracts:
            raise ValueError("No valid option contracts were returned")
        now = market_now()
        usable = self._usable_contracts(contracts, now)
        if not usable:
            raise ValueError("No valid option contracts were returned")
        # The headline number: net dealer gamma at the price the underlying
        # is actually trading at, using broker gamma wherever it exists.
        net_gex = self._net_at(usable, stock_price, now, at_current_spot=True)
        # gex_status is the SIGN of that number, full stop. It must never be
        # derived from `stock_price > zero_gamma`: when the profile has no
        # crossing there is no zero_gamma to compare against, and even when
        # there is one, the comparison is an indirect proxy for a quantity
        # we already have exactly.
        status = GEXStatus.POS_GAMMA if net_gex > 0 else GEXStatus.NEG_GAMMA
        zero_gamma = self._zero_gamma(usable, stock_price, now)
        call_wall, put_wall = self._walls(usable, stock_price, now)
        return OptionGEXSummary(
            ticker=ticker.upper(),
            stock_price=round(stock_price, 4),
            zero_gamma=None if zero_gamma is None else round(zero_gamma, 4),
            call_wall=None if call_wall is None else round(call_wall, 4),
            put_wall=None if put_wall is None else round(put_wall, 4),
            iv_rank=round(self._iv_rank_proxy(usable, stock_price), 2),
            net_gex=round(net_gex, 2),
            gex_status=status,
        )


def _aggregate_oi_by_strike(contracts: list[OptionContract]) -> list[dict]:
    """把逐合約的 OptionContract 清單彙整成 pinning_engine 要的
    [{"strike":, "call_oi":, "put_oi":}, ...] 形狀——同一履約價的 Call/Put
    未平倉量分開加總（同一履約價可能橫跨多個到期日一起傳進來，例如聚合
    模式，這裡自然加總，不假設輸入已經去重）。
    """
    by_strike: dict[float, dict] = {}
    for contract in contracts:
        row = by_strike.setdefault(
            contract.strike, {"strike": contract.strike, "call_oi": 0.0, "put_oi": 0.0}
        )
        if contract.option_type == "CALL":
            row["call_oi"] += contract.open_interest
        else:
            row["put_oi"] += contract.open_interest
    return list(by_strike.values())


def _calculate_max_pain(gex_by_strike: list[dict]) -> float:
    """Max Pain：找出讓所有未平倉期權「到期內在價值總和」最小的履約價——
    理論上這是做市商避險成本最低、因此有誘因把股價釘在附近的價位。
    對每個候選履約價 S，計算：
        sum(call_oi_k * max(0, S-k)) + sum(put_oi_k * max(0, k-S))
    取讓這個總和最小的 S。
    """
    strikes = [row["strike"] for row in gex_by_strike]
    call_oi = {row["strike"]: row["call_oi"] for row in gex_by_strike}
    put_oi = {row["strike"]: row["put_oi"] for row in gex_by_strike}

    best_strike, best_loss = strikes[0], float("inf")
    for candidate in strikes:
        loss = sum(call_oi[k] * max(0.0, candidate - k) for k in strikes)
        loss += sum(put_oi[k] * max(0.0, k - candidate) for k in strikes)
        if loss < best_loss:
            best_strike, best_loss = candidate, loss
    return best_strike


def compute_pinning_for_contracts(
    contracts: list[OptionContract], summary: OptionGEXSummary
) -> PinningAnalysis | None:
    """從同一批已經抓好的 contracts（跟 GEXCalculator.calculate() 用的是
    同一份資料，不會多打任何 Moomoo API）算出 Pinning 判斷，附加在
    OptionGEXSummary 上給前端顯示卡片用。

    加分項——計算失敗（理論上只有 contracts 為空或現貨價無效這種邊界
    情況才會發生，正常情況 GEXCalculator.calculate() 已經先擋掉空清單）
    不該讓整個 GEX 摘要連帶失敗，只記警告、回傳 None，前端優雅跳過
    Pinning 卡片。
    """
    try:
        gex_by_strike = _aggregate_oi_by_strike(contracts)
        if not gex_by_strike:
            return None
        max_pain = _calculate_max_pain(gex_by_strike)
        in_positive_gamma = summary.gex_status == GEXStatus.POS_GAMMA
        result = pinning_engine.compute_pinning_analysis(
            gex_by_strike, summary.stock_price, max_pain,
            summary.call_wall, summary.put_wall, in_positive_gamma,
        )
        return PinningAnalysis(**result) if result else None
    except Exception:
        logger.exception("Pinning analysis failed; omitting pinning card")
        return None

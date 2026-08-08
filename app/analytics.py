import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal

from app import pinning_engine
from app.models import GEXStatus, OptionGEXSummary, PinningAnalysis, RiskProfile


logger = logging.getLogger(__name__)


HIGH_RISK_WARNING = "High risk/high volatility; accelerated theta decay."


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
    locked = days_to_expiration < 7 and is_negative
    return RiskProfile(
        gex_status=summary.gex_status,
        volatility_regime=(
            "HIGH_VOL_TRENDING" if is_negative else "LOW_VOL_MEAN_REVERSION"
        ),
        risk_level="HIGH" if locked else "NORMAL",
        warnings=[HIGH_RISK_WARNING] if locked else [],
        locked_warning=locked,
    )


class GEXCalculator:
    """Calculate GEX using the conventional positive-call/negative-put sign."""

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
        spot: float,
        valuation_date: date,
        prefer_market: bool = False,
    ) -> float:
        if prefer_market and contract.market_gamma > 0:
            return contract.market_gamma
        days = max((contract.expiration_date - valuation_date).days, 1)
        return self._bs_gamma(
            spot, contract.strike, days / 365.0, max(contract.implied_volatility, 0.01)
        )

    def _contract_gex(
        self,
        contract: OptionContract,
        spot: float,
        valuation_date: date,
        prefer_market: bool = False,
    ) -> float:
        sign = 1.0 if contract.option_type == "CALL" else -1.0
        return (
            sign
            * self._gamma(contract, spot, valuation_date, prefer_market)
            * contract.open_interest
            * contract.contract_size
            * spot**2
            * 0.01
        )

    def _net_at(
        self, contracts: list[OptionContract], spot: float, valuation_date: date
    ) -> float:
        return sum(self._contract_gex(c, spot, valuation_date) for c in contracts)

    def _zero_gamma(
        self, contracts: list[OptionContract], spot: float, valuation_date: date
    ) -> float:
        strikes = sorted({contract.strike for contract in contracts})
        if not strikes:
            return spot
        lower = min(min(strikes) * 0.9, spot * 0.7)
        upper = max(max(strikes) * 1.1, spot * 1.3)
        levels = [lower + (upper - lower) * index / 160 for index in range(161)]
        values = [self._net_at(contracts, level, valuation_date) for level in levels]
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
        return levels[min(range(len(values)), key=lambda index: abs(values[index]))]

    def _walls(
        self, contracts: list[OptionContract], spot: float, valuation_date: date
    ) -> tuple[float, float]:
        calls: dict[float, float] = defaultdict(float)
        puts: dict[float, float] = defaultdict(float)
        for contract in contracts:
            exposure = abs(
                self._contract_gex(contract, spot, valuation_date, prefer_market=True)
            )
            (calls if contract.option_type == "CALL" else puts)[contract.strike] += exposure
        return (
            max(calls, key=calls.get, default=spot),
            max(puts, key=puts.get, default=spot),
        )

    @staticmethod
    def _iv_rank_proxy(contracts: list[OptionContract], spot: float) -> float:
        valid = [c for c in contracts if 0 < c.implied_volatility < 5]
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
        today = datetime.now(timezone.utc).date()
        zero_gamma = self._zero_gamma(contracts, stock_price, today)
        call_wall, put_wall = self._walls(contracts, stock_price, today)
        net_gex = sum(
            self._contract_gex(c, stock_price, today, prefer_market=True)
            for c in contracts
        )
        status = (
            GEXStatus.POS_GAMMA
            if stock_price > zero_gamma
            else GEXStatus.NEG_GAMMA
        )
        return OptionGEXSummary(
            ticker=ticker.upper(),
            stock_price=round(stock_price, 4),
            zero_gamma=round(zero_gamma, 4),
            call_wall=round(call_wall, 4),
            put_wall=round(put_wall, 4),
            iv_rank=round(self._iv_rank_proxy(contracts, stock_price), 2),
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

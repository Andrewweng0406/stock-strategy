import asyncio
import hashlib
import logging
import math
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from typing import Any, Literal

import pandas as pd

from app.analytics import GEXCalculator, OptionContract
from app.models import GEXStatus, OptionGEXSummary


logger = logging.getLogger(__name__)


class MarketDataClient(ABC):
    @abstractmethod
    async def get_gex_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        raise NotImplementedError


class MockMarketDataClient(MarketDataClient):
    async def get_gex_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        seed = int(hashlib.sha256(ticker.upper().encode()).hexdigest()[:8], 16)
        stock_price = float(80 + seed % 250)
        zero_gamma = max(1.0, stock_price + ((seed // 7) % 21) - 10)
        return OptionGEXSummary(
            ticker=ticker.upper(),
            stock_price=stock_price,
            zero_gamma=zero_gamma,
            call_wall=round(stock_price * 1.05, 2),
            put_wall=round(stock_price * 0.95, 2),
            iv_rank=float(30 + seed % 60),
            net_gex=float(((seed % 200) - 100) * 1_000_000),
            gex_status=(
                GEXStatus.POS_GAMMA
                if stock_price > zero_gamma
                else GEXStatus.NEG_GAMMA
            ),
        )


class MoomooMarketDataClient(MarketDataClient):
    def __init__(self, host: str, port: int, calculator: GEXCalculator) -> None:
        self.host = host
        self.port = port
        self.calculator = calculator

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        normalized = ticker.strip().upper()
        return normalized if "." in normalized else f"US.{normalized}"

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            return default if math.isnan(parsed) else parsed
        except (TypeError, ValueError):
            return default

    def _fetch_sync(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        from futu import OpenQuoteContext, RET_OK

        quote_context = OpenQuoteContext(host=self.host, port=self.port)
        code = self._normalize_ticker(ticker)
        try:
            ret, stock_snapshot = quote_context.get_market_snapshot([code])
            if ret != RET_OK or stock_snapshot.empty:
                raise RuntimeError(f"Stock snapshot failed: {stock_snapshot}")
            stock_price = self._number(stock_snapshot.iloc[0].get("last_price"))
            if stock_price <= 0:
                raise RuntimeError("OpenD returned an invalid stock price")

            ret, expiration_data = quote_context.get_option_expiration_date(code=code)
            if ret != RET_OK or expiration_data.empty:
                raise RuntimeError(f"Expiration lookup failed: {expiration_data}")
            available_dates = [
                date.fromisoformat(str(value)[:10])
                for value in expiration_data["strike_time"].tolist()
                if str(value)
            ]
            today = datetime.now(timezone.utc).date()
            selected = min(
                available_dates,
                key=lambda expiry: abs(
                    max((expiry - today).days, 0) - days_to_expiration
                ),
            )
            selected_text = selected.isoformat()
            ret, chain = quote_context.get_option_chain(
                code=code, start=selected_text, end=selected_text
            )
            if ret != RET_OK or chain.empty:
                raise RuntimeError(f"Option chain failed: {chain}")

            option_codes = chain["code"].dropna().astype(str).tolist()
            snapshots: list[pd.DataFrame] = []
            for start in range(0, len(option_codes), 200):
                ret, snapshot = quote_context.get_market_snapshot(
                    option_codes[start : start + 200]
                )
                if ret != RET_OK:
                    raise RuntimeError(f"Option snapshot failed: {snapshot}")
                snapshots.append(snapshot)
            if not snapshots:
                raise RuntimeError("Option chain did not contain contracts")

            contracts: list[OptionContract] = []
            for _, row in pd.concat(snapshots, ignore_index=True).iterrows():
                raw_type = str(row.get("option_type", "")).upper()
                option_type: Literal["CALL", "PUT"]
                if "CALL" in raw_type:
                    option_type = "CALL"
                elif "PUT" in raw_type:
                    option_type = "PUT"
                else:
                    continue
                strike = self._number(row.get("option_strike_price"))
                open_interest = int(self._number(row.get("option_open_interest")))
                contract_size = self._number(row.get("option_contract_size"), 100)
                iv = self._number(row.get("option_implied_volatility"))
                if iv > 3:
                    iv /= 100
                if strike <= 0 or open_interest <= 0:
                    continue
                contracts.append(
                    OptionContract(
                        code=str(row.get("code", "")),
                        option_type=option_type,
                        strike=strike,
                        expiration_date=selected,
                        implied_volatility=max(iv, 0.01),
                        delta=self._number(row.get("option_delta")),
                        market_gamma=self._number(row.get("option_gamma")),
                        open_interest=open_interest,
                        contract_size=contract_size if contract_size > 0 else 100,
                    )
                )
            return self.calculator.calculate(ticker, stock_price, contracts)
        finally:
            quote_context.close()

    async def get_gex_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        return await asyncio.to_thread(self._fetch_sync, ticker, days_to_expiration)


class FallbackMarketDataClient(MarketDataClient):
    def __init__(
        self,
        primary: MarketDataClient,
        fallback: MarketDataClient,
        primary_mode: str = "moomoo",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_mode = primary_mode
        self.using_fallback = False

    @property
    def active_mode(self) -> str:
        return "mock" if self.using_fallback else self.primary_mode

    async def get_gex_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        try:
            result = await self.primary.get_gex_summary(ticker, days_to_expiration)
            self.using_fallback = False
            return result
        except Exception:
            self.using_fallback = True
            logger.exception("Moomoo OpenD failed; using mock market data")
            return await self.fallback.get_gex_summary(ticker, days_to_expiration)

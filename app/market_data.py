import asyncio
import hashlib
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

import pandas as pd

from app import pinning_engine
from app.analytics import GEXCalculator, OptionContract, compute_pinning_for_contracts
from app.models import (
    ExpirationInfo,
    ExpirationType,
    GEXStatus,
    OptionGEXSummary,
    PinningAnalysis,
)


logger = logging.getLogger(__name__)


def _is_third_friday(value: date) -> bool:
    return value.weekday() == 4 and 15 <= value.day <= 21


def classify_expiration(value: date, today: date) -> ExpirationType:
    dte = (value - today).days
    if dte <= 0:
        return ExpirationType.ZERO_DTE
    if dte == 1:
        return ExpirationType.ONE_DTE
    if _is_third_friday(value):
        return ExpirationType.MONTHLY
    return ExpirationType.WEEKLY


class MarketDataClient(ABC):
    @abstractmethod
    async def get_gex_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        raise NotImplementedError

    @abstractmethod
    async def get_available_expirations(self, ticker: str) -> list[ExpirationInfo]:
        raise NotImplementedError

    @abstractmethod
    async def get_gex_summary_multi(
        self, ticker: str, expiration_dates: list[date]
    ) -> OptionGEXSummary:
        raise NotImplementedError


class MockMarketDataClient(MarketDataClient):
    @staticmethod
    def _next_fridays(start: date, count: int) -> list[date]:
        current = start + timedelta(days=1)
        while current.weekday() != 4:
            current += timedelta(days=1)
        fridays = []
        for _ in range(count):
            fridays.append(current)
            current += timedelta(days=7)
        return fridays

    async def get_gex_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        seed = int(hashlib.sha256(ticker.upper().encode()).hexdigest()[:8], 16)
        stock_price = float(80 + seed % 250)
        zero_gamma = max(1.0, stock_price + ((seed // 7) % 21) - 10)
        call_wall = round(stock_price * 1.05, 2)
        put_wall = round(stock_price * 0.95, 2)
        gex_status = (
            GEXStatus.POS_GAMMA if stock_price > zero_gamma else GEXStatus.NEG_GAMMA
        )
        return OptionGEXSummary(
            ticker=ticker.upper(),
            stock_price=stock_price,
            zero_gamma=zero_gamma,
            call_wall=call_wall,
            put_wall=put_wall,
            iv_rank=float(30 + seed % 60),
            net_gex=float(((seed % 200) - 100) * 1_000_000),
            gex_status=gex_status,
            pinning=self._mock_pinning(
                seed, stock_price, call_wall, put_wall, gex_status
            ),
        )

    @staticmethod
    def _mock_pinning(
        seed: int,
        stock_price: float,
        call_wall: float,
        put_wall: float,
        gex_status: GEXStatus,
    ) -> PinningAnalysis | None:
        """示意用的合成 Pinning 卡片——跟其他 mock 欄位一樣，用 ticker 的
        hash 決定性地產生，而不是硬寫死一組固定值，讓不同標的看起來不會
        完全一樣。實際評分邏輯呼叫跟真實 Moomoo 模式完全相同的
        pinning_engine，只有輸入的期權鏈是編造的三個履約價，不是重新
        寫一套假算法——避免 mock 卡片跟真實卡片的判斷邏輯兩邊不一致。
        """
        pin_strike = round(stock_price + ((seed // 13) % 5) - 2)
        gex_by_strike = [
            {"strike": put_wall, "call_oi": 500 + seed % 300, "put_oi": 4000 + seed % 2000},
            {"strike": pin_strike, "call_oi": 3000 + seed % 3000, "put_oi": 3000 + seed % 3000},
            {"strike": call_wall, "call_oi": 4000 + seed % 2000, "put_oi": 500 + seed % 300},
        ]
        result = pinning_engine.compute_pinning_analysis(
            gex_by_strike, stock_price, max_pain=pin_strike,
            call_wall=call_wall, put_wall=put_wall,
            in_positive_gamma=(gex_status == GEXStatus.POS_GAMMA),
        )
        return PinningAnalysis(**result) if result else None

    async def get_available_expirations(self, ticker: str) -> list[ExpirationInfo]:
        today = datetime.now(timezone.utc).date()
        dates = sorted({today, today + timedelta(days=1), *self._next_fridays(today, 8)})
        return [
            ExpirationInfo(
                date=d,
                days_to_expiration=max((d - today).days, 0),
                expiration_type=classify_expiration(d, today),
            )
            for d in dates
        ]

    async def get_gex_summary_multi(
        self, ticker: str, expiration_dates: list[date]
    ) -> OptionGEXSummary:
        base = await self.get_gex_summary(ticker, 30)
        multiplier = max(len(expiration_dates), 1)
        return base.model_copy(update={"net_gex": base.net_gex * multiplier})


class MoomooMarketDataClient(MarketDataClient):
    def __init__(
        self,
        host: str,
        port: int,
        calculator: GEXCalculator,
        connect_timeout_seconds: float = 8.0,
    ) -> None:
        self.host = host
        self.port = port
        self.calculator = calculator
        self.connect_timeout_seconds = connect_timeout_seconds

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

    def _quote_context(self):
        from futu import OpenQuoteContext

        context = OpenQuoteContext(host=self.host, port=self.port)
        # Without this, a sync call issued while OpenD isn't reachable hangs
        # indefinitely (futu-api's auto-reconnect has no ceiling by default)
        # instead of promptly raising so FallbackMarketDataClient can fail
        # over to mock data — see app/config.py's moomoo_connect_timeout_seconds.
        context.set_sync_query_connect_timeout(self.connect_timeout_seconds)
        return context

    def _stock_price_sync(self, quote_context: Any, code: str) -> float:
        from futu import RET_OK

        ret, stock_snapshot = quote_context.get_market_snapshot([code])
        if ret != RET_OK or stock_snapshot.empty:
            raise RuntimeError(f"Stock snapshot failed: {stock_snapshot}")
        stock_price = self._number(stock_snapshot.iloc[0].get("last_price"))
        if stock_price <= 0:
            raise RuntimeError("OpenD returned an invalid stock price")
        return stock_price

    def _expiration_dates_sync(self, quote_context: Any, code: str) -> list[date]:
        from futu import RET_OK

        ret, expiration_data = quote_context.get_option_expiration_date(code=code)
        if ret != RET_OK or expiration_data.empty:
            raise RuntimeError(f"Expiration lookup failed: {expiration_data}")
        dates = [
            date.fromisoformat(str(value)[:10])
            for value in expiration_data["strike_time"].tolist()
            if str(value)
        ]
        return sorted(set(dates))

    def _contracts_for_expiration_sync(
        self, quote_context: Any, code: str, expiration: date
    ) -> list[OptionContract]:
        from futu import RET_OK

        selected_text = expiration.isoformat()
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
            return []

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
                    expiration_date=expiration,
                    implied_volatility=max(iv, 0.01),
                    delta=self._number(row.get("option_delta")),
                    market_gamma=self._number(row.get("option_gamma")),
                    open_interest=open_interest,
                    contract_size=contract_size if contract_size > 0 else 100,
                )
            )
        return contracts

    def _fetch_sync(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        quote_context = self._quote_context()
        code = self._normalize_ticker(ticker)
        try:
            stock_price = self._stock_price_sync(quote_context, code)
            available_dates = self._expiration_dates_sync(quote_context, code)
            today = datetime.now(timezone.utc).date()
            selected = min(
                available_dates,
                key=lambda expiry: abs(
                    max((expiry - today).days, 0) - days_to_expiration
                ),
            )
            contracts = self._contracts_for_expiration_sync(
                quote_context, code, selected
            )
            if not contracts:
                raise RuntimeError("Option chain did not contain contracts")
            summary = self.calculator.calculate(ticker, stock_price, contracts)
            pinning = compute_pinning_for_contracts(contracts, summary)
            return summary.model_copy(update={"pinning": pinning})
        finally:
            quote_context.close()

    def _fetch_expirations_sync(self, ticker: str) -> list[ExpirationInfo]:
        quote_context = self._quote_context()
        code = self._normalize_ticker(ticker)
        try:
            dates = self._expiration_dates_sync(quote_context, code)
        finally:
            quote_context.close()
        today = datetime.now(timezone.utc).date()
        return [
            ExpirationInfo(
                date=d,
                days_to_expiration=max((d - today).days, 0),
                expiration_type=classify_expiration(d, today),
            )
            for d in dates
        ]

    def _fetch_multi_sync(
        self, ticker: str, expiration_dates: list[date]
    ) -> OptionGEXSummary:
        quote_context = self._quote_context()
        code = self._normalize_ticker(ticker)
        try:
            stock_price = self._stock_price_sync(quote_context, code)
            contracts: list[OptionContract] = []
            for expiration in expiration_dates:
                contracts.extend(
                    self._contracts_for_expiration_sync(quote_context, code, expiration)
                )
            if not contracts:
                raise RuntimeError("Aggregate option chain did not contain contracts")
            summary = self.calculator.calculate(ticker, stock_price, contracts)
            pinning = compute_pinning_for_contracts(contracts, summary)
            return summary.model_copy(update={"pinning": pinning})
        finally:
            quote_context.close()

    async def get_gex_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        return await asyncio.to_thread(self._fetch_sync, ticker, days_to_expiration)

    async def get_available_expirations(self, ticker: str) -> list[ExpirationInfo]:
        return await asyncio.to_thread(self._fetch_expirations_sync, ticker)

    async def get_gex_summary_multi(
        self, ticker: str, expiration_dates: list[date]
    ) -> OptionGEXSummary:
        return await asyncio.to_thread(
            self._fetch_multi_sync, ticker, expiration_dates
        )


class YFinanceMarketDataClient(MarketDataClient):
    """No login, no local credentials, works from any server — the cloud
    deployment uses this as its primary source (see main.py's lifespan)
    precisely because Moomoo/Futu OpenD cannot run there without putting
    real brokerage login credentials on a third-party host, which this
    project deliberately never does. Data is whatever yfinance/Yahoo
    Finance serves publicly: typically ~15-20 minutes delayed, and no
    broker-calculated Greeks — GEXCalculator already falls back to
    computing Black-Scholes gamma itself whenever a contract's
    market_gamma is <=0, so that's a pre-existing code path this client
    relies on, not something added specially for it.

    yfinance is an unofficial library that scrapes Yahoo Finance's public
    endpoints (no documented SLA, could change or get rate-limited without
    notice) — acceptable here because this is a secondary/tertiary data
    source behind FallbackMarketDataClient, not the primary trading
    surface; a failure here just falls through to mock, same as any other
    fetch failure.
    """

    def __init__(
        self,
        calculator: GEXCalculator,
        min_request_interval_seconds: float = 0.5,
    ) -> None:
        self.calculator = calculator
        self.min_request_interval_seconds = min_request_interval_seconds
        self._throttle_lock = threading.Lock()
        self._last_request_at: float = 0.0

    def _throttle(self) -> None:
        """Yahoo has no documented rate limit, but hitting it too fast in a
        tight loop (e.g. aggregate mode's per-expiration chain fetches)
        risks a temporary IP block — same defensive pattern already proven
        out in the smart-money project's data_fetcher.py.
        """
        with self._throttle_lock:
            now = time.monotonic()
            wait = self.min_request_interval_seconds - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            return default if math.isnan(parsed) else parsed
        except (TypeError, ValueError):
            return default

    def _spot_price_sync(self, ticker: str) -> float:
        import yfinance as yf

        handle = yf.Ticker(ticker)
        self._throttle()
        try:
            hist = handle.history(period="5d", interval="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance daily close lookup failed for %s: %s", ticker, exc)

        self._throttle()
        try:
            price = handle.fast_info.get("lastPrice")
            if price:
                return float(price)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance quote lookup failed for %s: %s", ticker, exc)

        raise RuntimeError(f"yfinance returned no spot price for {ticker}")

    def _expiration_dates_sync(self, ticker: str) -> list[date]:
        import yfinance as yf

        self._throttle()
        handle = yf.Ticker(ticker)
        raw = list(handle.options)
        return sorted({date.fromisoformat(value) for value in raw if value})

    def _contracts_for_expiration_sync(
        self, ticker: str, expiration: date
    ) -> list[OptionContract]:
        import yfinance as yf

        self._throttle()
        handle = yf.Ticker(ticker)
        chain = handle.option_chain(expiration.isoformat())

        contracts: list[OptionContract] = []
        for option_type, rows in (("CALL", chain.calls), ("PUT", chain.puts)):
            for _, row in rows.iterrows():
                strike = self._number(row.get("strike"))
                open_interest = int(self._number(row.get("openInterest")))
                iv = self._number(row.get("impliedVolatility"))
                if strike <= 0 or open_interest <= 0:
                    continue
                contracts.append(
                    OptionContract(
                        code=str(row.get("contractSymbol", "")),
                        option_type=option_type,
                        strike=strike,
                        expiration_date=expiration,
                        implied_volatility=max(iv, 0.01),
                        delta=0.0,
                        # yfinance doesn't provide broker-calculated Greeks;
                        # leaving this at 0 makes GEXCalculator compute
                        # Black-Scholes gamma itself (see _gamma()'s
                        # prefer_market check) instead of trusting a value
                        # that was never actually supplied.
                        market_gamma=0.0,
                        open_interest=open_interest,
                        contract_size=100,
                    )
                )
        return contracts

    def _fetch_sync(self, ticker: str, days_to_expiration: int) -> OptionGEXSummary:
        symbol = ticker.strip().upper()
        stock_price = self._spot_price_sync(symbol)
        available_dates = self._expiration_dates_sync(symbol)
        if not available_dates:
            raise RuntimeError(f"No option expirations available for {symbol}")
        today = datetime.now(timezone.utc).date()
        selected = min(
            available_dates,
            key=lambda expiry: abs(max((expiry - today).days, 0) - days_to_expiration),
        )
        contracts = self._contracts_for_expiration_sync(symbol, selected)
        if not contracts:
            raise RuntimeError("Option chain did not contain contracts")
        summary = self.calculator.calculate(ticker, stock_price, contracts)
        pinning = compute_pinning_for_contracts(contracts, summary)
        return summary.model_copy(update={"pinning": pinning})

    def _fetch_expirations_sync(self, ticker: str) -> list[ExpirationInfo]:
        symbol = ticker.strip().upper()
        dates = self._expiration_dates_sync(symbol)
        today = datetime.now(timezone.utc).date()
        return [
            ExpirationInfo(
                date=d,
                days_to_expiration=max((d - today).days, 0),
                expiration_type=classify_expiration(d, today),
            )
            for d in dates
        ]

    def _fetch_multi_sync(
        self, ticker: str, expiration_dates: list[date]
    ) -> OptionGEXSummary:
        symbol = ticker.strip().upper()
        stock_price = self._spot_price_sync(symbol)
        contracts: list[OptionContract] = []
        for expiration in expiration_dates:
            contracts.extend(self._contracts_for_expiration_sync(symbol, expiration))
        if not contracts:
            raise RuntimeError("Aggregate option chain did not contain contracts")
        summary = self.calculator.calculate(ticker, stock_price, contracts)
        pinning = compute_pinning_for_contracts(contracts, summary)
        return summary.model_copy(update={"pinning": pinning})

    async def get_gex_summary(
        self, ticker: str, days_to_expiration: int
    ) -> OptionGEXSummary:
        return await asyncio.to_thread(self._fetch_sync, ticker, days_to_expiration)

    async def get_available_expirations(self, ticker: str) -> list[ExpirationInfo]:
        return await asyncio.to_thread(self._fetch_expirations_sync, ticker)

    async def get_gex_summary_multi(
        self, ticker: str, expiration_dates: list[date]
    ) -> OptionGEXSummary:
        return await asyncio.to_thread(
            self._fetch_multi_sync, ticker, expiration_dates
        )


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

    async def get_available_expirations(self, ticker: str) -> list[ExpirationInfo]:
        try:
            result = await self.primary.get_available_expirations(ticker)
            self.using_fallback = False
            return result
        except Exception:
            self.using_fallback = True
            logger.exception("Moomoo OpenD failed; using mock expirations")
            return await self.fallback.get_available_expirations(ticker)

    async def get_gex_summary_multi(
        self, ticker: str, expiration_dates: list[date]
    ) -> OptionGEXSummary:
        try:
            result = await self.primary.get_gex_summary_multi(
                ticker, expiration_dates
            )
            self.using_fallback = False
            return result
        except Exception:
            self.using_fallback = True
            logger.exception("Moomoo OpenD failed; using mock aggregate market data")
            return await self.fallback.get_gex_summary_multi(
                ticker, expiration_dates
            )

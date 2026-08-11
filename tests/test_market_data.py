from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.analytics import GEXCalculator
from app.market_data import (
    FallbackMarketDataClient,
    MarketDataUnavailableError,
    MockMarketDataClient,
    MoomooMarketDataClient,
    UnavailableMarketDataClient,
    YFinanceMarketDataClient,
    _is_third_friday,
    classify_expiration,
)
from app.models import ExpirationType


def test_third_friday_detection() -> None:
    # 2026-08-21 is a Friday in the 15-21 window.
    assert _is_third_friday(date(2026, 8, 21))
    # 2026-08-14 is a Friday but before day 15.
    assert not _is_third_friday(date(2026, 8, 14))
    # 2026-08-19 is in the window but a Wednesday, not a Friday.
    assert not _is_third_friday(date(2026, 8, 19))


def test_classify_expiration_zero_and_one_dte() -> None:
    today = date(2026, 8, 6)
    assert classify_expiration(today, today) == ExpirationType.ZERO_DTE
    assert classify_expiration(today - timedelta(days=1), today) == ExpirationType.ZERO_DTE
    assert classify_expiration(today + timedelta(days=1), today) == ExpirationType.ONE_DTE


def test_classify_expiration_monthly_vs_weekly() -> None:
    today = date(2026, 8, 6)
    assert classify_expiration(date(2026, 8, 21), today) == ExpirationType.MONTHLY
    assert classify_expiration(date(2026, 8, 28), today) == ExpirationType.WEEKLY


@pytest.mark.asyncio
async def test_mock_client_available_expirations_are_classified_and_sorted() -> None:
    client = MockMarketDataClient()
    expirations = await client.get_available_expirations("AAPL")
    dates = [e.date for e in expirations]
    assert dates == sorted(dates)
    assert all(e.days_to_expiration >= 0 for e in expirations)
    assert all(
        e.expiration_type
        == classify_expiration(e.date, expirations[0].date - timedelta(days=expirations[0].days_to_expiration))
        for e in expirations
    )


@pytest.mark.asyncio
async def test_mock_client_aggregate_scales_net_gex_by_expiration_count() -> None:
    client = MockMarketDataClient()
    single = await client.get_gex_summary("AAPL", 30)
    dates = [date(2026, 8, 21), date(2026, 8, 28), date(2026, 9, 4)]
    aggregate = await client.get_gex_summary_multi("AAPL", dates)
    assert aggregate.net_gex == single.net_gex * 3
    # Only net_gex should change; the rest of the summary is the base lookup.
    assert aggregate.stock_price == single.stock_price
    assert aggregate.zero_gamma == single.zero_gamma


@pytest.mark.asyncio
async def test_mock_client_aggregate_with_empty_dates_uses_multiplier_one() -> None:
    client = MockMarketDataClient()
    single = await client.get_gex_summary("AAPL", 30)
    aggregate = await client.get_gex_summary_multi("AAPL", [])
    assert aggregate.net_gex == single.net_gex


@pytest.mark.asyncio
async def test_mock_client_populates_pinning_card() -> None:
    """mock 模式也該產生一組示意用的 Pinning 卡片，跟其他既有欄位一樣，
    讓雲端/demo 環境的前端有東西可以顯示，不會因為沒有真實 Moomoo 資料
    整張卡片消失。
    """
    client = MockMarketDataClient()
    summary = await client.get_gex_summary("AAPL", 30)
    assert summary.pinning is not None
    assert summary.pinning.regime in {"PINNING", "BREAKOUT", "NEUTRAL"}
    assert 0 <= summary.pinning.score <= 100
    assert summary.data_source == "MOCK"
    assert summary.is_synthetic is True
    assert summary.is_delayed is False


@pytest.mark.asyncio
async def test_mock_client_pinning_is_deterministic_per_ticker() -> None:
    client = MockMarketDataClient()
    first = await client.get_gex_summary("AAPL", 30)
    second = await client.get_gex_summary("AAPL", 30)
    assert first.pinning.model_dump() == second.pinning.model_dump()


@pytest.mark.asyncio
async def test_mock_client_aggregate_inherits_pinning_from_base() -> None:
    client = MockMarketDataClient()
    single = await client.get_gex_summary("AAPL", 30)
    aggregate = await client.get_gex_summary_multi("AAPL", [date(2026, 8, 21)])
    assert aggregate.pinning.model_dump() == single.pinning.model_dump()


def test_moomoo_client_sets_a_bounded_connect_timeout() -> None:
    """futu-api's OpenQuoteContext has no connect timeout by default and
    auto-reconnects forever, so a sync call issued while OpenD isn't
    reachable would hang indefinitely instead of promptly raising for the
    trusted-data path to fail closed or use explicitly enabled demo data.
    """
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = MoomooMarketDataClient(
        "127.0.0.1", 11111, calculator, connect_timeout_seconds=8.0
    )
    with patch("futu.OpenQuoteContext") as mock_context_cls:
        mock_context = mock_context_cls.return_value
        result = client._quote_context()

    mock_context.set_sync_query_connect_timeout.assert_called_once_with(8.0)
    assert result is mock_context


def test_moomoo_option_chain_calls_are_throttled(monkeypatch) -> None:
    import app.market_data as market_data_module

    calculator = GEXCalculator(risk_free_rate=0.045)
    client = MoomooMarketDataClient(
        "127.0.0.1",
        11111,
        calculator,
        option_chain_max_calls=2,
        option_chain_window_seconds=10.0,
    )
    now = [100.0]
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(market_data_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(market_data_module.time, "sleep", fake_sleep)

    client._throttle_option_chain_sync()
    client._throttle_option_chain_sync()
    client._throttle_option_chain_sync()

    assert sleep_calls == [10.0]


class _FailingMarketDataClient(MockMarketDataClient):
    async def get_gex_summary(self, ticker: str, days_to_expiration: int):
        raise RuntimeError("primary down")

    async def get_available_expirations(self, ticker: str):
        raise RuntimeError("primary down")

    async def get_gex_summary_multi(self, ticker: str, expiration_dates: list[date]):
        raise RuntimeError("primary down")


@pytest.mark.asyncio
async def test_fallback_client_does_not_use_mock_unless_explicitly_enabled() -> None:
    client = FallbackMarketDataClient(
        _FailingMarketDataClient(),
        MockMarketDataClient(),
        primary_mode="yfinance",
    )

    with pytest.raises(MarketDataUnavailableError):
        await client.get_gex_summary("AAPL", 6)
    assert client.active_mode == "yfinance"


@pytest.mark.asyncio
async def test_fallback_client_can_use_mock_for_explicit_demo_mode() -> None:
    client = FallbackMarketDataClient(
        _FailingMarketDataClient(),
        MockMarketDataClient(),
        primary_mode="yfinance",
        allow_synthetic_fallback=True,
    )

    summary = await client.get_gex_summary("AAPL", 6)

    assert summary.ticker == "AAPL"
    assert client.active_mode == "mock"
    assert summary.data_source == "MOCK"
    assert summary.is_synthetic is True


@pytest.mark.asyncio
async def test_unavailable_market_data_client_raises_clear_error() -> None:
    client = UnavailableMarketDataClient()

    with pytest.raises(MarketDataUnavailableError, match="No trusted"):
        await client.get_available_expirations("AAPL")


# ---------- YFinanceMarketDataClient ----------

def _option_chain_frame(strikes_oi_iv: list[tuple[float, int, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "strike": [s for s, _, _ in strikes_oi_iv],
        "openInterest": [oi for _, oi, _ in strikes_oi_iv],
        "impliedVolatility": [iv for _, _, iv in strikes_oi_iv],
        "contractSymbol": [f"TEST{s}" for s, _, _ in strikes_oi_iv],
    })


def _fake_yf_ticker(
    close_price: float = 100.0,
    options: tuple[str, ...] = ("2026-08-14",),
    calls: pd.DataFrame | None = None,
    puts: pd.DataFrame | None = None,
    history_raises: bool = False,
) -> MagicMock:
    handle = MagicMock()
    if history_raises:
        handle.history.side_effect = ConnectionError("network down")
    else:
        handle.history.return_value = pd.DataFrame({"Close": [close_price]})
    handle.fast_info = {"lastPrice": close_price}
    handle.options = options
    chain = MagicMock()
    chain.calls = calls if calls is not None else _option_chain_frame([(100.0, 500, 0.4)])
    chain.puts = puts if puts is not None else _option_chain_frame([(100.0, 500, 0.4)])
    handle.option_chain.return_value = chain
    return handle


@pytest.mark.asyncio
async def test_yfinance_client_computes_gex_summary_from_option_chain() -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.0)
    handle = _fake_yf_ticker(
        close_price=100.0,
        calls=_option_chain_frame([(105.0, 4000, 0.4)]),
        puts=_option_chain_frame([(95.0, 4000, 0.4)]),
    )
    with patch("yfinance.Ticker", return_value=handle):
        summary = await client.get_gex_summary("AAPL", 6)

    assert summary.ticker == "AAPL"
    assert summary.stock_price == 100.0
    assert summary.pinning is not None
    assert summary.data_source == "YFINANCE"
    assert summary.is_delayed is True
    assert summary.is_synthetic is False


@pytest.mark.asyncio
async def test_yfinance_client_falls_back_to_fast_info_when_history_fails() -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.0)
    handle = _fake_yf_ticker(close_price=250.0, history_raises=True)
    with patch("yfinance.Ticker", return_value=handle):
        summary = await client.get_gex_summary("MSFT", 6)

    assert summary.stock_price == 250.0


@pytest.mark.asyncio
async def test_yfinance_client_raises_clear_error_when_totally_unavailable() -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.0)
    handle = MagicMock()
    handle.history.side_effect = ConnectionError("network down")
    handle.fast_info = {}
    with patch("yfinance.Ticker", return_value=handle):
        with pytest.raises(RuntimeError, match="no spot price"):
            await client.get_gex_summary("AAPL", 6)


def test_yfinance_client_skips_contracts_with_invalid_strike_or_zero_oi() -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.0)
    calls = _option_chain_frame([(105.0, 4000, 0.4), (0.0, 500, 0.4), (110.0, 0, 0.4)])
    puts = _option_chain_frame([(95.0, 4000, 0.4)])
    handle = _fake_yf_ticker(close_price=100.0, calls=calls, puts=puts)
    with patch("yfinance.Ticker", return_value=handle):
        contracts = client._contracts_for_expiration_sync("AAPL", date(2026, 8, 14))

    strikes = sorted(c.strike for c in contracts)
    assert strikes == [95.0, 105.0]  # zero-strike and zero-OI rows dropped


def test_yfinance_client_leaves_market_gamma_zero_so_calculator_uses_black_scholes() -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.0)
    handle = _fake_yf_ticker(close_price=100.0)
    with patch("yfinance.Ticker", return_value=handle):
        contracts = client._contracts_for_expiration_sync("AAPL", date(2026, 8, 14))

    assert all(c.market_gamma == 0.0 and c.delta == 0.0 for c in contracts)


def test_yfinance_client_expirations_are_sorted_and_deduplicated() -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.0)
    handle = _fake_yf_ticker(options=("2026-08-21", "2026-08-14", "2026-08-14"))
    with patch("yfinance.Ticker", return_value=handle):
        dates = client._expiration_dates_sync("AAPL")

    assert dates == [date(2026, 8, 14), date(2026, 8, 21)]


@pytest.mark.asyncio
async def test_yfinance_client_selects_expiration_closest_to_requested_dte() -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.0)
    today = date.today()
    near = (today + timedelta(days=2)).isoformat()
    far = (today + timedelta(days=30)).isoformat()
    handle = _fake_yf_ticker(close_price=100.0, options=(near, far))
    with patch("yfinance.Ticker", return_value=handle):
        await client.get_gex_summary("AAPL", 30)
        called_with = handle.option_chain.call_args.args[0]

    assert called_with == far


def test_yfinance_client_throttle_enforces_minimum_interval(monkeypatch) -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.5)
    monkeypatch.setattr(client, "_last_request_at", 100.0)

    fake_now = [100.1]
    import app.market_data as market_data_module
    monkeypatch.setattr(market_data_module.time, "monotonic", lambda: fake_now[0])
    sleep_calls = []
    monkeypatch.setattr(market_data_module.time, "sleep", lambda s: sleep_calls.append(s))

    client._throttle()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(0.4, abs=1e-9)


@pytest.mark.asyncio
async def test_yfinance_client_aggregate_combines_multiple_expirations() -> None:
    calculator = GEXCalculator(risk_free_rate=0.045)
    client = YFinanceMarketDataClient(calculator, min_request_interval_seconds=0.0)
    handle = _fake_yf_ticker(
        close_price=100.0,
        calls=_option_chain_frame([(105.0, 4000, 0.4)]),
        puts=_option_chain_frame([(95.0, 4000, 0.4)]),
    )
    with patch("yfinance.Ticker", return_value=handle):
        summary = await client.get_gex_summary_multi("AAPL", [date(2026, 8, 14), date(2026, 8, 21)])

    assert summary.stock_price == 100.0
    assert handle.option_chain.call_count == 2
    assert summary.data_source == "YFINANCE"
    assert summary.is_delayed is True
    assert summary.is_synthetic is False

from datetime import date, timedelta

import pytest

from app.market_data import MockMarketDataClient, _is_third_friday, classify_expiration
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

import pytest

from app.main import app


@pytest.fixture(autouse=True)
def _disable_rate_limiting():
    """Turn off slowapi's IP-keyed limiter for the duration of the suite.

    /api/v1/chat and /api/v1/trades/{id}/review share a 10-per-minute budget
    per client IP, and every test in this process arrives from the same
    "testclient" address. Once enough endpoint tests exercise those two
    routes, whichever ones happen to run last start getting 429s — a failure
    with nothing to do with what they assert. No test here covers the rate
    limiter itself.
    """
    limiter = app.state.limiter
    previously_enabled = limiter.enabled
    limiter.enabled = False
    try:
        yield
    finally:
        limiter.enabled = previously_enabled

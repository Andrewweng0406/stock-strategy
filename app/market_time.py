from datetime import time as clock_time
from zoneinfo import ZoneInfo

# US listed options use the US equity-options market date and regular
# 4:00 PM ET close for expiry-sensitive calculations.
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE = clock_time(16, 0)

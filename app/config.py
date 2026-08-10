from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Options Trading Copilot"
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4"
    default_max_loss_usd: float = 250.0
    database_url: str = "sqlite+aiosqlite:///./trading_copilot.db"
    redis_url: str | None = "redis://127.0.0.1:6379/0"
    cache_ttl_seconds: int = 30
    moomoo_enabled: bool = True
    moomoo_host: str = "127.0.0.1"
    moomoo_port: int = 11111
    # futu-api's OpenQuoteContext has no connect timeout by default (auto
    # reconnect keeps retrying forever) — a sync call issued while OpenD
    # isn't reachable (not started yet, still logging in, network hiccup)
    # would otherwise hang indefinitely instead of promptly failing closed
    # or falling back to an explicitly enabled demo data source.
    moomoo_connect_timeout_seconds: float = 8.0
    # When Moomoo is disabled (the cloud deployment — OpenD needs real
    # brokerage login credentials, which never leave the local machine),
    # fall back to yfinance instead of going straight to synthetic mock
    # data. yfinance needs no login/credentials at all, so this is safe to
    # run anywhere; the tradeoff is ~15-20min delayed data and no
    # broker-calculated Greeks (GEXCalculator computes Black-Scholes gamma
    # itself in that case — see YFinanceMarketDataClient's docstring).
    yfinance_fallback_enabled: bool = True
    yfinance_min_request_interval_seconds: float = 0.5
    # Synthetic market data is useful for local UI development and tests, but
    # a paid trading product must never silently replace unavailable market
    # data with deterministic fake numbers. Keep this off unless a deployment
    # is explicitly a demo/sandbox.
    synthetic_market_data_enabled: bool = False
    risk_free_rate: float = 0.045

    # Cloud sync: when both are set, this instance pushes real (non-mock) GEX
    # summaries it computes to `{cloud_sync_url}/api/v1/sync/gex`. Used by the
    # local instance (real Moomoo data) to keep the cloud deployment's cache
    # warm with real data, without ever sending Moomoo credentials to the cloud.
    cloud_sync_url: str | None = None
    sync_token: str | None = None
    # How often the local instance re-fetches + pushes each actively-viewed
    # ticker/expiry. Kept well above 1-2s on purpose: a full GEX recompute
    # walks the entire option chain, and polling that fast risks tripping
    # Moomoo/Futu API rate limits on a real brokerage connection.
    sync_poll_seconds: int = 10
    # How long a ticker/expiry stays "active" (and gets polled) after the
    # local frontend last asked for it.
    active_window_seconds: int = 300

    # Aggregate GEX is deliberately excluded from the poller's continuous
    # refresh (walking one full option chain per expiration risks tripping
    # Moomoo/Futu's 10-calls/30s limit — see GEXService.get_aggregate_summary),
    # so on the cloud side its synced cache entry only ever gets refreshed
    # when the local instance happens to compute that exact same expiration
    # combination again. cache_ttl_seconds (30s) is far too short a window
    # for that to realistically happen — real data would flip back to mock
    # a few seconds after every push. Aggregate results are already the
    # "expensive, less time-sensitive" view by design, so a longer TTL here
    # is a natural fit, not a special case: matches active_window_seconds so
    # a synced aggregate view stays real for as long as a ticker would stay
    # "active" anyway.
    aggregate_cache_ttl_seconds: int = 300

    # /api/v1/chat is the only endpoint that spends real OpenAI budget per
    # call; cap it per client IP so a bug or abusive client can't run up the
    # bill. slowapi's rate-string format: "<count>/<second|minute|hour|day>".
    chat_rate_limit: str = "10/minute"

    # How often a real (non-mock) GEX calculation gets written to
    # gex_snapshots per ticker — a throttle, not the compute cadence, so the
    # table doesn't fill with near-duplicate rows every poller tick.
    snapshot_interval_seconds: int = 3600

    @property
    def allowed_cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def effective_database_url(self) -> str:
        """Railway's Postgres addon hands back `postgres://` or
        `postgresql://`; SQLAlchemy's async engine needs the `+asyncpg`
        driver named explicitly. SQLite URLs pass through unchanged.
        """
        url = self.database_url
        if url.startswith("postgres://"):
            return "postgresql+asyncpg://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            return "postgresql+asyncpg://" + url[len("postgresql://") :]
        return url


settings = Settings()

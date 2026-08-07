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

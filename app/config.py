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
    cache_ttl_seconds: int = 300
    moomoo_enabled: bool = True
    moomoo_host: str = "127.0.0.1"
    moomoo_port: int = 11111
    risk_free_rate: float = 0.045

    @property
    def allowed_cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


settings = Settings()

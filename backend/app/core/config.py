"""Application settings loaded from environment variables (see root .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAMEI API"
    app_env: str = "local"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://ramei:change-me@localhost:5432/ramei"
    redis_url: str = "redis://localhost:6379/0"

    model_bundle_dir: str = "./ml/artifacts/bundles"
    active_model: str = "ramei-rgcmt"
    inference_device: str = "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()

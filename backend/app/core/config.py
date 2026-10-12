"""Application settings loaded from environment variables (see root .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"), extra="ignore"
    )

    app_name: str = "RAMEI API"
    app_env: str = "local"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # SQLite for local development; PostgreSQL in staging/production (same ORM models)
    database_url: str = f"sqlite:///{(BACKEND_ROOT / 'ramei.db').as_posix()}"

    jwt_secret: str = "dev-only-change-me"  # noqa: S105 - refused outside local, see below
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60 * 8

    upload_dir: Path = BACKEND_ROOT / "uploads"
    max_upload_mb: int = 200
    max_upload_minutes: int = 60
    store_raw_media: bool = False  # PRD privacy default: delete uploads after analysis

    model_bundle_dir: Path = REPO_ROOT / "ml" / "artifacts" / "bundles"
    active_model: str = "ramei-rgcmt-1.0.0"
    inference_device: str = "auto"  # auto | cpu | cuda
    load_models_on_startup: bool = False  # models load lazily on first analysis


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.app_env != "local" and settings.jwt_secret == "dev-only-change-me":  # noqa: S105
        raise RuntimeError("Set JWT_SECRET to a long random value outside local development")
    return settings

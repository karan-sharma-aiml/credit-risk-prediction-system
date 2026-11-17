"""Application settings loaded from environment variables / ``.env``.

Nothing sensitive is hardcoded: credentials come from ``DATABASE_URL`` and
``SECRET_KEY``. The ML artifact location defaults to the path used by Part 2.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.config import PATHS, PROJECT_ROOT

APP_NAME = "Credit Risk Prediction Platform"
APP_VERSION = "1.0.0"
INSECURE_SECRET = "change-me"


class Settings(BaseSettings):
    """Typed configuration (environment variables are case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True
    api_host: str = "0.0.0.0"  # noqa: S104 (container default)
    api_port: int = 8000
    secret_key: SecretStr = SecretStr(INSECURE_SECRET)
    cors_origins: str = ""

    model_path: Path = PATHS.model_bundle
    reports_dir: Path = PATHS.summaries

    database_url: str = "sqlite:///./credit_risk.db"
    auto_create_tables: bool = True
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.environment == "production" and self.secret_key.get_secret_value() == INSECURE_SECRET:
            raise ValueError("SECRET_KEY must be changed from the default in production.")
        for name in ("model_path", "reports_dir"):
            path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, PROJECT_ROOT / path)
        return self

    @property
    def sqlalchemy_url(self) -> str:
        """``DATABASE_URL`` normalised for SQLAlchemy (Render/Heroku use ``postgres://``)."""
        url = self.database_url
        if url.startswith("postgres://"):
            url = "postgresql://" + url.removeprefix("postgres://")
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url.removeprefix("postgresql://")
        return url

    @property
    def is_sqlite(self) -> bool:
        """Whether the configured database is SQLite (development / tests)."""
        return self.sqlalchemy_url.startswith("sqlite")

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed ``CORS_ORIGINS`` (comma separated)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (override in tests via ``create_app(settings=...)``)."""
    return Settings()

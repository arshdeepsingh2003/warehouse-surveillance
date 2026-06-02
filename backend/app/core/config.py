"""
core/config.py
──────────────
Central settings file. All environment variables are read from the .env file
and exposed as typed Python attributes via Pydantic Settings.

Usage anywhere in the app:
    from app.core.config import settings
    print(settings.APP_NAME)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "Warehouse AI Surveillance System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins for the React frontend
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # ── Mock mode ─────────────────────────────────────────────────────────────
    USE_MOCK_DATA: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # FUTURE database / redis settings go here
    # DATABASE_URL: str = ""
    # REDIS_URL: str = ""

    # Tells Pydantic to read from the .env file in the project root
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def get_allowed_origins(self) -> list[str]:
        """Return ALLOWED_ORIGINS as a Python list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]


# Single shared instance — import this everywhere
settings = Settings()

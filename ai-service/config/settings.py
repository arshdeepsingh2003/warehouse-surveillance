"""
config/settings.py
──────────────────
Typed settings for the AI service, loaded from .env.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Backend ───────────────────────────────────────────────────────────────
    BACKEND_URL:    str = "http://localhost:8000"
    BACKEND_WS_URL: str = "ws://localhost:8000/ws"
    API_PREFIX:     str = "/api/v1"

    # ── Stream server ─────────────────────────────────────────────────────────
    STREAM_SERVER_HOST: str = "0.0.0.0"
    STREAM_SERVER_PORT: int = 8001
    STREAM_FPS:         int = 10
    FRAME_WIDTH:        int = 640
    FRAME_HEIGHT:       int = 360
    JPEG_QUALITY:       int = 75

    # ── Paths ─────────────────────────────────────────────────────────────────
    MOCK_VIDEO_DIR: str = "./mock_sources"

    # ── Behaviour ─────────────────────────────────────────────────────────────
    USE_MOCK_SOURCES:  bool  = True
    LOOP_VIDEO:        bool  = True
    ALERT_PROBABILITY: float = 0.05
    BATCH_SIZE:        int   = 30
    LOITERING_SECONDS: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cameras_api(self) -> str:
        return f"{self.BACKEND_URL}{self.API_PREFIX}/cameras"

    @property
    def alerts_api(self) -> str:
        return f"{self.BACKEND_URL}{self.API_PREFIX}/alerts"

    @property
    def activities_api(self) -> str:
        return f"{self.BACKEND_URL}{self.API_PREFIX}/activities"


settings = Settings()

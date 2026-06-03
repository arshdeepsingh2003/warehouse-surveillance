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

    # ── LLM / Summaries ────────────────────────────────────────────────────────
    SUMMARY_INTERVAL_SECONDS: int = 30
    LLM_BACKEND:              str = "mock"
    VLM_BACKEND:              str = "mock"
    USE_LLM:                  bool = False
    USE_VLM:                  bool = False

    # ── API Keys ────────────────────────────────────────────────────────────────
    GROQ_API_KEY:     str = ""
    OPENAI_API_KEY:   str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY:   str = ""

    # ── Models ──────────────────────────────────────────────────────────────────
    LLM_MODEL:       str = "llama-3.3-70b-versatile"
    GROQ_VLM_MODEL:  str = "meta-llama/llama-4-scout-17b-16e-instruct"
    VLM_MODEL:       str = ""
    VLM_EVERY_N_FRAMES:        int = 30
    VLM_MAX_PERSONS_PER_FRAME: int = 5
    VLM_JPEG_QUALITY:          int = 75
    VLM_CACHE_TTL_SECONDS:     int = 300
    VLM_MOCK_LATENCY_MS:       int = 80

    # ── Ollama ──────────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL:  str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "llama3.2:3b"
    OLLAMA_MODEL:     str = "llava"

    # ── Detection / Processing ──────────────────────────────────────────────────
    YOLO_MODEL_PATH:        str = "./models/yolov8n.pt"
    YOLO_CONFIDENCE:        float = 0.25
    DEVICE:                 str = "cpu"
    DETECTOR_BACKEND:       str = "auto"
    PROCESS_EVERY_N_FRAMES: int = 3
    ALERT_COOLDOWN_SECONDS: float = 30.0

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

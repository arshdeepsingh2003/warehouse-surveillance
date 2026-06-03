"""
main.py
───────
FastAPI application entry point.

This file:
  1. Creates the FastAPI app instance.
  2. Configures CORS (so the React frontend can call our API).
  3. Registers all route routers (cameras, alerts, activities, analytics, ws).
  4. Starts the mock event broadcaster as a background task on startup.
  5. Exposes a /health endpoint for load-balancer checks.

Run with:
    uvicorn app.main:app --reload
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import cameras, alerts, activities, analytics, websocket, ingest, summaries
from app.api.routes.websocket import mock_event_broadcaster

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code inside the `async with` block runs at startup.
    Code after `yield` runs at shutdown.

    Here we launch the mock WebSocket broadcaster as a background task.
    In production this would also initialise the database connection pool.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Mock data mode: {settings.USE_MOCK_DATA}")

    # Start background task: fires mock WS events on a schedule (demo mode only)
    if settings.USE_MOCK_DATA:
        broadcaster_task = asyncio.create_task(mock_event_broadcaster())
        logger.info("Mock WebSocket broadcaster started (demo mode).")
    else:
        broadcaster_task = None
        logger.info("Mock WebSocket broadcaster disabled (production mode).")

    yield  # ← Application is running while we're here

    # Shutdown: cancel the background task cleanly
    if broadcaster_task:
        broadcaster_task.cancel()
        try:
            await broadcaster_task
        except asyncio.CancelledError:
            pass
    logger.info("Application shutdown complete.")


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Backend API for the AI-powered Warehouse Surveillance and "
        "Anomalous Activity Detection System. Provides REST APIs for "
        "cameras, alerts, activities, and analytics, plus a WebSocket "
        "endpoint for real-time dashboard updates."
    ),
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI  → http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc UI    → http://localhost:8000/redoc
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# Allows the React frontend (running on a different port) to call this API.
# In production restrict allowed_origins to your actual frontend domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),  # ["http://localhost:3000", ...]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
# Each router handles one resource group. The prefix is defined in the router.

API_PREFIX = "/api/v1"

app.include_router(cameras.router,    prefix=API_PREFIX)  # /api/v1/cameras
app.include_router(alerts.router,     prefix=API_PREFIX)  # /api/v1/alerts
app.include_router(activities.router, prefix=API_PREFIX)  # /api/v1/activities
app.include_router(analytics.router,  prefix=API_PREFIX)  # /api/v1/analytics
app.include_router(websocket.router)                       # /ws  (no prefix — WS standard)
app.include_router(ingest.router,     prefix=API_PREFIX)  # /api/v1/ ingest from AI service
app.include_router(summaries.router,  prefix=API_PREFIX)  # /api/v1/summaries (VLM/LLM)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
async def health_check() -> dict:
    """
    Returns 200 OK when the server is running.
    Used by Docker, Kubernetes, and load balancers to verify the service is alive.
    """
    return {
        "status":  "healthy",
        "app":     settings.APP_NAME,
        "version": settings.APP_VERSION,
        "mode":    "mock" if settings.USE_MOCK_DATA else "production",
    }


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"], summary="API root")
async def root() -> dict:
    return {
        "message":   f"Welcome to {settings.APP_NAME}",
        "docs":      "/docs",
        "websocket": "ws://localhost:8000/ws",
        "api_base":  API_PREFIX,
    }

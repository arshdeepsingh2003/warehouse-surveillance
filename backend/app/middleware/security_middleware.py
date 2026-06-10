"""
app/middleware/security_middleware.py
──────────────────────────────────────
Production security middleware stack:

1. RateLimitMiddleware   — per-IP sliding window rate limiting (in-memory)
2. SecurityHeadersMiddleware — adds OWASP-recommended HTTP security headers
3. RequestLoggingMiddleware  — structured request/response logging

In production replace in-memory rate limiter with Redis for multi-worker support:
  pip install slowapi redis
  Swap _RateLimiter with SlowAPI + Redis backend
"""

from __future__ import annotations

import time
import logging
import uuid
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


# ── 1. In-memory rate limiter ─────────────────────────────────────────────────

class _RateLimiter:
    """
    Sliding window rate limiter per IP address.

    Stores request timestamps in a deque per IP.
    Thread-safe for asyncio (single event loop).
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests     = max_requests
        self.window_seconds   = window_seconds
        self._windows: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, ip: str) -> tuple[bool, int]:
        """
        Check if request from `ip` is within rate limit.

        Returns:
            (allowed: bool, remaining: int)
        """
        now    = time.monotonic()
        window = self._windows[ip]

        # Remove timestamps older than the window
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            return False, 0

        window.append(now)
        return True, self.max_requests - len(window)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Apply rate limiting to all requests.

    Endpoints with stricter limits (login, etc.) use tighter windows.
    Exempt paths: /health, /metrics (monitoring tools poll these frequently).
    """

    _EXEMPT = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    _AUTH_LIMIT   = _RateLimiter(max_requests=10,  window_seconds=60)   # login: 10/min
    _GLOBAL_LIMIT = _RateLimiter(max_requests=300, window_seconds=60)   # API:  300/min

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._EXEMPT:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"

        # Stricter limit for auth endpoints
        if request.url.path.startswith("/api/v1/auth"):
            allowed, remaining = self._AUTH_LIMIT.is_allowed(ip)
        else:
            allowed, remaining = self._GLOBAL_LIMIT.is_allowed(ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# ── 2. Security headers middleware ────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add OWASP-recommended security headers to every response.

    These prevent common web attacks:
      XSS, clickjacking, MIME sniffing, information leakage.
    """

    _HEADERS = {
        "X-Content-Type-Options":    "nosniff",
        "X-Frame-Options":           "DENY",
        "X-XSS-Protection":          "1; mode=block",
        "Referrer-Policy":           "strict-origin-when-cross-origin",
        "Permissions-Policy":        "camera=(), microphone=(), geolocation=()",
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
        "Cache-Control":             "no-store",
        # CSP: restrict sources to self + our stream service
        "Content-Security-Policy": (
            "default-src 'self'; "
            "img-src 'self' data: http://localhost:8002 blob:; "
            "connect-src 'self' ws://localhost:8000 http://localhost:8002; "
            "script-src 'self' 'unsafe-inline'; "  # relaxed for dev — tighten in prod
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com;"
        ),
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for header, value in self._HEADERS.items():
            response.headers[header] = value
        # Remove server fingerprinting
        response.headers.pop("server", None)
        return response


# ── 3. Request logging middleware ─────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured request/response logging.

    Each request gets a unique request_id for distributed tracing.
    Sensitive paths (auth) have body logging suppressed.
    """

    _SENSITIVE = {"/api/v1/auth/login", "/api/v1/auth/refresh"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start      = time.monotonic()

        # Attach request ID for downstream use
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = int((time.monotonic() - start) * 1000)
        ip          = request.client.host if request.client else "-"

        logger.info(
            f"[{request_id}] {ip} {request.method} {request.url.path} "
            f"→ {response.status_code} ({duration_ms}ms)"
        )

        # Add request ID to response for client-side debugging
        response.headers["X-Request-ID"] = request_id
        return response

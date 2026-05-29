"""
streams/stream_server.py
─────────────────────────
MJPEG Stream Server  —  exposes camera frames as HTTP video streams.

What is MJPEG?
  Motion JPEG: a sequence of JPEG images sent over HTTP as a
  multipart/x-mixed-replace response. Every modern browser and
  most video players support it natively — no plugins needed.

Endpoints exposed by this FastAPI sub-app:
  GET /stream/{camera_id}        → MJPEG stream for one camera
  GET /snapshot/{camera_id}      → Single JPEG frame (for thumbnails)
  GET /status                    → JSON: all camera stats + FPS
  GET /health                    → liveness check

Why MJPEG instead of HLS or WebRTC?
  MJPEG is the simplest possible approach that works immediately
  in a browser <img> or <video> tag.

  When to upgrade:
    • HLS    — better for recording, scrubbing, adaptive bitrate
    • WebRTC — needed for sub-500ms latency or 2-way audio
  See the architecture notes in README for upgrade path.

Usage in React:
  <img src="http://localhost:8001/stream/cam-01" />
  That's it — the browser handles the multipart stream automatically.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config.settings import settings

logger = logging.getLogger(__name__)

# The FrameProcessor instance is injected at startup by main.py
# (avoids circular imports)
_processor = None


def create_stream_app(processor) -> FastAPI:
    """
    Factory: create the MJPEG stream FastAPI app with the processor injected.
    Called from main.py after the processor is initialised.
    """
    global _processor
    _processor = processor

    app = FastAPI(
        title="Warehouse Camera Stream Server",
        description="Serves live MJPEG camera feeds for the dashboard.",
        version="1.0.0",
    )

    # Allow the React dashboard (on :5173 or :3000) to fetch streams
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "stream-server"}

    @app.get("/status")
    async def stream_status():
        """JSON status for all cameras — polled by the dashboard."""
        if not _processor:
            return {"cameras": []}
        return {"cameras": list(_processor.get_all_stats().values())}

    @app.get(
        "/stream/{camera_id}",
        responses={200: {"content": {"multipart/x-mixed-replace": {}}}},
        summary="Live MJPEG stream for one camera",
        description=(
            "Use as `<img src='http://localhost:8001/stream/cam-01' />`.\n\n"
            "The browser receives an infinite multipart response where each "
            "part is a JPEG frame. No client-side JavaScript needed."
        ),
    )
    async def mjpeg_stream(camera_id: str):
        return StreamingResponse(
            _frame_generator(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma":        "no-cache",
                "Expires":       "0",
            },
        )

    @app.get(
        "/snapshot/{camera_id}",
        responses={200: {"content": {"image/jpeg": {}}}},
        summary="Single JPEG snapshot",
        description="Returns the latest frame for a camera as a plain JPEG. Good for thumbnails.",
    )
    async def snapshot(camera_id: str):
        if not _processor:
            return Response(status_code=503)
        jpeg = _processor.get_latest_jpeg(camera_id)
        if not jpeg:
            return Response(status_code=404, content=f"No frames yet for {camera_id}")
        return Response(content=jpeg, media_type="image/jpeg")

    return app


# ── MJPEG frame generator ─────────────────────────────────────────────────────

async def _frame_generator(camera_id: str) -> AsyncGenerator[bytes, None]:
    """
    Async generator that yields MJPEG boundary + JPEG data indefinitely.

    The HTTP response stays open; the browser reads each part as it arrives.
    If no frame is available yet, we wait briefly and retry.
    """
    BOUNDARY = b"--frame\r\n"
    HEADER   = b"Content-Type: image/jpeg\r\n\r\n"
    TAIL     = b"\r\n"

    # Yield a placeholder JPEG if processor isn't ready yet
    empty_sent = False

    while True:
        if not _processor:
            await asyncio.sleep(0.1)
            continue

        jpeg = _processor.get_latest_jpeg(camera_id)

        if jpeg is None:
            # Camera not started yet — send a tiny placeholder once
            if not empty_sent:
                placeholder = _make_placeholder_jpeg(camera_id)
                yield BOUNDARY + HEADER + placeholder + TAIL
                empty_sent = True
            await asyncio.sleep(0.2)
            continue

        empty_sent = False
        yield BOUNDARY + HEADER + jpeg + TAIL

        # Target frame rate: 10fps → sleep 100ms between yields
        target_interval = 1.0 / settings.STREAM_FPS
        await asyncio.sleep(target_interval)


def _make_placeholder_jpeg(camera_id: str) -> bytes:
    """
    Generate a simple gray JPEG with 'Connecting…' text.
    Shown in the dashboard while the camera is still starting.
    """
    import cv2
    import numpy as np

    h, w = settings.FRAME_HEIGHT, settings.FRAME_WIDTH
    frame = np.full((h, w, 3), 25, dtype=np.uint8)   # very dark gray

    # Camera ID text
    cv2.putText(
        frame, f"[ {camera_id} ]",
        (w // 2 - 60, h // 2 - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Connecting to stream...",
        (w // 2 - 100, h // 2 + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 80), 1, cv2.LINE_AA,
    )

    # Corner brackets — matches dashboard UI
    c = (0, 130, 150)
    t = 1
    s = 20
    cv2.line(frame, (12, 12),     (12 + s, 12),     c, t)
    cv2.line(frame, (12, 12),     (12, 12 + s),     c, t)
    cv2.line(frame, (w-12, 12),   (w-12-s, 12),     c, t)
    cv2.line(frame, (w-12, 12),   (w-12, 12+s),     c, t)
    cv2.line(frame, (12, h-12),   (12+s, h-12),     c, t)
    cv2.line(frame, (12, h-12),   (12, h-12-s),     c, t)
    cv2.line(frame, (w-12, h-12), (w-12-s, h-12),   c, t)
    cv2.line(frame, (w-12, h-12), (w-12, h-12-s),   c, t)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buf.tobytes()

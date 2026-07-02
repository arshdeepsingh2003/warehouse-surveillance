"""
main.py  —  Warehouse AI Service entry point
─────────────────────────────────────────────
Wires together:
  • StreamManager   — reads frames from mp4/RTSP sources
  • FrameProcessor  — mock AI analysis, posts results to backend
  • Stream Server   — MJPEG HTTP server (dashboard connects here)
  • Heartbeat       — periodic camera status push to backend
  • API Client      — shared aiohttp session for backend calls

Two servers run side-by-side in one process:
  Port 8000 → FastAPI backend (separate process, already running)
  Port 8001 → THIS service: MJPEG stream endpoints

Startup sequence:
  1. Create all objects
  2. Start aiohttp session (APIClient)
  3. Register cameras with StreamManager
  4. Start frame read loops (StreamManager)
  5. Launch MJPEG stream server (uvicorn on port 8001)
  6. Launch heartbeat task

Run:
  cd warehouse-ai-service
  python main.py

Or:
  uvicorn main:stream_app --host 0.0.0.0 --port 8001 --reload
  (then run the stream manager separately via `python run_streams.py`)
"""

import asyncio
import logging
import os
import sys
import signal

import uvicorn

from config.settings import settings
from pipeline.api_client import APIClient
from streams.stream_manager import StreamManager
from streams.stream_server import create_stream_app
from utils.heartbeat import run_heartbeat

# Use VLMAIFrameProcessor which includes ZoneSummarizer + real AI detection.
# Switch to mock FrameProcessor by importing it instead when testing streams only.
from pipeline.ai_frame_processor import VLMAIFrameProcessor as FrameProcessor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Main async entrypoint.
    Runs all components concurrently using asyncio.gather.
    """
    logger.info("="*55)
    logger.info("  Warehouse AI Service  —  Camera Stream Manager")
    logger.info(f"  Backend  : {settings.BACKEND_URL}")
    logger.info(f"  Streams  : http://localhost:{settings.STREAM_SERVER_PORT}")
    logger.info(f"  Sources  : {'mp4 files' if settings.USE_MOCK_SOURCES else 'RTSP streams'}")
    logger.info("="*55)

    # ── 1. Create shared objects ───────────────────────────────────────────────
    api_client = APIClient()
    processor  = FrameProcessor(api_client)
    processor.start_background_tasks()
    manager    = StreamManager()

    # ── 2. Start API client ────────────────────────────────────────────────────
    await api_client.start()

    # ── 3. Register cameras ────────────────────────────────────────────────────
    if settings.USE_MOCK_SOURCES:
        logger.info("Mock mode: registering cameras from mp4 files...")
        manager.register_all_from_config()
    else:
        # Real RTSP mode — read URLs from environment
        _register_rtsp_cameras(manager)

    if not manager._configs:
        logger.error("No cameras registered. Check MOCK_VIDEO_DIR or RTSP_URL_* env vars.")
        sys.exit(1)

    logger.info(f"Registered {len(manager._configs)} cameras.")

    # ── 4. Start frame read loops ──────────────────────────────────────────────
    await manager.start_all(frame_callback=processor.process)
    logger.info("Stream loops started.")

    # ── 5. Create MJPEG stream server ─────────────────────────────────────────
    stream_app = create_stream_app(processor)

    server_config = uvicorn.Config(
        app=stream_app,
        host=settings.STREAM_SERVER_HOST,
        port=settings.STREAM_SERVER_PORT,
        log_level="warning",   # Suppress per-request logs
        access_log=False,
    )
    server = uvicorn.Server(server_config)

    # ── 6. Run everything concurrently ────────────────────────────────────────
    # asyncio.gather runs all coroutines simultaneously:
    #   • server.serve()  — MJPEG HTTP server (blocks until stopped)
    #   • heartbeat       — pushes camera status every 10 s
    logger.info(
        f"MJPEG stream server starting on "
        f"http://{settings.STREAM_SERVER_HOST}:{settings.STREAM_SERVER_PORT}"
    )
    logger.info("Streams ready:")
    for cam_id in manager._configs:
        logger.info(
            f"  http://localhost:{settings.STREAM_SERVER_PORT}/stream/{cam_id}"
        )

    try:
        await asyncio.gather(
            server.serve(),
            run_heartbeat(manager, api_client),
        )
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logger.info("Shutting down...")
        await manager.stop_all()
        await api_client.stop()
        logger.info("AI service stopped.")


def _register_rtsp_cameras(manager: StreamManager) -> None:
    """
    Register real RTSP cameras from environment variables.
    Set RTSP_URL_CAM01, RTSP_URL_CAM02, etc. in .env.
    """
    from streams.stream_manager import CameraConfig

    ZONE_MAP = {
        "CAM01": ("Main Gate",       "entry_zone",      "cam-01"),
        "CAM02": ("Warehouse Aisle", "storage_area",    "cam-02"),
        "CAM03": ("Loading Zone",    "loading_zone",    "cam-03"),
        "CAM04": ("Storage Area",    "storage_area",    "cam-04"),
        "CAM05": ("Restricted Area", "restricted_area", "cam-05"),
        "CAM06": ("Packing Area",    "packing_area",    "cam-06"),
    }

    for key, (name, zone, cam_id) in ZONE_MAP.items():
        url = os.environ.get(f"RTSP_URL_{key}")
        if url:
            manager.register(CameraConfig(
                camera_id=   cam_id,
                name=        name,
                location=    zone.replace("_", " ").title(),
                zone=        zone,
                source=      url,
                source_type= "rtsp",
                enabled=     True,
            ))
            logger.info(f"Registered RTSP: {cam_id} → {url}")


if __name__ == "__main__":
    # Handle Ctrl+C cleanly
    def handle_sigint(sig, frame):
        logger.info("Received SIGINT — shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    asyncio.run(main())

"""
utils/heartbeat.py
───────────────────
Periodic heartbeat task.

Every N seconds: reads the current status of all cameras from the
StreamManager and pushes it to:
  1. The backend REST API  →  PATCH /api/v1/cameras/{id}/status
  2. The backend WS hub    →  POST /api/v1/events/broadcast
     (so the dashboard camera grid updates its fps / latency display)

This runs as an asyncio background task launched from main.py.
"""

import asyncio
import logging
import time

from config.settings import settings

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 10   # seconds between status pushes


async def run_heartbeat(stream_manager, api_client) -> None:
    """
    Background task: broadcast camera statuses every HEARTBEAT_INTERVAL seconds.

    Args:
        stream_manager: StreamManager instance
        api_client:     APIClient instance
    """
    logger.info(f"Heartbeat started (interval={HEARTBEAT_INTERVAL}s)")

    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)

        try:
            statuses = stream_manager.get_all_statuses()

            for cam_status in statuses:
                cam_id = cam_status["camera_id"]
                status = cam_status["status"]
                fps    = cam_status.get("fps", 0)

                # Estimate latency from last frame timestamp
                last_frame = cam_status.get("last_frame_at", 0)
                latency_ms = int((time.time() - last_frame) * 1000) if last_frame else 0
                latency_ms = min(latency_ms, 9999)   # cap at ~10 s

                # Push to backend REST (updates camera table)
                await api_client.post_camera_status(
                    camera_id=  cam_id,
                    status=     status,
                    fps=        fps,
                    latency_ms= latency_ms,
                )

                # Push to WS hub (live dashboard update)
                await api_client._post("/api/v1/events/broadcast", {
                    "type":       "camera_status",
                    "camera_id":  cam_id,
                    "status":     status,
                    "fps":        round(fps, 1),
                    "latency_ms": latency_ms,
                    "timestamp":  __import__("datetime").datetime.utcnow().isoformat() + "Z",
                })

            logger.debug(f"Heartbeat: pushed {len(statuses)} camera statuses.")

        except Exception as e:
            logger.error(f"Heartbeat error: {e}")

"""
api/routes/websocket.py
────────────────────────
WebSocket endpoint for real-time dashboard updates.

The dashboard connects once and receives events as they happen:
  • alert_triggered  – a new anomaly was detected by the AI pipeline
  • alert_resolved   – an operator resolved an alert
  • frame_update     – latest person tracking data per camera
  • camera_status    – a camera came online or went offline
  • ping             – keep-alive, sent every 30 s

Architecture note:
  In this mock version a background task fires fake events on a schedule.
  In production the AI pipeline and alert engine will call
  `await manager.broadcast(event)` directly when real events occur.

WebSocket URL: ws://localhost:8000/ws
"""

import asyncio
import random
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.connection_manager import manager

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")


# ── Mock event generators ─────────────────────────────────────────────────────

def _random_alert_event() -> dict:
    """Generate a plausible fake alert event payload."""
    alert_types = [
        ("unauthorized_access",   "high",   "Unauthorized person detected entering restricted zone."),
        ("ppe_violation",         "low",    "Worker detected without safety helmet in PPE zone."),
        ("loitering",             "medium", "Individual loitering near loading dock for 15+ minutes."),
        ("worker_fall",           "high",   "Worker fall detected. Person stationary for 30+ seconds."),
        ("suspicious_activity",   "medium", "Suspicious behaviour detected near storage rack."),
    ]
    cameras = ["cam-01", "cam-02", "cam-03", "cam-04", "cam-05"]
    zones   = ["entry_zone", "storage_area", "restricted_area", "loading_zone", "packing_area"]

    alert_type, severity, description = random.choice(alert_types)
    cam  = random.choice(cameras)
    zone = random.choice(zones)

    return {
        "type":        "alert_triggered",
        "alert_id":    f"alert-{random.randint(1000, 9999)}",
        "camera_id":   cam,
        "zone":        zone,
        "alert_type":  alert_type,
        "severity":    severity,
        "description": description,
        "person_id":   f"P-{random.randint(1000, 1099)}",
        "confidence":  round(random.uniform(0.70, 0.99), 2),
        "snapshot_url": f"https://placehold.co/640x360?text={alert_type.replace('_', '+')}",
        "timestamp":   datetime.now(UTC).isoformat(),
    }


def _random_frame_update() -> dict:
    """Generate a fake frame update (person tracking data)."""
    cameras = ["cam-01", "cam-02", "cam-03", "cam-04", "cam-05"]
    zones   = ["entry_zone", "storage_area", "restricted_area", "loading_zone", "packing_area"]
    activities = ["walking", "standing", "carrying_object", "handling_items"]

    cam = random.choice(cameras)
    n_persons = random.randint(1, 4)

    persons = [
        {
            "person_id":     f"P-{random.randint(1000, 1099)}",
            "zone":          random.choice(zones),
            "activity":      random.choice(activities),
            "dwell_seconds": random.randint(5, 600),
        }
        for _ in range(n_persons)
    ]

    return {
        "type":      "frame_update",
        "camera_id": cam,
        "persons":   persons,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _camera_status_event() -> dict:
    """Simulate an occasional camera status change."""
    return {
        "type":       "camera_status",
        "camera_id":  random.choice(["cam-01", "cam-02", "cam-03", "cam-04", "cam-05"]),
        "status":     random.choice(["online", "online", "online", "offline"]),  # online 3× more likely
        "fps":        random.randint(8, 15),
        "latency_ms": random.randint(30, 120),
        "timestamp":  datetime.now(UTC).isoformat(),
    }


# ── Background broadcaster ────────────────────────────────────────────────────

async def mock_event_broadcaster() -> None:
    """
    Runs forever in the background.
    Periodically broadcasts mock events to all connected WebSocket clients.

    Timeline (approximate):
      Every  5 s → frame update
      Every 20 s → random alert
      Every 45 s → camera status update
      Every 30 s → ping (keep-alive)
    """
    tick = 0
    while True:
        await asyncio.sleep(5)
        tick += 5

        # Frame update every 5 s
        if manager.connection_count() > 0:
            await manager.broadcast(_random_frame_update())

        # Alert event every ~20 s
        if tick % 20 == 0 and manager.connection_count() > 0:
            await manager.broadcast(_random_alert_event())

        # Camera status every ~45 s
        if tick % 45 == 0 and manager.connection_count() > 0:
            await manager.broadcast(_camera_status_event())

        # Ping every 30 s (keeps the connection alive through proxies)
        if tick % 30 == 0:
            await manager.broadcast({"type": "ping", "timestamp": datetime.now(UTC).isoformat()})


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Main WebSocket endpoint.

    Connect from the React dashboard:
      const ws = new WebSocket("ws://localhost:8000/ws");
      ws.onmessage = (e) => console.log(JSON.parse(e.data));

    The client can also send messages (e.g. to subscribe to a specific camera):
      ws.send(JSON.stringify({ action: "subscribe", camera_id: "cam-01" }))
    """
    await manager.connect(websocket)
    logger.info(f"WebSocket client connected. Total: {manager.connection_count()}")

    try:
        # Send a welcome message so the client knows the connection is live
        await manager.send_personal(websocket, {
            "type":    "connected",
            "message": "Connected to Warehouse AI Surveillance WebSocket",
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Keep the connection open and listen for client messages
        while True:
            data = await websocket.receive_text()
            logger.debug(f"WS message received: {data}")
            # TODO: handle client messages (e.g. subscribe/unsubscribe to rooms)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket client disconnected. Remaining: {manager.connection_count()}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
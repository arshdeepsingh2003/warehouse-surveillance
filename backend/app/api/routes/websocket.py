"""
api/routes/websocket.py
────────────────────────
WebSocket endpoint for real-time dashboard updates.

The dashboard connects once and receives events as they happen:
  • alert_triggered  – a new anomaly was detected by the AI pipeline
                      (via POST /api/v1/alerts/ingest from the AI service)
  • alert_resolved   – an operator resolved an alert
  • frame_update     – latest person tracking data per camera
                      (via POST /api/v1/events/broadcast from the AI service)
  • camera_status    – a camera came online or went offline
                      (via PATCH /api/v1/cameras/{id}/status from heartbeat)
  • ping             – keep-alive, sent every 30 s

All dashboard alerts originate exclusively from the real AI pipeline:
  Detection → Tracking → ActivityAnalyzer → RulesEngine → API Client
  → POST /api/v1/alerts/ingest → WebSocket Broadcast → Dashboard

WebSocket URL: ws://localhost:8000/ws
"""

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.connection_manager import manager

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")


# ── Keep-alive broadcaster ────────────────────────────────────────────────────

async def ws_keepalive_broadcaster() -> None:
    """
    Runs forever in the background.
    Sends a ping every 30 s to keep WebSocket connections alive through proxies.
    No fake alert, frame, or camera status events are generated here.
    All real-time data comes from the AI pipeline via REST ingest endpoints.
    """
    while True:
        await asyncio.sleep(30)
        if manager.connection_count() > 0:
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

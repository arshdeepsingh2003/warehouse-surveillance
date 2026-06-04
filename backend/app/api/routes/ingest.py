"""
api/routes/ingest.py
─────────────────────
Ingest endpoints called by the AI service (not the dashboard).

The AI service (warehouse-ai-service) POSTs raw events here.
The backend:
  1. Validates the payload (Pydantic)
  2. Appends to the in-memory store (mock mode) or DB (production)
  3. Broadcasts to all WebSocket clients via the ConnectionManager

This is the bridge between the AI pipeline and the live dashboard.

Endpoints:
  POST /api/v1/activities/ingest   ← activity log events from FrameProcessor
  POST /api/v1/alerts/ingest       ← anomaly alerts from FrameProcessor
  POST /api/v1/events/broadcast    ← generic WS broadcast (frame updates, camera status)
  PATCH /api/v1/cameras/{id}/status ← camera online/offline/fps updates from heartbeat
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from app.ws.connection_manager import manager as ws_manager
from app.services.mock_data import MOCK_ACTIVITIES, MOCK_ALERTS, MOCK_CAMERAS

router = APIRouter(tags=["Ingest (AI Service)"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ActivityIngest(BaseModel):
    id:            str
    person_id:     str
    camera_id:     str
    zone:          str
    activity_type: str
    description:   str
    anomaly_label: str
    dwell_seconds: int
    confidence:    float
    timestamp:     str


class AlertIngest(BaseModel):
    id:           str
    camera_id:    str
    zone:         str
    alert_type:   str
    severity:     str
    description:  str
    person_id:    str
    confidence:   float
    status:       str = "active"
    triggered_at: str
    snapshot_b64: Optional[str] = None   # base64 JPEG — store or forward
    snapshot_url: Optional[str] = None


class BroadcastEvent(BaseModel):
    """Generic event forwarded straight to WebSocket clients."""
    type:      str
    timestamp: str
    # All other fields are passed through as-is
    model_config = {"extra": "allow"}


class CameraStatusUpdate(BaseModel):
    status:     str
    fps:        float = 0
    latency_ms: int   = 0


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/activities/ingest", status_code=201)
async def ingest_activity(data: ActivityIngest):
    """
    Receive an activity log entry from the AI service.
    Appends to the in-memory activity list and broadcasts via WebSocket.
    """
    record = data.model_dump()
    # Prepend so GET /activities returns newest first
    MOCK_ACTIVITIES.insert(0, record)

    # Keep list bounded (last 500 entries)
    if len(MOCK_ACTIVITIES) > 500:
        del MOCK_ACTIVITIES[500:]

    # Broadcast to dashboard (Activity Log page live updates)
    await ws_manager.broadcast({
        "type":        "activity_update",
        "activity_id": data.id,
        "person_id":   data.person_id,
        "camera_id":   data.camera_id,
        "zone":        data.zone,
        "activity":    data.activity_type,
        "label":       data.anomaly_label,
        "timestamp":   data.timestamp,
    })

    return {"status": "ok", "id": data.id}


@router.post("/alerts/ingest", status_code=201)
async def ingest_alert(data: AlertIngest):
    """
    Receive an anomaly alert from the AI service.
    Appends to in-memory alerts and broadcasts to dashboard.
    """
    # Build the record (strip raw base64 — too large for the list)
    record = {
        "id":           data.id,
        "camera_id":    data.camera_id,
        "zone":         data.zone,
        "alert_type":   data.alert_type,
        "severity":     data.severity,
        "description":  data.description,
        "person_id":    data.person_id,
        "snapshot_url": data.snapshot_url,  # None for now; wire S3 later
        "status":       "active",
        "confidence":   data.confidence,
        "triggered_at": data.triggered_at,
        "resolved_at":  None,
        "resolved_by":  None,
    }

    MOCK_ALERTS.insert(0, record)
    if len(MOCK_ALERTS) > 200:
        del MOCK_ALERTS[200:]

    # Broadcast alert to dashboard Alert Panel (real-time)
    await ws_manager.broadcast({
        "type":        "alert_triggered",
        "alert_id":    data.id,
        "camera_id":   data.camera_id,
        "zone":        data.zone,
        "alert_type":  data.alert_type,
        "severity":    data.severity,
        "description": data.description,
        "person_id":   data.person_id,
        "confidence":  data.confidence,
        "snapshot_url": data.snapshot_url,
        "timestamp":   data.triggered_at,
    })

    return {"status": "ok", "id": data.id}


@router.post("/events/broadcast", status_code=200)
async def broadcast_event(data: BroadcastEvent):
    """
    Generic broadcast: forward any JSON payload to all WebSocket clients.
    Used for frame_update, camera_status, and other real-time events.
    """
    payload = data.model_dump()
    if payload.get("type") == "frame_update":
        cam_id = payload.get("camera_id", "?")
        persons = payload.get("persons", [])
        logger.info(
            f"🔍 TRACE[backend-ingest] RECEIVED frame_update camera={cam_id} | "
            f"persons={len(persons)}"
        )
    await ws_manager.broadcast(payload)
    logger.info(
        f"🔍 TRACE[backend-ws] BROADCAST type={payload.get('type')} "
        f"camera={payload.get('camera_id', '?')} "
        f"persons={len(payload.get('persons', [])) if payload.get('type') == 'frame_update' else 'N/A'}"
    )
    return {"status": "broadcast_sent", "type": data.type}


@router.patch("/cameras/{camera_id}/status", status_code=200)
async def update_camera_status(camera_id: str, data: CameraStatusUpdate):
    """
    Update a camera's live status, FPS, and latency.
    Called by the AI service heartbeat every 10 seconds.
    """
    for cam in MOCK_CAMERAS:
        if cam["id"] == camera_id:
            cam["status"]     = data.status
            cam["fps"]        = int(data.fps)
            cam["latency_ms"] = data.latency_ms
            cam["updated_at"] = datetime.now(timezone.utc)
            break

    return {"status": "ok", "camera_id": camera_id}

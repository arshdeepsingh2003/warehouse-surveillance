"""
api/routes/vlm_insights.py
───────────────────────────
REST API routes for the VLMInsight resource.

VLM insights are AI-generated descriptions produced asynchronously
by Vision Language Models. They live in a separate store from Activity
records to avoid duplication and overwriting issues.

Endpoints:
  GET  /vlm-insights                     – list VLM insights (filterable)
  POST /vlm-insights/ingest              – ingest from AI service
"""

import logging
from typing import Optional
from fastapi import APIRouter, Query

from app.schemas.vlm_insight import VLMInsightOut
from app.services import vlm_insight_service
from app.ws.connection_manager import manager as ws_manager
from app.services.mock_data import MOCK_VLM_INSIGHTS

logger = logging.getLogger(__name__)

from datetime import datetime, timezone
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/vlm-insights",
    tags=["VLM Insights"],
)


# ── Ingest schema ───────────────────────────────────────────────────────────

class VLMInsightIngest(BaseModel):
    id:               str
    person_id:        str
    camera_id:        str
    zone:             str
    activity_type:    str
    description:      str
    anomaly_label:    str
    confidence:       float
    timestamp:        str
    objects_detected: list[str] = []
    backend_used:     str       = ""
    latency_ms:       int       = 0
    source:           str       = "vlm"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=list[VLMInsightOut],
    summary="List VLM insight records",
    description=(
        "Returns VLM-generated insights in reverse-chronological order. "
        "Filter by camera, zone, person, or anomalies only. "
        "This endpoint powers the AI Insights page."
    ),
)
async def list_vlm_insights(
    camera_id:    Optional[str] = Query(None, description="Filter by camera ID"),
    zone:         Optional[str] = Query(None, description="Filter by zone name"),
    person_id:    Optional[str] = Query(None, description="Filter by person ID, e.g. P-1025"),
    anomaly_only: bool          = Query(False, description="Return only anomalous insights"),
    limit:        int           = Query(100, ge=1, le=1000),
) -> list[VLMInsightOut]:
    return await vlm_insight_service.get_all_vlm_insights(
        camera_id=camera_id, zone=zone, person_id=person_id,
        anomaly_only=anomaly_only, limit=limit,
    )


@router.post("/ingest", status_code=201)
async def ingest_vlm_insight(data: VLMInsightIngest):
    """
    Receive a VLM insight entry from the AI service.
    Appends to the in-memory VLM insights list and broadcasts via WebSocket.
    """
    record = data.model_dump()
    MOCK_VLM_INSIGHTS.insert(0, record)

    if len(MOCK_VLM_INSIGHTS) > 500:
        del MOCK_VLM_INSIGHTS[500:]

    # Broadcast to dashboard (AI Insights page live updates)
    await ws_manager.broadcast({
        "type":             "vlm_insight",
        "insight_id":       data.id,
        "person_id":        data.person_id,
        "camera_id":        data.camera_id,
        "zone":             data.zone,
        "activity_type":    data.activity_type,
        "anomaly_label":    data.anomaly_label,
        "description":      data.description,
        "confidence":       data.confidence,
        "objects_detected": data.objects_detected,
        "backend_used":     data.backend_used,
        "latency_ms":       data.latency_ms,
        "source":           data.source,
        "timestamp":        data.timestamp,
    })
    logger.info(
        f"VLM insight ingested: {data.id} person={data.person_id} "
        f"type={data.activity_type} backend={data.backend_used}"
    )

    return {"status": "ok", "id": data.id}


@router.get("/metrics", summary="VLM event engine metrics")
async def vlm_metrics():
    """
    Return VLM event-driven engine metrics.
    The AI service at port 8001 serves the live metrics.
    This endpoint returns a summary of stored VLM insight counts.
    """
    return {
        "total_insights_stored": len(MOCK_VLM_INSIGHTS),
        "anomaly_insights": sum(
            1 for i in MOCK_VLM_INSIGHTS if i.get("anomaly_label") == "anomaly"
        ),
        "backends_used": list(set(
            i.get("backend_used", "unknown") for i in MOCK_VLM_INSIGHTS
        )),
        "note": "Live EventEngine metrics available at AI service /vlm/metrics",
    }


@router.post("/clear", status_code=200)
async def clear_vlm_insights():
    """
    Clear all in-memory VLM insights.
    Useful for resetting state between test runs or after reconfiguring the AI service.
    """
    count = len(MOCK_VLM_INSIGHTS)
    MOCK_VLM_INSIGHTS.clear()
    logger.info(f"Cleared {count} VLM insights from memory")
    return {"status": "ok", "cleared": count}

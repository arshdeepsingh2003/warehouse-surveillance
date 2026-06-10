"""
services/vlm_insight_service.py
─────────────────────────────────
Business logic for the VLMInsight resource.

VLM insights are produced asynchronously by the VLM pipeline and stored
separately from Activity records. The AI Insights page reads from here.
"""

from typing import Optional
from app.core.config import settings
from app.services.mock_data import MOCK_VLM_INSIGHTS
from app.schemas.vlm_insight import VLMInsightOut


async def get_all_vlm_insights(
    camera_id:    Optional[str] = None,
    zone:         Optional[str] = None,
    person_id:    Optional[str] = None,
    anomaly_only: bool = False,
    limit:        int = 100,
) -> list[VLMInsightOut]:
    """
    Return VLM insight records with optional filters.

    Args:
        camera_id:    filter by specific camera
        zone:         filter by zone name
        person_id:    filter to a single person
        anomaly_only: if True, only return anomalous insights
        limit:        max results

    Returns:
        List of VLMInsightOut, newest first.
    """
    if settings.USE_MOCK_DATA:
        items = MOCK_VLM_INSIGHTS.copy()

        if camera_id:   items = [a for a in items if a["camera_id"] == camera_id]
        if zone:        items = [a for a in items if a["zone"]       == zone]
        if person_id:   items = [a for a in items if a["person_id"]  == person_id]
        if anomaly_only:
            items = [a for a in items if a["anomaly_label"] == "anomaly"]

        items.sort(key=lambda a: a["timestamp"], reverse=True)
        return [VLMInsightOut(**a) for a in items[:limit]]

    raise NotImplementedError("Database not yet connected.")

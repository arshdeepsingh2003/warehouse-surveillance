"""
schemas/vlm_insight.py
──────────────────────
Pydantic schemas for the VLMInsight resource.

A VLMInsight is an AI-generated understanding of a person's activity,
produced asynchronously by a Vision Language Model (VLM). Unlike Activity
records (which are generated continuously per frame), VLM insights are
created infrequently and live in a separate store.

The AI Insights page consumes VLMInsight records only.
The Activity Log page consumes Activity records only.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class VLMInsightOut(BaseModel):
    """VLM insight entry returned by the API."""
    id:               str
    person_id:        str
    camera_id:        str
    zone:             str
    activity_type:    str
    description:      str
    anomaly_label:    str
    confidence:       float = Field(..., ge=0.0, le=1.0)
    objects_detected: list[str] = []
    backend_used:     str   = ""
    latency_ms:       int   = 0
    source:           str   = "vlm"   # "vlm" or "hybrid"
    timestamp:        datetime

    model_config = {"from_attributes": True}

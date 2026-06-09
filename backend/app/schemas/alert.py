"""
schemas/alert.py
────────────────
Pydantic schemas for the Alert resource.

Alerts are created by the AI anomaly classifier and consumed by
the dashboard's Alert Panel. They can be acknowledged or resolved
by a security operator.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class AlertStatus(str, Enum):
    ACTIVE   = "active"    # Not yet acknowledged
    RESOLVED = "resolved"  # Operator marked it resolved


class AlertType(str, Enum):
    """
    All anomaly categories the AI pipeline can detect.
    Add new categories here as the model improves.
    """
    UNAUTHORIZED_ACCESS   = "unauthorized_access"
    LOITERING             = "loitering"
    PPE_VIOLATION         = "ppe_violation"
    WORKER_FALL           = "worker_fall"
    RESTRICTED_ZONE_ENTRY = "restricted_zone_entry"
    SUSPICIOUS_ACTIVITY   = "suspicious_activity"
    THEFT_ATTEMPT         = "theft_attempt"
    UNKNOWN               = "unknown"


# ── Base ──────────────────────────────────────────────────────────────────────

class AlertBase(BaseModel):
    camera_id:    str        = Field(..., description="Which camera triggered the alert")
    zone:         str        = Field(..., description="Zone where the anomaly occurred")
    alert_type:   AlertType
    severity:     AlertSeverity
    description:  str        = Field(..., description="Human-readable description from the LLM")
    person_id:    Optional[str] = Field(None, description="Tracked person ID if identified")
    snapshot_url: Optional[str] = Field(None, description="S3 / static URL of the frame snapshot")


# ── Response ──────────────────────────────────────────────────────────────────

class AlertOut(AlertBase):
    """Full alert object returned by the API."""
    id:             str
    status:         AlertStatus = AlertStatus.ACTIVE
    confidence:     float       = Field(..., ge=0.0, le=1.0, description="AI confidence score 0–1")
    triggered_at:   datetime
    resolved_at:    Optional[datetime] = None
    resolved_by:    Optional[str]      = None   # Email or username of resolving operator
    source:         str = "other"              # rules_engine | activity_analyzer | manual_test | other

    model_config = {"from_attributes": True}


# ── Resolve action ────────────────────────────────────────────────────────────

class AlertResolve(BaseModel):
    """Body expected when PATCH /alerts/{id}/resolve is called."""
    resolved_by: str = Field(..., description="Email of the operator resolving the alert")


# ── WebSocket event ───────────────────────────────────────────────────────────

class AlertWSEvent(BaseModel):
    """
    Payload broadcast over WebSocket when a new alert fires.
    Clients subscribe to ws://host/ws and receive these in real time.
    """
    type:      str = "alert_triggered"   # event type discriminator
    alert:     AlertOut

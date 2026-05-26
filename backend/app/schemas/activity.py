"""
Pydantic schemas for the Activity resource.

An Activity is a single detected action for one tracked person in one frame.
The VLM generates the description; the activity is then persisted and served
by the Activity Log API.

This file defines the Activity Tracking System for your AI surveillance platform.
It stores and serves information about:

what a person is doing
where they are
how long they stayed
whether activity is suspicious
movement timeline of a person

Think of this as the behavior logging system of your project 

"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ActivityType(str, Enum):
    """
    High-level activity categories (from VLM classification).
    The 'description' field carries the full natural-language sentence.
    """
    WALKING            = "walking"
    STANDING           = "standing"
    CARRYING_OBJECT    = "carrying_object"
    LOITERING          = "loitering"
    RUNNING            = "running"
    FALLING            = "falling"
    HANDLING_ITEMS     = "handling_items"
    UNAUTHORIZED_ENTRY = "unauthorized_entry"
    UNKNOWN            = "unknown"


class AnomalyLabel(str, Enum):
    NORMAL  = "normal"
    ANOMALY = "anomaly"


# Base

class ActivityBase(BaseModel):
    person_id:     str           = Field(..., description="Tracked person ID, e.g. 'P-1025'")
    camera_id:     str           = Field(..., description="Camera that captured the frame")
    zone:          str           = Field(..., description="Zone the person was in")
    activity_type: ActivityType
    description:   str           = Field(..., description="VLM natural-language description")
    anomaly_label: AnomalyLabel  = AnomalyLabel.NORMAL
    dwell_seconds: int           = Field(0, description="Seconds person has been in this zone")


# Response 

class ActivityOut(ActivityBase):
    """Full activity log entry returned by the API."""
    id:           str
    confidence:   float    = Field(..., ge=0.0, le=1.0)
    timestamp:    datetime

    model_config = {"from_attributes": True}


# Person timeline entry

class PersonTimelineEntry(BaseModel):
    """
    One step in a person's movement timeline.
    Used by GET /persons/{person_id}/timeline.
    """
    zone:          str
    camera_id:     str
    activity_type: ActivityType
    description:   str
    entry_time:    datetime
    exit_time:     Optional[datetime] = None
    dwell_seconds: int


class PersonTimeline(BaseModel):
    person_id: str
    timeline:  list[PersonTimelineEntry]
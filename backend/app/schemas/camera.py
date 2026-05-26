"""
Pydantic schemas for the Camera resource.

A "schema" describes the shape of data going IN (request body) and OUT
(response body) of the API. FastAPI uses these to auto-validate and
auto-document your endpoints.

Naming convention used here:
  CameraBase     – shared fields
  CameraCreate   – fields required to CREATE a camera (POST body)
  CameraOut      – what we send BACK to the client (response)

This file defines the data structure for your Camera API using Pydantic schemas in a FastAPI project.
Think of it as the rulebook for what camera data should look like when:

a client sends data to the API
the API sends data back
FastAPI validates requests automatically
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class CameraStatus(str, Enum):
    """Possible live states for a camera stream."""
    ONLINE  = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class CameraType(str, Enum):
    """How the video is sourced."""
    RTSP   = "rtsp"    # Live IP camera
    FILE   = "file"    # Recorded video file
    MOCK   = "mock"    # Simulated stream for testing


class CameraBase(BaseModel):
    """Fields shared between create and read schemas."""
    name:        str = Field(..., description="Human-readable label, e.g. 'Main Gate'")
    location:    str = Field(..., description="Physical location, e.g. 'Gate A'")
    zone:        str = Field(..., description="Logical monitoring zone, e.g. 'restricted_area'")
    stream_url:  str = Field(..., description="RTSP URL or file path")
    camera_type: CameraType = Field(CameraType.MOCK, description="Source type")


class CameraCreate(CameraBase):
    """Body expected when a client POSTs a new camera."""
    pass  # Inherits everything from CameraBase; add extra fields here later


# ── Response ──────────────────────────────────────────────────────────────────

class CameraOut(CameraBase):
    """
    What the API returns for a camera.
    Adds server-managed fields like id, status, and timestamps.
    """
    id:          str          = Field(..., description="Unique camera ID, e.g. 'cam-01'")
    status:      CameraStatus = Field(CameraStatus.UNKNOWN)
    fps:         int          = Field(0,   description="Current frames per second")
    latency_ms:  int          = Field(0,   description="Stream latency in milliseconds")
    created_at:  datetime
    updated_at:  datetime

    # Allows FastAPI to read data from ORM objects (not just plain dicts)
    model_config = {"from_attributes": True}
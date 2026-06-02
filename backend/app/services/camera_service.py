"""
services/camera_service.py
──────────────────────────
Business logic for the Camera resource.

Services sit between routes (HTTP layer) and the data layer (database or mock).
Routes call service functions; services decide where to get/store data.

This makes it easy to swap mock data for real database queries later —
the routes don't change at all, only the service implementation.
"""

from typing import Optional
from app.core.config import settings
from app.services.mock_data import MOCK_CAMERAS
from app.schemas.camera import CameraOut, CameraCreate, CameraStatus


async def get_all_cameras(
    status: Optional[str] = None,
    zone:   Optional[str] = None,
) -> list[CameraOut]:
    """
    Return all cameras, optionally filtered by status and/or zone.

    Args:
        status: Filter by "online" | "offline" | "unknown"
        zone:   Filter by zone name, e.g. "restricted_area"

    Returns:
        List of CameraOut objects validated by Pydantic.
    """
    if settings.USE_MOCK_DATA:
        cameras = MOCK_CAMERAS.copy()

        # Apply filters
        if status:
            cameras = [c for c in cameras if c["status"] == status]
        if zone:
            cameras = [c for c in cameras if c["zone"] == zone]

        # Validate and return as Pydantic models
        return [CameraOut(**c) for c in cameras]

    # ── FUTURE: replace with real DB query ────────────────────────────────────
    # async with get_db() as db:
    #     query = select(Camera)
    #     if status: query = query.where(Camera.status == status)
    #     if zone:   query = query.where(Camera.zone == zone)
    #     result = await db.execute(query)
    #     return result.scalars().all()
    raise NotImplementedError("Database not yet connected. Set USE_MOCK_DATA=true.")


async def get_camera_by_id(camera_id: str) -> Optional[CameraOut]:
    """Return a single camera by its ID, or None if not found."""
    if settings.USE_MOCK_DATA:
        for c in MOCK_CAMERAS:
            if c["id"] == camera_id:
                return CameraOut(**c)
        return None

    raise NotImplementedError("Database not yet connected.")


async def create_camera(data: CameraCreate) -> CameraOut:
    """
    Add a new camera to the system.

    In mock mode we just echo the data back with a generated ID.
    In production this would INSERT into the database.
    """
    if settings.USE_MOCK_DATA:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("UTC"))
        new_camera = {
            **data.model_dump(),
            "id":         f"cam-{len(MOCK_CAMERAS) + 1:02d}",
            "status":     CameraStatus.UNKNOWN,
            "fps":        0,
            "latency_ms": 0,
            "created_at": now,
            "updated_at": now,
        }
        MOCK_CAMERAS.append(new_camera)  # persists for the session only
        return CameraOut(**new_camera)

    raise NotImplementedError("Database not yet connected.")

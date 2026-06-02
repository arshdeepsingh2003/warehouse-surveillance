"""
api/routes/activities.py
────────────────────────
REST API routes for the Activity Log.

Endpoints:
  GET /activities                      – full activity log (filterable)
  GET /activities/persons/{person_id}  – all activities for one person
  GET /persons/{person_id}/timeline    – person movement timeline
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.schemas.activity import ActivityOut, PersonTimeline
from app.services import activity_service

router = APIRouter(
    prefix="/activities",
    tags=["Activities"],
)


@router.get(
    "/",
    response_model=list[ActivityOut],
    summary="List activity log entries",
    description=(
        "Returns detected activities in reverse-chronological order. "
        "Filter by camera, zone, person, or anomalies only."
    ),
)
async def list_activities(
    camera_id:    Optional[str] = Query(None, description="Filter by camera ID"),
    zone:         Optional[str] = Query(None, description="Filter by zone name"),
    person_id:    Optional[str] = Query(None, description="Filter by person ID, e.g. P-1025"),
    anomaly_only: bool          = Query(False, description="Return only anomalous activities"),
    limit:        int           = Query(100, ge=1, le=1000),
) -> list[ActivityOut]:
    return await activity_service.get_all_activities(
        camera_id=camera_id, zone=zone, person_id=person_id,
        anomaly_only=anomaly_only, limit=limit,
    )


@router.get(
    "/persons/{person_id}",
    response_model=list[ActivityOut],
    summary="All activities for a person",
)
async def get_person_activities(person_id: str) -> list[ActivityOut]:
    return await activity_service.get_all_activities(person_id=person_id)


@router.get(
    "/persons/{person_id}/timeline",
    response_model=PersonTimeline,
    summary="Person movement timeline",
    description=(
        "Returns a step-by-step timeline of a person's movement through "
        "warehouse zones. Used by the Person Timeline dashboard page."
    ),
)
async def get_person_timeline(person_id: str) -> PersonTimeline:
    timeline = await activity_service.get_person_timeline(person_id)
    if not timeline:
        raise HTTPException(
            status_code=404,
            detail=f"No activities found for person '{person_id}'.",
        )
    return timeline

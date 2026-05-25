from fastapi import APIRouter, HTTPException

from app.schemas.activity import ActivityOut, PersonTimeline
from app.services.activity_service import ActivityService

router = APIRouter(prefix="/activities", tags=["activities"])


@router.get("/", response_model=list[ActivityOut])
async def list_activities():
    return ActivityService.list()


@router.get("/{activity_id}", response_model=ActivityOut)
async def get_activity(activity_id: str):
    act = ActivityService.get(activity_id)
    if not act:
        raise HTTPException(status_code=404, detail="Activity not found")
    return act


@router.get("/person/{person_id}", response_model=PersonTimeline)
async def person_timeline(person_id: str):
    acts = ActivityService.timeline(person_id)
    if not acts:
        raise HTTPException(status_code=404, detail="Person not found")
    return PersonTimeline(
        person_id=person_id,
        activities=acts,
        first_seen=acts[-1]["timestamp"],
        last_seen=acts[0]["timestamp"],
    )

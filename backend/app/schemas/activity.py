from pydantic import BaseModel
from datetime import datetime


class ActivityOut(BaseModel):
    id: str
    camera_id: str
    camera_name: str
    person_id: str | None
    activity_type: str
    timestamp: datetime
    description: str
    metadata: dict


class PersonTimeline(BaseModel):
    person_id: str
    activities: list[ActivityOut]
    first_seen: datetime
    last_seen: datetime

from pydantic import BaseModel
from datetime import datetime


class AlertOut(BaseModel):
    id: str
    camera_id: str
    camera_name: str
    type: str
    severity: str
    description: str
    timestamp: datetime
    acknowledged: bool


class AlertResolve(BaseModel):
    resolved: bool
    notes: str | None = None

from pydantic import BaseModel
from datetime import datetime


class CameraCreate(BaseModel):
    name: str
    rtsp_url: str
    zone_id: str | None = None


class CameraOut(BaseModel):
    id: str
    name: str
    rtsp_url: str
    zone_id: str | None
    status: str
    created_at: datetime

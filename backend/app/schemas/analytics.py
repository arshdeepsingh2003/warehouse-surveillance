from pydantic import BaseModel


class AnalyticsSummary(BaseModel):
    total_alerts: int
    unresolved_alerts: int
    active_cameras: int
    total_activities: int
    unique_persons: int

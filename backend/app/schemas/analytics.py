"""
schemas/analytics.py
────────────────────
Pydantic schemas for the Analytics API.

These are aggregated/summarised views consumed by the dashboard
Analytics page — charts, KPI cards, zone risk maps, etc.
"""

from pydantic import BaseModel, Field


class DashboardSummary(BaseModel):
    """KPI cards shown at the top of the dashboard."""
    total_cameras:        int
    cameras_online:       int
    total_alerts_today:   int
    active_alerts:        int
    high_severity_alerts: int
    people_detected:      int
    most_risky_zone:      str
    peak_activity_hour:   str   # e.g. "14:00 – 15:00"
    system_status:        str   # "healthy" | "degraded" | "critical"


class AlertTrendPoint(BaseModel):
    """One data point for the 'Alerts over time' line chart."""
    hour:   str   # e.g. "08:00"
    count:  int


class ZoneRisk(BaseModel):
    """One slice in the zone-wise incident donut chart."""
    zone:       str
    incidents:  int
    percentage: float


class AnalyticsSummary(BaseModel):
    """Full analytics payload returned by GET /analytics/summary."""
    summary:       DashboardSummary
    alert_trend:   list[AlertTrendPoint]  = Field(description="Last 24 h, hourly buckets")
    zone_risk:     list[ZoneRisk]

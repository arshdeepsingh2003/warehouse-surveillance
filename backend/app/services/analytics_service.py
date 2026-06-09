"""
services/analytics_service.py
──────────────────────────────
Business logic for the Analytics API.

Aggregates data from cameras, alerts, and activities into
KPI summaries and chart data for the dashboard Analytics page.

All values are computed live from the in-memory lists:
  MOCK_CAMERAS    — camera status / totals
  MOCK_ALERTS     — alert counts, zone risk, hourly trend
  MOCK_ACTIVITIES — unique people detected
"""

from datetime import datetime
from app.core.config import settings
from app.services.mock_data import MOCK_CAMERAS, MOCK_ALERTS, MOCK_ACTIVITIES
from app.schemas.analytics import AnalyticsSummary, DashboardSummary, AlertTrendPoint, ZoneRisk


def _parse_dt(val) -> datetime:
    """Accept datetime or ISO string, return datetime."""
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(val)


def _compute_summary() -> DashboardSummary:
    cameras = MOCK_CAMERAS
    alerts  = MOCK_ALERTS
    acts    = MOCK_ACTIVITIES

    total_cameras  = len(cameras)
    cameras_online = sum(1 for c in cameras if c.get("status") == "online")

    total_alerts   = len(alerts)
    active_alerts  = sum(1 for a in alerts if a.get("status") == "active")
    high_severity  = sum(1 for a in alerts if a.get("severity") == "high")

    people = {a.get("person_id") for a in acts if a.get("person_id")}
    people_detected = len(people)

    # Zone risk from alerts
    zone_incidents: dict[str, int] = {}
    for a in alerts:
        z = a.get("zone") or "unknown"
        zone_incidents[z] = zone_incidents.get(z, 0) + 1

    sorted_zones = sorted(zone_incidents.items(), key=lambda x: x[1], reverse=True)
    most_risky_zone = sorted_zones[0][0] if sorted_zones else "N/A"
    total_incidents = sum(zone_incidents.values()) or 1

    # Peak activity hour from alert timestamps
    hour_counts: dict[int, int] = {}
    for a in alerts:
        ts = _parse_dt(a["triggered_at"])
        h = ts.hour
        hour_counts[h] = hour_counts.get(h, 0) + 1

    if hour_counts:
        peak_h = max(hour_counts, key=hour_counts.get)
        peak_hour_range = f"{peak_h:02d}:00 – {(peak_h + 1) % 24:02d}:00"
    else:
        peak_hour_range = "N/A"

    return DashboardSummary(
        total_cameras        = total_cameras,
        cameras_online       = cameras_online,
        total_alerts_today   = total_alerts,
        active_alerts        = active_alerts,
        high_severity_alerts = high_severity,
        people_detected      = people_detected,
        most_risky_zone      = most_risky_zone,
        peak_activity_hour   = peak_hour_range,
        system_status        = "healthy" if cameras_online > 0 else "critical",
    )


def _compute_alert_trend() -> list[AlertTrendPoint]:
    hour_counts: dict[int, int] = {}
    for a in MOCK_ALERTS:
        ts = _parse_dt(a["triggered_at"])
        hour_counts[ts.hour] = hour_counts.get(ts.hour, 0) + 1
    return [AlertTrendPoint(hour=f"{h:02d}:00", count=hour_counts.get(h, 0)) for h in range(24)]


def _compute_zone_risk() -> list[ZoneRisk]:
    zone_incidents: dict[str, int] = {}
    for a in MOCK_ALERTS:
        z = a.get("zone") or "unknown"
        zone_incidents[z] = zone_incidents.get(z, 0) + 1

    total = sum(zone_incidents.values()) or 1
    sorted_zones = sorted(zone_incidents.items(), key=lambda x: x[1], reverse=True)
    return [
        ZoneRisk(zone=z, incidents=c, percentage=round(c / total * 100, 1))
        for z, c in sorted_zones
    ]


async def get_analytics_summary() -> AnalyticsSummary:
    """
    Return the full analytics payload used by the dashboard.

    All values are computed live from the in-memory lists so the
    analytics page always reflects the current state of real data.
    """
    if settings.USE_MOCK_DATA:
        return AnalyticsSummary(
            summary    = _compute_summary(),
            alert_trend= _compute_alert_trend(),
            zone_risk  = _compute_zone_risk(),
        )

    raise NotImplementedError("Database not yet connected.")

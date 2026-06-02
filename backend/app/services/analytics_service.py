"""
services/analytics_service.py
──────────────────────────────
Business logic for the Analytics API.

Aggregates data from cameras, alerts, and activities into
KPI summaries and chart data for the dashboard Analytics page.
"""

from app.core.config import settings
from app.services.mock_data import MOCK_ANALYTICS
from app.schemas.analytics import AnalyticsSummary


async def get_analytics_summary() -> AnalyticsSummary:
    """
    Return the full analytics payload used by the dashboard.

    Includes:
    - KPI summary cards (total alerts, active alerts, etc.)
    - Hourly alert trend for the line chart
    - Zone-wise risk breakdown for the donut chart
    """
    if settings.USE_MOCK_DATA:
        return AnalyticsSummary(**MOCK_ANALYTICS)

    # ── FUTURE: compute from real data ────────────────────────────────────────
    # summary   = await compute_kpi_summary()
    # trend     = await compute_hourly_alert_trend()
    # zone_risk = await compute_zone_risk()
    # return AnalyticsSummary(summary=summary, alert_trend=trend, zone_risk=zone_risk)
    raise NotImplementedError("Database not yet connected.")

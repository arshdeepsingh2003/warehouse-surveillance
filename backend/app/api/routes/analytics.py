"""
api/routes/analytics.py
────────────────────────
REST API routes for the Analytics dashboard.

Endpoints:
  GET /analytics/summary  – KPI cards + chart data for the analytics page
"""

from fastapi import APIRouter
from app.schemas.analytics import AnalyticsSummary
from app.services import analytics_service

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)


@router.get(
    "/summary",
    response_model=AnalyticsSummary,
    summary="Analytics dashboard summary",
    description=(
        "Returns everything the Analytics page needs: KPI summary cards, "
        "hourly alert trend for the line chart, and zone-wise risk breakdown."
    ),
)
async def get_analytics_summary() -> AnalyticsSummary:
    return await analytics_service.get_analytics_summary()

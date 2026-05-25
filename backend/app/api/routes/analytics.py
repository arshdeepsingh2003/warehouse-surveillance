from fastapi import APIRouter

from app.schemas.analytics import AnalyticsSummary
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def summary():
    return AnalyticsService.summary()

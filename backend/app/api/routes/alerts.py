"""
api/routes/alerts.py
──────────────────────
REST API routes for Alert management.

Endpoints:
  GET   /alerts               – list alerts (filterable)
  GET   /alerts/live          – only active/unresolved alerts
  GET   /alerts/{id}          – alert detail + snapshot
  PATCH /alerts/{id}/resolve  – mark an alert as resolved
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.alert import AlertOut, AlertResolve
from app.services import alert_service

router = APIRouter(
    prefix="/alerts",
    tags=["Alerts"],
)


@router.get(
    "/",
    response_model=list[AlertOut],
    summary="List alerts",
    description=(
        "Returns alerts in reverse-chronological order. "
        "Filter by status, severity, zone, or camera_id."
    ),
)
async def list_alerts(
    status:    Optional[str] = Query(None, description="active | resolved"),
    severity:  Optional[str] = Query(None, description="low | medium | high"),
    zone:      Optional[str] = Query(None, description="e.g. restricted_area"),
    camera_id: Optional[str] = Query(None, description="e.g. cam-01"),
    limit:     int           = Query(50,   ge=1, le=500, description="Max results"),
) -> list[AlertOut]:
    return await alert_service.get_all_alerts(
        status=status, severity=severity, zone=zone,
        camera_id=camera_id, limit=limit,
    )


@router.get(
    "/live",
    response_model=list[AlertOut],
    summary="Live (active) alerts only",
    description="Shortcut for GET /alerts?status=active. Used by the Alert Panel live tab.",
)
async def get_live_alerts() -> list[AlertOut]:
    return await alert_service.get_all_alerts(status="active")


@router.get(
    "/{alert_id}",
    response_model=AlertOut,
    summary="Get alert detail",
)
async def get_alert(alert_id: str) -> AlertOut:
    alert = await alert_service.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )
    return alert


@router.patch(
    "/{alert_id}/resolve",
    response_model=AlertOut,
    summary="Resolve an alert",
    description="Mark an alert as resolved. Requires the operator's email.",
)
async def resolve_alert(alert_id: str, body: AlertResolve) -> AlertOut:
    updated = await alert_service.resolve_alert(
        alert_id=alert_id, resolved_by=body.resolved_by
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )
    return updated
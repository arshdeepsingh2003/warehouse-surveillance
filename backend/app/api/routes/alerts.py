from fastapi import APIRouter, HTTPException

from app.schemas.alert import AlertOut, AlertResolve
from app.services.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/", response_model=list[AlertOut])
async def list_alerts():
    return AlertService.list()


@router.get("/{alert_id}", response_model=AlertOut)
async def get_alert(alert_id: str):
    alert = AlertService.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.patch("/{alert_id}/acknowledge", response_model=AlertOut)
async def acknowledge_alert(alert_id: str):
    alert = AlertService.acknowledge(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

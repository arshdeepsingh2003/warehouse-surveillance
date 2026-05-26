"""
api/routes/cameras.py
──────────────────────
REST API routes for Camera management.

Endpoints:
  GET  /cameras           – list all cameras (with optional filters)
  GET  /cameras/{id}      – get one camera by ID
  POST /cameras           – add a new camera
  POST /cameras/{id}/start – (stub) start streaming
  POST /cameras/{id}/stop  – (stub) stop streaming

All endpoint functions are async — they can await database queries,
HTTP calls, etc. without blocking the server.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.camera import CameraOut, CameraCreate
from app.services import camera_service

# APIRouter groups related endpoints. We'll include this in main.py.
router = APIRouter(
    prefix="/cameras",
    tags=["Cameras"],                    # Groups routes in /docs Swagger UI
)


@router.get(
    "/",
    response_model=list[CameraOut],
    summary="List all cameras",
    description="Returns all registered cameras. Optionally filter by status or zone.",
)
async def list_cameras(
    status: Optional[str] = Query(None, description="Filter: online | offline | unknown"),
    zone:   Optional[str] = Query(None, description="Filter by zone name, e.g. restricted_area"),
) -> list[CameraOut]:
    return await camera_service.get_all_cameras(status=status, zone=zone)


@router.get(
    "/{camera_id}",
    response_model=CameraOut,
    summary="Get camera by ID",
)
async def get_camera(camera_id: str) -> CameraOut:
    camera = await camera_service.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found.",
        )
    return camera


@router.post(
    "/",
    response_model=CameraOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new camera",
    description="Register a new CCTV camera or video stream source.",
)
async def create_camera(data: CameraCreate) -> CameraOut:
    return await camera_service.create_camera(data)


@router.post(
    "/{camera_id}/start",
    summary="Start camera stream",
    description="Signal the AI pipeline to start processing this camera's stream.",
)
async def start_camera(camera_id: str) -> dict:
    camera = await camera_service.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    # TODO: send start command to AI pipeline worker
    return {
        "message":   f"Stream start requested for camera '{camera_id}'.",
        "camera_id": camera_id,
        "status":    "starting",
    }


@router.post(
    "/{camera_id}/stop",
    summary="Stop camera stream",
)
async def stop_camera(camera_id: str) -> dict:
    camera = await camera_service.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    # TODO: send stop command to AI pipeline worker
    return {
        "message":   f"Stream stop requested for camera '{camera_id}'.",
        "camera_id": camera_id,
        "status":    "stopping",
    }
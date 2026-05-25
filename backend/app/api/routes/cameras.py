from fastapi import APIRouter, HTTPException

from app.schemas.camera import CameraCreate, CameraOut
from app.services.camera_service import CameraService

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("/", response_model=list[CameraOut])
async def list_cameras():
    return CameraService.list()


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: str):
    camera = CameraService.get(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.post("/", response_model=CameraOut, status_code=201)
async def create_camera(body: CameraCreate):
    return CameraService.create(body)

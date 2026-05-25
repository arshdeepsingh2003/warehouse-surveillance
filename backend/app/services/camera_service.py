from app.services.mock_data import cameras


from datetime import datetime

from app.schemas.camera import CameraCreate


class CameraService:
    @staticmethod
    def list() -> list[dict]:
        return cameras

    @staticmethod
    def get(camera_id: str) -> dict | None:
        return next((c for c in cameras if c["id"] == camera_id), None)

    @staticmethod
    def create(body: CameraCreate) -> dict:
        camera = {
            "id": f"cam-{len(cameras) + 1:03d}",
            **body.model_dump(),
            "status": "online",
            "created_at": datetime.utcnow(),
        }
        cameras.append(camera)
        return camera

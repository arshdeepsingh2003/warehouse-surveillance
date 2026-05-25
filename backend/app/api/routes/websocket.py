import asyncio
import random
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.connection_manager import manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str):
    await websocket.accept()
    await manager.connect(room, websocket)

    async def mock_broadcaster():
        types = ["alert", "activity", "camera_status"]
        while True:
            await asyncio.sleep(random.uniform(3, 8))
            event = {
                "type": random.choice(types),
                "data": {"timestamp": datetime.utcnow().isoformat(), "id": random.randint(100, 999)},
            }
            await manager.broadcast(room, event)

    task = asyncio.create_task(mock_broadcaster())

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        task.cancel()
        await manager.disconnect(room, websocket)

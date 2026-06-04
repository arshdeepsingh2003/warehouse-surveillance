"""
ws/connection_manager.py
────────────────────────
WebSocket Connection Manager

Manages all active WebSocket client connections.

Key responsibilities:
  1. Track who is connected (list of WebSocket objects).
  2. Broadcast messages to ALL connected clients (e.g. new alert events).
  3. Broadcast to a specific "room" (e.g. only subscribers to a specific camera).
  4. Cleanly remove disconnected clients.

This is intentionally simple (in-memory, single process).
For multi-worker production deployments, replace the `active_connections`
list with a Redis pub/sub channel so that any backend worker can broadcast
to any frontend client, regardless of which worker they're connected to.
"""

import json
import logging
from typing import Optional
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Central registry of all live WebSocket connections.

    Usage:
        manager = ConnectionManager()

        # In your WebSocket route:
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                ...
        except WebSocketDisconnect:
            manager.disconnect(websocket)

        # From anywhere (e.g. when an alert fires):
        await manager.broadcast({"type": "alert_triggered", ...})
    """

    def __init__(self) -> None:
        # Maps room_id → list of connected WebSockets
        # The special room "global" gets every broadcast.
        self._rooms: dict[str, list[WebSocket]] = {"global": []}

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def connect(self, websocket: WebSocket, room: str = "global") -> None:
        """
        Accept a new WebSocket connection and register it.

        Args:
            websocket: The FastAPI WebSocket object.
            room:      Optional room name (e.g. "cam-01") for targeted broadcasts.
                       Every client is also added to "global".
        """
        await websocket.accept()

        # Always join global room
        self._rooms["global"].append(websocket)

        # Also join the specified room if it's not global
        if room != "global":
            if room not in self._rooms:
                self._rooms[room] = []
            self._rooms[room].append(websocket)

        logger.info(f"Client connected to room '{room}'. Total in global: {len(self._rooms['global'])}")

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket from all rooms.
        Call this when the client disconnects or an error occurs.
        """
        for room_clients in self._rooms.values():
            if websocket in room_clients:
                room_clients.remove(websocket)
        logger.info(f"Client disconnected. Total in global: {len(self._rooms['global'])}")

    # ── Sending messages ──────────────────────────────────────────────────────

    async def broadcast(self, message: dict, room: str = "global") -> None:
        """
        Send a JSON message to all clients in a room.

        Args:
            message: Python dict — will be JSON-serialised automatically.
            room:    Target room. Defaults to "global" (all connected clients).
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)
        if message.get("type") == "frame_update":
            _log.info(
                f"🔍 TRACE[ws-broadcast] type=frame_update "
                f"camera={message.get('camera_id', '?')} "
                f"persons={len(message.get('persons', []))} "
                f"room={room} clients={len(self._rooms.get(room, []))}"
            )

        payload = json.dumps(message, default=str)   # default=str handles datetime objects
        targets = self._rooms.get(room, [])
        disconnected = []

        for websocket in targets:
            try:
                await websocket.send_text(payload)
            except Exception:
                # Client dropped unexpectedly — mark for cleanup
                disconnected.append(websocket)

        # Clean up dead connections
        for ws in disconnected:
            self.disconnect(ws)

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        """Send a message to ONE specific client."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception:
            self.disconnect(websocket)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def connection_count(self, room: str = "global") -> int:
        """How many clients are in a given room."""
        return len(self._rooms.get(room, []))


# Single shared instance used across the whole application
manager = ConnectionManager()

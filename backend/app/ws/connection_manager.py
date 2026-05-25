from collections import defaultdict


class ConnectionManager:
    def __init__(self):
        self.rooms: dict[str, list] = defaultdict(list)

    async def connect(self, room: str, websocket):
        self.rooms[room].append(websocket)

    async def disconnect(self, room: str, websocket):
        self.rooms[room].remove(websocket)
        if not self.rooms[room]:
            del self.rooms[room]

    async def broadcast(self, room: str, message: dict):
        for ws in self.rooms.get(room, []):
            try:
                await ws.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()

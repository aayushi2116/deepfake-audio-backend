from typing import Dict, List
from fastapi import WebSocket
import json

class SignalingManager:
    """Manages WebSockets for WebRTC signaling across rooms."""
    def __init__(self):
        # Maps room_id -> {client_id: WebSocket}
        self.rooms: Dict[str, Dict[str, WebSocket]] = {}

    async def connect(self, room_id: str, client_id: str, websocket: WebSocket):
        await websocket.accept()
        if room_id not in self.rooms:
            self.rooms[room_id] = {}
        self.rooms[room_id][client_id] = websocket

        # Notify other participants in the room
        await self.broadcast(room_id, {
            "type": "user-joined",
            "clientId": client_id,
            "sender": client_id,
            "peers": list(self.rooms[room_id].keys())
        }, exclude_client=client_id)

    def disconnect(self, room_id: str, client_id: str):
        if room_id in self.rooms and client_id in self.rooms[room_id]:
            del self.rooms[room_id][client_id]
            if not self.rooms[room_id]:
                del self.rooms[room_id]

    async def send_to_client(self, room_id: str, target_client_id: str, message: dict):
        if room_id in self.rooms and target_client_id in self.rooms[room_id]:
            websocket = self.rooms[room_id][target_client_id]
            await websocket.send_text(json.dumps(message))

    async def broadcast(self, room_id: str, message: dict, exclude_client: str = None):
        if room_id in self.rooms:
            for client_id, websocket in list(self.rooms[room_id].items()):
                if client_id != exclude_client:
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception:
                        pass

signaling_manager = SignalingManager()

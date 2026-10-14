"""
WebSocket Connection Manager — Broadcasts real-time events to connected clients

Manages open WebSocket connections, client lifecycle, and event broadcasting.
"""

import json
import logging
from typing import Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket client connections for live map updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register an incoming WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast_event(self, event: dict) -> None:
        """Broadcast an acoustic event JSON payload to all active clients."""
        if not self.active_connections:
            return

        payload_str = json.dumps(event)
        disconnected = []

        for conn in self.active_connections:
            try:
                await conn.send_text(payload_str)
            except Exception as e:
                logger.warning(f"Failed to send to client ({e}), queuing for removal.")
                disconnected.append(conn)

        for conn in disconnected:
            self.disconnect(conn)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


# Global singleton instance
ws_manager = ConnectionManager()

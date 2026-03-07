"""SSE broadcast manager — manages client connections and event dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

log = logging.getLogger(__name__)

# Heartbeat interval in seconds
HEARTBEAT_INTERVAL = 30


class SSEManager:
    """Manages SSE client connections, broadcasting, and heartbeat."""

    def __init__(self) -> None:
        self._clients: dict[str, asyncio.Queue[dict[str, str] | None]] = {}
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def connection_count(self) -> int:
        return len(self._clients)

    def connect(self) -> str:
        """Register a new client. Returns client_id."""
        client_id = uuid.uuid4().hex[:12]
        self._clients[client_id] = asyncio.Queue()
        log.info("SSE client connected: %s (total: %d)", client_id, len(self._clients))
        return client_id

    def disconnect(self, client_id: str) -> None:
        """Remove a client connection."""
        q = self._clients.pop(client_id, None)
        if q is not None:
            # Signal the subscriber generator to stop
            q.put_nowait(None)
            log.info("SSE client disconnected: %s (total: %d)", client_id, len(self._clients))

    async def subscribe(self, client_id: str) -> AsyncGenerator[dict[str, str]]:
        """Async generator that yields SSE events for a client.

        Yields dicts compatible with sse-starlette: {"event": ..., "data": ...}.
        Terminates when client is disconnected (receives None sentinel).
        """
        q = self._clients.get(client_id)
        if q is None:
            return

        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            # Ensure cleanup even if generator is abandoned
            self._clients.pop(client_id, None)

    def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Send an event to ALL connected clients."""
        if not self._clients:
            return

        msg = {"event": event_type, "data": json.dumps(data)}
        dead: list[str] = []
        for client_id, q in self._clients.items():
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                log.warning("SSE queue full for client %s — dropping", client_id)
                dead.append(client_id)

        for cid in dead:
            self.disconnect(cid)

    def send_to(self, client_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Send an event to a single client."""
        q = self._clients.get(client_id)
        if q is None:
            return
        msg = {"event": event_type, "data": json.dumps(data)}
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            log.warning("SSE queue full for client %s — dropping", client_id)
            self.disconnect(client_id)

    async def _heartbeat_loop(self) -> None:
        """Send ping events to all clients periodically to keep connections alive."""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if self._clients:
                    self.broadcast("ping", {"ts": time.time()})
        except asyncio.CancelledError:
            pass

    async def start(self) -> None:
        """Start the heartbeat background task."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            log.info("SSE heartbeat started (interval=%ds)", HEARTBEAT_INTERVAL)

    async def shutdown(self) -> None:
        """Stop heartbeat and disconnect all clients."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Disconnect all clients
        for client_id in list(self._clients):
            self.disconnect(client_id)
        log.info("SSE manager shut down")

"""Session monitor — watches live_signals.jsonl for new signals."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ui_v2.sse_manager import SSEManager

log = logging.getLogger(__name__)

# Polling interval in seconds (Windows-safe, no inotify)
POLL_INTERVAL = 0.5

# Signal types that map to SSE events
_EXIT_TYPES = {"SIGNAL_EXIT", "ORDER_EXIT"}
_ENTRY_TYPES = {"SIGNAL_ENTRY"}
_MANUAL_TYPES = {"MANUAL_ENTRY", "MANUAL_EXIT"}


class SessionMonitor:
    """Watches live_signals.jsonl for new lines and broadcasts via SSE."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._file_offset: int = 0

    def start(self, sse_manager: SSEManager, signals_path: Path) -> None:
        """Begin watching the signals file. Idempotent."""
        if self._task is not None:
            return
        self._running = True
        # Start at end of file so we only see NEW lines
        if signals_path.exists():
            self._file_offset = signals_path.stat().st_size
        else:
            self._file_offset = 0
        self._task = asyncio.create_task(self._poll_loop(sse_manager, signals_path))
        log.info("SessionMonitor started watching %s", signals_path)

    def stop(self) -> None:
        """Stop watching. Idempotent."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
            log.info("SessionMonitor stopped")

    async def _poll_loop(self, sse_manager: SSEManager, signals_path: Path) -> None:
        """Poll the signals file for new lines."""
        try:
            while self._running:
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    if not signals_path.exists():
                        continue

                    stat = signals_path.stat()

                    # File was truncated or replaced — reset offset
                    if stat.st_size < self._file_offset:
                        self._file_offset = 0

                    # Use file size as primary change detector (mtime too coarse on Windows)
                    if stat.st_size <= self._file_offset:
                        continue

                    # Read only the new bytes
                    with open(signals_path, "r", encoding="utf-8") as fh:
                        fh.seek(self._file_offset)
                        new_data = fh.read()
                        self._file_offset = fh.tell()

                    # Process each new line
                    for line in new_data.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        self._process_line(sse_manager, line)

                except OSError as exc:
                    log.warning("SessionMonitor file read error: %s", exc)
                except Exception as exc:
                    log.error("SessionMonitor unexpected error: %s", exc, exc_info=True)

        except asyncio.CancelledError:
            pass

    def _process_line(self, sse_manager: SSEManager, line: str) -> None:
        """Parse a JSON line and broadcast the appropriate SSE event."""
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            log.warning("SessionMonitor: invalid JSON line: %s", line[:120])
            return

        signal_type = record.get("type", "")

        # Broadcast as a signal event
        sse_manager.broadcast("signal", record)
        log.info("SessionMonitor broadcast signal: %s", signal_type)

        # Exit signals also trigger a debrief_required event
        if signal_type in _EXIT_TYPES:
            sse_manager.broadcast(
                "debrief_required",
                {
                    "strategy_id": record.get("strategy_id"),
                    "exit_ts": record.get("ts"),
                    "signal_type": signal_type,
                },
            )
            log.info("SessionMonitor broadcast debrief_required for %s", record.get("strategy_id"))

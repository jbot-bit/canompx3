"""Background state ticker — broadcasts state changes and clock ticks via SSE."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from ui_v2.sse_manager import SSEManager
from ui_v2.state_machine import (
    StateName,
    get_app_state,
    get_et_time,
    get_refresh_seconds,
    resolve_global_state,
)

log = logging.getLogger(__name__)

BRISBANE = ZoneInfo("Australia/Brisbane")

# Interval bounds
_FAST_INTERVAL = 1  # seconds — during ALERT+
_SLOW_INTERVAL = 5  # seconds — during IDLE/OVERNIGHT


def _serialize_date(val) -> str | None:
    if val is None:
        return None
    return val.isoformat()


class StateBroadcaster:
    """Periodically computes app state and broadcasts changes via SSE."""

    def __init__(self, sse_manager: SSEManager) -> None:
        self._sse = sse_manager
        self._task: asyncio.Task | None = None
        self._prev_state_name: str | None = None

    def start(self) -> None:
        """Start the broadcast loop. Idempotent."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        log.info("StateBroadcaster started")

    def stop(self) -> None:
        """Stop the broadcast loop. Idempotent."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
            log.info("StateBroadcaster stopped")

    async def _loop(self) -> None:
        try:
            while True:
                now = datetime.now(BRISBANE)
                state = get_app_state(now)
                global_state = resolve_global_state(state)
                current_name = global_state.value

                # Determine adaptive interval
                minutes_to_next = state.minutes_to_next or 999
                is_wknd = state.name == StateName.WEEKEND
                refresh_interval = get_refresh_seconds(minutes_to_next, is_weekend=is_wknd)
                interval = _FAST_INTERVAL if refresh_interval <= 5 else _SLOW_INTERVAL

                # Always broadcast clock tick
                clock_data = {
                    "bris_time": now.strftime("%I:%M:%S %p BRIS"),
                    "et_time": get_et_time(now),
                    "state": current_name,
                    "next_session": state.next_session,
                    "minutes_to_next": round(minutes_to_next, 1) if minutes_to_next < 999 else None,
                }
                self._sse.broadcast("clock_tick", clock_data)

                # Broadcast state_change only when state actually changes
                if current_name != self._prev_state_name:
                    state_data = {
                        "previous": self._prev_state_name,
                        "current": current_name,
                        "next_session": state.next_session,
                        "next_session_dt": _serialize_date(state.next_session_dt),
                        "minutes_to_next": round(minutes_to_next, 1) if minutes_to_next < 999 else None,
                        "trading_day": _serialize_date(state.trading_day),
                    }
                    self._sse.broadcast("state_change", state_data)
                    log.info("State changed: %s -> %s", self._prev_state_name, current_name)
                    self._prev_state_name = current_name

                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            pass

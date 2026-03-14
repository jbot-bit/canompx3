"""
Aggregates real-time ticks into 1-minute OHLCV bars.

Bar.ts_utc = start of minute in UTC.
Bar.as_dict() produces the exact format ExecutionEngine.on_bar() expects
(key 'ts_utc', not 'ts_event').
"""

import logging
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class Bar:
    ts_utc: datetime  # UTC, truncated to minute start
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str = ""

    def as_dict(self) -> dict:
        """Return dict for ExecutionEngine.on_bar() — uses key 'ts_utc'."""
        return {
            "ts_utc": self.ts_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BarAggregator:
    def __init__(self):
        self._current: Bar | None = None
        self._bar_minute: datetime | None = None

    def on_tick(self, price: float, volume: int, ts: datetime) -> Bar | None:
        """Process one tick. Returns completed Bar when minute boundary crossed, else None."""
        tick_minute = ts.replace(second=0, microsecond=0)

        if self._current is None:
            self._open_bar(price, volume, tick_minute)
            return None

        if tick_minute < self._bar_minute:
            log.warning("Dropped out-of-order tick: %s < current bar %s", tick_minute, self._bar_minute)
            return None

        if tick_minute == self._bar_minute:
            self._current.high = max(self._current.high, price)
            self._current.low = min(self._current.low, price)
            self._current.close = price
            self._current.volume += volume
            return None

        completed = self._current
        self._open_bar(price, volume, tick_minute)
        return completed

    def _open_bar(self, price: float, volume: int, minute: datetime) -> None:
        self._bar_minute = minute
        self._current = Bar(
            ts_utc=minute,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
        )

    def flush(self) -> Bar | None:
        """Force-close current in-progress bar (call at session end)."""
        bar = self._current
        self._current = None
        self._bar_minute = None
        return bar

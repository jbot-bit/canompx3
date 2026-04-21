"""
Aggregates real-time ticks into 1-minute OHLCV bars.

Bar.ts_utc = start of minute in UTC.
Bar.as_dict() produces the exact format ExecutionEngine.on_bar() expects
(key 'ts_utc', not 'ts_event').
"""

import logging
import math
import threading
from collections.abc import Callable
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

    def is_valid(self) -> bool:
        """Validate bar integrity. Returns False if any field is corrupt."""
        for field in (self.open, self.high, self.low, self.close):
            if not isinstance(field, int | float):
                return False
            if math.isnan(field) or math.isinf(field):
                return False
            if field <= 0:
                return False
        if self.high < self.low:
            return False
        return True


# Spike filter: reject ticks > 5x the last known price
_SPIKE_MULTIPLIER = 5.0
# Alert after N consecutive bad bars
_BAD_BAR_ALERT_THRESHOLD = 3


class BarAggregator:
    def __init__(self, on_bad_bar_alert: Callable[[str], None] | None = None):
        self._current: Bar | None = None
        self._bar_minute: datetime | None = None
        self._lock = threading.Lock()  # R2-C1: protects OHLCV from cross-thread corruption
        self._last_good_price: float | None = None
        self._consecutive_bad_bars = 0
        self._bad_tick_count = 0
        self._on_bad_bar_alert = on_bad_bar_alert

    def on_tick(self, price: float, volume: int, ts: datetime) -> Bar | None:
        """Process one tick. Returns completed Bar when minute boundary crossed, else None.

        Thread-safe: signalrcore callbacks fire on a foreign thread. The lock ensures
        concurrent ticks cannot corrupt high/low/volume via read-modify-write races.
        """
        # Tick validation — reject before any processing
        if not self._validate_tick(price, volume):
            return None

        self._last_good_price = price
        tick_minute = ts.replace(second=0, microsecond=0)

        with self._lock:
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

            # Validate completed bar before returning
            return self._validate_bar(completed)

    def _validate_tick(self, price: float, volume: int) -> bool:
        """Reject bad ticks. Returns True if tick is valid."""
        if not isinstance(price, int | float):
            self._bad_tick_count += 1
            log.warning("BAD TICK: price is not numeric: %r (total bad: %d)", price, self._bad_tick_count)
            return False

        if math.isnan(price) or math.isinf(price):
            self._bad_tick_count += 1
            log.warning("BAD TICK: price is NaN/inf: %r (total bad: %d)", price, self._bad_tick_count)
            return False

        if price <= 0:
            self._bad_tick_count += 1
            log.warning("BAD TICK: price <= 0: %s (total bad: %d)", price, self._bad_tick_count)
            return False

        if volume < 0:
            self._bad_tick_count += 1
            log.warning("BAD TICK: negative volume: %d (total bad: %d)", volume, self._bad_tick_count)
            return False

        # Spike filter: 5x last known price is clearly corrupt data
        if self._last_good_price is not None:
            ratio = price / self._last_good_price
            if ratio > _SPIKE_MULTIPLIER or ratio < (1.0 / _SPIKE_MULTIPLIER):
                self._bad_tick_count += 1
                log.warning(
                    "BAD TICK: price spike %s vs last good %s (ratio %.2f, total bad: %d)",
                    price,
                    self._last_good_price,
                    ratio,
                    self._bad_tick_count,
                )
                return False

        return True

    def _validate_bar(self, bar: Bar) -> Bar | None:
        """Validate completed bar. Returns bar if valid, None if corrupt."""
        if bar.is_valid():
            self._consecutive_bad_bars = 0
            return bar

        self._consecutive_bad_bars += 1
        log.critical(
            "BAD BAR REJECTED: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%d (consecutive: %d)",
            bar.ts_utc,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            self._consecutive_bad_bars,
        )

        if self._consecutive_bad_bars >= _BAD_BAR_ALERT_THRESHOLD and self._on_bad_bar_alert:
            self._on_bad_bar_alert(f"Data quality warning: {self._consecutive_bad_bars} consecutive bad bars rejected")

        return None

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
        with self._lock:
            bar = self._current
            self._current = None
            self._bar_minute = None
            if bar is not None:
                return self._validate_bar(bar)
            return None

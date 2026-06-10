"""Accumulate live bid/ask quotes into per-minute spread summaries.

Diagnostic-only sibling of BarAggregator. Captures the quoted spread
(ask - bid, in raw PRICE POINTS) that the ProjectX feed receives for free
on every GatewayQuote but otherwise discards, so a later validation query
can compare measured spread against the modelled ``COST_SPECS[I].spread_doubled``
(see docs/audit/2026-06-10-data-pipeline-gap-report.md Defect B).

This NEVER touches bars_1m, the Bar contract, or the order path. It is gated
OFF by default at the DataFeed boundary; when on, it runs in its own
try/except branch after the bar/order path.

QuoteMinute.ts_utc = start of minute in UTC (mirrors Bar.ts_utc).
Units are raw price points — validation converts to dollars at query time via
the canonical point_value, so no unit is ever baked into the table.
"""

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class QuoteMinute:
    ts_utc: datetime  # UTC, truncated to minute start
    avg_spread: float  # mean (ask - bid) over the minute, price points
    close_spread: float  # spread at the last valid tick of the minute
    min_spread: float
    max_spread: float
    n_quotes: int  # valid (non-crossed, both-sided) quotes counted this minute
    symbol: str = ""

    def is_valid(self) -> bool:
        """True if the minute holds at least one valid quote and finite spreads.

        Mirrors Bar.is_valid() (used at bar_persister.py before INSERT). A
        zero-quote minute or a non-finite spread is dropped at persist time.
        """
        if self.n_quotes <= 0:
            return False
        for field in (self.avg_spread, self.close_spread, self.min_spread, self.max_spread):
            if not isinstance(field, int | float):
                return False
            if math.isnan(field) or math.isinf(field):
                return False
            if field < 0:
                return False
        return True


class SpreadAccumulator:
    """Roll bid/ask ticks into per-minute QuoteMinute summaries.

    Thread-safe: signalrcore fires quote callbacks on a foreign thread (same
    reality BarAggregator guards against). The lock protects the in-progress
    minute's running stats from cross-thread read-modify-write races.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._minute: datetime | None = None
        self._n: int = 0
        self._sum: float = 0.0
        self._min: float = math.inf
        self._max: float = -math.inf
        self._close: float = 0.0
        # Diagnostics — dropped-tick visibility without per-tick log spam.
        self._dropped: int = 0

    def add(self, bid: float | None, ask: float | None, ts: datetime) -> QuoteMinute | None:
        """Feed one quote. Returns a completed QuoteMinute when the minute
        boundary is crossed, else None.

        Crossed/locked/one-sided guard: bid or ask missing, or ask <= bid,
        drops the tick — it is NOT counted in n_quotes. Coercing such a quote
        to a zero or negative spread would understate measured friction (the
        dangerous direction for validating the cost model), so we drop instead.
        """
        if bid is None or ask is None:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 1000 == 0:
                log.debug("SpreadAccumulator: dropped one-sided quote (total dropped: %d)", self._dropped)
            return None
        spread = float(ask) - float(bid)
        if spread <= 0:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 1000 == 0:
                log.debug(
                    "SpreadAccumulator: dropped crossed/locked quote bid=%s ask=%s (total dropped: %d)",
                    bid,
                    ask,
                    self._dropped,
                )
            return None

        tick_minute = ts.replace(second=0, microsecond=0)

        with self._lock:
            if self._minute is None:
                self._open_minute(spread, tick_minute)
                return None

            if tick_minute < self._minute:
                # Out-of-order tick — same handling as BarAggregator: drop it.
                log.debug("SpreadAccumulator: dropped out-of-order quote %s < %s", tick_minute, self._minute)
                return None

            if tick_minute == self._minute:
                self._accumulate(spread)
                return None

            completed = self._snapshot()
            self._open_minute(spread, tick_minute)
            return completed

    def flush(self, symbol: str = "") -> QuoteMinute | None:
        """Force-close the in-progress minute (call at session end).

        Without this the final partial minute of every session is lost — the
        same reason DataFeed.flush() force-closes the BarAggregator's last bar.
        """
        with self._lock:
            if self._minute is None:
                return None
            completed = self._snapshot()
            self._reset()
            if completed is not None:
                completed.symbol = symbol
            return completed

    # ------------------------------------------------------------------
    # Internal — every caller holds self._lock (add() opens the first minute
    # inside its `with self._lock` block; flush() holds it too).
    # ------------------------------------------------------------------

    def _open_minute(self, spread: float, minute: datetime) -> None:
        self._minute = minute
        self._n = 1
        self._sum = spread
        self._min = spread
        self._max = spread
        self._close = spread

    def _accumulate(self, spread: float) -> None:
        self._n += 1
        self._sum += spread
        self._min = min(self._min, spread)
        self._max = max(self._max, spread)
        self._close = spread

    def _snapshot(self) -> QuoteMinute | None:
        if self._minute is None or self._n <= 0:
            return None
        return QuoteMinute(
            ts_utc=self._minute,
            avg_spread=self._sum / self._n,
            close_spread=self._close,
            min_spread=self._min,
            max_spread=self._max,
            n_quotes=self._n,
        )

    def _reset(self) -> None:
        self._minute = None
        self._n = 0
        self._sum = 0.0
        self._min = math.inf
        self._max = -math.inf
        self._close = 0.0

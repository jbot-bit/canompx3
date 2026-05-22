"""Persist live 1-minute bars to bars_1m table.

Captures broker-feed bars during live trading and batch-writes them
after session end. Eliminates Databento dependency for daily bar data.

Architecture:
  DataFeed -> BarAggregator -> Bar -> SessionOrchestrator._on_bar()
                                       |-> ExecutionEngine (trading)
                                       |-> BarPersister.append(bar)
  Session end -> BarPersister.flush_to_db()

Safety:
  - Bars collected in memory during session (list append, thread-safe)
  - Batch write at session end only (no concurrent DuckDB writes)
  - Idempotent: removes existing bars for time range, then writes new
  - Fail-open: persister failure does NOT block trading
"""

import logging
import threading

import duckdb

from pipeline.db_config import configure_connection
from trading_app.live import bar_ring
from trading_app.live.bar_aggregator import Bar

log = logging.getLogger(__name__)


class BarPersister:
    """Collect live bars and batch-persist to bars_1m at session end."""

    def __init__(self, symbol: str, db_path: str | None = None, session_id: str | None = None):
        self.symbol = symbol
        self._db_path = db_path
        self._session_id = session_id
        self._bars: list[Bar] = []
        self._lock = threading.Lock()

    def append(self, bar: Bar, enqueue_to_ring: bool = True) -> None:
        """Append a completed bar. Thread-safe. Called from _on_bar.

        Also enqueues the bar to the live-bars ring file for the dashboard
        chart (atomic-file IPC; see trading_app.live.bar_ring). Trading hot
        path adds one put_nowait; disk write happens off-path. Failures in
        the ring writer are fail-open per institutional-rigor § 6.
        """
        with self._lock:
            self._bars.append(bar)
        if enqueue_to_ring:
            try:
                bar_ring.enqueue_bar(self.symbol, bar, session_id=self._session_id)
            except TypeError:
                # Mock-contamination guard — re-raise: tests must see this.
                raise
            except Exception:
                log.warning(
                    "BarPersister(%s): ring enqueue failed (trading unaffected)",
                    self.symbol,
                    exc_info=True,
                )

    def clear_ring(self) -> None:
        """Stop the background ring writer and delete the ring file.

        Called from session shutdown after flush_to_db() succeeds so the
        dashboard sees the gold.db historical view rather than a stale ring.
        """
        bar_ring.clear_ring(self.symbol)

    @property
    def bar_count(self) -> int:
        return len(self._bars)

    def flush_to_db(self) -> int:
        """Batch-insert collected bars into bars_1m. Returns count inserted.

        Uses DELETE+INSERT (idempotent) for the captured time range.
        Call AFTER session ends and all trading is complete.
        """
        with self._lock:
            bars = list(self._bars)

        if not bars:
            log.info("BarPersister: no bars to flush")
            return 0

        db_path = self._db_path
        if db_path is None:
            from pipeline.paths import GOLD_DB_PATH

            db_path = str(GOLD_DB_PATH)

        ts_min = min(b.ts_utc for b in bars)
        ts_max = max(b.ts_utc for b in bars)

        try:
            con = duckdb.connect(db_path)
            configure_connection(con, writing=True)
            con.execute(
                "DELETE FROM bars_1m WHERE symbol = ? AND ts_utc >= ? AND ts_utc <= ?",
                [self.symbol, ts_min, ts_max],
            )
            rows = [
                (b.ts_utc, self.symbol, self.symbol, b.open, b.high, b.low, b.close, b.volume)
                for b in bars
                if b.is_valid()
            ]
            con.executemany(
                "INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.close()
            log.info(
                "BarPersister: flushed %d bars for %s (%s to %s)",
                len(rows),
                self.symbol,
                ts_min,
                ts_max,
            )
            return len(rows)
        except (duckdb.Error, OSError) as e:
            log.error("BarPersister: flush failed (trading unaffected): %s", e)
            return 0

    def clear(self) -> None:
        """Clear collected bars without persisting."""
        with self._lock:
            self._bars.clear()

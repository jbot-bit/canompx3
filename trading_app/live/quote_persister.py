"""Persist live per-minute spread summaries to the live_quotes table.

Diagnostic sibling of BarPersister. Collects QuoteMinute summaries during a
live session and batch-writes them at session end so a later validation query
can compare measured quoted spread against the modelled cost-model spread
(docs/audit/2026-06-10-data-pipeline-gap-report.md Defect B).

Architecture (mirrors BarPersister, separate table):
  DataFeed -> SpreadAccumulator -> QuoteMinute -> SessionOrchestrator._on_quote_minute()
                                                   |-> QuotePersister.append(qm)
  Session end -> QuotePersister.flush_to_db()   (AFTER the bar flush)

Safety:
  - QuoteMinutes collected in memory during session (list append, thread-safe)
  - Batch write at session end only (no concurrent DuckDB writes)
  - Idempotent: removes existing rows for the captured time range, then writes
  - Fail-open: persister failure does NOT block trading (diagnostic data only)
  - Writes ONLY to live_quotes — never bars_1m
"""

import logging
import threading

import duckdb

from pipeline.db_config import configure_connection
from pipeline.db_connect import open_writer_with_retry
from trading_app.live.spread_accumulator import QuoteMinute

log = logging.getLogger(__name__)


class QuotePersister:
    """Collect live QuoteMinutes and batch-persist to live_quotes at session end."""

    def __init__(self, symbol: str, db_path: str | None = None, session_id: str | None = None):
        self.symbol = symbol
        self._db_path = db_path
        self._session_id = session_id
        self._quotes: list[QuoteMinute] = []
        self._lock = threading.Lock()

    def append(self, qm: QuoteMinute) -> None:
        """Append a completed QuoteMinute. Thread-safe. Called from _on_quote_minute.

        Synchronous and lock-guarded so it is safe to invoke from BOTH the
        pysignalr async quote path and the signalrcore foreign-thread path.
        """
        with self._lock:
            self._quotes.append(qm)

    @property
    def quote_count(self) -> int:
        return len(self._quotes)

    def flush_to_db(self) -> int:
        """Batch-insert collected QuoteMinutes into live_quotes. Returns count inserted.

        Uses DELETE+INSERT (idempotent) for the captured time range.
        Call AFTER session ends, AFTER the bar flush (bars are the capital
        artifact and flush first; quotes are diagnostic).
        """
        with self._lock:
            quotes = [q for q in self._quotes if q.is_valid()]

        if not quotes:
            log.info("QuotePersister: no valid quote-minutes to flush")
            return 0

        db_path = self._db_path
        if db_path is None:
            from pipeline.paths import GOLD_DB_PATH

            db_path = str(GOLD_DB_PATH)

        ts_min = min(q.ts_utc for q in quotes)
        ts_max = max(q.ts_utc for q in quotes)

        try:
            con = open_writer_with_retry(db_path)
            configure_connection(con, writing=True)
            con.execute(
                "DELETE FROM live_quotes WHERE symbol = ? AND ts_utc >= ? AND ts_utc <= ?",
                [self.symbol, ts_min, ts_max],
            )
            rows = [
                (
                    q.ts_utc,
                    self.symbol,
                    self.symbol,
                    q.avg_spread,
                    q.close_spread,
                    q.min_spread,
                    q.max_spread,
                    q.n_quotes,
                )
                for q in quotes
            ]
            con.executemany(
                "INSERT INTO live_quotes "
                "(ts_utc, symbol, source_symbol, avg_spread, close_spread, min_spread, max_spread, n_quotes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.close()
            log.info(
                "QuotePersister: flushed %d quote-minutes for %s (%s to %s)",
                len(rows),
                self.symbol,
                ts_min,
                ts_max,
            )
            return len(rows)
        except (duckdb.Error, OSError) as e:
            # Fail-open: a failed diagnostic flush must never abort shutdown.
            # Log at CRITICAL with the exception class/detail so the loss is
            # unmistakable in the shutdown trace. Return 0 — trading unaffected.
            log.critical(
                "QuotePersister: flush FAILED — %d quote-minutes NOT persisted for %s (trading unaffected; %s: %s)",
                len(quotes),
                self.symbol,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return 0

    def clear(self) -> None:
        """Clear collected quote-minutes without persisting."""
        with self._lock:
            self._quotes.clear()

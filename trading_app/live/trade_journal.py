"""Persistent trade journal for live trading sessions.

Writes every entry/exit to a DuckDB table so trade records survive process crashes.
Uses a SEPARATE database (live_journal.db) to avoid write contention with gold.db
pipeline rebuilds.

Design: fail-open — journal write failures log CRITICAL but NEVER block trading.
"""

import logging
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from pipeline.db_config import configure_connection

log = logging.getLogger(__name__)

LIVE_TRADES_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_trades (
    trade_id        TEXT PRIMARY KEY,
    trading_day     DATE NOT NULL,
    instrument      TEXT NOT NULL,
    strategy_id     TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_model     TEXT NOT NULL,
    engine_entry    DOUBLE,
    engine_exit     DOUBLE,
    fill_entry      DOUBLE,
    fill_exit       DOUBLE,
    actual_r        DOUBLE,
    expected_r      DOUBLE,
    slippage_pts    DOUBLE,
    pnl_dollars     DOUBLE,
    exit_reason     TEXT,
    cusum_alarm     BOOLEAN DEFAULT FALSE,
    broker          TEXT,
    order_id_entry  TEXT,
    order_id_exit   TEXT,
    contracts       INTEGER DEFAULT 1,
    session_mode    TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    exited_at       TIMESTAMPTZ
);
"""


def generate_trade_id() -> str:
    """Generate a unique trade ID (UUID4)."""
    return str(uuid.uuid4())


class TradeJournal:
    """Append-only trade journal backed by DuckDB.

    Fail-open: every public method catches exceptions and logs CRITICAL.
    A journal failure must NEVER prevent a trade from executing.
    """

    def __init__(self, db_path: Path | str, mode: str = "live"):
        self._mode = mode
        self._db_path = Path(db_path)
        try:
            self._con = duckdb.connect(str(self._db_path))
            configure_connection(self._con, writing=True)
            self._con.execute(LIVE_TRADES_SCHEMA)
            log.info("TradeJournal opened: %s (mode=%s)", self._db_path, mode)
        except Exception:
            log.critical("TradeJournal FAILED to open %s — trades will NOT be persisted", db_path, exc_info=True)
            self._con = None

    def record_entry(
        self,
        *,
        trade_id: str,
        trading_day: date,
        instrument: str,
        strategy_id: str,
        direction: str,
        entry_model: str,
        engine_entry: float,
        fill_entry: float | None = None,
        broker: str | None = None,
        order_id_entry: str | int | None = None,
        contracts: int = 1,
    ) -> None:
        """Insert a partial row when an entry order is submitted/filled."""
        if self._con is None:
            return
        try:
            self._con.execute(
                """
                INSERT INTO live_trades (
                    trade_id, trading_day, instrument, strategy_id, direction,
                    entry_model, engine_entry, fill_entry, broker,
                    order_id_entry, contracts, session_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    trade_id,
                    trading_day,
                    instrument,
                    strategy_id,
                    direction,
                    entry_model,
                    engine_entry,
                    fill_entry,
                    broker,
                    str(order_id_entry) if order_id_entry is not None else None,
                    contracts,
                    self._mode,
                ],
            )
        except Exception:
            log.critical("TradeJournal.record_entry FAILED for %s", trade_id, exc_info=True)

    def update_entry_fill(self, *, trade_id: str, fill_entry: float) -> None:
        """Update fill price when E2 stop-market order fills via poller."""
        if self._con is None:
            return
        try:
            self._con.execute(
                "UPDATE live_trades SET fill_entry = ? WHERE trade_id = ?",
                [fill_entry, trade_id],
            )
        except Exception:
            log.critical("TradeJournal.update_entry_fill FAILED for %s", trade_id, exc_info=True)

    def record_exit(
        self,
        *,
        trade_id: str,
        engine_exit: float | None = None,
        fill_exit: float | None = None,
        actual_r: float | None = None,
        expected_r: float | None = None,
        slippage_pts: float | None = None,
        pnl_dollars: float | None = None,
        exit_reason: str | None = None,
        order_id_exit: str | int | None = None,
        cusum_alarm: bool = False,
    ) -> None:
        """Update an existing row when a trade exits."""
        if self._con is None:
            return
        try:
            # Check entry exists before updating (detect orphaned exits)
            exists = self._con.execute("SELECT 1 FROM live_trades WHERE trade_id = ?", [trade_id]).fetchone()
            if exists is None:
                log.critical(
                    "TradeJournal.record_exit ORPHANED: trade_id=%s not found — "
                    "entry write may have failed, exit data lost",
                    trade_id,
                )
                return
            self._con.execute(
                """
                UPDATE live_trades SET
                    engine_exit = ?,
                    fill_exit = ?,
                    actual_r = ?,
                    expected_r = ?,
                    slippage_pts = ?,
                    pnl_dollars = ?,
                    exit_reason = ?,
                    order_id_exit = ?,
                    cusum_alarm = ?,
                    exited_at = ?
                WHERE trade_id = ?
                """,
                [
                    engine_exit,
                    fill_exit,
                    actual_r,
                    expected_r,
                    slippage_pts,
                    pnl_dollars,
                    exit_reason,
                    str(order_id_exit) if order_id_exit is not None else None,
                    cusum_alarm,
                    datetime.now(UTC),
                    trade_id,
                ],
            )
        except Exception:
            log.critical("TradeJournal.record_exit FAILED for %s", trade_id, exc_info=True)

    def daily_summary(self, trading_day: date) -> dict:
        """Query completed trades for a trading day. Returns summary dict."""
        if self._con is None:
            return {"error": "journal unavailable", "n_trades": 0}
        try:
            rows = self._con.execute(
                """
                SELECT strategy_id, direction, actual_r, slippage_pts, pnl_dollars,
                       exit_reason, cusum_alarm
                FROM live_trades
                WHERE trading_day = ? AND exited_at IS NOT NULL
                ORDER BY created_at
                """,
                [trading_day],
            ).fetchall()
            total_r = sum(r[2] or 0.0 for r in rows)
            total_slip = sum(r[3] or 0.0 for r in rows)
            total_pnl = sum(r[4] or 0.0 for r in rows)
            alarms = [r[0] for r in rows if r[6]]
            return {
                "date": trading_day.isoformat(),
                "n_trades": len(rows),
                "total_r": round(total_r, 4),
                "total_slippage_pts": round(total_slip, 4),
                "total_pnl_dollars": round(total_pnl, 2),
                "alarms": alarms,
                "by_strategy": {r[0]: round(r[2] or 0.0, 4) for r in rows},
            }
        except Exception:
            log.critical("TradeJournal.daily_summary FAILED", exc_info=True)
            return {"error": "query failed", "n_trades": 0}

    def incomplete_trades(self) -> list[dict]:
        """Return trades with entry but no exit (crash detection)."""
        if self._con is None:
            return []
        try:
            rows = self._con.execute(
                """
                SELECT trade_id, strategy_id, instrument, direction, engine_entry, fill_entry
                FROM live_trades
                WHERE exited_at IS NULL
                ORDER BY created_at
                """
            ).fetchall()
            return [
                {
                    "trade_id": r[0],
                    "strategy_id": r[1],
                    "instrument": r[2],
                    "direction": r[3],
                    "engine_entry": r[4],
                    "fill_entry": r[5],
                }
                for r in rows
            ]
        except Exception:
            log.critical("TradeJournal.incomplete_trades FAILED", exc_info=True)
            return []

    def close(self) -> None:
        """Close the DB connection."""
        if self._con is not None:
            try:
                self._con.close()
                log.info("TradeJournal closed: %s", self._db_path)
            except Exception:
                log.critical("TradeJournal.close FAILED", exc_info=True)
            self._con = None

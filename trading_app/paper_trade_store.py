"""Shared paper_trades write helpers.

Single write surface for:
- modeled backfill (`paper_trade_logger`)
- manual live logging (`log_trade`)
- automated live/shadow execution bridges

The key rule is simple:
- backfill rows must never overwrite real execution rows
- real execution rows may replace modeled rows for the same strategy/day
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH


@dataclass(frozen=True)
class PaperTradeRecord:
    trading_day: date
    orb_label: str
    strategy_id: str
    instrument: str
    orb_minutes: int
    rr_target: float
    filter_type: str
    entry_model: str
    direction: str | None = None
    entry_time: datetime | str | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    exit_price: float | None = None
    exit_time: datetime | str | None = None
    exit_reason: str | None = None
    pnl_r: float | None = None
    slippage_ticks: float = 0.0
    lane_name: str | None = None
    execution_source: str = "backfill"
    pnl_dollar: float | None = None
    notes: str = ""


_INSERT_SQL = """
    INSERT INTO paper_trades (
        trading_day, orb_label, entry_time, direction,
        entry_price, stop_price, target_price, exit_price,
        exit_time, exit_reason, pnl_r, slippage_ticks,
        strategy_id, lane_name, instrument,
        orb_minutes, rr_target, filter_type, entry_model,
        execution_source, pnl_dollar, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_CREATE_PAPER_TRADES_SQL = """
    CREATE TABLE IF NOT EXISTS paper_trades (
        trading_day DATE NOT NULL,
        orb_label TEXT NOT NULL,
        entry_time TIMESTAMPTZ,
        direction TEXT,
        entry_price DOUBLE,
        stop_price DOUBLE,
        target_price DOUBLE,
        exit_price DOUBLE,
        exit_time TIMESTAMPTZ,
        exit_reason TEXT,
        pnl_r DOUBLE,
        slippage_ticks DOUBLE DEFAULT 0,
        strategy_id TEXT NOT NULL,
        lane_name TEXT,
        instrument TEXT DEFAULT 'MNQ',
        orb_minutes INTEGER,
        rr_target DOUBLE,
        filter_type TEXT,
        entry_model TEXT,
        execution_source VARCHAR DEFAULT 'backfill',
        pnl_dollar DOUBLE,
        notes VARCHAR DEFAULT '',
        PRIMARY KEY (strategy_id, trading_day)
    )
"""


def _as_db_timestamp(value: datetime | str | None) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _record_values(record: PaperTradeRecord) -> list:
    return [
        record.trading_day,
        record.orb_label,
        _as_db_timestamp(record.entry_time),
        record.direction,
        record.entry_price,
        record.stop_price,
        record.target_price,
        record.exit_price,
        _as_db_timestamp(record.exit_time),
        record.exit_reason,
        record.pnl_r,
        record.slippage_ticks,
        record.strategy_id,
        record.lane_name,
        record.instrument,
        record.orb_minutes,
        record.rr_target,
        record.filter_type,
        record.entry_model,
        record.execution_source,
        record.pnl_dollar,
        record.notes,
    ]


def ensure_paper_trades_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure paper_trades has the full live-attribution schema."""
    con.execute(_CREATE_PAPER_TRADES_SQL)
    for col, typedef in [
        ("execution_source", "VARCHAR DEFAULT 'backfill'"),
        ("pnl_dollar", "DOUBLE"),
        ("notes", "VARCHAR DEFAULT ''"),
    ]:
        try:
            con.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {typedef}")
        except duckdb.CatalogException:
            pass


def delete_backfill_rows(
    con: duckdb.DuckDBPyConnection,
    *,
    strategy_id: str,
    since: date | None = None,
) -> None:
    """Delete only modeled rows, preserving any live/shadow/manual evidence."""
    ensure_paper_trades_schema(con)
    if since is not None:
        con.execute(
            """
            DELETE FROM paper_trades
            WHERE strategy_id = ?
              AND trading_day >= ?
              AND COALESCE(execution_source, 'backfill') = 'backfill'
            """,
            [strategy_id, since],
        )
        return

    con.execute(
        """
        DELETE FROM paper_trades
        WHERE strategy_id = ?
          AND COALESCE(execution_source, 'backfill') = 'backfill'
        """,
        [strategy_id],
    )


def upsert_backfill_trade(
    con: duckdb.DuckDBPyConnection,
    record: PaperTradeRecord,
) -> bool:
    """Insert/update a modeled row unless real execution evidence already owns the key."""
    ensure_paper_trades_schema(con)
    existing = con.execute(
        """
        SELECT COALESCE(execution_source, 'backfill')
        FROM paper_trades
        WHERE strategy_id = ? AND trading_day = ?
        """,
        [record.strategy_id, record.trading_day],
    ).fetchone()
    if existing is not None and existing[0] != "backfill":
        return False

    con.execute(
        "DELETE FROM paper_trades WHERE strategy_id = ? AND trading_day = ?",
        [record.strategy_id, record.trading_day],
    )
    con.execute(_INSERT_SQL, _record_values(record))
    return True


class PaperTradeCollisionError(RuntimeError):
    """Raised when a real-execution row already owns the (strategy_id, trading_day) key."""


def write_completed_trade(
    record: PaperTradeRecord,
    *,
    db_path: Path | str | None = None,
) -> None:
    """Write real execution evidence, replacing any modeled row for the same key.

    Fail-closed: if a non-backfill row already owns (strategy_id, trading_day),
    raise PaperTradeCollisionError rather than silently overwrite. The live
    journal remains the durable event record; attribution must not lose a
    completed trade through a silent DELETE.
    """
    target = Path(db_path) if db_path else GOLD_DB_PATH
    with duckdb.connect(str(target)) as con:
        configure_connection(con, writing=True)
        ensure_paper_trades_schema(con)
        existing = con.execute(
            """
            SELECT COALESCE(execution_source, 'backfill')
            FROM paper_trades
            WHERE strategy_id = ? AND trading_day = ?
            """,
            [record.strategy_id, record.trading_day],
        ).fetchone()
        if existing is not None and existing[0] != "backfill":
            raise PaperTradeCollisionError(
                f"paper_trades already has a {existing[0]} row for "
                f"strategy_id={record.strategy_id} trading_day={record.trading_day}; "
                f"refusing to overwrite. Resolve upstream (enforce one-trade-per-day "
                f"at the engine tier) or replay from live_journal.db."
            )
        con.execute(
            "DELETE FROM paper_trades WHERE strategy_id = ? AND trading_day = ?",
            [record.strategy_id, record.trading_day],
        )
        con.execute(_INSERT_SQL, _record_values(record))

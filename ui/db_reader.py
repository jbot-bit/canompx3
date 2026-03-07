"""
Read-only database helper for the Streamlit dashboard.

All connections use read_only=True to prevent accidental writes.
Uses a cached connection per db_path to avoid connection-per-query overhead.
"""

import atexit
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

_DB_CONNECTIONS: dict[str, duckdb.DuckDBPyConnection] = {}


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Get or create a cached read-only DuckDB connection."""
    path = str(db_path or GOLD_DB_PATH)
    if path not in _DB_CONNECTIONS:
        _DB_CONNECTIONS[path] = duckdb.connect(path, read_only=True)
    return _DB_CONNECTIONS[path]


def _cleanup_connections():
    for con in _DB_CONNECTIONS.values():
        try:
            con.close()
        except Exception:
            pass
    _DB_CONNECTIONS.clear()


atexit.register(_cleanup_connections)


def query_df(
    sql: str,
    db_path: Path | None = None,
    params: list | None = None,
) -> pd.DataFrame:
    """Execute a SELECT query and return a DataFrame.

    Only SELECT statements are allowed. Raises ValueError for anything else.
    Use $1, $2, ... placeholders for parameterized queries.
    """
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        raise ValueError(f"Only SELECT/WITH queries allowed, got: {sql[:40]}...")

    con = get_connection(db_path)
    if params:
        return con.execute(sql, params).fetchdf()
    return con.execute(sql).fetchdf()


def get_table_counts(db_path: Path | None = None) -> dict[str, int]:
    """Return row counts for all known tables."""
    tables = [
        "bars_1m",
        "bars_5m",
        "daily_features",
        "orb_outcomes",
        "experimental_strategies",
        "validated_setups",
    ]
    conn = get_connection(db_path)
    counts = {}
    for t in tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {t}").fetchone()
            counts[t] = row[0] if row else 0
        except duckdb.CatalogException:
            counts[t] = -1  # table doesn't exist
    return counts


def get_date_ranges(db_path: Path | None = None) -> dict[str, dict]:
    """Return min/max dates for key tables."""
    conn = get_connection(db_path)
    ranges = {}
    # bars_1m
    try:
        row = conn.execute("SELECT MIN(ts_utc)::DATE AS min_d, MAX(ts_utc)::DATE AS max_d FROM bars_1m").fetchone()
        ranges["bars_1m"] = {"min": str(row[0]), "max": str(row[1])} if row else {}
    except Exception:
        ranges["bars_1m"] = {}
    # daily_features
    try:
        row = conn.execute("SELECT MIN(trading_day) AS min_d, MAX(trading_day) AS max_d FROM daily_features").fetchone()
        ranges["daily_features"] = {"min": str(row[0]), "max": str(row[1])} if row else {}
    except Exception:
        ranges["daily_features"] = {}
    # orb_outcomes
    try:
        row = conn.execute("SELECT MIN(trading_day) AS min_d, MAX(trading_day) AS max_d FROM orb_outcomes").fetchone()
        ranges["orb_outcomes"] = {"min": str(row[0]), "max": str(row[1])} if row else {}
    except Exception:
        ranges["orb_outcomes"] = {}
    return ranges


def get_validated_strategies(
    db_path: Path | None = None,
    min_expectancy_r: float = 0.0,
) -> pd.DataFrame:
    """Load validated_setups as a DataFrame, optionally filtered by min ExpR."""
    sql = """
        SELECT * FROM validated_setups
        WHERE expectancy_r >= $1
        ORDER BY expectancy_r DESC
    """
    return query_df(sql, db_path, params=[min_expectancy_r])


def get_daily_features(
    trading_day: str,
    orb_minutes: int = 5,
    db_path: Path | None = None,
) -> dict | None:
    """Load daily_features for a single trading day. Returns dict or None."""
    sql = """
        SELECT * FROM daily_features
        WHERE trading_day = $1
          AND orb_minutes = $2
        LIMIT 1
    """
    df = query_df(sql, db_path, params=[trading_day, orb_minutes])
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_bars_per_day(db_path: Path | None = None) -> pd.DataFrame:
    """Return bar count per trading day for coverage analysis."""
    sql = """
        SELECT trading_day, bar_count_1m
        FROM daily_features
        WHERE orb_minutes = 5
        ORDER BY trading_day
    """
    return query_df(sql, db_path)


def get_contract_timeline(db_path: Path | None = None) -> pd.DataFrame:
    """Return source_symbol usage over time (contract rolls)."""
    sql = """
        SELECT source_symbol,
               MIN(ts_utc)::DATE AS first_seen,
               MAX(ts_utc)::DATE AS last_seen,
               COUNT(*) AS bar_count
        FROM bars_1m
        WHERE symbol = 'MGC'
        GROUP BY source_symbol
        ORDER BY first_seen
    """
    return query_df(sql, db_path)


def get_gap_days(db_path: Path | None = None) -> pd.DataFrame:
    """Find trading days with unusually low bar counts (potential gaps)."""
    sql = """
        SELECT trading_day, bar_count_1m
        FROM daily_features
        WHERE orb_minutes = 5
          AND bar_count_1m < 500
        ORDER BY trading_day
    """
    return query_df(sql, db_path)


def get_schema_summary(db_path: Path | None = None) -> str:
    """Return a text summary of DB tables and columns for AI context."""
    conn = get_connection(db_path)
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchdf()
    lines = []
    for t in tables["table_name"]:
        cols = conn.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name='{t}' ORDER BY ordinal_position"
        ).fetchdf()
        col_strs = [f"{r['column_name']} ({r['data_type']})" for _, r in cols.iterrows()]
        lines.append(f"{t}: {', '.join(col_strs)}")
    return "\n".join(lines)


def get_prior_day_atr(
    instrument: str,
    orb_minutes: int = 5,
    db_path: Path | None = None,
) -> float | None:
    """Get the most recent ATR-20 for an instrument.

    Returns the atr_20 value from the latest trading day in daily_features.
    Used by the co-pilot to set expectations: "Prior day ATR: 28pts."
    """
    sql = """
        SELECT atr_20
        FROM daily_features
        WHERE symbol = $1
          AND orb_minutes = $2
        ORDER BY trading_day DESC
        LIMIT 1
    """
    try:
        df = query_df(sql, db_path, params=[instrument, orb_minutes])
        if df.empty:
            return None
        val = df.iloc[0]["atr_20"]
        return float(val) if val is not None else None
    except Exception:
        return None


def get_today_completed_sessions(
    trading_day: date,
    db_path: Path | None = None,
) -> list[dict]:
    """Get ORB outcomes for a trading day, grouped by session.

    Returns list of dicts with keys: orb_label, symbol, break_dir, pnl_r, outcome.
    Used by the co-pilot's day summary section.
    """
    sql = """
        SELECT orb_label, symbol, break_dir, pnl_r, outcome,
               entry_model, rr_target
        FROM orb_outcomes
        WHERE trading_day = $1
          AND orb_minutes = 5
        ORDER BY orb_label, symbol
    """
    try:
        df = query_df(sql, db_path, params=[trading_day.isoformat()])
        return df.to_dict("records") if not df.empty else []
    except Exception:
        return []


def get_previous_trading_day(
    before: date,
    db_path: Path | None = None,
) -> date | None:
    """Find the most recent trading day before the given date.

    Queries daily_features for the latest trading_day < before.
    Used by the co-pilot for "Last trading day" summary.
    """
    sql = """
        SELECT MAX(trading_day) as prev_day
        FROM daily_features
        WHERE trading_day < $1
          AND orb_minutes = 5
    """
    try:
        df = query_df(sql, db_path, params=[before.isoformat()])
        if df.empty or df.iloc[0]["prev_day"] is None:
            return None
        val = df.iloc[0]["prev_day"]
        # DuckDB returns pd.Timestamp — convert to date
        if hasattr(val, "date"):
            return val.date()
        return val
    except Exception:
        return None

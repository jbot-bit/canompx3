"""
Read-only data layer for the V2 dashboard backend.

All connections use read_only=True to prevent accidental writes.
Uses a cached connection per db_path with retry-backoff for lock contention.
"""

from __future__ import annotations

import atexit
import logging
import time
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

log = logging.getLogger(__name__)

_DB_CONNECTIONS: dict[str, duckdb.DuckDBPyConnection] = {}


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Get or create a cached read-only DuckDB connection with retry."""
    path = str(db_path or GOLD_DB_PATH)
    if path not in _DB_CONNECTIONS:
        for attempt in range(3):
            try:
                _DB_CONNECTIONS[path] = duckdb.connect(path, read_only=True)
                break
            except duckdb.IOException:
                if attempt == 2:
                    raise
                wait = 0.5 * (2**attempt)
                log.warning("DuckDB locked, retrying in %.1fs (attempt %d/3)", wait, attempt + 1)
                time.sleep(wait)
    return _DB_CONNECTIONS[path]


def _cleanup_connections():
    for con in _DB_CONNECTIONS.values():
        try:
            con.close()
        except Exception:
            pass
    _DB_CONNECTIONS.clear()


atexit.register(_cleanup_connections)


def query_df(sql: str, db_path: Path | None = None) -> pd.DataFrame:
    """Execute a SELECT query and return a DataFrame.

    Only SELECT statements are allowed. Raises ValueError for anything else.
    """
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        raise ValueError(f"Only SELECT/WITH queries allowed, got: {sql[:40]}...")

    con = get_connection(db_path)
    return con.execute(sql).fetchdf()


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe list of dicts (NaN/NaT → None)."""
    if df.empty:
        return []
    import math

    records = df.to_dict("records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and math.isnan(v) or pd.isna(v):
                rec[k] = None
    return records


# ---------------------------------------------------------------------------
# Ported queries from ui/db_reader.py
# ---------------------------------------------------------------------------


def get_prior_day_atr(
    instrument: str,
    orb_minutes: int = 5,
    db_path: Path | None = None,
) -> float | None:
    """Get the most recent ATR-20 for an instrument.

    Returns the atr_20 value from the latest trading day in daily_features.
    Used by the co-pilot to set expectations: "Prior day ATR: 28pts."
    """
    sql = f"""
        SELECT atr_20
        FROM daily_features
        WHERE symbol = '{instrument}'
          AND orb_minutes = {orb_minutes}
        ORDER BY trading_day DESC
        LIMIT 1
    """
    try:
        df = query_df(sql, db_path)
        if df.empty:
            return None
        val = df.iloc[0]["atr_20"]
        return float(val) if val is not None else None
    except Exception:
        return None


def get_previous_trading_day(
    before: date,
    db_path: Path | None = None,
) -> date | None:
    """Find the most recent trading day before the given date.

    Queries daily_features for the latest trading_day < before.
    """
    sql = f"""
        SELECT MAX(trading_day) as prev_day
        FROM daily_features
        WHERE trading_day < '{before.isoformat()}'
          AND orb_minutes = 5
    """
    try:
        df = query_df(sql, db_path)
        if df.empty or pd.isna(df.iloc[0]["prev_day"]):
            return None
        val = df.iloc[0]["prev_day"]
        # DuckDB returns pd.Timestamp — convert to date
        if hasattr(val, "date"):
            return val.date()
        return val
    except Exception:
        return None


def get_today_completed_sessions(
    trading_day: date,
    db_path: Path | None = None,
) -> list[dict]:
    """Get ORB outcomes for a trading day, grouped by session.

    Returns list of dicts with keys: orb_label, symbol, pnl_r, outcome,
    entry_model, rr_target.
    """
    sql = f"""
        SELECT orb_label, symbol, pnl_r, outcome,
               entry_model, rr_target
        FROM orb_outcomes
        WHERE trading_day = '{trading_day.isoformat()}'
          AND orb_minutes = 5
        ORDER BY orb_label, symbol
    """
    try:
        df = query_df(sql, db_path)
        return _df_to_records(df)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# New V2 queries
# ---------------------------------------------------------------------------


def get_session_history(
    session_name: str,
    limit: int = 10,
    db_path: Path | None = None,
) -> list[dict]:
    """Last N occurrences of a session with outcomes.

    Returns list of dicts: trading_day, symbol, pnl_r, outcome,
    entry_model, rr_target.
    """
    sql = f"""
        SELECT trading_day, symbol, pnl_r, outcome,
               entry_model, rr_target
        FROM orb_outcomes
        WHERE orb_label = '{session_name}'
          AND orb_minutes = 5
        ORDER BY trading_day DESC
        LIMIT {limit}
    """
    try:
        df = query_df(sql, db_path)
        return _df_to_records(df)
    except Exception:
        return []


def get_rolling_pnl(
    days: int = 20,
    db_path: Path | None = None,
) -> dict:
    """Daily R totals for sparkline + week/month aggregates.

    Returns dict with keys: daily (list of {date, total_r}), week_r, month_r.
    """
    sql = f"""
        SELECT trading_day,
               SUM(pnl_r) AS total_r,
               COUNT(*) AS trade_count
        FROM orb_outcomes
        WHERE orb_minutes = 5
          AND pnl_r IS NOT NULL
        GROUP BY trading_day
        ORDER BY trading_day DESC
        LIMIT {days}
    """
    try:
        df = query_df(sql, db_path)
        if df.empty:
            return {"daily": [], "week_r": 0.0, "month_r": 0.0}

        daily = [{"date": str(row["trading_day"]), "total_r": float(row["total_r"])} for _, row in df.iterrows()]

        # Week = last 5 trading days, month = last 20
        week_r = sum(d["total_r"] for d in daily[:5])
        month_r = sum(d["total_r"] for d in daily[:20])

        return {"daily": daily, "week_r": round(week_r, 2), "month_r": round(month_r, 2)}
    except Exception:
        return {"daily": [], "week_r": 0.0, "month_r": 0.0}


def _get_overnight_sessions(for_date: date) -> tuple[str, ...]:
    """Derive overnight sessions dynamically from SESSION_CATALOG.

    A session is 'overnight' if its Brisbane hour falls outside AWAKE_START..AWAKE_END.
    """
    from pipeline.dst import SESSION_CATALOG
    from ui_v2.state_machine import AWAKE_END, AWAKE_START

    overnight: list[str] = []
    for name, entry in SESSION_CATALOG.items():
        h, _m = entry["resolver"](for_date)
        if not (AWAKE_START <= h < AWAKE_END):
            overnight.append(name)
    return tuple(overnight)


def get_overnight_recap(
    trading_day: date,
    db_path: Path | None = None,
) -> list[dict]:
    """Overnight session outcomes for a trading day.

    Returns outcomes for sessions outside awake hours, derived dynamically
    from SESSION_CATALOG + AWAKE_START/AWAKE_END.
    """
    overnight_sessions = _get_overnight_sessions(trading_day)
    if not overnight_sessions:
        return []
    placeholders = ", ".join(f"'{s}'" for s in overnight_sessions)
    sql = f"""
        SELECT orb_label, symbol, pnl_r, outcome
        FROM orb_outcomes
        WHERE trading_day = '{trading_day.isoformat()}'
          AND orb_label IN ({placeholders})
          AND orb_minutes = 5
        ORDER BY orb_label, symbol
    """
    try:
        df = query_df(sql, db_path)
        return _df_to_records(df)
    except Exception:
        return []


def get_fitness_regimes(db_path: Path | None = None) -> list[dict]:
    """Fitness status for all validated strategies.

    Returns list of dicts with strategy_id, instrument, orb_label, entry_model,
    filter_type, rr_target, orb_minutes, expectancy_r, sharpe, win_rate, sample_size.
    """
    sql = """
        SELECT strategy_id, instrument, orb_label, entry_model,
               filter_type, rr_target, orb_minutes,
               expectancy_r, sharpe, win_rate, sample_size
        FROM validated_setups
        ORDER BY instrument, orb_label
    """
    try:
        df = query_df(sql, db_path)
        return _df_to_records(df)
    except Exception:
        return []

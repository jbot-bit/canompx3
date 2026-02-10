"""
Read-only database helper for the Streamlit dashboard.

All connections use read_only=True to prevent accidental writes.
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection."""
    path = db_path or GOLD_DB_PATH
    return duckdb.connect(str(path), read_only=True)


def query_df(sql: str, db_path: Path | None = None) -> pd.DataFrame:
    """Execute a SELECT query and return a DataFrame.

    Only SELECT statements are allowed. Raises ValueError for anything else.
    """
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        raise ValueError(f"Only SELECT/WITH queries allowed, got: {sql[:40]}...")

    conn = get_connection(db_path)
    try:
        return conn.execute(sql).fetchdf()
    finally:
        conn.close()


def get_table_counts(db_path: Path | None = None) -> dict[str, int]:
    """Return row counts for all known tables."""
    tables = [
        "bars_1m", "bars_5m", "daily_features",
        "orb_outcomes", "experimental_strategies", "validated_setups",
    ]
    conn = get_connection(db_path)
    try:
        counts = {}
        for t in tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {t}").fetchone()
                counts[t] = row[0] if row else 0
            except duckdb.CatalogException:
                counts[t] = -1  # table doesn't exist
        return counts
    finally:
        conn.close()


def get_date_ranges(db_path: Path | None = None) -> dict[str, dict]:
    """Return min/max dates for key tables."""
    conn = get_connection(db_path)
    try:
        ranges = {}
        # bars_1m
        try:
            row = conn.execute(
                "SELECT MIN(ts_utc)::DATE AS min_d, MAX(ts_utc)::DATE AS max_d FROM bars_1m"
            ).fetchone()
            ranges["bars_1m"] = {"min": str(row[0]), "max": str(row[1])} if row else {}
        except Exception:
            ranges["bars_1m"] = {}
        # daily_features
        try:
            row = conn.execute(
                "SELECT MIN(trading_day) AS min_d, MAX(trading_day) AS max_d FROM daily_features"
            ).fetchone()
            ranges["daily_features"] = {"min": str(row[0]), "max": str(row[1])} if row else {}
        except Exception:
            ranges["daily_features"] = {}
        # orb_outcomes
        try:
            row = conn.execute(
                "SELECT MIN(trading_day) AS min_d, MAX(trading_day) AS max_d FROM orb_outcomes"
            ).fetchone()
            ranges["orb_outcomes"] = {"min": str(row[0]), "max": str(row[1])} if row else {}
        except Exception:
            ranges["orb_outcomes"] = {}
        return ranges
    finally:
        conn.close()


def get_validated_strategies(
    db_path: Path | None = None,
    min_expectancy_r: float = 0.0,
) -> pd.DataFrame:
    """Load validated_setups as a DataFrame, optionally filtered by min ExpR."""
    sql = f"""
        SELECT * FROM validated_setups
        WHERE expectancy_r >= {min_expectancy_r}
        ORDER BY expectancy_r DESC
    """
    return query_df(sql, db_path)


def get_daily_features(
    trading_day: str,
    orb_minutes: int = 5,
    db_path: Path | None = None,
) -> dict | None:
    """Load daily_features for a single trading day. Returns dict or None."""
    sql = f"""
        SELECT * FROM daily_features
        WHERE trading_day = '{trading_day}'
          AND orb_minutes = {orb_minutes}
        LIMIT 1
    """
    df = query_df(sql, db_path)
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
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchdf()
        lines = []
        for t in tables["table_name"]:
            cols = conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name='{t}' ORDER BY ordinal_position"
            ).fetchdf()
            col_strs = [f"{r['column_name']} ({r['data_type']})" for _, r in cols.iterrows()]
            lines.append(f"{t}: {', '.join(col_strs)}")
        return "\n".join(lines)
    finally:
        conn.close()

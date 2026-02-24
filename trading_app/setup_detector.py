"""
Filter trading days by market conditions.

Given a StrategyFilter, queries daily_features and returns matching
(trading_day, row_dict) tuples. Used by strategy_discovery to scope
backtests to specific market regimes.

Usage:
    from trading_app.setup_detector import detect_setups
    setups = detect_setups(con, filter, "CME_REOPEN", "MGC", start, end)
"""

from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb

from trading_app.config import StrategyFilter

def detect_setups(
    con: duckdb.DuckDBPyConnection,
    strategy_filter: StrategyFilter,
    orb_label: str,
    instrument: str = "MGC",
    orb_minutes: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[tuple[date, dict]]:
    """
    Find trading days matching a filter for a given ORB label.

    Args:
        con: Open DuckDB connection
        strategy_filter: Filter to apply to each row
        orb_label: ORB label (e.g., "CME_REOPEN", "TOKYO_OPEN")
        instrument: Instrument symbol
        orb_minutes: ORB duration
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        List of (trading_day, row_dict) for matching days that have a break.
    """
    # Build query
    params = [instrument, orb_minutes]
    where_clauses = ["symbol = ?", "orb_minutes = ?"]

    if start_date:
        where_clauses.append("trading_day >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("trading_day <= ?")
        params.append(end_date)

    where_sql = " AND ".join(where_clauses)

    rows = con.execute(
        f"SELECT * FROM daily_features WHERE {where_sql} ORDER BY trading_day",
        params,
    ).fetchall()
    col_names = [desc[0] for desc in con.description]

    results = []
    for row in rows:
        row_dict = dict(zip(col_names, row))

        # Must have a break for this ORB
        break_dir = row_dict.get(f"orb_{orb_label}_break_dir")
        if break_dir is None:
            continue

        # Apply filter
        if not strategy_filter.matches_row(row_dict, orb_label):
            continue

        results.append((row_dict["trading_day"], row_dict))

    return results

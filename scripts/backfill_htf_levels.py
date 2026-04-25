#!/usr/bin/env python3
"""One-shot backfill for HTF prev_week_* / prev_month_* fields on existing DB.

Purpose:
    After `pipeline/init_db.py` has added the 12 HTF columns as NULL,
    populate them for all existing daily_features rows by running the
    canonical ``_apply_htf_level_fields()`` helper on rows loaded from
    the DB, then UPDATE the DB in place.

Why we reuse the canonical helper (not a SQL-only rewrite):
    A SQL re-encoding of the same aggregation is a silent-drift risk per
    .claude/rules/institutional-rigor.md rule #4. This script delegates to
    the SAME Python function that build_daily_features() runs, guaranteeing
    byte-identical semantics. The post-backfill drift check
    ``check_htf_levels_integrity`` independently verifies the result against
    a DuckDB DATE_TRUNC('week')/'month' aggregation, catching any divergence.

Idempotency:
    Safe to re-run. Always recomputes from current daily_open/high/low/close
    and overwrites the 12 HTF columns. No external side effects.

Usage:
    DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python scripts/backfill_htf_levels.py
    python scripts/backfill_htf_levels.py --dry-run
    python scripts/backfill_htf_levels.py --symbols MNQ,MES
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.build_daily_features import _apply_htf_level_fields
from pipeline.paths import GOLD_DB_PATH

HTF_COLUMNS = [
    "prev_week_high",
    "prev_week_low",
    "prev_week_open",
    "prev_week_close",
    "prev_week_range",
    "prev_week_mid",
    "prev_month_high",
    "prev_month_low",
    "prev_month_open",
    "prev_month_close",
    "prev_month_range",
    "prev_month_mid",
]


def _ensure_columns(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotently add HTF columns if init_db migration hasn't been run."""
    for col in HTF_COLUMNS:
        try:
            con.execute(f"ALTER TABLE daily_features ADD COLUMN {col} DOUBLE")
            print(f"  Added column: {col}")
        except duckdb.CatalogException:
            pass  # already present


def _rows_for_symbol(con: duckdb.DuckDBPyConnection, symbol: str) -> list[dict]:
    """Load (trading_day, daily_*) rows at orb_minutes=5 sorted ascending.

    HTF fields are orb_minutes-agnostic so one representative slice suffices;
    the UPDATE step replicates the result across all three orb_minutes
    triplets per (symbol, trading_day) pair.
    """
    records = con.execute(
        """
        SELECT trading_day, daily_open, daily_high, daily_low, daily_close
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
        ORDER BY trading_day
        """,
        [symbol],
    ).fetchall()
    return [
        {
            "trading_day": r[0] if isinstance(r[0], date) else None,
            "daily_open": r[1],
            "daily_high": r[2],
            "daily_low": r[3],
            "daily_close": r[4],
        }
        for r in records
    ]


def _backfill_symbol(con: duckdb.DuckDBPyConnection, symbol: str, dry_run: bool) -> int:
    rows = _rows_for_symbol(con, symbol)
    if not rows:
        print(f"  {symbol}: no rows, skipping")
        return 0

    _apply_htf_level_fields(rows)
    non_null = sum(1 for r in rows if r.get("prev_week_high") is not None)
    print(f"  {symbol}: {len(rows)} rows, {non_null} non-null prev_week_high")

    if dry_run:
        return non_null

    update_sql = """
        UPDATE daily_features
           SET prev_week_high  = ?,
               prev_week_low   = ?,
               prev_week_open  = ?,
               prev_week_close = ?,
               prev_week_range = ?,
               prev_week_mid   = ?,
               prev_month_high  = ?,
               prev_month_low   = ?,
               prev_month_open  = ?,
               prev_month_close = ?,
               prev_month_range = ?,
               prev_month_mid   = ?
         WHERE symbol = ? AND trading_day = ?
    """

    for r in rows:
        td = r.get("trading_day")
        if td is None:
            continue
        params = [
            r.get("prev_week_high"),
            r.get("prev_week_low"),
            r.get("prev_week_open"),
            r.get("prev_week_close"),
            r.get("prev_week_range"),
            r.get("prev_week_mid"),
            r.get("prev_month_high"),
            r.get("prev_month_low"),
            r.get("prev_month_open"),
            r.get("prev_month_close"),
            r.get("prev_month_range"),
            r.get("prev_month_mid"),
            symbol,
            td,
        ]
        con.execute(update_sql, params)

    return non_null


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill HTF prev-week/prev-month fields on daily_features")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't UPDATE")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: all distinct in daily_features)",
    )
    args = parser.parse_args()

    db_path = GOLD_DB_PATH
    print(f"Canonical DB: {db_path}")
    if not db_path.exists():
        print(f"FATAL: DB does not exist at {db_path}", file=sys.stderr)
        return 2

    with duckdb.connect(str(db_path)) as con:
        _ensure_columns(con)

        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = sorted(r[0] for r in con.execute("SELECT DISTINCT symbol FROM daily_features").fetchall())

        total = 0
        for sym in symbols:
            total += _backfill_symbol(con, sym, args.dry_run)

        if not args.dry_run:
            con.commit()
        print(f"Done. Non-null prev_week_high rows across run: {total}. dry_run={args.dry_run}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

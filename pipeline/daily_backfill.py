"""
Nightly data backfill: ingest new bars → 5m bars → daily features → outcomes.
Does NOT trigger strategy discovery or validation (those are periodic/manual).

Usage:
    python pipeline/daily_backfill.py --instrument MGC
    python pipeline/daily_backfill.py  # all active instruments
"""

import argparse
import subprocess
import sys
from datetime import UTC, date, timedelta

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH


def get_last_ingested_date(db_path: str, symbol: str):
    """Return the datetime of the most recent bar for symbol, or None."""
    con = duckdb.connect(db_path, read_only=True)
    configure_connection(con)
    try:
        row = con.execute("SELECT MAX(ts_utc) FROM bars_1m WHERE symbol = ?", [symbol]).fetchone()
        ts = row[0] if row and row[0] is not None else None
        # Normalize to UTC — DuckDB returns TIMESTAMPTZ in system local timezone
        if ts is not None and hasattr(ts, "astimezone"):
            ts = ts.astimezone(UTC)
        return ts
    finally:
        con.close()


def is_up_to_date(db_path: str, symbol: str, as_of: date) -> bool:
    """True if bars_1m has data through as_of date for symbol."""
    last = get_last_ingested_date(db_path, symbol)
    if last is None:
        return False
    return last.date() >= as_of


def _run(cmd: list[str], desc: str) -> None:
    """Run subprocess, raise on non-zero exit."""
    print(f"\n▶ {desc}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FAILED: {desc} (exit {result.returncode})")
    print(f"✓ {desc}")


def run_backfill_for_instrument(
    instrument: str,
    db_path: str | None = None,
    as_of: date | None = None,
) -> None:
    """Run incremental backfill for one instrument. Idempotent."""
    db = db_path or str(GOLD_DB_PATH)
    target = as_of or (date.today() - timedelta(days=1))  # last complete session

    if is_up_to_date(db, instrument, target):
        print(f"✓ {instrument}: already current through {target}, skipping")
        return

    last = get_last_ingested_date(db, instrument)
    start = (last.date() + timedelta(days=1)).isoformat() if last else "2021-01-01"
    end = target.isoformat()
    print(f"▶ {instrument}: backfilling {start} → {end}")

    py = sys.executable
    _run(
        [py, "pipeline/ingest_dbn.py", "--instrument", instrument, "--start", start, "--end", end],
        f"ingest_dbn {instrument}",
    )
    _run(
        [py, "pipeline/build_bars_5m.py", "--instrument", instrument, "--start", start, "--end", end],
        f"build_bars_5m {instrument}",
    )
    _run(
        [py, "pipeline/build_daily_features.py", "--instrument", instrument, "--start", start, "--end", end],
        f"build_daily_features {instrument}",
    )
    # Outcomes: incremental only. DO NOT trigger discovery/validation.
    _run(
        [py, "trading_app/outcome_builder.py", "--instrument", instrument, "--start", start, "--end", end],
        f"outcome_builder {instrument}",
    )


def main():
    parser = argparse.ArgumentParser(description="Nightly data backfill")
    parser.add_argument("--instrument", help="Single instrument (default: all active)")
    parser.add_argument("--db-path", help="Override DB path")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else list(ACTIVE_ORB_INSTRUMENTS)
    db = args.db_path or str(GOLD_DB_PATH)
    errors = []
    for inst in instruments:
        try:
            run_backfill_for_instrument(inst, db_path=db)
        except Exception as e:
            print(f"✗ {inst}: {e}")
            errors.append(inst)
    if errors:
        print(f"\n✗ Backfill failed for: {errors}")
        sys.exit(1)
    print("\n✓ Backfill complete")


if __name__ == "__main__":
    main()

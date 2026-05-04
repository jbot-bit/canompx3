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


def _patch_atr_percentiles(db_path: str, instrument: str) -> None:
    """Fill atr_20_pct NULLs using row-by-row percentile rank from full history.

    build_daily_features computes atr_20_pct from a rolling 252-day window,
    but incremental builds often process too few rows for the percentile to
    compute. This post-pass fills NULLs using the full history via single-row
    UPDATEs (avoids DuckDB FK constraints that block CTE-based UPDATEs).
    """
    con = duckdb.connect(db_path)
    configure_connection(con)
    patched = 0
    try:
        for om in [5, 15, 30]:
            nulls = con.execute(
                """SELECT trading_day, atr_20 FROM daily_features
                   WHERE symbol = ? AND orb_minutes = ?
                     AND atr_20_pct IS NULL AND atr_20 IS NOT NULL
                   ORDER BY trading_day""",
                [instrument, om],
            ).fetchall()
            for td, atr in nulls:
                td_str = td.isoformat() if hasattr(td, "isoformat") else str(td)
                rank = con.execute(
                    """SELECT COUNT(*) FROM daily_features
                       WHERE symbol=? AND orb_minutes=? AND trading_day < ?
                         AND atr_20 IS NOT NULL
                         AND trading_day >= ?::DATE - INTERVAL '504 DAY'
                         AND atr_20 < ?""",
                    [instrument, om, td_str, td_str, atr],
                ).fetchone()[0]
                total = con.execute(
                    """SELECT COUNT(*) FROM daily_features
                       WHERE symbol=? AND orb_minutes=? AND trading_day < ?
                         AND atr_20 IS NOT NULL
                         AND trading_day >= ?::DATE - INTERVAL '504 DAY'""",
                    [instrument, om, td_str, td_str],
                ).fetchone()[0]
                if total > 0:
                    pct = round(rank / total * 100, 2)
                    con.execute(
                        "UPDATE daily_features SET atr_20_pct=? WHERE symbol=? AND orb_minutes=? AND trading_day=?",
                        [pct, instrument, om, td_str],
                    )
                    patched += 1
    finally:
        con.close()
    if patched:
        print(f"  Patched {patched} atr_20_pct NULLs for {instrument}")


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

    # Subprocess invocations MUST use python -m <module> form.
    # The target modules (pipeline.ingest_dbn, pipeline.build_bars_5m,
    # pipeline.build_daily_features, trading_app.outcome_builder) all use
    # `from pipeline.X import Y` / `from trading_app.X import Y` imports, which
    # require the project root in sys.path. Python executed as `python foo.py`
    # puts foo.py's directory in sys.path, breaking the package import.
    # Regression guard: tests/test_pipeline/test_daily_backfill_subprocess_form.py
    py = sys.executable
    _run(
        [py, "-m", "pipeline.ingest_dbn", "--instrument", instrument, "--start", start, "--end", end],
        f"ingest_dbn {instrument}",
    )
    _run(
        [py, "-m", "pipeline.build_bars_5m", "--instrument", instrument, "--start", start, "--end", end],
        f"build_bars_5m {instrument}",
    )
    _run(
        [py, "-m", "pipeline.build_daily_features", "--instrument", instrument, "--start", start, "--end", end],
        f"build_daily_features {instrument}",
    )
    # Outcomes: incremental only. DO NOT trigger discovery/validation.
    _run(
        [py, "-m", "trading_app.outcome_builder", "--instrument", instrument, "--start", start, "--end", end],
        f"outcome_builder {instrument}",
    )

    # Patch atr_20_pct for any NULL rows.
    # build_daily_features computes atr_20_pct from a rolling 252-day window,
    # but incremental builds only process the new date range (too few rows).
    # This post-pass fills in NULLs using the full history via row-by-row UPDATE
    # (avoids DuckDB FK constraint issues that block CTE-based UPDATE).
    _patch_atr_percentiles(db, instrument)


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

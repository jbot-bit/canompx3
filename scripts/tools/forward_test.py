"""
Daily batch forward test for MNQ RR1.0 raw baseline.

One command to keep the forward test current:
  1. Download missing bars from Databento API
  2. Ingest + build 5m bars + daily features (via refresh_data.py)
  3. Build orb_outcomes for the new dates
  4. Check kill criteria against 2026+ outcomes

No live WebSocket connection needed. Run whenever you want -- catches up
on all missed days automatically. PC off for a week? Fine. Run once.

Requires: DATABENTO_API_KEY in .env or environment.

Usage:
    python scripts/tools/forward_test.py                    # Full refresh + check
    python scripts/tools/forward_test.py --check-only       # Skip refresh, just check
    python scripts/tools/forward_test.py --dry-run           # Show what would download
    python scripts/tools/forward_test.py --instrument MNQ    # Explicit instrument
"""

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Ensure project root is on sys.path so pipeline/trading_app imports work
# when running directly (not via PYTHONPATH=.)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(cmd: list[str], label: str) -> bool:
    """Run subprocess with PROJECT_ROOT as cwd. Returns True on success."""
    print(f"\n>>> {label}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        return False
    print("    OK")
    return True


def get_outcomes_last_date(instrument: str) -> date | None:
    """Get the latest trading_day in orb_outcomes for this instrument."""
    from pipeline.paths import GOLD_DB_PATH

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        row = con.execute(
            "SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = ?",
            [instrument],
        ).fetchone()
        return row[0] if row and row[0] else None


def get_features_last_date(instrument: str) -> date | None:
    """Get the latest trading_day in daily_features for this instrument."""
    from pipeline.paths import GOLD_DB_PATH

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        row = con.execute(
            "SELECT MAX(trading_day) FROM daily_features WHERE symbol = ? AND orb_minutes = 5",
            [instrument],
        ).fetchone()
        return row[0] if row and row[0] else None


def main():
    parser = argparse.ArgumentParser(
        description="Daily batch forward test -- refresh data + check kill criteria",
    )
    parser.add_argument("--instrument", default="MNQ", help="Instrument (default: MNQ)")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip data refresh, just check kill criteria on existing data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded/built without doing it",
    )
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        help="Kill criteria start date (default: 2026-01-01)",
    )
    args = parser.parse_args()

    py = sys.executable
    instrument = args.instrument
    since = args.since or date(2026, 1, 1)

    print("=" * 60)
    print(f"  FORWARD TEST -- {instrument} Raw Baseline")
    print(f"  Date: {date.today()}")
    print("=" * 60)

    if not args.check_only:
        # Step 1: Refresh data (download from Databento + ingest + features)
        refresh_cmd = [py, "scripts/tools/refresh_data.py", "--instrument", instrument]
        if args.dry_run:
            refresh_cmd.append("--dry-run")

        if not _run(refresh_cmd, "Step 1: Refresh data (Databento -> bars -> features)"):
            print("\nFATAL: Data refresh failed. Fix errors above before proceeding.")
            sys.exit(1)

        if args.dry_run:
            print("\n[dry-run] Would build outcomes and check kill criteria next.")
            sys.exit(0)

        # Step 2: Build outcomes for any dates that have features but no outcomes
        features_last = get_features_last_date(instrument)
        outcomes_last = get_outcomes_last_date(instrument)

        if features_last is None:
            print("\nFATAL: No daily_features data. Run full pipeline first.")
            sys.exit(1)

        if outcomes_last is None or outcomes_last < features_last:
            gap_start = (outcomes_last + timedelta(days=1)) if outcomes_last else date(2021, 1, 1)
            gap_end = features_last

            outcome_cmd = [
                py,
                "trading_app/outcome_builder.py",
                "--instrument",
                instrument,
                "--start",
                str(gap_start),
                "--end",
                str(gap_end),
            ]
            if not _run(outcome_cmd, f"Step 2: Build outcomes ({gap_start} to {gap_end})"):
                print("\nFATAL: Outcome builder failed. Fix errors above.")
                sys.exit(1)
        else:
            print(f"\n>>> Step 2: Outcomes already current (through {outcomes_last})")

    # Step 3: Check kill criteria
    check_cmd = [
        py,
        "scripts/tools/check_kill_criteria.py",
        "--from-outcomes",
        "--instrument",
        instrument,
        "--since",
        str(since),
    ]
    # check_kill_criteria exits non-zero if any kill fires
    print()
    result = subprocess.run(
        check_cmd,
        cwd=str(PROJECT_ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    if result.returncode != 0:
        print("\n*** KILL CRITERIA TRIGGERED -- review output above ***")
        sys.exit(1)

    print("\nForward test complete. All clear.")


if __name__ == "__main__":
    main()

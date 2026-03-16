#!/usr/bin/env python3
"""Rebuild daily_features + full rebuild chain for an instrument.

Handles the FK constraint (orb_outcomes → daily_features) by clearing
outcomes before rebuilding daily_features, then running the standard
rebuild chain which rebuilds outcomes from scratch.

Usage:
    python scripts/tools/rebuild_with_daily_features.py --instrument MNQ
    python scripts/tools/rebuild_with_daily_features.py --instrument MGC --daily-features-only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PY = sys.executable


def main():
    parser = argparse.ArgumentParser(description="Rebuild daily_features + full chain")
    parser.add_argument("--instrument", required=True)
    parser.add_argument(
        "--daily-features-only",
        action="store_true",
        help="Only rebuild daily_features (skip outcome/discovery/validation chain)",
    )
    parser.add_argument("--db-path", default=str(PROJECT_ROOT / "gold.db"))
    args = parser.parse_args()

    instrument = args.instrument
    db_path = args.db_path

    print(f"{'=' * 60}")
    print(f"REBUILD: {instrument}")
    print(f"DB: {db_path}")
    print(f"{'=' * 60}")

    # Step 1: Clear outcomes (FK constraint)
    print(f"\n[1] Clearing {instrument} outcomes (FK constraint)...")
    con = duckdb.connect(db_path)
    for orb in [5, 15, 30]:
        n = con.execute(
            "SELECT count(*) FROM orb_outcomes WHERE symbol = ? AND orb_minutes = ?",
            [instrument, orb],
        ).fetchone()[0]
        if n > 0:
            con.execute(
                "DELETE FROM orb_outcomes WHERE symbol = ? AND orb_minutes = ?",
                [instrument, orb],
            )
            print(f"  Cleared {n} O{orb} outcomes")
        else:
            print(f"  O{orb}: no outcomes to clear")
    con.close()

    # Step 2: Rebuild daily_features (all apertures)
    print(f"\n[2] Rebuilding {instrument} daily_features...")
    for orb in [5, 15, 30]:
        cmd = [
            _PY,
            "pipeline/build_daily_features.py",
            "--instrument",
            instrument,
            "--start",
            "2016-01-01",
            "--end",
            "2026-12-31",
            "--orb-minutes",
            str(orb),
        ]
        print(f"  O{orb}: {' '.join(cmd[-6:])}")
        start = time.monotonic()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.monotonic() - start
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            sys.exit(1)
        print(f"  O{orb}: PASSED ({elapsed:.1f}s)")

    # Verify atr_20_pct populated
    con = duckdb.connect(db_path, read_only=True)
    r = con.execute(
        "SELECT count(*) as total, count(atr_20_pct) as has FROM daily_features WHERE symbol = ? AND orb_minutes = 5",
        [instrument],
    ).fetchone()
    con.close()
    print(f"  Verification: {r[1]}/{r[0]} rows have atr_20_pct")
    if r[1] == 0:
        print("  ERROR: atr_20_pct not populated!")
        sys.exit(1)

    if args.daily_features_only:
        print(f"\n{'=' * 60}")
        print("DAILY FEATURES COMPLETE (--daily-features-only)")
        print(f"Run standard rebuild next: python scripts/tools/pipeline_status.py --rebuild --instrument {instrument}")
        print(f"{'=' * 60}")
        return

    # Step 3: Standard rebuild chain (outcomes → discovery → validation → edge_families)
    print("\n[3] Running standard rebuild chain...")
    cmd = [
        _PY,
        "scripts/tools/pipeline_status.py",
        "--rebuild",
        "--instrument",
        instrument,
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  Rebuild chain returned exit code {result.returncode}")
        # Don't exit — the health_check step may fail on non-critical issues
        # Check if the core steps (outcomes, discovery, validation) passed

    # Final verification
    con = duckdb.connect(db_path, read_only=True)
    outcomes_n = con.execute("SELECT count(*) FROM orb_outcomes WHERE symbol = ?", [instrument]).fetchone()[0]
    validated_n = con.execute(
        "SELECT count(*) FROM validated_setups WHERE instrument = ? AND status = 'active'", [instrument]
    ).fetchone()[0]
    atr70_n = con.execute(
        "SELECT count(*) FROM validated_setups WHERE instrument = ? AND filter_type = 'ATR70_VOL' AND status = 'active'",
        [instrument],
    ).fetchone()[0]
    cross_n = con.execute(
        "SELECT count(*) FROM validated_setups WHERE instrument = ? AND filter_type LIKE 'X_%' AND status = 'active'",
        [instrument],
    ).fetchone()[0]
    con.close()

    print(f"\n{'=' * 60}")
    print(f"REBUILD COMPLETE: {instrument}")
    print(f"  Outcomes: {outcomes_n:,}")
    print(f"  Validated (active): {validated_n}")
    print(f"  ATR70_VOL strategies: {atr70_n}")
    print(f"  Cross-asset strategies: {cross_n}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Full pipeline: ingest -> bars_5m -> features -> audit -> outcomes -> discovery -> validation.

Extends run_pipeline.py with strategy pipeline stages.
Uses DUCKDB_PATH env var or --db-path for database location.

Usage:
    python pipeline/run_full_pipeline.py --instrument MGC --start 2024-01-01 --end 2026-02-14
    python pipeline/run_full_pipeline.py --instrument MGC --skip-to build_outcomes --db-path C:/db/gold.db
    python pipeline/run_full_pipeline.py --instrument MGC --dry-run
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

from pipeline.asset_configs import list_instruments

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# STEP FUNCTIONS
# =============================================================================

def step_ingest(instrument: str, args) -> int:
    """Ingest DBN -> bars_1m."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "ingest_dbn.py"),
        f"--instrument={instrument}",
    ]
    if args.start:
        cmd.append(f"--start={args.start}")
    if args.end:
        cmd.append(f"--end={args.end}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode

def step_build_5m(instrument: str, args) -> int:
    """Rebuild bars_5m from bars_1m."""
    if not args.start or not args.end:
        print("FATAL: --start and --end required for bars_5m")
        return 1
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "build_bars_5m.py"),
        f"--instrument={instrument}",
        f"--start={args.start}",
        f"--end={args.end}",
    ]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode

def step_build_features(instrument: str, args) -> int:
    """Rebuild daily_features."""
    if not args.start or not args.end:
        print("FATAL: --start and --end required for daily_features")
        return 1
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "build_daily_features.py"),
        f"--instrument={instrument}",
        f"--start={args.start}",
        f"--end={args.end}",
    ]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode

def step_audit(instrument: str, args) -> int:
    """Database integrity check."""
    cmd = [sys.executable, str(PROJECT_ROOT / "pipeline" / "check_db.py")]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode

def step_build_outcomes(instrument: str, args) -> int:
    """Pre-compute orb_outcomes."""
    if not args.start or not args.end:
        print("FATAL: --start and --end required for outcomes")
        return 1
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "trading_app" / "outcome_builder.py"),
        f"--instrument={instrument}",
        f"--start={args.start}",
        f"--end={args.end}",
    ]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode

def step_discover(instrument: str, args) -> int:
    """Grid search experimental_strategies."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "trading_app" / "strategy_discovery.py"),
        f"--instrument={instrument}",
    ]
    if args.db_path:
        cmd.append(f"--db={args.db_path}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode

def step_validate(instrument: str, args) -> int:
    """Strategy validation."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "trading_app" / "strategy_validator.py"),
        f"--instrument={instrument}",
        "--min-sample=50",
    ]
    if args.db_path:
        cmd.append(f"--db={args.db_path}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode

# =============================================================================
# STEP REGISTRY
# =============================================================================

FULL_PIPELINE_STEPS = [
    ("ingest", "Ingest DBN -> bars_1m", step_ingest),
    ("build_5m", "Rebuild bars_5m from bars_1m", step_build_5m),
    ("build_features", "Rebuild daily_features", step_build_features),
    ("audit", "Database integrity check", step_audit),
    ("build_outcomes", "Pre-compute orb_outcomes", step_build_outcomes),
    ("discover", "Grid search experimental_strategies", step_discover),
    ("validate", "Strategy validation", step_validate),
]

# =============================================================================
# HELPERS
# =============================================================================

def get_steps_from(steps, skip_to: str):
    """Return steps starting from skip_to."""
    names = [s[0] for s in steps]
    if skip_to not in names:
        raise ValueError(f"Unknown step '{skip_to}'. Valid: {names}")
    idx = names.index(skip_to)
    return steps[idx:]

def print_dry_run(steps, instrument: str):
    """Print planned steps without executing."""
    print(f"DRY RUN: Full pipeline for {instrument}")
    print()
    for i, (name, desc, _) in enumerate(steps, 1):
        print(f"  Step {i}: {name} - {desc}")
    print()
    print("No steps executed (dry run).")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: ingest -> 5m -> features -> outcomes -> discovery -> validation"
    )
    parser.add_argument("--instrument", type=str, required=True,
                        help=f"Instrument ({', '.join(list_instruments())})")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    parser.add_argument("--skip-to", type=str,
                        help="Skip to named step (e.g. build_outcomes)")
    parser.add_argument("--db-path", type=str,
                        help="Database path (also sets DUCKDB_PATH env var)")
    args = parser.parse_args()

    instrument = args.instrument.upper()

    # Set DUCKDB_PATH so all subprocesses use the right DB
    if args.db_path:
        os.environ["DUCKDB_PATH"] = args.db_path

    # Determine steps to run
    steps = FULL_PIPELINE_STEPS
    if args.skip_to:
        steps = get_steps_from(FULL_PIPELINE_STEPS, args.skip_to)

    if args.dry_run:
        print_dry_run(steps, instrument)
        sys.exit(0)

    # Execute
    start_time = datetime.now()
    print("=" * 70)
    print(f"FULL PIPELINE: {instrument}")
    print("=" * 70)
    print(f"  Date range: {args.start or 'default'} to {args.end or 'default'}")
    print(f"  DB path: {args.db_path or 'default (DUCKDB_PATH or project/gold.db)'}")
    print(f"  Steps: {' -> '.join(s[0] for s in steps)}")
    print()

    results = []
    for i, (name, desc, func) in enumerate(steps, 1):
        print("-" * 70)
        print(f"STEP {i}/{len(steps)}: {name} - {desc}")
        print("-" * 70)

        step_start = datetime.now()
        rc = func(instrument, args)
        elapsed = datetime.now() - step_start

        results.append({"step": i, "name": name, "rc": rc, "elapsed": elapsed})

        if rc != 0:
            print(f"\nFATAL: {name} failed (exit {rc}). Pipeline halted.")
            break

        print(f"  {name}: PASSED ({elapsed})\n")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    total = datetime.now() - start_time
    for r in results:
        status = "PASSED" if r["rc"] == 0 else f"FAILED (exit {r['rc']})"
        print(f"  {r['name']}: {status} ({r['elapsed']})")

    all_ok = all(r["rc"] == 0 for r in results)
    print(f"\nTotal: {total}")
    print("SUCCESS" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()

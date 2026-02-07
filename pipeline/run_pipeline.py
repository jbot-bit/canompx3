#!/usr/bin/env python3
"""
Pipeline runner: orchestrates ingest -> build_5m -> build_features -> audit.

Runs steps in strict order. Fail-closed: any step failure aborts the pipeline.
Derived steps (build_5m, build_features) are NOT IMPLEMENTED stubs that fail-closed.

Usage:
    python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31
    python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31 --dry-run
    python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31 --resume

Options:
    --instrument INST     Instrument to process (MGC, MNQ, NQ)
    --start YYYY-MM-DD    Start date (inclusive)
    --end YYYY-MM-DD      End date (inclusive)
    --resume              Resume ingest from checkpoint
    --retry-failed        Retry failed ingest chunks
    --dry-run             Show planned steps without executing
    --chunk-days N        Trading days per ingest commit (default: 7)
    --batch-size N        Rows per DBN read batch (default: 50000)
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.asset_configs import list_instruments

# Project root for resolving script paths
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# PIPELINE STEP DEFINITIONS
# =============================================================================

def step_ingest(instrument: str, args) -> int:
    """Step 1: Ingest DBN -> bars_1m. Delegates to pipeline/ingest_dbn.py."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "ingest_dbn.py"),
        f"--instrument={instrument}",
    ]

    if args.start:
        cmd.append(f"--start={args.start}")
    if args.end:
        cmd.append(f"--end={args.end}")
    if args.resume:
        cmd.append("--resume")
    if args.retry_failed:
        cmd.append("--retry-failed")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.chunk_days != 7:
        cmd.append(f"--chunk-days={args.chunk_days}")
    if args.batch_size != 50000:
        cmd.append(f"--batch-size={args.batch_size}")

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def step_build_5m(instrument: str, args) -> int:
    """Step 2: Rebuild bars_5m from bars_1m. Delegates to pipeline/build_bars_5m.py."""
    if not args.start or not args.end:
        print("FATAL: --start and --end are required for bars_5m build")
        return 1

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "build_bars_5m.py"),
        f"--instrument={instrument}",
        f"--start={args.start}",
        f"--end={args.end}",
    ]

    if args.dry_run:
        cmd.append("--dry-run")

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def step_build_features(instrument: str, args) -> int:
    """Step 3: Rebuild daily_features from bars_1m/bars_5m."""
    if not args.start or not args.end:
        print("FATAL: --start and --end are required for daily_features build")
        return 1

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "build_daily_features.py"),
        f"--instrument={instrument}",
        f"--start={args.start}",
        f"--end={args.end}",
    ]

    if args.dry_run:
        cmd.append("--dry-run")

    # Pass orb_minutes if available
    orb_minutes = getattr(args, 'orb_minutes', 5)
    cmd.append(f"--orb-minutes={orb_minutes}")

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def step_audit(instrument: str, args) -> int:
    """Step 4: Run database integrity check."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "check_db.py"),
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


# =============================================================================
# PIPELINE STEPS REGISTRY (ordered)
# =============================================================================

PIPELINE_STEPS = [
    ("ingest", "Ingest DBN -> bars_1m", step_ingest),
    ("build_5m", "Rebuild bars_5m from bars_1m", step_build_5m),
    ("build_features", "Rebuild daily_features from bars", step_build_features),
    ("audit", "Database integrity check", step_audit),
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline runner: ingest -> build_5m -> build_features -> audit"
    )
    parser.add_argument(
        "--instrument", type=str, required=True,
        help=f"Instrument to process ({', '.join(list_instruments())})"
    )
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--resume", action="store_true", help="Resume ingest from checkpoint")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed ingest chunks")
    parser.add_argument("--dry-run", action="store_true", help="Show planned steps without executing")
    parser.add_argument("--chunk-days", type=int, default=7, help="Trading days per ingest commit")
    parser.add_argument("--batch-size", type=int, default=50000, help="Rows per DBN read batch")
    parser.add_argument("--orb-minutes", type=int, default=5, choices=[5, 15, 30],
                        help="ORB duration in minutes (default: 5)")
    args = parser.parse_args()

    instrument = args.instrument.upper()

    start_time = datetime.now()

    print("=" * 70)
    print("PIPELINE RUNNER")
    print("=" * 70)
    print()
    print(f"Instrument: {instrument}")
    print(f"Date range: {args.start or 'default'} to {args.end or 'default'}")
    print(f"Resume: {args.resume}")
    print(f"Retry failed: {args.retry_failed}")
    print(f"Dry run: {args.dry_run}")
    print()

    # =========================================================================
    # DRY RUN: Show plan and exit
    # =========================================================================
    if args.dry_run:
        print("DRY RUN: Showing planned pipeline steps")
        print()
        for i, (name, desc, _) in enumerate(PIPELINE_STEPS, 1):
            print(f"  Step {i}: {name} - {desc}")
        print()
        print("No steps executed (dry run).")
        print()

        # Still validate that the instrument config is loadable
        print("Validating instrument config...")
        from pipeline.asset_configs import get_asset_config
        config = get_asset_config(instrument)
        print(f"  Config loaded for {instrument} [OK]")
        print(f"  DBN: {config['dbn_path']}")
        print(f"  Pattern: {config['outright_pattern'].pattern}")
        print(f"  Min start: {config['minimum_start_date']}")
        print()
        print("DRY RUN COMPLETE.")
        sys.exit(0)

    # =========================================================================
    # EXECUTE PIPELINE STEPS (FAIL-CLOSED)
    # =========================================================================
    results = []

    for i, (name, desc, step_func) in enumerate(PIPELINE_STEPS, 1):
        print("-" * 70)
        print(f"STEP {i}/{len(PIPELINE_STEPS)}: {name} - {desc}")
        print("-" * 70)
        print()

        step_start = datetime.now()
        returncode = step_func(instrument, args)
        step_elapsed = datetime.now() - step_start

        results.append({
            'step': i,
            'name': name,
            'desc': desc,
            'returncode': returncode,
            'elapsed': step_elapsed,
        })

        if returncode != 0:
            print()
            print(f"FATAL: Step {i} ({name}) failed with exit code {returncode}")
            print("ABORT: Pipeline halted (FAIL-CLOSED)")
            print()
            break

        print()
        print(f"  Step {i} ({name}): PASSED [OK] ({step_elapsed})")
        print()

    # =========================================================================
    # PIPELINE SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print()

    total_elapsed = datetime.now() - start_time
    all_passed = all(r['returncode'] == 0 for r in results)
    steps_run = len(results)
    steps_total = len(PIPELINE_STEPS)

    for r in results:
        status = "PASSED" if r['returncode'] == 0 else f"FAILED (exit {r['returncode']})"
        print(f"  Step {r['step']}: {r['name']} - {status} ({r['elapsed']})")

    if steps_run < steps_total:
        for i in range(steps_run + 1, steps_total + 1):
            name = PIPELINE_STEPS[i - 1][0]
            print(f"  Step {i}: {name} - SKIPPED (prior step failed)")

    print()
    print(f"Steps completed: {steps_run}/{steps_total}")
    print(f"Total wall time: {total_elapsed}")
    print()

    if all_passed:
        print("SUCCESS: All pipeline steps completed.")
        sys.exit(0)
    else:
        print("FAILED: Pipeline did not complete. See step failures above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

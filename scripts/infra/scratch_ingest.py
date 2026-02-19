#!/usr/bin/env python3
"""One-command wrapper for full scratch-DB pipeline workflow.

Copies gold.db to a scratch location, runs the full pipeline there,
verifies the result, and optionally merges back. Safer than manual
copy-run-copy because it includes pre-flight checks and integrity
verification before any merge-back.

Usage:
    # Full pipeline on scratch DB
    python scripts/infra/scratch_ingest.py --instrument MES --start 2019-05-01 --end 2026-02-18

    # Skip to a later step
    python scripts/infra/scratch_ingest.py --instrument MES --start 2019-05-01 --end 2026-02-18 --skip-to build_outcomes

    # Just check the scratch DB state
    python scripts/infra/scratch_ingest.py --verify-only

    # Merge scratch DB back to main after manual inspection
    python scripts/infra/scratch_ingest.py --merge-back

    # Dry run: show what would happen
    python scripts/infra/scratch_ingest.py --instrument MES --start 2019-05-01 --end 2026-02-18 --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Import the canonical DB path resolution (respects DUCKDB_PATH, local_db junction, etc.)
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.paths import GOLD_DB_PATH

DEFAULT_GOLD = GOLD_DB_PATH
DEFAULT_SCRATCH = Path("C:/db/gold.db")
BACKUP_SUFFIX = ".bak"


def _fmt_size(path: Path) -> str:
    """Format file size in MB."""
    if not path.exists():
        return "missing"
    return f"{path.stat().st_size / 1024 / 1024:.0f} MB"


def _verify_db(db_path: Path, instrument: str = None) -> dict:
    """Run basic integrity checks on a DuckDB file. Returns stats dict."""
    import duckdb

    stats = {}
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            # Total bar count
            stats["total_bars"] = con.execute(
                "SELECT COUNT(*) FROM bars_1m"
            ).fetchone()[0]

            # Per-instrument breakdown
            rows = con.execute(
                "SELECT symbol, COUNT(*), MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) "
                "FROM bars_1m GROUP BY symbol ORDER BY symbol"
            ).fetchall()
            stats["instruments"] = {
                r[0]: {"rows": r[1], "min_date": str(r[2]), "max_date": str(r[3])}
                for r in rows
            }

            # Check for specific instrument if requested
            if instrument:
                inst_stats = con.execute(
                    "SELECT COUNT(*), MIN(DATE(ts_utc)), MAX(DATE(ts_utc)) "
                    "FROM bars_1m WHERE symbol = ?",
                    [instrument],
                ).fetchone()
                stats["target"] = {
                    "instrument": instrument,
                    "rows": inst_stats[0],
                    "min_date": str(inst_stats[1]),
                    "max_date": str(inst_stats[2]),
                }

            # Check for duplicates
            dupes = con.execute(
                "SELECT COUNT(*) FROM ("
                "  SELECT symbol, ts_utc FROM bars_1m"
                "  GROUP BY symbol, ts_utc HAVING COUNT(*) > 1"
                ")"
            ).fetchone()[0]
            stats["duplicates"] = dupes

            # Check for NULL source_symbol
            nulls = con.execute(
                "SELECT COUNT(*) FROM bars_1m WHERE source_symbol IS NULL"
            ).fetchone()[0]
            stats["null_source"] = nulls

            stats["ok"] = dupes == 0 and nulls == 0
        finally:
            con.close()
    except Exception as e:
        stats["error"] = str(e)
        stats["ok"] = False

    return stats


def _print_verify(stats: dict, label: str):
    """Print verification results."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        return

    print(f"  Total bars: {stats['total_bars']:,}")
    print(f"  Duplicates: {stats['duplicates']}")
    print(f"  NULL source_symbol: {stats['null_source']}")
    print(f"  Integrity: {'PASS' if stats['ok'] else 'FAIL'}")
    print()

    for sym, info in stats.get("instruments", {}).items():
        print(f"  {sym}: {info['rows']:,} rows  ({info['min_date']} to {info['max_date']})")

    if "target" in stats:
        t = stats["target"]
        print(f"\n  Target ({t['instrument']}): {t['rows']:,} rows  ({t['min_date']} to {t['max_date']})")


def cmd_verify_only(args):
    """Just verify the scratch DB state."""
    scratch = Path(args.scratch)
    if not scratch.exists():
        print(f"Scratch DB not found: {scratch}")
        return 1

    stats = _verify_db(scratch, instrument=args.instrument)
    _print_verify(stats, f"Scratch DB: {scratch} ({_fmt_size(scratch)})")
    return 0 if stats.get("ok") else 1


def cmd_merge_back(args):
    """Merge scratch DB back to main with integrity checks."""
    gold = Path(args.gold)
    scratch = Path(args.scratch)

    if not scratch.exists():
        print(f"FATAL: Scratch DB not found: {scratch}")
        return 1

    # Verify scratch DB integrity
    print("Verifying scratch DB integrity...")
    stats = _verify_db(scratch, instrument=args.instrument)
    _print_verify(stats, f"Scratch DB: {scratch} ({_fmt_size(scratch)})")

    if not stats.get("ok"):
        print("\nFATAL: Scratch DB failed integrity checks. Aborting merge.")
        return 1

    # Back up main DB
    backup = gold.with_suffix(gold.suffix + BACKUP_SUFFIX)
    if gold.exists():
        print(f"\nBacking up {gold} -> {backup}...")
        shutil.copy2(gold, backup)
        print(f"Backup created: {_fmt_size(backup)}")

    # Copy scratch to main
    print(f"Copying {scratch} -> {gold}...")
    shutil.copy2(scratch, gold)
    print(f"Merge complete: {_fmt_size(gold)}")

    # Verify the merged copy
    print("\nVerifying merged DB...")
    merged_stats = _verify_db(gold, instrument=args.instrument)
    if not merged_stats.get("ok"):
        print("WARNING: Merged DB failed integrity checks!")
        print(f"Backup available at: {backup}")
        return 1

    print("Merged DB integrity: PASS")
    return 0


def cmd_run_pipeline(args):
    """Run full pipeline on scratch DB."""
    gold = Path(args.gold)
    scratch = Path(args.scratch)

    # =====================================================
    # PRE-FLIGHT CHECKS
    # =====================================================
    print("=" * 60)
    print("  SCRATCH INGEST — PRE-FLIGHT CHECKS")
    print("=" * 60)

    if not args.instrument:
        print("FATAL: --instrument is required")
        return 1

    instrument = args.instrument.upper()

    # Check main DB exists
    if not gold.exists():
        print(f"FATAL: Main DB not found: {gold}")
        return 1
    print(f"  Main DB: {gold} ({_fmt_size(gold)})")

    # Ensure scratch directory exists
    scratch.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Scratch DB: {scratch}")

    # Check if scratch DB already exists (don't overwrite unless --fresh-copy)
    do_copy = args.fresh_copy
    if scratch.exists() and not do_copy:
        print(f"  Scratch DB exists ({_fmt_size(scratch)}), reusing (use --fresh-copy to overwrite)")
    elif scratch.exists() and do_copy:
        print(f"  Scratch DB exists, --fresh-copy: will overwrite")
    else:
        print(f"  Scratch DB does not exist, will copy from main")
        do_copy = True

    print(f"  Instrument: {instrument}")
    print(f"  Date range: {args.start or 'default'} to {args.end or 'default'}")
    if args.skip_to:
        print(f"  Skip to: {args.skip_to}")
    print()

    if args.dry_run:
        print("DRY RUN: Would execute the following steps:")
        if do_copy:
            print(f"  1. Copy {gold} -> {scratch}")
        print(f"  2. Set DUCKDB_PATH={scratch}")
        print(f"  3. Run: python pipeline/run_full_pipeline.py --instrument {instrument}", end="")
        if args.start:
            print(f" --start {args.start}", end="")
        if args.end:
            print(f" --end {args.end}", end="")
        if args.skip_to:
            print(f" --skip-to {args.skip_to}", end="")
        print(f" --db-path {scratch}")
        print(f"  4. Verify scratch DB")
        print(f"  5. Print merge-back command")
        print("\nNo actions taken (dry run).")
        return 0

    # =====================================================
    # COPY TO SCRATCH
    # =====================================================
    if do_copy:
        print(f"Copying {gold} -> {scratch}...")
        start = datetime.now()
        shutil.copy2(gold, scratch)
        elapsed = datetime.now() - start
        print(f"Copy complete ({elapsed.total_seconds():.1f}s, {_fmt_size(scratch)})")
        print()

    # =====================================================
    # RUN PIPELINE
    # =====================================================
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "run_full_pipeline.py"),
        f"--instrument={instrument}",
        f"--db-path={scratch}",
    ]
    if args.start:
        cmd.append(f"--start={args.start}")
    if args.end:
        cmd.append(f"--end={args.end}")
    if args.skip_to:
        cmd.append(f"--skip-to={args.skip_to}")

    env = {**os.environ, "DUCKDB_PATH": str(scratch)}

    print("=" * 60)
    print(f"  RUNNING PIPELINE")
    print("=" * 60)
    print(f"  Command: {' '.join(cmd)}")
    print(f"  DUCKDB_PATH={scratch}")
    print()

    pipeline_start = datetime.now()
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    pipeline_elapsed = datetime.now() - pipeline_start

    print()
    print("=" * 60)
    if result.returncode != 0:
        print(f"  PIPELINE FAILED (exit {result.returncode}) after {pipeline_elapsed}")
        print(f"  Main DB unchanged: {gold}")
        print(f"  Scratch DB left at: {scratch}")
        print("=" * 60)
        return result.returncode

    print(f"  PIPELINE SUCCEEDED ({pipeline_elapsed})")
    print("=" * 60)

    # =====================================================
    # VERIFY SCRATCH DB
    # =====================================================
    stats = _verify_db(scratch, instrument=instrument)
    _print_verify(stats, f"Scratch DB verification: {scratch}")

    if not stats.get("ok"):
        print("\nWARNING: Scratch DB failed integrity checks!")
        print("Do NOT merge back until investigated.")
        return 1

    # =====================================================
    # PRINT MERGE-BACK COMMAND
    # =====================================================
    print()
    print("=" * 60)
    print("  NEXT STEP: Merge back when ready")
    print("=" * 60)
    print()
    print("  When you're satisfied with the results, run:")
    print()
    merge_cmd = f"  python scripts/infra/scratch_ingest.py --merge-back"
    if instrument:
        merge_cmd += f" --instrument {instrument}"
    print(merge_cmd)
    print()
    print(f"  This will: verify scratch DB -> backup main DB -> copy scratch -> verify merged")
    print()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Scratch DB pipeline workflow wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode flags (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true",
                       help="Just verify the scratch DB state")
    mode.add_argument("--merge-back", action="store_true",
                       help="Merge scratch DB back to main with integrity checks")

    # Common args
    parser.add_argument("--instrument", type=str,
                        help="Instrument to ingest (MGC, MNQ, MCL, MES, SIL)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--skip-to", type=str,
                        help="Skip to pipeline step (e.g. build_outcomes)")
    parser.add_argument("--fresh-copy", action="store_true",
                        help="Force fresh copy of main DB to scratch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without executing")
    parser.add_argument("--gold", type=str, default=str(DEFAULT_GOLD),
                        help=f"Main DB path (default: {DEFAULT_GOLD})")
    parser.add_argument("--scratch", type=str, default=str(DEFAULT_SCRATCH),
                        help=f"Scratch DB path (default: {DEFAULT_SCRATCH})")

    args = parser.parse_args()

    if args.verify_only:
        sys.exit(cmd_verify_only(args))
    elif args.merge_back:
        sys.exit(cmd_merge_back(args))
    else:
        sys.exit(cmd_run_pipeline(args))


if __name__ == "__main__":
    main()

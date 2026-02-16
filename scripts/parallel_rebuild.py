#!/usr/bin/env python3
"""Parallel per-instrument pipeline rebuild using isolated DB copies.

DuckDB allows only one writer per file. This script creates per-instrument
copies, runs rebuilds in parallel subprocesses, then merges results back
using DuckDB ATTACH.

Architecture:
    1. COPY: master DB -> C:\\db\\rebuild\\gold_{instrument}.db per instrument
    2. BUILD: parallel subprocesses per instrument, sequential steps:
       features -> outcome -> discovery -> validation
    3. MERGE: ATTACH each copy to master, DELETE+INSERT per instrument (FK-safe order)
    4. CLEANUP: remove temp copies (unless --keep-copies)

Usage:
    python scripts/parallel_rebuild.py --all
    python scripts/parallel_rebuild.py --all --steps features outcome discovery validation
    python scripts/parallel_rebuild.py --instruments MGC MNQ
    python scripts/parallel_rebuild.py --instruments MGC --steps outcome discovery
    python scripts/parallel_rebuild.py --merge-only --instruments MGC MNQ
    python scripts/parallel_rebuild.py --all --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MASTER = Path(r"C:\db\gold.db")
REBUILD_DIR = Path(r"C:\db\rebuild")
ALL_STEPS = ["features", "outcome", "discovery", "validation"]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------
def _stream_pipe(pipe, prefix: str, log_file) -> None:
    """Read subprocess stdout line-by-line, print with prefix, write to log."""
    for line in iter(pipe.readline, ""):
        if line.strip():
            msg = f"  [{prefix}] {line}"
            print(msg, end="", flush=True)
            log_file.write(msg)
    pipe.close()


def _get_instrument_date_range(instrument: str) -> tuple[str, str]:
    """Return (start_date, end_date) strings for an instrument.

    Uses minimum_start_date from asset_configs. End date is today.
    """
    from pipeline.asset_configs import ASSET_CONFIGS

    config = ASSET_CONFIGS.get(instrument.upper(), {})
    start = config.get("minimum_start_date", date(2016, 1, 1))
    end = date.today()
    return start.isoformat(), end.isoformat()


def _build_command(
    step: str, instrument: str, db_path: Path, extra: dict,
) -> list[str]:
    """Build the CLI command for a single pipeline step."""
    py = sys.executable

    if step == "features":
        # build_daily_features has no --db; uses DUCKDB_PATH env var
        start_dt = extra.get("start")
        end_dt = extra.get("end")
        if not start_dt or not end_dt:
            start_dt, end_dt = _get_instrument_date_range(instrument)
        cmd = [
            py, str(PROJECT_ROOT / "pipeline" / "build_daily_features.py"),
            "--instrument", instrument,
            "--start", start_dt,
            "--end", end_dt,
            "--orb-minutes", str(extra.get("orb_minutes", 5)),
        ]

    elif step == "outcome":
        # outcome_builder has no --db; uses DUCKDB_PATH env var
        cmd = [
            py, str(PROJECT_ROOT / "trading_app" / "outcome_builder.py"),
            "--instrument", instrument,
        ]
        if extra.get("start"):
            cmd += ["--start", extra["start"]]
        if extra.get("end"):
            cmd += ["--end", extra["end"]]
        if extra.get("orb_minutes"):
            cmd += ["--orb-minutes", str(extra["orb_minutes"])]

    elif step == "discovery":
        cmd = [
            py, str(PROJECT_ROOT / "trading_app" / "strategy_discovery.py"),
            "--instrument", instrument,
            "--db", str(db_path),
        ]
        if extra.get("start"):
            cmd += ["--start", extra["start"]]
        if extra.get("end"):
            cmd += ["--end", extra["end"]]

    elif step == "validation":
        cmd = [
            py, str(PROJECT_ROOT / "trading_app" / "strategy_validator.py"),
            "--instrument", instrument,
            "--db", str(db_path),
            "--exclude-years", "2021",
            "--min-years-positive-pct", "0.8",
            "--min-sample", str(extra.get("min_sample", 30)),
        ]

    else:
        raise ValueError(f"Unknown step: {step}")

    return cmd


def _run_step(
    instrument: str, step: str, db_path: Path, extra: dict, log_path: Path,
) -> int:
    """Run one pipeline step as a subprocess with live output streaming."""
    cmd = _build_command(step, instrument, db_path, extra)
    label = f"{instrument}/{step}"

    env = os.environ.copy()
    env["DUCKDB_PATH"] = str(db_path)

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"{label} started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"CMD: {' '.join(cmd)}\n")
        log.write(f"{'='*60}\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        reader = threading.Thread(
            target=_stream_pipe, args=(proc.stdout, label, log), daemon=True,
        )
        reader.start()
        proc.wait()
        reader.join(timeout=10)

        log.write(f"\n--- {label} exit code: {proc.returncode} ---\n")

    return proc.returncode


# ---------------------------------------------------------------------------
# Pre-clear: remove FK-dependent tables before daily_features rebuild
# ---------------------------------------------------------------------------
def _clear_downstream_tables(db_path: Path, instrument: str) -> None:
    """Delete orb_outcomes, strategies, and validated for this instrument.

    Required before daily_features rebuild because orb_outcomes has a FK
    to daily_features. Without this, DELETE FROM daily_features fails.
    Only operates on the per-instrument copy, never the master.
    """
    con = duckdb.connect(str(db_path))
    try:
        # FK-safe order: child -> parent
        con.execute(
            "DELETE FROM validated_setups_archive "
            "WHERE original_strategy_id IN "
            "(SELECT strategy_id FROM validated_setups WHERE instrument = ?)",
            [instrument],
        )
        con.execute(
            "DELETE FROM validated_setups WHERE instrument = ?", [instrument],
        )
        con.execute(
            "DELETE FROM experimental_strategies WHERE instrument = ?",
            [instrument],
        )
        n = con.execute(
            "SELECT COUNT(*) FROM orb_outcomes WHERE symbol = ?", [instrument],
        ).fetchone()[0]
        con.execute(
            "DELETE FROM orb_outcomes WHERE symbol = ?", [instrument],
        )
        con.commit()
        if n > 0:
            print(f"  [{instrument}] Cleared {n:,} orb_outcomes (FK pre-clear)")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Per-instrument pipeline (runs in a thread, calls subprocesses)
# ---------------------------------------------------------------------------
def run_instrument(
    instrument: str,
    db_path: Path,
    steps: list[str],
    extra: dict,
    log_dir: Path,
) -> dict[str, int]:
    """Run all pipeline steps for one instrument sequentially.

    Returns dict of {step_name: return_code}. Stops on first failure.
    """
    results: dict[str, int] = {}
    log_path = log_dir / f"{instrument}.log"

    # Pre-clear: if rebuilding features, clear downstream FK-dependent tables
    if "features" in steps:
        _clear_downstream_tables(db_path, instrument)

    for step in steps:
        print(f"\n>>> Starting {instrument}/{step}")
        t0 = time.time()

        rc = _run_step(instrument, step, db_path, extra, log_path)
        elapsed = time.time() - t0
        results[step] = rc

        if rc != 0:
            print(f"  FAILED: {instrument}/{step} (exit {rc}, {elapsed:.0f}s)")
            print(f"  Log: {log_path}")
            break
        else:
            print(f"  OK: {instrument}/{step} ({elapsed:.0f}s)")

    return results


# ---------------------------------------------------------------------------
# Merge: ATTACH temp DB, DELETE+INSERT per instrument
# ---------------------------------------------------------------------------
def merge_instrument(
    master_db: Path,
    instrument_db: Path,
    instrument: str,
    steps: list[str],
) -> None:
    """Merge one instrument's results back into the master DB.

    Uses DuckDB ATTACH (read-only on source) and respects FK ordering:
      Delete (child -> parent):
        archive -> validated -> experimental -> outcomes -> daily_features
      Insert (parent -> child):
        daily_features -> outcomes -> experimental -> validated
    """
    print(f"\n  Merging {instrument} into {master_db.name}...")

    con = duckdb.connect(str(master_db))
    try:
        con.execute(f"ATTACH '{instrument_db}' AS src (READ_ONLY)")

        # --- DELETE in FK-safe order (child -> parent) ---

        if "validation" in steps:
            # Archive references validated_setups via FK
            con.execute(
                "DELETE FROM validated_setups_archive "
                "WHERE original_strategy_id IN "
                "(SELECT strategy_id FROM validated_setups WHERE instrument = ?)",
                [instrument],
            )
            con.execute(
                "DELETE FROM validated_setups WHERE instrument = ?", [instrument],
            )

        if "discovery" in steps:
            con.execute(
                "DELETE FROM experimental_strategies WHERE instrument = ?",
                [instrument],
            )

        if "outcome" in steps:
            con.execute(
                "DELETE FROM orb_outcomes WHERE symbol = ?", [instrument],
            )

        if "features" in steps:
            # orb_outcomes FK -> daily_features, so outcomes must be deleted first
            # (handled above if "outcome" in steps; if not, delete outcomes too)
            if "outcome" not in steps:
                con.execute(
                    "DELETE FROM orb_outcomes WHERE symbol = ?", [instrument],
                )
            con.execute(
                "DELETE FROM daily_features WHERE symbol = ?", [instrument],
            )

        # --- INSERT in FK-safe order (parent -> child) ---

        if "features" in steps:
            n = con.execute(
                "SELECT COUNT(*) FROM src.daily_features WHERE symbol = ?",
                [instrument],
            ).fetchone()[0]
            con.execute(
                "INSERT INTO daily_features "
                "SELECT * FROM src.daily_features WHERE symbol = ?",
                [instrument],
            )
            print(f"    daily_features: {n:,} rows")

        if "outcome" in steps:
            n = con.execute(
                "SELECT COUNT(*) FROM src.orb_outcomes WHERE symbol = ?",
                [instrument],
            ).fetchone()[0]
            con.execute(
                "INSERT INTO orb_outcomes "
                "SELECT * FROM src.orb_outcomes WHERE symbol = ?",
                [instrument],
            )
            print(f"    orb_outcomes: {n:,} rows")

        if "discovery" in steps:
            n = con.execute(
                "SELECT COUNT(*) FROM src.experimental_strategies "
                "WHERE instrument = ?",
                [instrument],
            ).fetchone()[0]
            con.execute(
                "INSERT INTO experimental_strategies "
                "SELECT * FROM src.experimental_strategies WHERE instrument = ?",
                [instrument],
            )
            print(f"    experimental_strategies: {n:,} rows")

        if "validation" in steps:
            n = con.execute(
                "SELECT COUNT(*) FROM src.validated_setups WHERE instrument = ?",
                [instrument],
            ).fetchone()[0]
            con.execute(
                "INSERT INTO validated_setups "
                "SELECT * FROM src.validated_setups WHERE instrument = ?",
                [instrument],
            )
            print(f"    validated_setups: {n:,} rows")

        con.commit()
        con.execute("DETACH src")
        print(f"  OK: {instrument} merged")

    finally:
        con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel per-instrument pipeline rebuild",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/parallel_rebuild.py --all
  python scripts/parallel_rebuild.py --all --steps features outcome discovery validation
  python scripts/parallel_rebuild.py --instruments MGC MNQ --steps outcome discovery
  python scripts/parallel_rebuild.py --merge-only --instruments MGC MNQ
  python scripts/parallel_rebuild.py --all --keep-copies --dry-run
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", help="Rebuild all available instruments",
    )
    group.add_argument(
        "--instruments", nargs="+", help="Specific instruments to rebuild",
    )
    group.add_argument(
        "--merge-only", action="store_true",
        help="Skip rebuild; merge existing temp copies into master",
    )

    parser.add_argument(
        "--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS,
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--master", type=Path, default=DEFAULT_MASTER,
        help=f"Master DB path (default: {DEFAULT_MASTER})",
    )
    parser.add_argument(
        "--keep-copies", action="store_true",
        help="Keep per-instrument temp copies after merge",
    )
    parser.add_argument("--start", help="Start date for outcome_builder (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date for outcome_builder (YYYY-MM-DD)")
    parser.add_argument("--orb-minutes", type=int, default=5, help="ORB minutes")
    parser.add_argument("--min-sample", type=int, default=30, help="Min sample for validator")
    parser.add_argument("--dry-run", action="store_true", help="Show plan, don't execute")

    args = parser.parse_args()

    # -- Resolve instrument list --
    if args.all:
        from pipeline.asset_configs import list_available_instruments
        instruments = list_available_instruments()
    elif args.merge_only:
        if not REBUILD_DIR.exists():
            print(f"FATAL: No rebuild directory at {REBUILD_DIR}")
            sys.exit(1)
        instruments = sorted(
            f.stem.replace("gold_", "").upper()
            for f in REBUILD_DIR.glob("gold_*.db")
        )
        if not instruments:
            print(f"FATAL: No temp copies found in {REBUILD_DIR}")
            sys.exit(1)
    else:
        instruments = [i.upper() for i in args.instruments]

    # -- Validate master DB --
    if not args.master.exists():
        print(f"FATAL: Master DB not found: {args.master}")
        print(f"  Copy from project first:")
        print(f'  cp "C:\\canodrive\\canompx3\\gold.db" "{args.master}"')
        sys.exit(1)

    master_mb = args.master.stat().st_size / (1024 * 1024)

    print(f"Master DB:   {args.master} ({master_mb:.0f} MB)")
    print(f"Instruments: {instruments}")
    print(f"Steps:       {args.steps}")
    print(f"Rebuild dir: {REBUILD_DIR}")

    extra = {
        "start": args.start,
        "end": args.end,
        "orb_minutes": args.orb_minutes,
        "min_sample": args.min_sample,
    }

    # -- Dry run --
    if args.dry_run:
        print("\nDRY RUN -- would execute:\n")
        for inst in instruments:
            db_path = REBUILD_DIR / f"gold_{inst.lower()}.db"
            print(f"  {inst}:")
            print(f"    copy {args.master} -> {db_path}")
            for step in args.steps:
                cmd = _build_command(step, inst, db_path, extra)
                print(f"    {step}: {' '.join(cmd)}")
            print(f"    merge {db_path} -> {args.master}")
        return

    # -- PHASE 1: Create per-instrument copies --
    db_paths: dict[str, Path] = {}

    if not args.merge_only:
        REBUILD_DIR.mkdir(parents=True, exist_ok=True)
        log_dir = REBUILD_DIR / "logs"
        log_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print("PHASE 1: Creating per-instrument DB copies")
        print(f"{'='*60}")

        for inst in instruments:
            dst = REBUILD_DIR / f"gold_{inst.lower()}.db"
            print(f"  {inst}: {args.master} -> {dst} ({master_mb:.0f} MB)")
            shutil.copy2(args.master, dst)
            db_paths[inst] = dst

        # -- PHASE 2: Run rebuilds in parallel --
        print(f"\n{'='*60}")
        print(f"PHASE 2: Running {len(instruments)} instrument(s) in parallel")
        print(f"{'='*60}")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_results: dict[str, dict[str, int]] = {}

        with ThreadPoolExecutor(max_workers=len(instruments)) as executor:
            futures = {
                executor.submit(
                    run_instrument, inst, db_paths[inst], args.steps, extra, log_dir,
                ): inst
                for inst in instruments
            }

            for future in as_completed(futures):
                inst = futures[future]
                try:
                    all_results[inst] = future.result()
                except Exception as exc:
                    print(f"\n  EXCEPTION in {inst}: {exc}")
                    all_results[inst] = {s: -99 for s in args.steps}

    else:
        # merge-only mode: assume all steps succeeded
        log_dir = REBUILD_DIR / "logs"
        db_paths = {
            inst: REBUILD_DIR / f"gold_{inst.lower()}.db" for inst in instruments
        }
        all_results = {inst: {s: 0 for s in args.steps} for inst in instruments}

    # -- PHASE 3: Merge results into master --
    print(f"\n{'='*60}")
    print("PHASE 3: Merging results into master")
    print(f"{'='*60}")

    merged = []
    skipped = []

    for inst in instruments:
        results = all_results.get(inst, {})

        succeeded = [s for s in args.steps if results.get(s) == 0]
        failed = [s for s in args.steps if results.get(s, -1) != 0]

        if failed:
            print(f"\n  SKIP {inst}: failed steps {failed}")
            skipped.append(inst)
            continue

        if not succeeded:
            print(f"\n  SKIP {inst}: no successful steps")
            skipped.append(inst)
            continue

        merge_instrument(args.master, db_paths[inst], inst, succeeded)
        merged.append(inst)

    # -- PHASE 4: Cleanup --
    if not args.keep_copies:
        print(f"\nCleaning up temp copies...")
        for inst in instruments:
            p = db_paths.get(inst)
            if p and p.exists():
                p.unlink()
                print(f"  Removed: {p}")
        # Remove dirs if empty
        if log_dir.exists() and not any(log_dir.iterdir()):
            log_dir.rmdir()
        if REBUILD_DIR.exists() and not any(REBUILD_DIR.iterdir()):
            REBUILD_DIR.rmdir()
    else:
        print(f"\nTemp copies kept in: {REBUILD_DIR}")

    # -- Summary --
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for inst in instruments:
        results = all_results.get(inst, {})
        parts = []
        for step in args.steps:
            rc = results.get(step, "?")
            parts.append(f"{step}: {'OK' if rc == 0 else f'FAIL({rc})'}")
        tag = "MERGED" if inst in merged else "SKIPPED"
        print(f"  {inst} [{tag}]: {' | '.join(parts)}")

    if merged:
        print(f"\nMaster DB updated: {args.master}")
        print(f"Remember to copy back to project:")
        print(f'  cp "{args.master}" '
              f'"C:\\canodrive\\canompx3\\gold.db"')


if __name__ == "__main__":
    main()

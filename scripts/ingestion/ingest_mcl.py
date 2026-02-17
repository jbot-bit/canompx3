#!/usr/bin/env python3
"""
MCL Ingestion Pipeline - Full ingest + build on local disk.

Copies gold.db to C:\\db\\, runs the full pipeline,
then copies the result back. Crash-safe: if interrupted, re-run
with --resume to continue from checkpoint.

Steps:
  1. Copy gold.db from project to C:\\db\\gold.db
  2. Ingest MCL bars_1m from daily DBN files
  3. Build bars_5m
  4. Build daily_features
  5. Copy gold.db back to project

Usage:
    python scripts/ingest_mcl.py
    python scripts/ingest_mcl.py --resume
    python scripts/ingest_mcl.py --dry-run
    python scripts/ingest_mcl.py --skip-copy  # Already on local disk
"""

import subprocess
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Configuration ---
INSTRUMENT = "MCL"
START_DATE = "2021-07-11"
END_DATE = "2026-02-10"
CHUNK_DAYS = 50

MASTER_DB = PROJECT_ROOT / "gold.db"
LOCAL_DB = Path(r"C:\db\gold.db")
LOCAL_DB_DIR = LOCAL_DB.parent

LOG_FILE = PROJECT_ROOT / "scripts" / "ingest_mcl.log"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_step(label: str, cmd: list[str]) -> int:
    log(f"START: {label}")
    log(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={
            **__import__("os").environ,
            "DUCKDB_PATH": str(LOCAL_DB),
        },
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        log(f"  DONE: {label} ({elapsed:.0f}s)")
    else:
        log(f"  FAILED: {label} (exit {result.returncode}, {elapsed:.0f}s)")
    return result.returncode


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MCL full pipeline (local disk)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Validate only")
    parser.add_argument("--skip-copy", action="store_true", help="DB already at C:\\db\\")
    parser.add_argument("--no-copy-back", action="store_true", help="Skip copy back to project")
    args = parser.parse_args()

    log("=" * 60)
    log(f"MCL INGESTION PIPELINE")
    log(f"  Source DB: {MASTER_DB}")
    log(f"  Local DB:  {LOCAL_DB}")
    log(f"  Date range: {START_DATE} to {END_DATE}")
    log(f"  Resume: {args.resume}")
    log(f"  Dry run: {args.dry_run}")
    log("=" * 60)

    # Step 1: Copy DB to local disk
    if not args.skip_copy and not args.dry_run:
        LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
        if MASTER_DB.exists():
            log(f"Copying gold.db to {LOCAL_DB} ...")
            t0 = time.time()
            shutil.copy2(str(MASTER_DB), str(LOCAL_DB))
            sz_gb = LOCAL_DB.stat().st_size / (1024**3)
            log(f"  Copied ({sz_gb:.2f} GB, {time.time() - t0:.0f}s)")
        elif LOCAL_DB.exists():
            log(f"Master DB not found, using existing {LOCAL_DB}")
        else:
            log("FATAL: No gold.db found at master or local path")
            sys.exit(1)

    # Step 2: Ingest bars_1m from daily DBN files
    ingest_cmd = [
        sys.executable, "pipeline/ingest_dbn_daily.py",
        "--instrument", INSTRUMENT,
        "--start", START_DATE,
        "--end", END_DATE,
        f"--chunk-days={CHUNK_DAYS}",
        "--db", str(LOCAL_DB),
    ]
    if args.resume:
        ingest_cmd.append("--resume")
        ingest_cmd.append("--retry-failed")
    if args.dry_run:
        ingest_cmd.append("--dry-run")

    rc = run_step("Ingest MCL bars_1m", ingest_cmd)
    if rc != 0:
        log("ABORT: Ingestion failed. Fix issue and re-run with --resume")
        sys.exit(rc)

    if args.dry_run:
        log("DRY RUN complete. No further steps.")
        sys.exit(0)

    # Step 3: Build bars_5m
    rc = run_step("Build MCL bars_5m", [
        sys.executable, "pipeline/build_bars_5m.py",
        "--instrument", INSTRUMENT,
        "--start", START_DATE,
        "--end", END_DATE,
    ])
    if rc != 0:
        log("ABORT: bars_5m build failed")
        sys.exit(rc)

    # Step 4: Build daily_features
    rc = run_step("Build MCL daily_features", [
        sys.executable, "pipeline/build_daily_features.py",
        "--instrument", INSTRUMENT,
        "--start", START_DATE,
        "--end", END_DATE,
    ])
    if rc != 0:
        log("ABORT: daily_features build failed")
        sys.exit(rc)

    # Step 5: Copy DB back to project
    if not args.no_copy_back:
        log(f"Copying gold.db back to {MASTER_DB} ...")
        t0 = time.time()
        shutil.copy2(str(LOCAL_DB), str(MASTER_DB))
        log(f"  Copied back ({time.time() - t0:.0f}s)")

    log("=" * 60)
    log("MCL PIPELINE COMPLETE")
    log("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

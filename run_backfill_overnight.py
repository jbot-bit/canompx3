#!/usr/bin/env python3
"""
Overnight Backfill Runner - Crash-proof wrapper for ingest_dbn_mgc.py

This script:
1. Runs the ingest script with auto-resume
2. If it crashes, waits and restarts automatically
3. Continues until the full backfill is complete
4. Logs all activity to a file

Usage:
    python run_backfill_overnight.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Example (full backfill):
    python run_backfill_overnight.py

Example (partial backfill):
    python run_backfill_overnight.py --start 2020-01-01 --end 2025-12-31
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Configuration
MAX_RETRIES = 100  # Maximum restart attempts
RETRY_DELAY = 30   # Seconds to wait before restart
CHUNK_DAYS = 7     # Trading days per commit
BATCH_SIZE = 50000 # Rows per DBN read

LOG_FILE = Path(__file__).parent / "backfill_overnight.log"


def log(message: str):
    """Log message to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def run_ingest(start_date: str = None, end_date: str = None) -> int:
    """Run the ingest script with resume. Returns exit code."""

    cmd = [
        sys.executable,
        "pipeline/ingest_dbn_daily.py",
        "--resume",
        "--retry-failed",
        f"--chunk-days={CHUNK_DAYS}",
        f"--batch-size={BATCH_SIZE}",
    ]

    if start_date:
        cmd.append(f"--start={start_date}")
    if end_date:
        cmd.append(f"--end={end_date}")

    log(f"Starting: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,  # Show output in real-time
        )
        return result.returncode
    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl+C)")
        return -1
    except Exception as e:
        log(f"Exception: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Overnight backfill runner")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    log("=" * 60)
    log("OVERNIGHT BACKFILL RUNNER STARTED")
    log("=" * 60)
    log(f"Max retries: {MAX_RETRIES}")
    log(f"Retry delay: {RETRY_DELAY}s")
    log(f"Date range: {args.start or 'beginning'} to {args.end or 'end'}")
    log("")

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        log(f"--- ATTEMPT {attempt}/{MAX_RETRIES} ---")

        exit_code = run_ingest(args.start, args.end)

        if exit_code == 0:
            log("SUCCESS: Backfill completed successfully!")
            log("=" * 60)
            return 0

        if exit_code == -1:
            log("Aborted by user")
            return 1

        log(f"FAILED: Exit code {exit_code}")

        if attempt < MAX_RETRIES:
            log(f"Waiting {RETRY_DELAY}s before retry...")
            time.sleep(RETRY_DELAY)
        else:
            log(f"Max retries ({MAX_RETRIES}) reached. Giving up.")

    log("FAILED: Could not complete backfill after all retries")
    return 1


if __name__ == "__main__":
    sys.exit(main())

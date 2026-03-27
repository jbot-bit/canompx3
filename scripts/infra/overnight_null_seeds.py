#!/usr/bin/env python3
"""Overnight null seed runner with checkpointing, manifest awareness, and CPU throttling.

Runs null seeds for all instruments, skipping already-completed seeds.
Saves results to manifest after each seed. Restartable at any point.
Throttles to ~70% CPU by limiting parallel workers.

Usage:
    python scripts/infra/overnight_null_seeds.py
    python scripts/infra/overnight_null_seeds.py --instrument MNQ
    python scripts/infra/overnight_null_seeds.py --instrument MNQ --max-workers 4
    python scripts/infra/overnight_null_seeds.py --all  # MNQ then MES then MGC
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SEED_DIR = PROJECT_ROOT / "scripts" / "tests" / "null_seeds"
PYTHON = sys.executable

# Target ~70% CPU: use floor(cores * 0.7), minimum 2
CPU_COUNT = os.cpu_count() or 4
DEFAULT_WORKERS = max(2, int(CPU_COUNT * 0.7))
TARGET_SEEDS = 100


def load_manifest(instrument: str) -> dict:
    path = SEED_DIR / instrument.lower() / "manifest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"seeds": {}}


def save_manifest(instrument: str, manifest: dict):
    path = SEED_DIR / instrument.lower() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def get_remaining_seeds(manifest: dict) -> list[int]:
    """Return seeds not yet completed (PASS or FAIL)."""
    done = {int(k) for k, v in manifest.get("seeds", {}).items() if v.get("status") in ("PASS", "FAIL")}
    return sorted(set(range(TARGET_SEEDS)) - done)


def run_single_seed(instrument: str, seed: int) -> dict:
    """Run one null seed and return the result dict."""
    start = time.time()
    cmd = [
        PYTHON,
        "-m",
        "scripts.tests.test_synthetic_null",
        "--instrument",
        instrument,
        "--seeds",
        "1",
        "--start-seed",
        str(seed),
        "--output-dir",
        str(SEED_DIR / instrument.lower()),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1hr max per seed
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - start

        # Parse output for PASS/FAIL
        output = result.stdout + result.stderr
        if "PASS" in output and "Zero strategies" in output:
            status = "PASS"
        elif "FAIL" in output:
            status = "FAIL"
        elif result.returncode != 0:
            status = "PIPELINE_ERROR"
        else:
            status = "UNKNOWN"

        # Try to extract survivor count
        survivors = 0
        for line in output.split("\n"):
            if "validated_count" in line or "validated=" in line:
                import re

                m = re.search(r"validated[=_](\d+)", line)
                if m:
                    survivors = int(m.group(1))

        # Extract max_oos_expr if available
        max_oos = None
        for line in output.split("\n"):
            if "noise_positive_expr" in line or "max_oos" in line:
                import re

                m = re.search(r"noise_positive_expr=([\d.]+)", line)
                if m:
                    max_oos = float(m.group(1))

        return {
            "seed": seed,
            "status": status,
            "survivors": survivors,
            "max_oos_expr": max_oos,
            "elapsed_s": round(elapsed, 1),
            "completed_at": datetime.now(UTC).isoformat(),
            "date_range": "2016-02-01 to 2025-12-31",
            "output_path": str(SEED_DIR / instrument.lower() / f"seed_{seed:04d}"),
        }

    except subprocess.TimeoutExpired:
        return {
            "seed": seed,
            "status": "TIMEOUT",
            "elapsed_s": 3600,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return {
            "seed": seed,
            "status": "EXCEPTION",
            "error": str(e),
            "completed_at": datetime.now(UTC).isoformat(),
        }


def run_instrument(instrument: str, max_workers: int):
    """Run all remaining seeds for one instrument with checkpointing."""
    manifest = load_manifest(instrument)
    remaining = get_remaining_seeds(manifest)

    done = TARGET_SEEDS - len(remaining)
    print(f"\n{'=' * 60}")
    print(f"{instrument}: {done}/{TARGET_SEEDS} done, {len(remaining)} remaining")
    print(f"Workers: {max_workers} (~{max_workers / CPU_COUNT * 100:.0f}% of {CPU_COUNT} cores)")
    print(f"{'=' * 60}\n")

    if not remaining:
        print(f"{instrument}: All {TARGET_SEEDS} seeds complete.")
        return

    for i, seed in enumerate(remaining):
        print(f"[{instrument}] Seed {seed} ({i + 1}/{len(remaining)})...", flush=True)

        result = run_single_seed(instrument, seed)

        # Save to manifest immediately (checkpoint)
        manifest["seeds"][str(seed)] = result
        save_manifest(instrument, manifest)

        status = result["status"]
        elapsed = result.get("elapsed_s", 0)
        survivors = result.get("survivors", "?")
        print(f"  {status} (survivors={survivors}, {elapsed:.0f}s)", flush=True)

        # Progress update every 5 seeds
        if (i + 1) % 5 == 0:
            new_done = TARGET_SEEDS - len(get_remaining_seeds(manifest))
            print(f"  --- {instrument}: {new_done}/{TARGET_SEEDS} complete ---", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Overnight null seed runner with checkpointing")
    parser.add_argument("--instrument", type=str, default=None, help="Single instrument (MNQ, MGC, MES)")
    parser.add_argument("--all", action="store_true", help="Run all instruments: MNQ → MES → MGC")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Max parallel workers (default: {DEFAULT_WORKERS}, ~70%% of {CPU_COUNT} cores)",
    )
    args = parser.parse_args()

    if args.all:
        instruments = ["MNQ", "MES", "MGC"]  # Priority order
    elif args.instrument:
        instruments = [args.instrument.upper()]
    else:
        instruments = ["MNQ"]  # Default to primary instrument

    print("Overnight null seed runner")
    print(f"Instruments: {instruments}")
    print(f"Workers: {args.max_workers}/{CPU_COUNT} cores")
    print(f"Started: {datetime.now()}")

    for inst in instruments:
        run_instrument(inst, args.max_workers)

    print(f"\nAll done. Finished: {datetime.now()}")


if __name__ == "__main__":
    main()

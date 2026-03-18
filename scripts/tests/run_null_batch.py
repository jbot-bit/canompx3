#!/usr/bin/env python3
"""
Batch null test runner — 50-seed White's Reality Check with proper infrastructure.

Handles:
  1. Checks prerequisites (noise floor must be zeroed, T80 must be removed)
  2. Runs N seeds sequentially, saving each DB to a permanent location
  3. After all seeds complete, runs envelope analysis and reports results
  4. Does NOT auto-update config — that's a manual decision after reviewing results

IMPORTANT: You must manually set NOISE_EXPR_FLOOR = {"E1": 0, "E2": 0} in
trading_app/config.py BEFORE running. The script aborts if floors are active.

Usage:
    python scripts/tests/run_null_batch.py                     # 50 seeds, all apertures
    python scripts/tests/run_null_batch.py --seeds 20           # quick run
    python scripts/tests/run_null_batch.py --seeds 50 --batch 1 --of 3  # parallel: terminal 1 of 3
    python scripts/tests/run_null_batch.py --analyze-only       # just analyze existing seed DBs

Seed assignment for parallel terminals:
    Terminal 1: --batch 1 --of 3  → seeds 0-16
    Terminal 2: --batch 2 --of 3  → seeds 17-33
    Terminal 3: --batch 3 --of 3  → seeds 34-49

Expected runtime: ~30 min/seed. 50 seeds / 3 terminals ≈ 8-9 hours.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEED_DIR = PROJECT_ROOT / "scripts" / "tests" / "null_seeds"
MANIFEST_PATH = SEED_DIR / "manifest.json"


def load_manifest() -> dict:
    """Load or create the seed manifest."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"seeds": {}, "config_snapshot": {}, "created": datetime.now(UTC).isoformat()}


def save_manifest(manifest: dict) -> None:
    """Save the seed manifest."""
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))


def check_prerequisites() -> list[str]:
    """Check that the pipeline is in the right state for null testing."""
    issues = []

    # Check noise floor is disabled (or will be)
    try:
        from trading_app.config import NOISE_EXPR_FLOOR
        for em, floor in NOISE_EXPR_FLOOR.items():
            if floor > 0.01:
                issues.append(
                    f"NOISE_EXPR_FLOOR[{em}]={floor} — must be 0 during seed runs. "
                    f"Noise floor gate will reject all noise survivors, making calibration impossible."
                )
    except ImportError:
        issues.append("Cannot import trading_app.config — check PYTHONPATH")

    # Check T80/early exit status
    try:
        from trading_app.config import EARLY_EXIT_MINUTES
        active_exits = {k: v for k, v in EARLY_EXIT_MINUTES.items() if v is not None}
        if active_exits:
            issues.append(
                f"EARLY_EXIT_MINUTES has active entries: {active_exits}. "
                f"If T80 is being removed, do that BEFORE running seeds."
            )
    except (ImportError, AttributeError):
        pass  # No early exit config — fine

    return issues


def get_seed_range(total_seeds: int, batch: int, of: int) -> tuple[int, int]:
    """Compute seed range for parallel batch execution."""
    per_batch = total_seeds // of
    remainder = total_seeds % of
    start = (batch - 1) * per_batch + min(batch - 1, remainder)
    end = start + per_batch + (1 if batch <= remainder else 0)
    return start, end


def run_seed(seed: int, output_dir: Path) -> dict:
    """Run one null test seed, saving DB to permanent location."""
    db_dir = output_dir / f"seed_{seed:04d}"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "null_test.db"

    # Skip if already completed
    if db_path.exists():
        import duckdb
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            n = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
            con.close()
            print(f"  Seed {seed}: ALREADY DONE ({n} survivors)")
            return {"seed": seed, "status": "CACHED", "survivors": n}
        except Exception:
            # Corrupt or incomplete — delete and rerun
            shutil.rmtree(db_dir, ignore_errors=True)
            db_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"# SEED {seed}")
    print(f"{'#'*60}")

    # Run the null test with --output-dir for direct permanent storage
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "tests" / "test_synthetic_null.py"),
            "--seeds", "1",
            "--start-seed", str(seed),
            "--output-dir", str(output_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=7200,  # 2 hour timeout per seed
    )

    elapsed = time.time() - t0

    # Count survivors
    survivors = -1
    if db_path.exists():
        import duckdb
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            survivors = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
            con.close()
        except Exception:
            survivors = -1

    status = "PASS" if survivors == 0 else "FAIL" if survivors > 0 else "ERROR"
    print(f"  Seed {seed}: {status} ({survivors} survivors, {elapsed:.0f}s)")

    if result.returncode != 0 and survivors < 0:
        print(f"  STDERR (last 500 chars): {result.stderr[-500:]}")
        status = "ERROR"

    return {
        "seed": seed,
        "status": status,
        "survivors": survivors,
        "elapsed_s": round(elapsed, 1),
        "completed_at": datetime.now(UTC).isoformat(),
    }


def analyze_seeds(seed_dir: Path) -> None:
    """Analyze all completed seed DBs and report the envelope."""
    import duckdb
    import numpy as np

    seed_dirs = sorted(seed_dir.glob("seed_*"))
    results = []

    for sd in seed_dirs:
        db = sd / "null_test.db"
        if not db.exists():
            continue
        try:
            con = duckdb.connect(str(db), read_only=True)
            rows = con.execute(
                "SELECT entry_model, expectancy_r, sharpe_ratio, sharpe_ann FROM validated_setups"
            ).fetchall()
            con.close()
            seed_num = int(sd.name.split("_")[1])
            results.append({"seed": seed_num, "survivors": rows})
        except Exception as e:
            print(f"  {sd.name}: ERROR reading — {e}")

    if not results:
        print("No completed seed DBs found.")
        return

    # Flatten
    all_survivors = {"E1": [], "E2": []}
    seed_maxima = {"E1": [], "E2": []}

    for r in results:
        e1 = [row[1] for row in r["survivors"] if row[0] == "E1"]
        e2 = [row[1] for row in r["survivors"] if row[0] == "E2"]
        all_survivors["E1"].extend(e1)
        all_survivors["E2"].extend(e2)
        if e1:
            seed_maxima["E1"].append(max(e1))
        if e2:
            seed_maxima["E2"].append(max(e2))

    n_seeds = len(results)
    total = sum(len(r["survivors"]) for r in results)

    print(f"\n{'='*60}")
    print(f"NULL ENVELOPE — {n_seeds} SEEDS, {total} TOTAL SURVIVORS")
    print(f"{'='*60}")

    for em in ["E1", "E2"]:
        subs = all_survivors[em]
        maxima = seed_maxima[em]
        if not subs:
            print(f"\n  {em}: 0 survivors")
            continue

        exprs = np.array(subs)
        raw_max = float(np.max(exprs))
        floor = float(np.ceil(raw_max * 100) / 100)

        print(f"\n  {em}: {len(subs)} survivors across {len(maxima)} seeds")
        print(f"    Max ExpR:    {raw_max:.4f}")
        print(f"    P95 ExpR:    {float(np.percentile(exprs, 95)):.4f}")
        print(f"    P90 ExpR:    {float(np.percentile(exprs, 90)):.4f}")
        print(f"    Median:      {float(np.median(exprs)):.4f}")
        print(f"    -> FLOOR (ceil of max): {floor}")

        # Per-seed maxima stats
        if len(maxima) >= 5:
            mx = np.array(maxima)
            print(f"    Per-seed max: mean={np.mean(mx):.4f}, std={np.std(mx, ddof=1):.4f}, "
                  f"min={np.min(mx):.4f}, max={np.max(mx):.4f}")
            # Parametric 95% CI on the true max (mean + 2*std)
            conservative = float(np.mean(mx) + 2 * np.std(mx, ddof=1))
            print(f"    Conservative (mean+2std of maxima): {conservative:.4f}")
            print(f"    -> CONSERVATIVE FLOOR: {float(np.ceil(conservative * 100) / 100)}")

    print(f"\n  RECOMMENDED: Update NOISE_EXPR_FLOOR in trading_app/config.py")
    print(f"  Then run: python scripts/tools/null_envelope.py --analyze scripts/tests/null_seeds --update-config")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch null test runner (White's Reality Check)")
    parser.add_argument("--seeds", type=int, default=50, help="Total seeds to run (default: 50)")
    parser.add_argument("--batch", type=int, default=1, help="Which batch (for parallel terminals)")
    parser.add_argument("--of", type=int, default=1, help="Total batches (for parallel terminals)")
    parser.add_argument("--analyze-only", action="store_true", help="Skip running, just analyze existing DBs")
    parser.add_argument("--check", action="store_true", help="Check prerequisites only, don't run")
    parser.add_argument("--force", action="store_true", help="Override prerequisite failures (results may be invalid)")
    args = parser.parse_args()

    SEED_DIR.mkdir(parents=True, exist_ok=True)

    # Prerequisites check
    issues = check_prerequisites()
    if issues:
        print("PREREQUISITE ISSUES:")
        for issue in issues:
            print(f"  FAIL: {issue}")
        if args.check:
            return 1
        if not args.force:
            print()
            print("ABORTED — fix prerequisites or use --force to override.")
            return 1
        print()
        print("--force: proceeding despite prerequisite failures. Results may be invalid.")
        print()

    if args.check:
        print("All prerequisites OK.")
        return 0

    if args.analyze_only:
        analyze_seeds(SEED_DIR)
        return 0

    # Compute seed range for this batch
    start, end = get_seed_range(args.seeds, args.batch, args.of)
    seed_range = list(range(start, end))

    print(f"{'='*60}")
    print(f"NULL TEST BATCH RUNNER")
    print(f"{'='*60}")
    print(f"  Total seeds:  {args.seeds}")
    print(f"  This batch:   {args.batch} of {args.of} (seeds {start}-{end-1})")
    print(f"  Output dir:   {SEED_DIR}")
    print(f"  Seeds to run: {len(seed_range)}")
    print()

    # Snapshot current config for the manifest
    manifest = load_manifest()
    try:
        from trading_app.config import NOISE_EXPR_FLOOR
        manifest["config_snapshot"]["noise_floors"] = dict(NOISE_EXPR_FLOOR)
    except ImportError:
        pass

    t_batch_start = time.time()

    for i, seed in enumerate(seed_range):
        print(f"\n[{i+1}/{len(seed_range)}] Running seed {seed}...")
        result = run_seed(seed, SEED_DIR)
        manifest["seeds"][str(seed)] = result
        save_manifest(manifest)

    batch_time = time.time() - t_batch_start
    print(f"\nBatch complete: {len(seed_range)} seeds in {batch_time:.0f}s ({batch_time/3600:.1f}h)")

    # Auto-analyze
    analyze_seeds(SEED_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())

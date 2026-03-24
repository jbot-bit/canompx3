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
    python scripts/tests/run_null_batch.py                      # 100 seeds, 8 parallel
    python scripts/tests/run_null_batch.py --seeds 100 --parallel 10  # 10 parallel workers
    python scripts/tests/run_null_batch.py --seeds 20           # quick run
    python scripts/tests/run_null_batch.py --analyze-only       # just analyze existing seed DBs

Hardware: i9-14900HX (24C/32T), 32GB RAM, NVMe SSD.
Each seed uses ~1 core + ~2GB RAM. Default 8 parallel = 16GB peak, safe.

Expected runtime: 100 seeds / 8 parallel x 30 min = ~6.25 hours.
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
BASE_SEED_DIR = PROJECT_ROOT / "scripts" / "tests" / "null_seeds"
# Per-instrument calibration: sigma from empirical 1m return std (p99 trimmed)
INSTRUMENT_NULL_PARAMS: dict[str, dict] = {
    "MGC": {"sigma": 1.2, "start_price": 2000.0, "tick_size": 0.10},
    "MNQ": {"sigma": 5.0, "start_price": 20000.0, "tick_size": 0.25},
    "MES": {"sigma": 1.1, "start_price": 5000.0, "tick_size": 0.25},
}

# C1 time-varying null: per-year trimmed_std from real bars_1m.
# Each year's sigma derived from that year's own data only. No cross-year pooling.
# 2026 excluded (holdout sacred).
# NOTE: Pre-holdout-era null audit. If reused in WF, sigma must be IS-only per fold.
# Calibrated via: python scripts/tools/calibrate_null_sigma.py --per-year
# Default null range is 2020-2025. Years 2016-2019 included for --start-date override.
SIGMA_BY_YEAR: dict[str, dict[int, float]] = {
    "MGC": {
        2016: 0.29, 2017: 0.20, 2018: 0.20, 2019: 0.24,
        2020: 0.53, 2021: 0.37, 2022: 0.41, 2023: 0.36,
        2024: 0.51, 2025: 0.99,
    },
}


def _seed_dir(instrument: str) -> Path:
    """Per-instrument seed directory."""
    return BASE_SEED_DIR / instrument.lower()


def load_manifest(seed_dir: Path) -> dict:
    """Load or create the seed manifest."""
    manifest_path = seed_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"seeds": {}, "config_snapshot": {}, "created": datetime.now(UTC).isoformat()}


def save_manifest(manifest: dict, seed_dir: Path) -> None:
    """Save the seed manifest."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


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


def run_seed(
    seed: int,
    output_dir: Path,
    instrument: str = "MGC",
    sigma: float | None = None,
    sigma_by_year: dict[int, float] | None = None,
) -> dict:
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
            dr = con.execute(
                "SELECT min(trading_day)::VARCHAR, max(trading_day)::VARCHAR FROM orb_outcomes"
            ).fetchone()
            cached_range = f"{dr[0]} to {dr[1]}" if dr[0] else None
            cached_max_expr = None
            if n > 0:
                row = con.execute("SELECT MAX(expectancy_r) FROM validated_setups").fetchone()
                cached_max_expr = round(float(row[0]), 4) if row[0] is not None else None
            con.close()
            print(f"  Seed {seed}: ALREADY DONE ({n} survivors, range={cached_range})")
            return {
                "seed": seed, "status": "CACHED", "survivors": n,
                "max_oos_expr": cached_max_expr, "date_range": cached_range,
                "output_path": str(db_dir),
            }
        except Exception:
            # Corrupt or incomplete — delete and rerun
            shutil.rmtree(db_dir, ignore_errors=True)
            db_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"# SEED {seed} ({instrument})")
    print(f"{'#'*60}")

    # Run the null test with --output-dir for direct permanent storage
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "tests" / "test_synthetic_null.py"),
        "--seeds", "1",
        "--start-seed", str(seed),
        "--output-dir", str(output_dir),
        "--instrument", instrument,
    ]
    if sigma_by_year:
        import json as _json
        cmd += ["--sigma-by-year", _json.dumps({str(k): v for k, v in sigma_by_year.items()})]
    elif sigma is not None:
        cmd += ["--sigma", str(sigma)]

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=7200,  # 2 hour timeout per seed
    )

    elapsed = time.time() - t0

    # Count survivors and verify date range
    survivors = -1
    date_range = None
    max_oos_expr = None
    if db_path.exists():
        import duckdb
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            survivors = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
            # Date range from orb_outcomes (the binding constraint)
            dr = con.execute(
                "SELECT min(trading_day)::VARCHAR, max(trading_day)::VARCHAR FROM orb_outcomes"
            ).fetchone()
            date_range = f"{dr[0]} to {dr[1]}" if dr[0] else None
            # Max OOS ExpR from survivors
            if survivors > 0:
                row = con.execute(
                    "SELECT MAX(expectancy_r) FROM validated_setups"
                ).fetchone()
                max_oos_expr = round(float(row[0]), 4) if row[0] is not None else None
            con.close()
        except Exception:
            survivors = -1

    # Fail-closed date range guard
    if date_range and date_range != "2020-01-01 to 2025-12-31":
        print(f"  FAIL-CLOSED: date range is {date_range}, expected 2020-01-01 to 2025-12-31")
        shutil.rmtree(db_dir, ignore_errors=True)
        return {
            "seed": seed,
            "status": "DATE_RANGE_ERROR",
            "survivors": -1,
            "date_range": date_range,
            "elapsed_s": round(elapsed, 1),
            "completed_at": datetime.now(UTC).isoformat(),
        }

    status = "PASS" if survivors == 0 else "FAIL" if survivors > 0 else "ERROR"
    print(f"  Seed {seed}: {status} ({survivors} survivors, {elapsed:.0f}s)")

    if result.returncode != 0 and survivors < 0:
        print(f"  STDERR (last 500 chars): {result.stderr[-500:]}")
        status = "ERROR"

    return {
        "seed": seed,
        "status": status,
        "survivors": survivors,
        "max_oos_expr": max_oos_expr,
        "date_range": date_range,
        "output_path": str(db_dir),
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
    print(f"  Then run: python scripts/tools/null_envelope.py --analyze {seed_dir} --update-config")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch null test runner (White's Reality Check)")
    parser.add_argument("--seeds", type=int, default=100, help="Total seeds to run (default: 100)")
    parser.add_argument("--parallel", type=int, default=8, help="Max parallel seeds (default: 8, safe for 32GB RAM)")
    parser.add_argument("--batch", type=int, default=1, help="Which batch (for parallel terminals)")
    parser.add_argument("--of", type=int, default=1, help="Total batches (for parallel terminals)")
    parser.add_argument("--instrument", type=str, default="MGC", help="Instrument (default: MGC). Calibrates sigma/price/tick.")
    parser.add_argument("--sigma", type=float, default=None, help="Override flat sigma (default: auto from INSTRUMENT_NULL_PARAMS)")
    parser.add_argument(
        "--time-varying",
        action="store_true",
        help="Use C1 time-varying null: per-year sigma from SIGMA_BY_YEAR. Overrides --sigma.",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Skip running, just analyze existing DBs")
    parser.add_argument("--check", action="store_true", help="Check prerequisites only, don't run")
    parser.add_argument("--force", action="store_true", help="Override prerequisite failures (results may be invalid)")
    args = parser.parse_args()

    instrument = args.instrument.upper()
    params = INSTRUMENT_NULL_PARAMS.get(instrument, INSTRUMENT_NULL_PARAMS["MGC"])
    sigma = args.sigma if args.sigma is not None else params["sigma"]

    # C1 time-varying null: use per-year sigma if available and requested
    sigma_by_year_dict = None
    if args.time_varying:
        sigma_by_year_dict = SIGMA_BY_YEAR.get(instrument)
        if not sigma_by_year_dict:
            print(f"ERROR: No SIGMA_BY_YEAR for {instrument}. Add to run_null_batch.py or use --sigma.")
            return 1
        print(f"C1 time-varying null: per-year sigma for {instrument}")
        for yr in sorted(sigma_by_year_dict):
            print(f"  {yr}: {sigma_by_year_dict[yr]}")
        print()

    SEED_DIR = _seed_dir(instrument)
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

    n_parallel = min(args.parallel, len(seed_range))

    print(f"{'='*60}")
    print(f"NULL TEST BATCH RUNNER — {instrument}")
    print(f"{'='*60}")
    print(f"  Instrument:   {instrument}")
    if sigma_by_year_dict:
        print(f"  Sigma:        C1 time-varying (per-year trimmed_std)")
    else:
        print(f"  Sigma:        {sigma} (calibrated {'auto' if args.sigma is None else 'manual'})")
    print(f"  Total seeds:  {args.seeds}")
    print(f"  This batch:   {args.batch} of {args.of} (seeds {start}-{end-1})")
    print(f"  Parallel:     {n_parallel} workers")
    print(f"  Output dir:   {SEED_DIR}")
    print(f"  Seeds to run: {len(seed_range)}")
    est_hours = len(seed_range) * 30 / 60 / n_parallel
    print(f"  ETA:          ~{est_hours:.1f} hours ({len(seed_range)} seeds / {n_parallel} parallel x ~30min)")
    print()

    # Snapshot current config for the manifest
    manifest = load_manifest(SEED_DIR)
    manifest["config_snapshot"]["instrument"] = instrument
    manifest["config_snapshot"]["sigma"] = sigma if not sigma_by_year_dict else "C1_time_varying"
    if sigma_by_year_dict:
        manifest["config_snapshot"]["sigma_by_year"] = {str(k): v for k, v in sigma_by_year_dict.items()}
    try:
        from trading_app.config import NOISE_EXPR_FLOOR
        manifest["config_snapshot"]["noise_floors"] = dict(NOISE_EXPR_FLOOR)
    except ImportError:
        pass

    t_batch_start = time.time()

    if n_parallel <= 1:
        # Sequential fallback
        for i, seed in enumerate(seed_range):
            print(f"\n[{i+1}/{len(seed_range)}] Running seed {seed}...")
            result = run_seed(seed, SEED_DIR, instrument=instrument, sigma=sigma, sigma_by_year=sigma_by_year_dict)
            manifest["seeds"][str(seed)] = result
            save_manifest(manifest, SEED_DIR)
    else:
        # Parallel execution
        from concurrent.futures import ProcessPoolExecutor, as_completed

        completed = 0
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {
                executor.submit(run_seed, seed, SEED_DIR, instrument, sigma, sigma_by_year_dict): seed
                for seed in seed_range
            }
            for future in as_completed(futures):
                seed = futures[future]
                completed += 1
                try:
                    result = future.result()
                    manifest["seeds"][str(seed)] = result
                    survivors = result.get("survivors", "?")
                    elapsed = result.get("elapsed_s", 0)
                    print(f"  [{completed}/{len(seed_range)}] Seed {seed}: {survivors} survivors ({elapsed:.0f}s)")
                except Exception as e:
                    print(f"  [{completed}/{len(seed_range)}] Seed {seed}: EXCEPTION — {e}")
                    manifest["seeds"][str(seed)] = {"seed": seed, "status": "EXCEPTION", "error": str(e)}

                # Save manifest after EVERY seed — no data loss on crash
                save_manifest(manifest, SEED_DIR)

        save_manifest(manifest, SEED_DIR)

    batch_time = time.time() - t_batch_start
    print(f"\nBatch complete: {len(seed_range)} seeds in {batch_time:.0f}s ({batch_time/3600:.1f}h)")

    # Auto-analyze
    analyze_seeds(SEED_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())

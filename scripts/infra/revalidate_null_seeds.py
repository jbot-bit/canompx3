#!/usr/bin/env python3
"""Re-validate null seed DBs with current validator code (stratified-K).

Does NOT regenerate bars/features/outcomes — only clears and reruns the
validation step on existing experimental_strategies. ~1-2 min per seed
vs ~30 min for full pipeline.

Checkpoints to manifest after each seed. Restartable.

Usage:
    python scripts/infra/revalidate_null_seeds.py --instrument MNQ
    python scripts/infra/revalidate_null_seeds.py --all
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SEED_DIR = PROJECT_ROOT / "scripts" / "tests" / "null_seeds"
PYTHON = sys.executable

# Real production per-session K values (from canonical gold.db, Mar 25 2026).
# Used to override the auto-counted K in null seed DBs, which only have one
# instrument and therefore undercount K by 4-8x.
# Source: SELECT orb_label, COUNT(*) FROM experimental_strategies
#         WHERE is_canonical AND p_value IS NOT NULL
#         AND instrument IN ('MNQ','MGC','MES') GROUP BY orb_label
PRODUCTION_SESSION_K = {
    "BRISBANE_1025": 1512,
    "CME_PRECLOSE": 3254,
    "CME_REOPEN": 12104,
    "COMEX_SETTLE": 5183,
    "EUROPE_FLOW": 4536,
    "LONDON_METALS": 14760,
    "NYSE_CLOSE": 3587,
    "NYSE_OPEN": 5184,
    "SINGAPORE_OPEN": 4752,
    "TOKYO_OPEN": 13032,
    "US_DATA_1000": 5184,
    "US_DATA_830": 5184,
}

# Path to the K override file (written at runtime, read by validator)
K_OVERRIDE_FILE = SEED_DIR / "_production_k_overrides.json"


def load_manifest(instrument: str) -> dict:
    path = SEED_DIR / instrument.lower() / "manifest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"seeds": {}}


def save_manifest(instrument: str, manifest: dict):
    path = SEED_DIR / instrument.lower() / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def revalidate_seed(instrument: str, seed_num: int) -> dict:
    """Clear and re-run validation on one seed DB."""
    import duckdb

    seed_dir = SEED_DIR / instrument.lower() / f"seed_{seed_num:04d}"
    db_path = seed_dir / "null_test.db"

    if not db_path.exists():
        return {"seed": seed_num, "status": "MISSING_DB"}

    start = time.time()

    try:
        # Clear old validation
        con = duckdb.connect(str(db_path))
        con.execute("DELETE FROM validated_setups")
        con.execute(
            "UPDATE experimental_strategies SET validation_status = NULL, validation_notes = NULL"
        )
        con.close()

        # Write K override file (production K values, not null DB K)
        with open(K_OVERRIDE_FILE, "w") as kf:
            json.dump(PRODUCTION_SESSION_K, kf)

        # Re-run validator with current code (stratified K + production K override)
        subprocess.run(
            [
                PYTHON,
                "-m",
                "trading_app.strategy_validator",
                "--instrument",
                instrument,
                "--db",
                str(db_path),
                "--fdr-k-file",
                str(K_OVERRIDE_FILE),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        elapsed = time.time() - start

        # Read result
        con = duckdb.connect(str(db_path), read_only=True)
        val_count = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]

        # Get max ExpR from survivors (for noise floor calculation)
        max_expr = None
        if val_count > 0:
            r = con.execute(
                "SELECT MAX(expectancy_r) FROM validated_setups"
            ).fetchone()
            max_expr = r[0] if r else None

        con.close()

        return {
            "seed": seed_num,
            "status": "PASS" if val_count == 0 else "FAIL",
            "survivors": val_count,
            "max_oos_expr": max_expr,
            "elapsed_s": round(elapsed, 1),
            "completed_at": datetime.now(UTC).isoformat(),
            "fdr_method": "stratified-K-by-session",
            "output_path": str(seed_dir),
        }

    except subprocess.TimeoutExpired:
        return {"seed": seed_num, "status": "TIMEOUT", "elapsed_s": 600}
    except Exception as e:
        return {"seed": seed_num, "status": "ERROR", "error": str(e)}


def run_instrument(instrument: str):
    manifest = load_manifest(instrument)

    # Find seeds that need re-validation (have DBs but used old global K)
    needs_revalidation = []
    for seed_dir in sorted((SEED_DIR / instrument.lower()).glob("seed_*")):
        seed_num = int(seed_dir.name.replace("seed_", ""))
        db_path = seed_dir / "null_test.db"
        if not db_path.exists():
            continue

        existing = manifest.get("seeds", {}).get(str(seed_num), {})
        if existing.get("fdr_method") == "stratified-K-by-session":
            continue  # Already re-validated

        needs_revalidation.append(seed_num)

    print(f"\n{'='*60}")
    print(f"{instrument}: {len(needs_revalidation)} seeds need re-validation")
    print(f"{'='*60}\n")

    if not needs_revalidation:
        print("All seeds already validated with stratified-K.")
        return

    for i, seed_num in enumerate(needs_revalidation):
        print(f"[{instrument}] Seed {seed_num} ({i+1}/{len(needs_revalidation)})...", end=" ", flush=True)

        result = revalidate_seed(instrument, seed_num)

        # Checkpoint
        manifest.setdefault("seeds", {})[str(seed_num)] = result
        save_manifest(instrument, manifest)

        status = result["status"]
        survivors = result.get("survivors", "?")
        elapsed = result.get("elapsed_s", 0)
        max_expr = result.get("max_oos_expr")
        expr_str = f" max_expr={max_expr:.4f}" if max_expr else ""
        print(f"{status} (survivors={survivors}{expr_str}, {elapsed:.0f}s)", flush=True)

        if (i + 1) % 10 == 0:
            done_strat = sum(
                1
                for v in manifest["seeds"].values()
                if v.get("fdr_method") == "stratified-K-by-session"
            )
            print(f"  --- {instrument}: {done_strat} re-validated ---", flush=True)

    # Final summary
    seeds = manifest.get("seeds", {})
    total = len(seeds)
    passes = sum(1 for v in seeds.values() if v.get("status") == "PASS")
    fails = sum(1 for v in seeds.values() if v.get("status") == "FAIL")
    stratified = sum(1 for v in seeds.values() if v.get("fdr_method") == "stratified-K-by-session")
    print(f"\n{instrument} DONE: {total} seeds, PASS={passes}, FAIL={fails}, stratified-K={stratified}")


def main():
    parser = argparse.ArgumentParser(description="Re-validate null seeds with stratified-K")
    parser.add_argument("--instrument", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="MNQ, MGC, MES")
    args = parser.parse_args()

    if args.all:
        instruments = ["MNQ", "MGC", "MES"]
    elif args.instrument:
        instruments = [args.instrument.upper()]
    else:
        instruments = ["MNQ"]

    print("Null seed re-validation (stratified-K)")
    print(f"Instruments: {instruments}")
    print(f"Started: {datetime.now()}")

    for inst in instruments:
        run_instrument(inst)

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()

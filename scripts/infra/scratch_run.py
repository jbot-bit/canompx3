"""Run a long job on a scratch copy of gold.db, then swap back on success.

Protects your production gold.db during long-running operations.
If the job crashes, gold.db is untouched. If it succeeds, the scratch
copy replaces gold.db via a single copy operation.

Usage:
    python scripts/infra/scratch_run.py -- python trading_app/strategy_discovery.py --instrument MGC
    python scripts/infra/scratch_run.py -- python trading_app/outcome_builder.py --instrument MGC
    python scripts/infra/scratch_run.py --scratch C:/db/gold.db -- python pipeline/build_daily_features.py --instrument MGC
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_GOLD = PROJECT_ROOT / "gold.db"
DEFAULT_SCRATCH = Path("C:/db/gold.db")


def run_on_scratch(cmd: list[str], gold: Path, scratch: Path) -> int:
    """Copy gold.db to scratch, run cmd with DUCKDB_PATH override, swap back on success."""
    if not gold.exists():
        print(f"FATAL: Source database not found: {gold}")
        return 1

    # Ensure scratch directory exists
    scratch.parent.mkdir(parents=True, exist_ok=True)

    # Copy to scratch
    print(f"Copying {gold} -> {scratch} ({gold.stat().st_size / 1024 / 1024:.0f} MB)...")
    shutil.copy2(gold, scratch)
    print("Copy complete.")

    # Run command with DUCKDB_PATH pointing to scratch
    env = {**os.environ, "DUCKDB_PATH": str(scratch)}
    print(f"Running: {' '.join(cmd)}")
    print(f"DUCKDB_PATH={scratch}")
    print()

    result = subprocess.run(cmd, env=env)

    if result.returncode == 0:
        print()
        print(f"SUCCESS — copying {scratch} -> {gold}...")
        shutil.copy2(scratch, gold)
        print("gold.db updated.")
        return 0
    else:
        print()
        print(f"FAILED (rc={result.returncode}) — gold.db unchanged.")
        return result.returncode


def main():
    # Split args on '--'
    if "--" not in sys.argv:
        print("Usage: python scripts/infra/scratch_run.py [--scratch PATH] -- <command>")
        print("Example: python scripts/infra/scratch_run.py -- python trading_app/strategy_discovery.py --instrument MGC")
        sys.exit(1)

    sep_idx = sys.argv.index("--")
    our_args = sys.argv[1:sep_idx]
    cmd = sys.argv[sep_idx + 1:]

    if not cmd:
        print("FATAL: No command specified after '--'")
        sys.exit(1)

    # Parse our args
    gold = DEFAULT_GOLD
    scratch = DEFAULT_SCRATCH

    i = 0
    while i < len(our_args):
        if our_args[i] == "--scratch" and i + 1 < len(our_args):
            scratch = Path(our_args[i + 1])
            i += 2
        elif our_args[i] == "--gold" and i + 1 < len(our_args):
            gold = Path(our_args[i + 1])
            i += 2
        else:
            print(f"Unknown arg: {our_args[i]}")
            sys.exit(1)

    rc = run_on_scratch(cmd, gold, scratch)
    sys.exit(rc)


if __name__ == "__main__":
    main()

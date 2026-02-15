# Canonical Pipeline Fixes — Priority 1-3

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Defuse destructive ingest landmine, extend drift guardrails, create single end-to-end pipeline command.

**Architecture:** Three independent fixes. Task 1 edits `scripts/run_parallel_ingest.py` to remove silent table drops. Task 2 adds `ingest_dbn_daily.py` to drift check coverage. Task 3 creates new `pipeline/run_full_pipeline.py` extending existing `run_pipeline.py` with strategy stages.

**Tech Stack:** Python, DuckDB, subprocess orchestration. No new dependencies.

---

### Task 1: Defuse destructive reset in run_parallel_ingest.py

**Files:**
- Modify: `scripts/run_parallel_ingest.py:69-84`
- Test: `tests/test_pipeline/test_parallel_ingest_safety.py` (create)

**Context:** Lines 76-84 of `merge_all()` drop `validated_setups_archive`, `validated_setups`, `experimental_strategies`, `orb_outcomes` then call `init_db(force=True)`. This silently wipes all trading_app work whenever someone re-ingests bars. Also misses `strategy_trade_days` and `edge_families` tables.

**Step 1: Write the failing test**

```python
"""Tests that run_parallel_ingest merge does NOT touch trading_app tables."""
import sys
from pathlib import Path
import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_merge_does_not_drop_trading_tables(tmp_path):
    """merge_bars_only() must NOT drop trading_app tables."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    # Create bars_1m + a fake trading table
    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc TIMESTAMPTZ, symbol TEXT, source_symbol TEXT,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id TEXT PRIMARY KEY, instrument TEXT NOT NULL
        )
    """)
    con.execute("""
        INSERT INTO validated_setups VALUES ('test_strat', 'MGC')
    """)
    con.commit()
    con.close()

    # Import and run the safe merge
    from scripts.run_parallel_ingest import merge_bars_only
    merge_bars_only(db_path=db_path, temp_dbs=[])

    # Trading table must survive
    con = duckdb.connect(str(db_path), read_only=True)
    count = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
    con.close()
    assert count == 1, "merge_bars_only() destroyed trading_app data!"


def test_force_rebuild_requires_flag(tmp_path):
    """--force-rebuild must be explicitly passed to allow table drops."""
    # This is a design contract test — force_rebuild defaults to False
    from scripts.run_parallel_ingest import merge_bars_only
    # Default call should NOT have force_rebuild behavior
    # (function signature enforces this)
    import inspect
    sig = inspect.signature(merge_bars_only)
    assert "force_rebuild" not in sig.parameters or \
        sig.parameters.get("force_rebuild").default is False
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline/test_parallel_ingest_safety.py -v`
Expected: FAIL — `merge_bars_only` doesn't exist yet

**Step 3: Rewrite merge_all() in run_parallel_ingest.py**

Replace lines 69-126 with safe `merge_bars_only()`:

```python
def merge_bars_only(db_path: Path = None, temp_dbs: list[Path] = None):
    """Merge temp DBs into gold.db — bars_1m ONLY. Never touches trading tables."""
    if db_path is None:
        db_path = GOLD_DB

    print("\n=== MERGING (bars_1m only) ===")

    con = duckdb.connect(str(db_path))
    total = 0

    if temp_dbs is None:
        temp_dbs = []
        for start, _ in YEAR_RANGES:
            year_label = start[:4]
            p = PROJECT_ROOT / f"temp_{year_label}.db"
            if p.exists():
                temp_dbs.append(p)

    for temp_db in temp_dbs:
        alias = f"t{temp_db.stem}"
        con.execute(f"ATTACH '{temp_db}' AS {alias} (READ_ONLY)")
        con.execute(f"INSERT OR REPLACE INTO bars_1m SELECT * FROM {alias}.bars_1m")
        count = con.execute(f"SELECT COUNT(*) FROM {alias}.bars_1m").fetchone()[0]
        con.execute(f"DETACH {alias}")
        total += count
        print(f"  {temp_db.stem}: {count:,} rows merged")

    con.commit()

    # Verify
    actual = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    sources = con.execute(
        "SELECT DISTINCT LEFT(source_symbol, 2) as prefix FROM bars_1m ORDER BY prefix"
    ).fetchall()
    nulls = con.execute(
        "SELECT COUNT(*) FROM bars_1m WHERE source_symbol IS NULL"
    ).fetchone()[0]
    print(f"\n  Total bars_1m:   {actual:,} (from {total:,} incl overlaps)")
    print(f"  Source prefixes: {[r[0] for r in sources]}")
    print(f"  NULL sources:    {nulls}")

    if nulls > 0:
        print("  WARNING: NULL source_symbols found!")

    con.close()
```

Update `main()` to call `merge_bars_only()` instead of `merge_all()`. Add `--force-rebuild` CLI flag that prints a warning and requires confirmation before doing full schema reset:

```python
def main():
    parser = argparse.ArgumentParser(description="Parallel GC re-ingest")
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="DROP all tables including trading_app before ingest (DANGEROUS)"
    )
    args = parser.parse_args()

    if args.force_rebuild:
        print("WARNING: --force-rebuild will DROP all trading_app tables!")
        print("  This destroys: validated_setups, experimental_strategies,")
        print("  orb_outcomes, strategy_trade_days, edge_families")
        confirm = input("  Type 'yes-destroy-everything' to confirm: ")
        if confirm != "yes-destroy-everything":
            print("Aborted.")
            sys.exit(1)

    t0 = time.time()
    # ... existing parallel ingest code ...

    # Safe merge (bars only)
    merge_bars_only()

    # ... rest of main ...
```

Delete the old `merge_all()` function entirely.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline/test_parallel_ingest_safety.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_parallel_ingest.py tests/test_pipeline/test_parallel_ingest_safety.py
git commit -m "fix(critical): defuse destructive table drops in parallel ingest

merge_all() silently dropped validated_setups, experimental_strategies,
orb_outcomes when re-ingesting bars. Replaced with merge_bars_only()
that only touches bars_1m. Full rebuild requires --force-rebuild flag
with explicit confirmation."
```

---

### Task 2: Extend drift guardrails to cover all ingest scripts

**Files:**
- Modify: `pipeline/check_drift.py:38-47`
- Test: Run `python pipeline/check_drift.py` (existing drift check)

**Context:** `INGEST_FILES` (line 38) and `INGEST_WRITE_FILES` (line 44) only list `ingest_dbn.py` and `ingest_dbn_mgc.py`. The daily file ingester (`ingest_dbn_daily.py`) and the parallel wrapper (`scripts/run_parallel_ingest.py`) are not monitored.

**Step 1: Add files to drift check lists**

In `pipeline/check_drift.py`, modify lines 38-47:

```python
# Ingest files: must NOT use .apply()/.iterrows() on large data
INGEST_FILES = [
    PIPELINE_DIR / "ingest_dbn.py",
    PIPELINE_DIR / "ingest_dbn_mgc.py",
    PIPELINE_DIR / "ingest_dbn_daily.py",
]

# Ingest files: must NOT write to any table other than bars_1m
INGEST_WRITE_FILES = [
    PIPELINE_DIR / "ingest_dbn.py",
    PIPELINE_DIR / "ingest_dbn_mgc.py",
    PIPELINE_DIR / "ingest_dbn_daily.py",
]
```

Also add `scripts/run_parallel_ingest.py` to the write guard. Since it's in `scripts/` not `pipeline/`, add it to a separate list or extend INGEST_WRITE_FILES:

```python
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

INGEST_WRITE_FILES = [
    PIPELINE_DIR / "ingest_dbn.py",
    PIPELINE_DIR / "ingest_dbn_mgc.py",
    PIPELINE_DIR / "ingest_dbn_daily.py",
    SCRIPTS_DIR / "run_parallel_ingest.py",
]
```

**Step 2: Run drift check to verify no false positives**

Run: `python pipeline/check_drift.py`
Expected: All 21 checks PASS (the new files should only write bars_1m after Task 1 fix)

Note: If `run_parallel_ingest.py` still has the old `merge_all()` with DROP TABLE, it WILL trigger the drift check — which proves the guard works. Task 1 must be done first.

**Step 3: Update drift check header comment**

Change the docstring to note the expanded coverage:

```python
"""
Drift detection for the multi-instrument pipeline.

Fails if anyone reintroduces:
1. Hardcoded 'MGC' SQL literals in generic pipeline code
2. .apply() or .iterrows() usage in ingest scripts
3. Any writes to tables other than bars_1m in ingest scripts
   (covers: ingest_dbn.py, ingest_dbn_mgc.py, ingest_dbn_daily.py,
    scripts/run_parallel_ingest.py)
...
"""
```

**Step 4: Commit**

```bash
git add pipeline/check_drift.py
git commit -m "fix: extend drift guardrails to cover all ingest scripts

Added ingest_dbn_daily.py and scripts/run_parallel_ingest.py to
INGEST_FILES and INGEST_WRITE_FILES. Drift check now catches
non-bars_1m writes in all ingest-capable modules."
```

---

### Task 3: Create single end-to-end pipeline orchestrator

**Files:**
- Create: `pipeline/run_full_pipeline.py`
- Test: `tests/test_pipeline/test_full_pipeline.py` (create)
- Modify: `CLAUDE.md` — add to Key Commands

**Context:** `run_pipeline.py` stops at ingest → 5m → features → audit. Strategy pipeline (outcomes → discovery → validation) is entirely manual. This task creates a new orchestrator that chains all stages.

**Step 1: Write the failing test**

```python
"""Tests for the full pipeline orchestrator step registry."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_step_registry_is_ordered():
    """Full pipeline steps must be in correct dependency order."""
    from pipeline.run_full_pipeline import FULL_PIPELINE_STEPS

    step_names = [s[0] for s in FULL_PIPELINE_STEPS]
    expected_order = [
        "ingest", "build_5m", "build_features", "audit",
        "build_outcomes", "discover", "validate",
    ]
    assert step_names == expected_order


def test_step_functions_are_callable():
    """Every step in the registry must be a callable."""
    from pipeline.run_full_pipeline import FULL_PIPELINE_STEPS

    for name, desc, func in FULL_PIPELINE_STEPS:
        assert callable(func), f"Step {name} is not callable"


def test_dry_run_does_not_execute(capsys):
    """--dry-run should print plan without executing."""
    from pipeline.run_full_pipeline import print_dry_run, FULL_PIPELINE_STEPS

    print_dry_run(FULL_PIPELINE_STEPS, "MGC")
    captured = capsys.readouterr()
    assert "build_outcomes" in captured.out
    assert "discover" in captured.out
    assert "validate" in captured.out


def test_skip_to_works():
    """--skip-to should skip steps before the named step."""
    from pipeline.run_full_pipeline import get_steps_from, FULL_PIPELINE_STEPS

    steps = get_steps_from(FULL_PIPELINE_STEPS, "build_outcomes")
    names = [s[0] for s in steps]
    assert names == ["build_outcomes", "discover", "validate"]
    assert "ingest" not in names
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline/test_full_pipeline.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Create pipeline/run_full_pipeline.py**

```python
#!/usr/bin/env python3
"""
Full pipeline: ingest -> bars_5m -> features -> audit -> outcomes -> discovery -> validation.

Extends run_pipeline.py with strategy pipeline stages.
Uses DUCKDB_PATH env var or --db-path for database location.

Usage:
    python pipeline/run_full_pipeline.py --instrument MGC --start 2024-01-01 --end 2026-02-14
    python pipeline/run_full_pipeline.py --instrument MGC --skip-to build_outcomes
    python pipeline/run_full_pipeline.py --instrument MGC --dry-run
    python pipeline/run_full_pipeline.py --instrument MGC --db-path C:/db/gold.db
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.asset_configs import list_instruments

PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# STEP FUNCTIONS
# =============================================================================

def step_ingest(instrument: str, args) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "ingest_dbn.py"),
        f"--instrument={instrument}",
    ]
    if args.start:
        cmd.append(f"--start={args.start}")
    if args.end:
        cmd.append(f"--end={args.end}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def step_build_5m(instrument: str, args) -> int:
    if not args.start or not args.end:
        print("FATAL: --start and --end required for bars_5m")
        return 1
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "build_bars_5m.py"),
        f"--instrument={instrument}",
        f"--start={args.start}",
        f"--end={args.end}",
    ]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def step_build_features(instrument: str, args) -> int:
    if not args.start or not args.end:
        print("FATAL: --start and --end required for daily_features")
        return 1
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "build_daily_features.py"),
        f"--instrument={instrument}",
        f"--start={args.start}",
        f"--end={args.end}",
    ]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def step_audit(instrument: str, args) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "pipeline" / "check_db.py")]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def step_build_outcomes(instrument: str, args) -> int:
    if not args.start or not args.end:
        print("FATAL: --start and --end required for outcomes")
        return 1
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "trading_app" / "outcome_builder.py"),
        f"--instrument={instrument}",
        f"--start={args.start}",
        f"--end={args.end}",
    ]
    if args.db_path:
        cmd.append(f"--db-path={args.db_path}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def step_discover(instrument: str, args) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "trading_app" / "strategy_discovery.py"),
        f"--instrument={instrument}",
    ]
    if args.db_path:
        cmd.append(f"--db-path={args.db_path}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def step_validate(instrument: str, args) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "trading_app" / "strategy_validator.py"),
        f"--instrument={instrument}",
        "--min-sample=50",
    ]
    if args.db_path:
        cmd.append(f"--db-path={args.db_path}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


# =============================================================================
# STEP REGISTRY
# =============================================================================

FULL_PIPELINE_STEPS = [
    ("ingest", "Ingest DBN -> bars_1m", step_ingest),
    ("build_5m", "Rebuild bars_5m from bars_1m", step_build_5m),
    ("build_features", "Rebuild daily_features", step_build_features),
    ("audit", "Database integrity check", step_audit),
    ("build_outcomes", "Pre-compute orb_outcomes", step_build_outcomes),
    ("discover", "Grid search experimental_strategies", step_discover),
    ("validate", "6-phase strategy validation", step_validate),
]


# =============================================================================
# HELPERS
# =============================================================================

def get_steps_from(steps, skip_to: str):
    """Return steps starting from skip_to."""
    names = [s[0] for s in steps]
    if skip_to not in names:
        raise ValueError(f"Unknown step '{skip_to}'. Valid: {names}")
    idx = names.index(skip_to)
    return steps[idx:]


def print_dry_run(steps, instrument: str):
    """Print planned steps without executing."""
    print(f"DRY RUN: Full pipeline for {instrument}")
    print()
    for i, (name, desc, _) in enumerate(steps, 1):
        print(f"  Step {i}: {name} - {desc}")
    print()
    print("No steps executed (dry run).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: ingest -> 5m -> features -> outcomes -> discovery -> validation"
    )
    parser.add_argument("--instrument", type=str, required=True,
                        help=f"Instrument ({', '.join(list_instruments())})")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    parser.add_argument("--skip-to", type=str,
                        help="Skip to named step (e.g. build_outcomes)")
    parser.add_argument("--db-path", type=str,
                        help="Database path (default: from DUCKDB_PATH or paths.py)")
    args = parser.parse_args()

    instrument = args.instrument.upper()

    # Determine steps to run
    steps = FULL_PIPELINE_STEPS
    if args.skip_to:
        steps = get_steps_from(FULL_PIPELINE_STEPS, args.skip_to)

    if args.dry_run:
        print_dry_run(steps, instrument)
        sys.exit(0)

    # Execute
    start_time = datetime.now()
    print("=" * 70)
    print(f"FULL PIPELINE: {instrument}")
    print("=" * 70)
    print(f"  Date range: {args.start or 'default'} to {args.end or 'default'}")
    print(f"  DB path: {args.db_path or 'default'}")
    print(f"  Steps: {' -> '.join(s[0] for s in steps)}")
    print()

    results = []
    for i, (name, desc, func) in enumerate(steps, 1):
        print("-" * 70)
        print(f"STEP {i}/{len(steps)}: {name} - {desc}")
        print("-" * 70)

        step_start = datetime.now()
        rc = func(instrument, args)
        elapsed = datetime.now() - step_start

        results.append({"step": i, "name": name, "rc": rc, "elapsed": elapsed})

        if rc != 0:
            print(f"\nFATAL: {name} failed (exit {rc}). Pipeline halted.")
            break

        print(f"  {name}: PASSED ({elapsed})\n")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    total = datetime.now() - start_time
    for r in results:
        status = "PASSED" if r["rc"] == 0 else f"FAILED (exit {r['rc']})"
        print(f"  {r['name']}: {status} ({r['elapsed']})")

    all_ok = all(r["rc"] == 0 for r in results)
    print(f"\nTotal: {total}")
    print("SUCCESS" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline/test_full_pipeline.py -v`
Expected: All 4 PASS

**Step 5: Run full drift check**

Run: `python pipeline/check_drift.py`
Expected: All checks PASS (new file is in pipeline/ but doesn't write to non-bars_1m tables itself — it delegates via subprocess)

**Step 6: Update CLAUDE.md Key Commands**

Add under Trading App section:

```markdown
### Full Pipeline (end-to-end)

```bash
python pipeline/run_full_pipeline.py --instrument MGC --start 2024-01-01 --end 2026-02-14
python pipeline/run_full_pipeline.py --instrument MGC --skip-to build_outcomes --db-path C:/db/gold.db
python pipeline/run_full_pipeline.py --instrument MGC --dry-run
```
```

**Step 7: Commit**

```bash
git add pipeline/run_full_pipeline.py tests/test_pipeline/test_full_pipeline.py CLAUDE.md
git commit -m "feat: add full pipeline orchestrator (ingest through validation)

New pipeline/run_full_pipeline.py chains all 7 stages:
ingest -> 5m -> features -> audit -> outcomes -> discovery -> validation

Supports --skip-to for partial runs, --db-path for working copy,
--dry-run for plan preview. Fail-closed on any step failure."
```

---

## Execution Order

1. Task 1 first (defuse landmine) — blocks Task 2 (drift check would flag old code)
2. Task 2 second (extend drift guardrails)
3. Task 3 last (independent, but commit after drift checks pass)

## Verification

```bash
# After all 3 tasks:
python -m pytest tests/test_pipeline/test_parallel_ingest_safety.py tests/test_pipeline/test_full_pipeline.py -v
python pipeline/check_drift.py
python pipeline/run_full_pipeline.py --instrument MGC --dry-run
```

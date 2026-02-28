# PASS 2 Audit Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Implement 4 sequential AMBER-fix phases from the "IS THIS REAL?" audit: E3 soft retirement, walk-forward soft gate columns, data years drift check, E2 slippage stress research.

**Architecture:** Each phase is a self-contained change with its own migration/script, drift check addition, test, and verification gate. Phases execute sequentially — no phase starts until the prior phase's drift check and pytest pass. All changes are additive (no schema drops, no data deletion).

**Tech Stack:** Python 3.12, DuckDB, pytest, pipeline/check_drift.py registry

---

### Task 1: Phase A — E3 Soft Retirement Migration Script

**Files:**
- Create: `scripts/migrations/retire_e3_strategies.py`
- Test: `tests/test_migrations/test_retire_e3.py`

**Step 1: Write the failing test**

Create `tests/test_migrations/__init__.py` (empty) and `tests/test_migrations/test_retire_e3.py`:

```python
"""Tests for E3 soft retirement migration."""
import duckdb
import pytest
from pathlib import Path


def _setup_test_db(tmp_path):
    """Create a minimal validated_setups table with E3 rows."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id TEXT PRIMARY KEY,
            instrument TEXT NOT NULL,
            entry_model TEXT NOT NULL,
            status TEXT NOT NULL,
            retired_at TIMESTAMPTZ,
            retirement_reason TEXT,
            fdr_significant BOOLEAN
        )
    """)
    # 3 E3 rows (none FDR-significant), 2 E1/E2 rows
    con.execute("""
        INSERT INTO validated_setups VALUES
        ('MGC_E3_1', 'MGC', 'E3', 'active', NULL, NULL, FALSE),
        ('MNQ_E3_2', 'MNQ', 'E3', 'active', NULL, NULL, FALSE),
        ('MES_E3_3', 'MES', 'E3', 'active', NULL, NULL, FALSE),
        ('MGC_E1_4', 'MGC', 'E1', 'active', NULL, NULL, TRUE),
        ('MNQ_E2_5', 'MNQ', 'E2', 'active', NULL, NULL, TRUE)
    """)
    con.close()
    return db_path


def test_retires_all_e3_rows(tmp_path):
    db_path = _setup_test_db(tmp_path)
    from scripts.migrations.retire_e3_strategies import retire_e3
    count = retire_e3(db_path, dry_run=False)
    assert count == 3

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT status, retirement_reason FROM validated_setups WHERE entry_model = 'E3'"
    ).fetchall()
    con.close()
    for status, reason in rows:
        assert status == "RETIRED"
        assert "0/50 FDR-sig" in reason


def test_does_not_touch_non_e3(tmp_path):
    db_path = _setup_test_db(tmp_path)
    from scripts.migrations.retire_e3_strategies import retire_e3
    retire_e3(db_path, dry_run=False)

    con = duckdb.connect(str(db_path), read_only=True)
    active = con.execute(
        "SELECT COUNT(*) FROM validated_setups WHERE status = 'active'"
    ).fetchone()[0]
    con.close()
    assert active == 2


def test_dry_run_no_changes(tmp_path):
    db_path = _setup_test_db(tmp_path)
    from scripts.migrations.retire_e3_strategies import retire_e3
    count = retire_e3(db_path, dry_run=True)
    assert count == 3

    con = duckdb.connect(str(db_path), read_only=True)
    active = con.execute(
        "SELECT COUNT(*) FROM validated_setups WHERE status = 'active'"
    ).fetchone()[0]
    con.close()
    assert active == 5  # All still active


def test_idempotent_rerun(tmp_path):
    db_path = _setup_test_db(tmp_path)
    from scripts.migrations.retire_e3_strategies import retire_e3
    retire_e3(db_path, dry_run=False)
    count = retire_e3(db_path, dry_run=False)
    assert count == 0  # Already retired
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_migrations/test_retire_e3.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.migrations.retire_e3_strategies'`

**Step 3: Write the migration script**

Create `scripts/migrations/retire_e3_strategies.py`:

```python
"""
Soft-retire all E3 (limit retrace) strategies in validated_setups.

PASS 2 audit finding: 0/50 E3 strategies are FDR-significant.
E3 has no timeout mechanism — 90-91% fill rate includes late adverse fills.

Sets status='RETIRED', retirement_reason, retired_at.
Does NOT delete rows — E3 remains for future timeout research.
Idempotent: re-running skips already-retired rows.

Usage:
    python scripts/migrations/retire_e3_strategies.py
    python scripts/migrations/retire_e3_strategies.py --dry-run
    python scripts/migrations/retire_e3_strategies.py --db C:/db/gold.db
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from pipeline.paths import GOLD_DB_PATH


RETIREMENT_REASON = "PASS2: 0/50 FDR-sig, no timeout mechanism"


def retire_e3(db_path: Path, dry_run: bool = False) -> int:
    """Retire all active E3 rows. Returns count of rows affected."""
    con = duckdb.connect(str(db_path))
    try:
        # Count candidates
        count = con.execute(
            "SELECT COUNT(*) FROM validated_setups "
            "WHERE entry_model = 'E3' AND status = 'active'"
        ).fetchone()[0]

        print(f"Found {count} active E3 strategies to retire")

        if count == 0:
            print("Nothing to do.")
            return 0

        if dry_run:
            # Show what would be retired
            rows = con.execute(
                "SELECT strategy_id, instrument FROM validated_setups "
                "WHERE entry_model = 'E3' AND status = 'active'"
            ).fetchall()
            for sid, inst in rows:
                print(f"  [DRY RUN] Would retire: {sid} ({inst})")
            return count

        now = datetime.now(timezone.utc).isoformat()
        con.execute(
            """UPDATE validated_setups
               SET status = 'RETIRED',
                   retired_at = ?,
                   retirement_reason = ?
               WHERE entry_model = 'E3' AND status = 'active'""",
            [now, RETIREMENT_REASON],
        )
        con.commit()
        print(f"Retired {count} E3 strategies")
        return count
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soft-retire E3 strategies")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    retire_e3(db_path, dry_run=args.dry_run)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_migrations/test_retire_e3.py -v`
Expected: 4 passed

**Step 5: Run migration on production DB (dry-run first)**

Run: `python scripts/migrations/retire_e3_strategies.py --dry-run`
Expected: "Found 50 active E3 strategies to retire" + list of IDs

Run: `python scripts/migrations/retire_e3_strategies.py`
Expected: "Retired 50 E3 strategies"

**Step 6: Commit**

```bash
git add scripts/migrations/retire_e3_strategies.py tests/test_migrations/
git commit -m "feat: E3 soft retirement migration (PASS2 Phase A)"
```

---

### Task 2: Phase A — E3 Drift Check Guard

**Files:**
- Modify: `pipeline/check_drift.py` (add check function + CHECKS entry)
- Modify: `tests/test_pipeline/test_check_drift.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_pipeline/test_check_drift.py`:

```python
from pipeline.check_drift import check_no_active_e3


class TestNoActiveE3:
    """Check #39: No active E3 rows in validated_setups."""

    def test_catches_active_e3(self, tmp_path, monkeypatch):
        """Active E3 rows should trigger violation."""
        import duckdb
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, entry_model TEXT, status TEXT
            )
        """)
        con.execute(
            "INSERT INTO validated_setups VALUES ('E3_1', 'E3', 'active')"
        )
        con.close()

        monkeypatch.setattr("pipeline.check_drift.GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_no_active_e3()
        assert len(violations) > 0
        assert "E3" in violations[0]

    def test_passes_retired_e3(self, tmp_path, monkeypatch):
        """Retired E3 rows should not trigger."""
        import duckdb
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, entry_model TEXT, status TEXT
            )
        """)
        con.execute(
            "INSERT INTO validated_setups VALUES ('E3_1', 'E3', 'RETIRED')"
        )
        con.close()

        monkeypatch.setattr("pipeline.check_drift.GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_no_active_e3()
        assert len(violations) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_check_drift.py::TestNoActiveE3 -v`
Expected: FAIL with `ImportError: cannot import name 'check_no_active_e3'`

**Step 3: Implement the drift check**

Add to `pipeline/check_drift.py`:

Near the top (after existing imports), add a module-level constant for test monkeypatching:

```python
GOLD_DB_PATH_FOR_CHECKS = None  # Set by tests; production uses GOLD_DB_PATH
```

Add the check function (before the CHECKS list):

```python
def check_no_active_e3() -> list[str]:
    """Check #39: No active E3 strategies in validated_setups.

    E3 (limit retrace at ORB) was soft-retired in PASS2 audit (Feb 2026).
    0/50 E3 strategies survived BH FDR. No timeout mechanism exists.
    Any active E3 rows indicate the migration wasn't run or was reverted.
    """
    violations = []
    try:
        import duckdb
        db_path = GOLD_DB_PATH_FOR_CHECKS
        if db_path is None:
            from pipeline.paths import GOLD_DB_PATH
            db_path = GOLD_DB_PATH
        if not Path(db_path).exists():
            return violations  # Skip if no DB (CI)
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            count = con.execute(
                "SELECT COUNT(*) FROM validated_setups "
                "WHERE entry_model = 'E3' AND status = 'active'"
            ).fetchone()[0]
            if count > 0:
                violations.append(
                    f"  validated_setups: {count} active E3 rows "
                    f"(should be RETIRED — run scripts/migrations/retire_e3_strategies.py)"
                )
        finally:
            con.close()
    except Exception:
        pass  # DB may not exist in CI
    return violations
```

Add to the CHECKS list (at the end, before the closing `]`):

```python
    ("No active E3 strategies (soft-retired Feb 2026)",
     check_no_active_e3),
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline/test_check_drift.py::TestNoActiveE3 -v`
Expected: 2 passed

**Step 5: Run full drift check to verify it passes with production DB**

Run: `python pipeline/check_drift.py`
Expected: Check 39 PASSED (because migration already ran in Task 1 step 5)

**Step 6: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 7: Commit**

```bash
git add pipeline/check_drift.py tests/test_pipeline/test_check_drift.py
git commit -m "feat: add E3 retirement drift guard (check #39, PASS2 Phase A)"
```

---

### Task 3: Phase A — Verification Gate

**Step 1: Verify E3 row counts in DB**

Run: `python -c "import duckdb; con=duckdb.connect('gold.db', read_only=True); print('Active E3:', con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE entry_model='E3' AND status='active'\").fetchone()[0]); print('Retired E3:', con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE entry_model='E3' AND status='RETIRED'\").fetchone()[0]); con.close()"`
Expected: Active E3: 0, Retired E3: 50

**Step 2: Run drift check**

Run: `python pipeline/check_drift.py`
Expected: All checks pass (including new check #39)

**Step 3: Run pytest**

Run: `pytest tests/ -x -q`
Expected: All pass

---

### Task 4: Phase B — WF Columns Schema Migration

**Files:**
- Modify: `trading_app/db_manager.py` (add migration block + schema verification)

**Step 1: Write the migration block in db_manager.py**

Add to `init_trading_app_schema()` in `trading_app/db_manager.py`, after the existing time-stop migration block (after line ~397):

```python
        # Migration: add walk-forward soft gate columns (PASS2 audit Phase B)
        wf_cols = [
            ("wf_tested", "BOOLEAN"),
            ("wf_passed", "BOOLEAN"),
            ("wf_windows", "INTEGER"),
        ]
        for col, typedef in wf_cols:
            try:
                con.execute(f"ALTER TABLE validated_setups ADD COLUMN {col} {typedef}")
            except duckdb.CatalogException:
                pass  # column already exists
```

Also add `"wf_tested", "wf_passed", "wf_windows"` to the `expected_cols` set in `verify_trading_app_schema()` (inside the `validated_setups` schema check block, around line ~496).

**Step 2: Run init_db to apply migration**

Run: `python trading_app/db_manager.py`
Expected: "Trading app schema initialized successfully"

**Step 3: Verify columns exist**

Run: `python -c "import duckdb; con=duckdb.connect('gold.db', read_only=True); print([r[0] for r in con.execute(\"SELECT column_name FROM information_schema.columns WHERE table_name='validated_setups' AND column_name LIKE 'wf_%'\").fetchall()]); con.close()"`
Expected: `['wf_tested', 'wf_passed', 'wf_windows']`

**Step 4: Commit**

```bash
git add trading_app/db_manager.py
git commit -m "feat: add wf_tested/wf_passed/wf_windows columns to validated_setups (PASS2 Phase B)"
```

---

### Task 5: Phase B — WF Backfill Migration Script

**Files:**
- Create: `scripts/migrations/backfill_wf_columns.py`
- Test: `tests/test_migrations/test_backfill_wf.py`

**Step 1: Write the failing test**

Create `tests/test_migrations/test_backfill_wf.py`:

```python
"""Tests for walk-forward column backfill migration."""
import json
import duckdb
import pytest
from pathlib import Path


def _setup_test_db_and_jsonl(tmp_path):
    """Create minimal DB + JSONL for testing."""
    db_path = tmp_path / "test.db"
    jsonl_path = tmp_path / "wf_results.jsonl"

    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id TEXT PRIMARY KEY,
            instrument TEXT,
            entry_model TEXT,
            status TEXT,
            wf_tested BOOLEAN,
            wf_passed BOOLEAN,
            wf_windows INTEGER
        )
    """)
    con.execute("""
        INSERT INTO validated_setups VALUES
        ('S1', 'MGC', 'E1', 'active', NULL, NULL, NULL),
        ('S2', 'MGC', 'E2', 'active', NULL, NULL, NULL),
        ('S3', 'MNQ', 'E2', 'active', NULL, NULL, NULL)
    """)
    con.close()

    # JSONL: S1 passed (4 windows), S2 failed
    records = [
        {"strategy_id": "S1", "passed": True, "n_valid_windows": 4},
        {"strategy_id": "S2", "passed": False, "n_valid_windows": 2},
    ]
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return db_path, jsonl_path


def test_backfills_from_jsonl(tmp_path):
    db_path, jsonl_path = _setup_test_db_and_jsonl(tmp_path)
    from scripts.migrations.backfill_wf_columns import backfill_wf
    updated = backfill_wf(db_path, jsonl_path, dry_run=False)
    assert updated == 2  # S1 and S2 found in JSONL

    con = duckdb.connect(str(db_path), read_only=True)
    rows = {r[0]: r for r in con.execute(
        "SELECT strategy_id, wf_tested, wf_passed, wf_windows FROM validated_setups"
    ).fetchall()}
    con.close()

    assert rows["S1"] == ("S1", True, True, 4)
    assert rows["S2"] == ("S2", True, False, 2)
    assert rows["S3"] == ("S3", None, None, None)  # Not in JSONL


def test_dry_run_no_changes(tmp_path):
    db_path, jsonl_path = _setup_test_db_and_jsonl(tmp_path)
    from scripts.migrations.backfill_wf_columns import backfill_wf
    backfill_wf(db_path, jsonl_path, dry_run=True)

    con = duckdb.connect(str(db_path), read_only=True)
    nulls = con.execute(
        "SELECT COUNT(*) FROM validated_setups WHERE wf_tested IS NULL"
    ).fetchone()[0]
    con.close()
    assert nulls == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_migrations/test_backfill_wf.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the backfill migration**

Create `scripts/migrations/backfill_wf_columns.py`:

```python
"""
Backfill wf_tested/wf_passed/wf_windows on validated_setups from JSONL.

Reads data/walkforward_results.jsonl and populates the 3 new columns
for each strategy_id found in both the JSONL and validated_setups.

Usage:
    python scripts/migrations/backfill_wf_columns.py
    python scripts/migrations/backfill_wf_columns.py --dry-run
"""

import json
import argparse
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from pipeline.paths import GOLD_DB_PATH

DEFAULT_JSONL = PROJECT_ROOT / "data" / "walkforward_results.jsonl"


def backfill_wf(db_path: Path, jsonl_path: Path, dry_run: bool = False) -> int:
    """Read JSONL, update validated_setups. Returns count updated."""
    # Parse JSONL — last entry per strategy_id wins (append-only log)
    wf_data = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record["strategy_id"]
            wf_data[sid] = {
                "passed": record["passed"],
                "n_valid_windows": record.get("n_valid_windows", 0),
            }
    print(f"Loaded {len(wf_data)} WF results from {jsonl_path}")

    con = duckdb.connect(str(db_path))
    try:
        # Get all strategy_ids in validated_setups
        vs_ids = {r[0] for r in con.execute(
            "SELECT strategy_id FROM validated_setups"
        ).fetchall()}

        updates = []
        for sid, data in wf_data.items():
            if sid in vs_ids:
                updates.append((
                    True,  # wf_tested
                    data["passed"],
                    data["n_valid_windows"],
                    sid,
                ))

        print(f"Matched {len(updates)} strategies in validated_setups")

        if dry_run:
            for tested, passed, windows, sid in updates[:5]:
                print(f"  [DRY RUN] {sid}: passed={passed}, windows={windows}")
            if len(updates) > 5:
                print(f"  ... ({len(updates)} total)")
            return len(updates)

        con.executemany(
            """UPDATE validated_setups
               SET wf_tested = ?, wf_passed = ?, wf_windows = ?
               WHERE strategy_id = ?""",
            updates,
        )
        con.commit()
        print(f"Updated {len(updates)} rows")
        return len(updates)
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill WF columns from JSONL")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--jsonl", type=str, default=None, help="JSONL path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    jsonl_path = Path(args.jsonl) if args.jsonl else DEFAULT_JSONL
    backfill_wf(db_path, jsonl_path, dry_run=args.dry_run)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_migrations/test_backfill_wf.py -v`
Expected: 2 passed

**Step 5: Run migration on production DB**

Run: `python scripts/migrations/backfill_wf_columns.py --dry-run`
Then: `python scripts/migrations/backfill_wf_columns.py`
Expected: Shows count of matched strategies

**Step 6: Commit**

```bash
git add scripts/migrations/backfill_wf_columns.py tests/test_migrations/test_backfill_wf.py
git commit -m "feat: backfill wf_tested/wf_passed/wf_windows from JSONL (PASS2 Phase B)"
```

---

### Task 6: Phase B — Validator WF Column Population

**Files:**
- Modify: `trading_app/strategy_validator.py` (populate wf columns during batch write)

**Step 1: Identify the insertion point**

In `strategy_validator.py`, the batch write at line ~877 inserts into `validated_setups`. The INSERT column list must include the 3 new WF columns. The WF result data is available in `sr.get("wf_result_dict")`.

**Step 2: Modify the INSERT statement**

In the INSERT OR REPLACE at line ~877, add 3 columns to the column list and 3 values:

Column list — add after `dst_verdict)`:
```
wf_tested, wf_passed, wf_windows
```

Values — extract from the walkforward result dict:
```python
# Before the INSERT, compute WF column values
wf_result_dict = sr.get("wf_result_dict")
wf_tested = wf_result_dict is not None
wf_passed = (wf_result_dict or {}).get("passed", False) if wf_tested else None
wf_windows_val = (wf_result_dict or {}).get("as_dict", {}).get("n_valid_windows") if wf_tested else None
```

Add `wf_tested, wf_passed, wf_windows_val` to the parameter list.

**Step 3: Run existing validator tests**

Run: `pytest tests/test_trading_app/test_strategy_validator.py -v`
Expected: All pass (existing tests use `validate_strategy()` not `run_validation()`)

**Step 4: Run full pytest**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 5: Commit**

```bash
git add trading_app/strategy_validator.py
git commit -m "feat: populate wf_tested/wf_passed/wf_windows during validation (PASS2 Phase B)"
```

---

### Task 7: Phase B — WF Drift Check

**Files:**
- Modify: `pipeline/check_drift.py` (add check function + CHECKS entry)
- Modify: `tests/test_pipeline/test_check_drift.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_pipeline/test_check_drift.py`:

```python
from pipeline.check_drift import check_wf_coverage


class TestWfCoverage:
    """Check #40: MGC/MES strategies should have WF data."""

    def test_warns_on_untested_mgc(self, tmp_path, monkeypatch):
        import duckdb
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, instrument TEXT, entry_model TEXT,
                status TEXT, wf_tested BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
            ('S1', 'MGC', 'E1', 'active', NULL),
            ('S2', 'MGC', 'E2', 'active', TRUE)
        """)
        con.close()

        monkeypatch.setattr("pipeline.check_drift.GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_wf_coverage()
        assert len(violations) > 0
        assert "MGC" in violations[0]

    def test_passes_all_tested(self, tmp_path, monkeypatch):
        import duckdb
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, instrument TEXT, entry_model TEXT,
                status TEXT, wf_tested BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
            ('S1', 'MGC', 'E2', 'active', TRUE)
        """)
        con.close()

        monkeypatch.setattr("pipeline.check_drift.GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_wf_coverage()
        assert len(violations) == 0

    def test_ignores_mnq_m2k(self, tmp_path, monkeypatch):
        """MNQ/M2K have too few years for WF — skip them."""
        import duckdb
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, instrument TEXT, entry_model TEXT,
                status TEXT, wf_tested BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
            ('S1', 'MNQ', 'E2', 'active', NULL)
        """)
        con.close()

        monkeypatch.setattr("pipeline.check_drift.GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_wf_coverage()
        assert len(violations) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_check_drift.py::TestWfCoverage -v`
Expected: FAIL with `ImportError`

**Step 3: Implement the drift check**

Add to `pipeline/check_drift.py` (before CHECKS list):

```python
def check_wf_coverage() -> list[str]:
    """Check #40: MGC/MES strategies should have walk-forward data.

    MGC (10yr) and MES (7yr) have enough data for 3+ WF windows.
    MNQ/M2K have ~5yr — not enough for meaningful WF, so SKIPPED.
    This is a WARNING, not a blocker (soft gate).
    """
    violations = []
    WF_REQUIRED_INSTRUMENTS = {"MGC", "MES"}
    try:
        import duckdb
        db_path = GOLD_DB_PATH_FOR_CHECKS
        if db_path is None:
            from pipeline.paths import GOLD_DB_PATH
            db_path = GOLD_DB_PATH
        if not Path(db_path).exists():
            return violations
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            for inst in sorted(WF_REQUIRED_INSTRUMENTS):
                rows = con.execute(
                    """SELECT COUNT(*) as total,
                              SUM(CASE WHEN wf_tested = TRUE THEN 1 ELSE 0 END) as tested
                       FROM validated_setups
                       WHERE instrument = ? AND status = 'active'""",
                    [inst],
                ).fetchone()
                total, tested = rows[0] or 0, rows[1] or 0
                if total > 0 and tested < total:
                    untested = total - tested
                    violations.append(
                        f"  {inst}: {untested}/{total} active strategies lack "
                        f"walk-forward data (wf_tested != TRUE). "
                        f"SKIPPED — run validation with --no-walkforward disabled"
                    )
        finally:
            con.close()
    except Exception:
        pass
    return violations
```

Add to CHECKS list:

```python
    ("WF coverage for MGC/MES (soft gate, SKIPPED warning)",
     check_wf_coverage),
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline/test_check_drift.py::TestWfCoverage -v`
Expected: 3 passed

**Step 5: Run full drift check**

Run: `python pipeline/check_drift.py`
Expected: Check 40 — may show SKIPPED warning if backfill didn't cover all strategies

**Step 6: Run full pytest**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 7: Commit**

```bash
git add pipeline/check_drift.py tests/test_pipeline/test_check_drift.py
git commit -m "feat: add WF coverage drift check for MGC/MES (check #40, PASS2 Phase B)"
```

---

### Task 8: Phase B — Verification Gate

**Step 1: Verify WF columns populated**

Run: `python -c "import duckdb; con=duckdb.connect('gold.db', read_only=True); print('Total active:', con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE status='active'\").fetchone()[0]); print('WF tested:', con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE status='active' AND wf_tested=TRUE\").fetchone()[0]); print('WF passed:', con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE status='active' AND wf_passed=TRUE\").fetchone()[0]); con.close()"`

**Step 2: Run drift check**

Run: `python pipeline/check_drift.py`
Expected: All checks pass

**Step 3: Run pytest**

Run: `pytest tests/ -x -q`
Expected: All pass

---

### Task 9: Phase C — Data Years Drift Check

**Files:**
- Modify: `pipeline/check_drift.py` (add check function + CHECKS entry)
- Modify: `tests/test_pipeline/test_check_drift.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_pipeline/test_check_drift.py`:

```python
from pipeline.check_drift import check_data_years_disclosure


class TestDataYearsDisclosure:
    """Check #41: Warn on instruments with years_tested < 7."""

    def test_warns_on_short_history(self, tmp_path, monkeypatch):
        import duckdb
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, instrument TEXT, status TEXT,
                years_tested INTEGER
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
            ('S1', 'MNQ', 'active', 5),
            ('S2', 'MGC', 'active', 10)
        """)
        con.close()

        monkeypatch.setattr("pipeline.check_drift.GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_data_years_disclosure()
        assert len(violations) == 1
        assert "MNQ" in violations[0]

    def test_passes_long_history(self, tmp_path, monkeypatch):
        import duckdb
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, instrument TEXT, status TEXT,
                years_tested INTEGER
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
            ('S1', 'MGC', 'active', 10)
        """)
        con.close()

        monkeypatch.setattr("pipeline.check_drift.GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_data_years_disclosure()
        assert len(violations) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_check_drift.py::TestDataYearsDisclosure -v`
Expected: FAIL with `ImportError`

**Step 3: Implement the drift check**

Add to `pipeline/check_drift.py` (before CHECKS list):

```python
def check_data_years_disclosure() -> list[str]:
    """Check #41: Warn on instruments with years_tested < 7.

    MNQ (~5yr) and M2K (~5yr) have shorter histories than MGC (10yr)
    and MES (7yr). Strategies validated on shorter data may not survive
    regime changes. Advisory warning only — not blocking.
    """
    violations = []
    MIN_YEARS = 7
    try:
        import duckdb
        db_path = GOLD_DB_PATH_FOR_CHECKS
        if db_path is None:
            from pipeline.paths import GOLD_DB_PATH
            db_path = GOLD_DB_PATH
        if not Path(db_path).exists():
            return violations
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                """SELECT instrument, MIN(years_tested) as min_years,
                          COUNT(*) as n_strategies
                   FROM validated_setups
                   WHERE status = 'active'
                   GROUP BY instrument
                   HAVING MIN(years_tested) < ?""",
                [MIN_YEARS],
            ).fetchall()
            for inst, min_years, n_strats in rows:
                violations.append(
                    f"  {inst}: {n_strats} active strategies with "
                    f"min years_tested={min_years} (< {MIN_YEARS}). "
                    f"Short data history — monitor for regime fragility"
                )
        finally:
            con.close()
    except Exception:
        pass
    return violations
```

Add to CHECKS list:

```python
    ("Data years disclosure (years_tested < 7 warning)",
     check_data_years_disclosure),
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline/test_check_drift.py::TestDataYearsDisclosure -v`
Expected: 2 passed

**Step 5: Run full drift check**

Run: `python pipeline/check_drift.py`
Expected: Check 41 warns for MNQ and M2K (5 years), passes for MGC/MES

**Step 6: Run full pytest**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 7: Commit**

```bash
git add pipeline/check_drift.py tests/test_pipeline/test_check_drift.py
git commit -m "feat: add data years disclosure drift check (check #41, PASS2 Phase C)"
```

---

### Task 10: Phase C — Verification Gate

**Step 1: Run drift check and confirm MNQ/M2K trigger**

Run: `python pipeline/check_drift.py`
Expected: Check 41 shows warnings for MNQ and M2K, no warning for MGC/MES

**Step 2: Run pytest**

Run: `pytest tests/ -x -q`
Expected: All pass

---

### Task 11: Phase D — E2 Slippage Stress Research Script

**Files:**
- Create: `research/research_e2_slippage_stress.py`

**Step 1: Write the research script**

Create `research/research_e2_slippage_stress.py`:

```python
#!/usr/bin/env python3
"""
E2 Slippage Stress Test — PASS2 Audit Phase D

Stress-tests E2 (stop-market) outcomes at 1x, 1.5x, 2x, 3x slippage
multiples. Reports Sharpe/winrate degradation per instrument/session.

E2 entry price = ORB high/low + slippage ticks. This script answers:
"How badly does E2 performance degrade if real slippage is worse than assumed?"

Uses cost_model.stress_test_costs() for consistent friction scaling.
Reads directly from gold.db (read-only) — no schema changes.

Usage:
    python research/research_e2_slippage_stress.py
    python research/research_e2_slippage_stress.py --instrument MGC
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import COST_SPECS, stress_test_costs
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS


SLIPPAGE_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]


def run_stress_test(instrument: str | None = None):
    """Run slippage stress test for one or all instruments."""
    instruments = [instrument] if instrument else list(ACTIVE_ORB_INSTRUMENTS)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        print("=" * 80)
        print("E2 SLIPPAGE STRESS TEST")
        print("=" * 80)
        print()

        for inst in instruments:
            cost_spec = COST_SPECS[inst]
            base_friction = cost_spec.total_friction

            # Get E2 validated strategies for this instrument
            strategies = con.execute(
                """SELECT strategy_id, orb_label, sample_size, win_rate,
                          expectancy_r, sharpe_ann, avg_risk_dollars
                   FROM validated_setups
                   WHERE instrument = ? AND entry_model = 'E2'
                   AND status = 'active'
                   ORDER BY sharpe_ann DESC NULLS LAST""",
                [inst],
            ).fetchall()

            if not strategies:
                print(f"{inst}: No active E2 strategies — skipping")
                print()
                continue

            print(f"{inst}: {len(strategies)} active E2 strategies")
            print(f"  Base friction: ${base_friction:.2f} RT")
            print(f"  Point value: ${cost_spec.point_value:.0f}")
            print()

            # Header
            header = f"  {'Multiplier':>10} | {'Friction':>10} | {'Avg ExpR':>10} | {'Avg WR':>8} | {'Pct Positive':>12}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for mult in SLIPPAGE_MULTIPLIERS:
                stressed = stress_test_costs(cost_spec, mult)
                extra_friction = stressed.total_friction - base_friction
                # extra_friction is in dollars; convert to R using avg_risk_dollars

                adjusted_exp_r_list = []
                for sid, orb_label, n, wr, exp_r, sharpe, avg_risk_d in strategies:
                    if exp_r is None or avg_risk_d is None or avg_risk_d <= 0:
                        continue
                    extra_r = extra_friction / avg_risk_d
                    adj_exp_r = exp_r - extra_r
                    adjusted_exp_r_list.append(adj_exp_r)

                if not adjusted_exp_r_list:
                    print(f"  {mult:>10.1f}x | ${stressed.total_friction:>8.2f} | {'N/A':>10} | {'N/A':>8} | {'N/A':>12}")
                    continue

                avg_exp_r = sum(adjusted_exp_r_list) / len(adjusted_exp_r_list)
                pct_positive = sum(1 for e in adjusted_exp_r_list if e > 0) / len(adjusted_exp_r_list)

                # Winrate doesn't change with slippage (same entry/exit, different accounting)
                avg_wr = sum(s[3] for s in strategies if s[3]) / len(strategies)

                print(
                    f"  {mult:>10.1f}x | ${stressed.total_friction:>8.2f} | "
                    f"{avg_exp_r:>10.4f} | {avg_wr:>7.1%} | {pct_positive:>11.1%}"
                )

            print()

            # Show worst-hit strategies at 2x
            stressed_2x = stress_test_costs(cost_spec, 2.0)
            extra_2x = stressed_2x.total_friction - base_friction
            print(f"  Strategies that go NEGATIVE at 2x slippage:")
            negative_at_2x = []
            for sid, orb_label, n, wr, exp_r, sharpe, avg_risk_d in strategies:
                if exp_r is None or avg_risk_d is None or avg_risk_d <= 0:
                    continue
                adj = exp_r - (extra_2x / avg_risk_d)
                if adj <= 0:
                    negative_at_2x.append((sid, orb_label, exp_r, adj))

            if negative_at_2x:
                for sid, sess, base_exp, adj_exp in negative_at_2x:
                    print(f"    {sid}: {sess} base={base_exp:.4f} -> 2x={adj_exp:.4f}")
            else:
                print("    None — all survive 2x slippage")

            print()

    finally:
        con.close()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="E2 slippage stress test")
    parser.add_argument("--instrument", type=str, default=None)
    args = parser.parse_args()
    run_stress_test(instrument=args.instrument)
```

**Step 2: Run the script**

Run: `python research/research_e2_slippage_stress.py`
Expected: Table output showing degradation curves at 1x/1.5x/2x/3x for each instrument

**Step 3: Commit**

```bash
git add research/research_e2_slippage_stress.py
git commit -m "feat: E2 slippage stress research script (PASS2 Phase D)"
```

---

### Task 12: Phase D — Final Verification Gate

**Step 1: Run full drift check**

Run: `python pipeline/check_drift.py`
Expected: All checks pass (39-41 + existing 38)

**Step 2: Run full pytest**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 3: Run health check**

Run: `python pipeline/health_check.py`
Expected: ALL PASSED

**Step 4: Update CLAUDE.md drift check count if needed**

If CLAUDE.md mentions a specific drift check count (e.g., "38 static checks"), update it to match the new total from `len(CHECKS)`. The doc-stats drift check (#36) enforces this.

Run: `python -c "from pipeline.check_drift import CHECKS; print(len(CHECKS))"`
Expected: 41 (38 original + 3 new)

Update CLAUDE.md if it hardcodes the count, then commit.

---

### Task 13: Final Commit and Summary

**Step 1: Verify all changes**

Run: `git log --oneline -10`
Expected: Shows all phase commits

**Step 2: Run all gates one final time**

Run: `python pipeline/check_drift.py && pytest tests/ -x -q && python pipeline/health_check.py`
Expected: All pass

---

## Notes for the Executor

- **GOLD_DB_PATH_FOR_CHECKS**: This module-level variable in `check_drift.py` enables monkeypatching in tests. Production code ignores it (falls through to `GOLD_DB_PATH`). This is a standard test isolation pattern.
- **Drift check numbering**: New checks get numbers 39/40/41 by position in the CHECKS list. The count is dynamic (`len(CHECKS)`), never hardcoded.
- **The E3 retirement migration must run on production DB BEFORE the drift check is committed**. Otherwise check #39 will fail on the next pre-commit.
- **JSONL path**: `data/walkforward_results.jsonl` is the canonical location. The backfill script reads from there by default.
- **CLAUDE.md**: If the doc-stats drift check (#36) references a hardcoded drift check count, it must be updated to match the new total (41). This is the only doc change needed.

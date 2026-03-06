# Update Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Build `pipeline_status.py` — a single CLI tool that detects staleness across the rebuild chain, validates prerequisites, orchestrates rebuilds with step-level resume, and logs results to a `rebuild_manifest` table.

**Architecture:** Single new script (`scripts/tools/pipeline_status.py`) with a staleness engine that queries `MAX(trading_day)` across all pipeline tables per instrument/aperture. Rebuild orchestration shells out to existing scripts (subprocess). A `rebuild_manifest` DB table tracks what ran, when, and whether it passed. Two new drift checks make staleness CI-visible.

**Tech Stack:** Python 3.13, DuckDB, argparse, subprocess, uuid, datetime, zoneinfo

---

### Task 0: Schema — Add `rebuild_manifest` Table to init_db.py

**Files:**
- Modify: `pipeline/init_db.py` (after line ~453, the `family_rr_locks` block)
- Test: `tests/test_pipeline/test_pipeline_status.py` (new file)

**Step 1: Write failing test — table creation**

Create `tests/test_pipeline/test_pipeline_status.py`:

```python
"""Tests for pipeline_status staleness engine and rebuild manifest."""

import duckdb
import pytest


def test_rebuild_manifest_table_exists():
    """init_db creates rebuild_manifest table."""
    con = duckdb.connect(":memory:")
    # Import and run init_db's schema creation
    from pipeline.init_db import initialize_database
    initialize_database(con)
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchdf()
    assert "rebuild_manifest" in tables.table_name.values
    con.close()


def test_rebuild_manifest_schema():
    """rebuild_manifest has correct columns."""
    con = duckdb.connect(":memory:")
    from pipeline.init_db import initialize_database
    initialize_database(con)
    cols = con.execute("DESCRIBE rebuild_manifest").fetchdf()
    expected = {
        "rebuild_id", "instrument", "started_at", "completed_at",
        "status", "failed_step", "steps_completed", "trigger",
    }
    assert expected == set(cols.column_name.values)
    con.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_pipeline_status.py::test_rebuild_manifest_table_exists -v`
Expected: FAIL — `rebuild_manifest` not in tables

**Step 3: Add schema constant to init_db.py**

Add after the `FAMILY_RR_LOCKS_SCHEMA` constant (near top of file, with other schema constants):

```python
REBUILD_MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS rebuild_manifest (
    rebuild_id TEXT PRIMARY KEY,
    instrument TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    failed_step TEXT,
    steps_completed TEXT[],
    trigger TEXT NOT NULL
);
"""
```

Then in `initialize_database()`, after the `family_rr_locks` block (line ~454):

```python
con.execute(REBUILD_MANIFEST_SCHEMA)
logger.info("  rebuild_manifest: created (or already exists)")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline/test_pipeline_status.py -v`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add pipeline/init_db.py tests/test_pipeline/test_pipeline_status.py
git commit -m "feat: add rebuild_manifest table to schema"
```

---

### Task 1: Staleness Engine — Core Query Logic

**Files:**
- Create: `scripts/tools/pipeline_status.py`
- Test: `tests/test_pipeline/test_pipeline_status.py` (append)

**Step 1: Write failing tests for staleness detection**

Append to `tests/test_pipeline/test_pipeline_status.py`:

```python
import datetime


def _create_test_db():
    """Create in-memory DB with minimal data for staleness testing."""
    con = duckdb.connect(":memory:")
    from pipeline.init_db import initialize_database
    initialize_database(con)
    return con


def _insert_bars_1m(con, symbol, max_date):
    """Insert a single bar_1m row at max_date."""
    con.execute(
        """INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
        VALUES ($1, $2, $2, 100.0, 101.0, 99.0, 100.5, 1000)""",
        [datetime.datetime(max_date.year, max_date.month, max_date.day, 10, 0, tzinfo=datetime.timezone.utc), symbol],
    )


def _insert_bars_5m(con, symbol, trading_day):
    """Insert a single bars_5m row."""
    con.execute(
        """INSERT INTO bars_5m (ts_utc, symbol, open, high, low, close, volume)
        VALUES ($1, $2, 100.0, 101.0, 99.0, 100.5, 5000)""",
        [datetime.datetime(trading_day.year, trading_day.month, trading_day.day, 10, 0, tzinfo=datetime.timezone.utc), symbol],
    )


def test_staleness_engine_fresh():
    """When all tables have same max date, nothing is stale."""
    # Import will be tested once pipeline_status.py exists
    from scripts.tools.pipeline_status import staleness_engine

    con = _create_test_db()
    d = datetime.date(2026, 3, 5)
    _insert_bars_1m(con, "MGC", d)
    _insert_bars_5m(con, "MGC", d)

    result = staleness_engine(con, "MGC")
    assert result["bars_1m"] == d
    assert result["bars_5m"] == d
    # orb_outcomes etc will be None (no data) — that's stale
    assert result["orb_outcomes"] is None
    con.close()


def test_staleness_engine_detects_gap():
    """When orb_outcomes is behind bars_1m, detect staleness."""
    from scripts.tools.pipeline_status import staleness_engine

    con = _create_test_db()
    d_fresh = datetime.date(2026, 3, 5)
    d_stale = datetime.date(2026, 2, 19)
    _insert_bars_1m(con, "MGC", d_fresh)
    _insert_bars_5m(con, "MGC", d_fresh)

    # Insert an old orb_outcome
    con.execute(
        """INSERT INTO orb_outcomes (trading_day, symbol, orb_label, orb_minutes, rr_target,
           confirm_bars, entry_model, outcome, pnl_r, risk_dollars, pnl_dollars)
        VALUES ($1, 'MGC', 'CME_REOPEN', 5, 1.5, 1, 'E2', 'win', 1.0, 10.0, 15.0)""",
        [d_stale],
    )

    result = staleness_engine(con, "MGC")
    assert result["bars_1m"] == d_fresh
    assert result["orb_outcomes"] == d_stale
    assert result["stale_steps"]  # should have stale steps listed
    con.close()


def test_staleness_weekend_not_false_positive():
    """Friday data + Monday query = not stale (weekend gap is normal)."""
    from scripts.tools.pipeline_status import is_stale

    friday = datetime.date(2026, 3, 6)  # Friday
    monday_check = datetime.date(2026, 3, 9)  # Monday

    # 1 trading day gap (Fri->Mon) should NOT be stale
    assert not is_stale(friday, monday_check, max_gap_trading_days=1)


def test_staleness_real_gap_is_stale():
    """10 trading days behind = stale."""
    from scripts.tools.pipeline_status import is_stale

    old = datetime.date(2026, 2, 19)
    now = datetime.date(2026, 3, 5)

    assert is_stale(old, now, max_gap_trading_days=7)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline/test_pipeline_status.py -v -k "staleness"`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.tools.pipeline_status'`

**Step 3: Create `scripts/tools/pipeline_status.py` — staleness engine only**

```python
"""Pipeline status — staleness detection, pre-flight checks, rebuild orchestration.

Usage:
    python scripts/tools/pipeline_status.py --status
    python scripts/tools/pipeline_status.py --status --instrument MGC
    python scripts/tools/pipeline_status.py --rebuild --instrument MGC
    python scripts/tools/pipeline_status.py --rebuild-all
    python scripts/tools/pipeline_status.py --resume --instrument MGC
    python scripts/tools/pipeline_status.py --rebuild --instrument MGC --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import duckdb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH

APERTURES = [5, 15, 30]

# Trading day = weekday (Mon-Fri). Used for gap calculation.
def _trading_days_between(d1: datetime.date, d2: datetime.date) -> int:
    """Count trading days (weekdays) between d1 and d2, exclusive of d1."""
    if d1 >= d2:
        return 0
    count = 0
    current = d1 + datetime.timedelta(days=1)
    while current <= d2:
        if current.weekday() < 5:  # Mon=0 .. Fri=4
            count += 1
        current += datetime.timedelta(days=1)
    return count


def is_stale(
    table_date: datetime.date | None,
    reference_date: datetime.date,
    max_gap_trading_days: int = 1,
) -> bool:
    """Check if table_date is stale relative to reference_date."""
    if table_date is None:
        return True
    gap = _trading_days_between(table_date, reference_date)
    return gap > max_gap_trading_days


def staleness_engine(con: duckdb.DuckDBPyConnection, instrument: str) -> dict:
    """Query max dates across all pipeline tables for one instrument.

    Returns dict with keys: bars_1m, bars_5m, daily_features, orb_outcomes,
    experimental, validated, edge_families, family_rr_locks, stale_steps, last_rebuild.
    """
    def _max_date(query: str) -> datetime.date | None:
        row = con.execute(query).fetchone()
        if row and row[0]:
            val = row[0]
            if isinstance(val, datetime.datetime):
                return val.date()
            return val
        return None

    result = {}

    # bars_1m: max ts_utc date for instrument
    result["bars_1m"] = _max_date(
        f"SELECT MAX(ts_utc::DATE) FROM bars_1m WHERE symbol = '{instrument}'"
    )

    # bars_5m
    result["bars_5m"] = _max_date(
        f"SELECT MAX(ts_utc::DATE) FROM bars_5m WHERE symbol = '{instrument}'"
    )

    # daily_features: check per aperture
    df_dates = {}
    for ap in APERTURES:
        df_dates[ap] = _max_date(
            f"SELECT MAX(trading_day) FROM daily_features WHERE symbol = '{instrument}' AND orb_minutes = {ap}"
        )
    result["daily_features"] = df_dates
    result["daily_features_min"] = min((d for d in df_dates.values() if d), default=None)

    # orb_outcomes
    result["orb_outcomes"] = _max_date(
        f"SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = '{instrument}'"
    )

    # experimental_strategies
    result["experimental"] = _max_date(
        f"SELECT MAX(trading_day) FROM experimental_strategies WHERE instrument = '{instrument}'"
    )

    # validated_setups (use promoted_at as freshness indicator)
    result["validated"] = _max_date(
        f"SELECT MAX(promoted_at::DATE) FROM validated_setups WHERE instrument = '{instrument}' AND status = 'active'"
    )

    # edge_families
    result["edge_families"] = _max_date(
        f"SELECT MAX(created_at::DATE) FROM edge_families WHERE instrument = '{instrument}'"
    )

    # family_rr_locks (may not have instrument column — check)
    try:
        result["family_rr_locks"] = _max_date(
            "SELECT MAX(locked_at::DATE) FROM family_rr_locks"
        )
    except Exception:
        result["family_rr_locks"] = None

    # Last rebuild from manifest
    try:
        row = con.execute(
            f"""SELECT status, completed_at, failed_step
            FROM rebuild_manifest
            WHERE instrument = '{instrument}'
            ORDER BY started_at DESC LIMIT 1"""
        ).fetchone()
        if row:
            result["last_rebuild"] = {
                "status": row[0],
                "completed_at": row[1],
                "failed_step": row[2],
            }
        else:
            result["last_rebuild"] = None
    except Exception:
        result["last_rebuild"] = None

    # Compute stale steps
    today = datetime.date.today()
    baseline = result["bars_1m"] or today
    stale = []

    if is_stale(result["bars_5m"], baseline):
        stale.append("bars_5m")
    if is_stale(result["daily_features_min"], baseline):
        stale.append("daily_features")
    if is_stale(result["orb_outcomes"], result["daily_features_min"] or baseline, max_gap_trading_days=1):
        stale.append("orb_outcomes")
    if is_stale(result["experimental"], result["orb_outcomes"] or baseline, max_gap_trading_days=1):
        stale.append("experimental")
    # validated/families/rr_locks: stale if experimental is newer
    if result["experimental"] and is_stale(result["validated"], result["experimental"] or baseline, max_gap_trading_days=1):
        stale.append("validated")
    if result["validated"] and is_stale(result["edge_families"], result["validated"] or baseline, max_gap_trading_days=1):
        stale.append("edge_families")

    result["stale_steps"] = stale
    return result


def format_status(instrument: str, status: dict) -> str:
    """Format staleness status as human-readable text."""
    lines = [f"\nInstrument: {instrument}"]
    today = datetime.date.today()

    def _fmt(label: str, date_val: datetime.date | None, ref: datetime.date | None = None) -> str:
        if date_val is None:
            return f"  {label:20s} NO DATA"
        ref = ref or today
        gap = _trading_days_between(date_val, ref)
        if gap <= 1:
            return f"  {label:20s} {date_val}  OK"
        return f"  {label:20s} {date_val}  STALE ({gap} trading days behind)"

    baseline = status["bars_1m"]
    lines.append(_fmt("bars_1m:", baseline, today))
    lines.append(_fmt("bars_5m:", status["bars_5m"], baseline))

    # daily_features per aperture
    df_dates = status["daily_features"]
    for ap in APERTURES:
        lines.append(_fmt(f"daily_features O{ap}:", df_dates.get(ap), baseline))

    lines.append(_fmt("orb_outcomes:", status["orb_outcomes"], status["daily_features_min"]))
    lines.append(_fmt("experimental:", status["experimental"], status["orb_outcomes"]))
    lines.append(_fmt("validated:", status["validated"], status["experimental"]))
    lines.append(_fmt("edge_families:", status["edge_families"], status["validated"]))
    lines.append(_fmt("family_rr_locks:", status["family_rr_locks"], status["edge_families"]))

    # Last rebuild
    lr = status.get("last_rebuild")
    if lr:
        lines.append(f"  Last rebuild:       {lr['status']}" + (f" (failed at {lr['failed_step']})" if lr["failed_step"] else ""))
    else:
        lines.append("  Last rebuild:       NEVER")

    # Recommendation
    stale = status["stale_steps"]
    if stale:
        lines.append(f"\n  ACTION NEEDED: {', '.join(stale)} are stale")
        lines.append(f"  Run: python scripts/tools/pipeline_status.py --rebuild --instrument {instrument}")
    else:
        lines.append("\n  STATUS: UP TO DATE")

    return "\n".join(lines)
```

**Step 4: Run tests**

Run: `pytest tests/test_pipeline/test_pipeline_status.py -v`
Expected: Most staleness tests PASS. Fix any import path issues.

**Step 5: Commit**

```bash
git add scripts/tools/pipeline_status.py tests/test_pipeline/test_pipeline_status.py
git commit -m "feat: staleness engine — core query logic for pipeline_status"
```

---

### Task 2: Pre-Flight Checks

**Files:**
- Modify: `scripts/tools/pipeline_status.py`
- Test: `tests/test_pipeline/test_pipeline_status.py` (append)

**Step 1: Write failing test**

Append to test file:

```python
def test_preflight_daily_features_missing():
    """Pre-flight fails when daily_features has no O15 data."""
    from scripts.tools.pipeline_status import preflight_check

    con = _create_test_db()
    # Only O5 data exists
    con.execute(
        """INSERT INTO daily_features (trading_day, symbol, orb_minutes)
        VALUES ('2026-03-05', 'MGC', 5)"""
    )

    ok, msg = preflight_check(con, "MGC", "outcome_builder", orb_minutes=15)
    assert not ok
    assert "daily_features" in msg
    assert "O15" in msg or "15" in msg
    con.close()


def test_preflight_daily_features_present():
    """Pre-flight passes when daily_features has data for aperture."""
    from scripts.tools.pipeline_status import preflight_check

    con = _create_test_db()
    con.execute(
        """INSERT INTO daily_features (trading_day, symbol, orb_minutes)
        VALUES ('2026-03-05', 'MGC', 15)"""
    )

    ok, msg = preflight_check(con, "MGC", "outcome_builder", orb_minutes=15)
    assert ok
    con.close()
```

**Step 2: Run test — verify fail**

Run: `pytest tests/test_pipeline/test_pipeline_status.py -v -k "preflight"`
Expected: FAIL — `preflight_check` not defined

**Step 3: Add preflight_check to pipeline_status.py**

```python
# Pre-flight checks — verify prerequisites before each rebuild step
PREFLIGHT_RULES = {
    "outcome_builder": {
        "table": "daily_features",
        "query": "SELECT COUNT(*) FROM daily_features WHERE symbol = '{instrument}' AND orb_minutes = {orb_minutes}",
        "msg": "daily_features missing for {instrument} O{orb_minutes} — run: python pipeline/build_daily_features.py --instrument {instrument} --orb-minutes {orb_minutes}",
    },
    "strategy_discovery": {
        "table": "orb_outcomes",
        "query": "SELECT COUNT(*) FROM orb_outcomes WHERE symbol = '{instrument}' AND orb_minutes = {orb_minutes}",
        "msg": "orb_outcomes missing for {instrument} O{orb_minutes} — run outcome_builder first",
    },
    "strategy_validator": {
        "table": "experimental_strategies",
        "query": "SELECT COUNT(*) FROM experimental_strategies WHERE instrument = '{instrument}'",
        "msg": "experimental_strategies empty for {instrument} — run strategy_discovery first",
    },
    "build_edge_families": {
        "table": "validated_setups",
        "query": "SELECT COUNT(*) FROM validated_setups WHERE instrument = '{instrument}' AND status = 'active'",
        "msg": "No active validated_setups for {instrument} — run strategy_validator first",
    },
    "select_family_rr": {
        "table": "edge_families",
        "query": "SELECT COUNT(*) FROM edge_families WHERE instrument = '{instrument}'",
        "msg": "No edge_families for {instrument} — run build_edge_families first",
    },
}


def preflight_check(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    step: str,
    orb_minutes: int = 5,
) -> tuple[bool, str]:
    """Check prerequisites for a rebuild step. Returns (ok, message)."""
    rule = PREFLIGHT_RULES.get(step)
    if not rule:
        return True, f"No pre-flight rule for {step}"

    query = rule["query"].format(instrument=instrument, orb_minutes=orb_minutes)
    count = con.execute(query).fetchone()[0]

    if count == 0:
        msg = rule["msg"].format(instrument=instrument, orb_minutes=orb_minutes)
        return False, f"PRE-FLIGHT FAIL: {msg}"

    return True, f"Pre-flight OK: {rule['table']} has {count} rows for {instrument}"
```

**Step 4: Run tests**

Run: `pytest tests/test_pipeline/test_pipeline_status.py -v -k "preflight"`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/tools/pipeline_status.py tests/test_pipeline/test_pipeline_status.py
git commit -m "feat: pre-flight checks for rebuild steps"
```

---

### Task 3: Rebuild Manifest — Write/Read/Resume

**Files:**
- Modify: `scripts/tools/pipeline_status.py`
- Test: `tests/test_pipeline/test_pipeline_status.py` (append)

**Step 1: Write failing tests**

```python
import uuid


def test_manifest_write_and_read():
    """Write a rebuild manifest record and read it back."""
    from scripts.tools.pipeline_status import write_manifest, read_last_manifest

    con = _create_test_db()
    rid = str(uuid.uuid4())
    write_manifest(con, rid, "MGC", "COMPLETED", steps_completed=["outcome_builder", "discovery"], trigger="MANUAL")

    last = read_last_manifest(con, "MGC")
    assert last is not None
    assert last["rebuild_id"] == rid
    assert last["status"] == "COMPLETED"
    assert "outcome_builder" in last["steps_completed"]
    con.close()


def test_manifest_resume_from_failed():
    """Resume reads last FAILED manifest and returns failed_step."""
    from scripts.tools.pipeline_status import write_manifest, get_resume_point

    con = _create_test_db()
    rid = str(uuid.uuid4())
    write_manifest(
        con, rid, "MGC", "FAILED",
        failed_step="strategy_validator",
        steps_completed=["outcome_builder", "discovery"],
        trigger="MANUAL",
    )

    resume = get_resume_point(con, "MGC")
    assert resume is not None
    assert resume["failed_step"] == "strategy_validator"
    assert resume["steps_completed"] == ["outcome_builder", "discovery"]
    con.close()
```

**Step 2: Run test — verify fail**

Run: `pytest tests/test_pipeline/test_pipeline_status.py -v -k "manifest"`

**Step 3: Add manifest functions to pipeline_status.py**

```python
def write_manifest(
    con: duckdb.DuckDBPyConnection,
    rebuild_id: str,
    instrument: str,
    status: str,
    failed_step: str | None = None,
    steps_completed: list[str] | None = None,
    trigger: str = "MANUAL",
) -> None:
    """Write or update a rebuild manifest record."""
    now = datetime.datetime.now(datetime.timezone.utc)
    completed_at = now if status in ("COMPLETED", "FAILED") else None
    steps = steps_completed or []

    con.execute(
        """INSERT OR REPLACE INTO rebuild_manifest
        (rebuild_id, instrument, started_at, completed_at, status, failed_step, steps_completed, trigger)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
        [rebuild_id, instrument, now, completed_at, status, failed_step, steps, trigger],
    )


def read_last_manifest(con: duckdb.DuckDBPyConnection, instrument: str) -> dict | None:
    """Read the most recent rebuild manifest for instrument."""
    row = con.execute(
        """SELECT rebuild_id, instrument, started_at, completed_at, status,
                  failed_step, steps_completed, trigger
        FROM rebuild_manifest
        WHERE instrument = $1
        ORDER BY started_at DESC LIMIT 1""",
        [instrument],
    ).fetchone()
    if not row:
        return None
    return {
        "rebuild_id": row[0],
        "instrument": row[1],
        "started_at": row[2],
        "completed_at": row[3],
        "status": row[4],
        "failed_step": row[5],
        "steps_completed": row[6] or [],
        "trigger": row[7],
    }


def get_resume_point(con: duckdb.DuckDBPyConnection, instrument: str) -> dict | None:
    """Get the resume point from the last FAILED rebuild."""
    row = con.execute(
        """SELECT rebuild_id, failed_step, steps_completed
        FROM rebuild_manifest
        WHERE instrument = $1 AND status = 'FAILED'
        ORDER BY started_at DESC LIMIT 1""",
        [instrument],
    ).fetchone()
    if not row:
        return None
    return {
        "rebuild_id": row[0],
        "failed_step": row[1],
        "steps_completed": row[2] or [],
    }
```

**Step 4: Run tests**

Run: `pytest tests/test_pipeline/test_pipeline_status.py -v -k "manifest"`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/tools/pipeline_status.py tests/test_pipeline/test_pipeline_status.py
git commit -m "feat: rebuild manifest write/read/resume"
```

---

### Task 4: Rebuild Orchestration — subprocess chain

**Files:**
- Modify: `scripts/tools/pipeline_status.py`
- Test: `tests/test_pipeline/test_pipeline_status.py` (append)

**Step 1: Write failing test**

```python
from unittest.mock import patch, MagicMock


def test_rebuild_chain_dry_run():
    """Dry run lists steps without executing."""
    from scripts.tools.pipeline_status import build_step_list

    steps = build_step_list("MGC", resume_from=None)
    step_names = [s["name"] for s in steps]
    assert "outcome_builder_O5" in step_names
    assert "outcome_builder_O15" in step_names
    assert "outcome_builder_O30" in step_names
    assert "discovery_O5" in step_names
    assert "validator" in step_names
    assert "retire_e3" in step_names
    assert "edge_families" in step_names
    assert "family_rr_locks" in step_names
    assert "health_check" in step_names


def test_rebuild_chain_resume_skips_completed():
    """Resume skips already-completed steps."""
    from scripts.tools.pipeline_status import build_step_list

    steps = build_step_list("MGC", resume_from=["outcome_builder_O5", "outcome_builder_O15", "outcome_builder_O30"])
    step_names = [s["name"] for s in steps]
    assert "outcome_builder_O5" not in step_names
    assert "discovery_O5" in step_names  # not yet completed
```

**Step 2: Run test — verify fail**

**Step 3: Add rebuild orchestration to pipeline_status.py**

```python
import subprocess
import uuid as _uuid

# The canonical rebuild chain — order matters
REBUILD_STEPS = [
    # (name, command_template)
    ("outcome_builder_O5", "python trading_app/outcome_builder.py --instrument {instrument} --force --orb-minutes 5"),
    ("outcome_builder_O15", "python trading_app/outcome_builder.py --instrument {instrument} --force --orb-minutes 15"),
    ("outcome_builder_O30", "python trading_app/outcome_builder.py --instrument {instrument} --force --orb-minutes 30"),
    ("discovery_O5", "python trading_app/strategy_discovery.py --instrument {instrument} --orb-minutes 5"),
    ("discovery_O15", "python trading_app/strategy_discovery.py --instrument {instrument} --orb-minutes 15"),
    ("discovery_O30", "python trading_app/strategy_discovery.py --instrument {instrument} --orb-minutes 30"),
    ("validator", "python trading_app/strategy_validator.py --instrument {instrument} --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75"),
    ("retire_e3", "python scripts/migrations/retire_e3_strategies.py"),
    ("edge_families", "python scripts/tools/build_edge_families.py --instrument {instrument}"),
    ("family_rr_locks", "python scripts/tools/select_family_rr.py"),
    ("repo_map", "python scripts/tools/gen_repo_map.py"),
    ("health_check", "python pipeline/health_check.py"),
    ("pinecone_sync", "python scripts/tools/sync_pinecone.py"),
]


def build_step_list(instrument: str, resume_from: list[str] | None = None) -> list[dict]:
    """Build the list of steps to execute, skipping completed ones on resume."""
    skip = set(resume_from or [])
    steps = []
    for name, cmd_template in REBUILD_STEPS:
        if name in skip:
            continue
        cmd = cmd_template.format(instrument=instrument)
        steps.append({"name": name, "cmd": cmd})
    return steps


def run_rebuild(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    dry_run: bool = False,
    resume: bool = False,
    trigger: str = "CLI",
) -> bool:
    """Execute the rebuild chain for one instrument. Returns True on success."""
    rebuild_id = str(_uuid.uuid4())

    # Resume logic
    resume_from = None
    if resume:
        rp = get_resume_point(con, instrument)
        if rp:
            resume_from = rp["steps_completed"]
            print(f"Resuming from after: {', '.join(resume_from)}")
        else:
            print(f"No failed rebuild found for {instrument} — starting fresh")

    steps = build_step_list(instrument, resume_from=resume_from)
    completed = list(resume_from or [])

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — Rebuild chain for {instrument}")
        print(f"{'='*60}")
        for s in steps:
            print(f"  [{s['name']}] {s['cmd']}")
        print(f"\n{len(steps)} steps would execute.")
        return True

    print(f"\n{'='*60}")
    print(f"REBUILD — {instrument} ({len(steps)} steps)")
    print(f"{'='*60}")

    write_manifest(con, rebuild_id, instrument, "RUNNING", trigger=trigger)

    for i, step in enumerate(steps):
        print(f"\n[{i+1}/{len(steps)}] {step['name']}")
        print(f"  CMD: {step['cmd']}")

        # Pre-flight check (for steps that have one)
        step_base = step["name"].split("_O")[0]  # e.g. "outcome_builder"
        orb_min = 5
        if "_O" in step["name"]:
            orb_min = int(step["name"].split("_O")[1])
        ok, msg = preflight_check(con, instrument, step_base, orb_minutes=orb_min)
        if not ok:
            print(f"  {msg}")
            write_manifest(con, rebuild_id, instrument, "FAILED",
                         failed_step=step["name"], steps_completed=completed, trigger=trigger)
            return False

        # Execute
        result = subprocess.run(step["cmd"], shell=True, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            write_manifest(con, rebuild_id, instrument, "FAILED",
                         failed_step=step["name"], steps_completed=completed, trigger=trigger)
            return False

        print(f"  PASSED")
        completed.append(step["name"])

    write_manifest(con, rebuild_id, instrument, "COMPLETED",
                 steps_completed=completed, trigger=trigger)
    print(f"\nREBUILD COMPLETE — {instrument}")
    return True
```

**Step 4: Run tests**

Run: `pytest tests/test_pipeline/test_pipeline_status.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add scripts/tools/pipeline_status.py tests/test_pipeline/test_pipeline_status.py
git commit -m "feat: rebuild orchestration with step-level resume"
```

---

### Task 5: CLI Interface — argparse + main()

**Files:**
- Modify: `scripts/tools/pipeline_status.py`

**Step 1: Add CLI**

```python
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline status — staleness detection and rebuild orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/tools/pipeline_status.py --status
  python scripts/tools/pipeline_status.py --status --instrument MGC
  python scripts/tools/pipeline_status.py --rebuild --instrument MGC
  python scripts/tools/pipeline_status.py --rebuild --instrument MGC --dry-run
  python scripts/tools/pipeline_status.py --rebuild-all
  python scripts/tools/pipeline_status.py --resume --instrument MGC
        """,
    )
    parser.add_argument("--status", action="store_true", help="Show staleness status")
    parser.add_argument("--rebuild", action="store_true", help="Run rebuild chain for one instrument")
    parser.add_argument("--rebuild-all", action="store_true", help="Run rebuild chain for all stale instruments")
    parser.add_argument("--resume", action="store_true", help="Resume last failed rebuild")
    parser.add_argument("--instrument", help="Instrument symbol (e.g. MGC)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would execute")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH, help="Database path")
    args = parser.parse_args()

    if not any([args.status, args.rebuild, args.rebuild_all, args.resume]):
        args.status = True  # Default to status

    con = duckdb.connect(str(args.db_path))

    try:
        if args.status:
            instruments = [args.instrument] if args.instrument else ACTIVE_ORB_INSTRUMENTS
            from zoneinfo import ZoneInfo
            now = datetime.datetime.now(ZoneInfo("Australia/Brisbane"))
            print(f"Pipeline Status -- {now.strftime('%Y-%m-%d %H:%M')} Brisbane")
            print("=" * 55)
            for inst in instruments:
                status = staleness_engine(con, inst)
                print(format_status(inst, status))

        elif args.rebuild:
            if not args.instrument:
                parser.error("--rebuild requires --instrument")
            success = run_rebuild(con, args.instrument, dry_run=args.dry_run, trigger="CLI")
            sys.exit(0 if success else 1)

        elif args.rebuild_all:
            for inst in ACTIVE_ORB_INSTRUMENTS:
                status = staleness_engine(con, inst)
                if status["stale_steps"]:
                    print(f"\n{inst}: stale ({', '.join(status['stale_steps'])})")
                    if not args.dry_run:
                        success = run_rebuild(con, inst, trigger="CLI")
                        if not success:
                            print(f"REBUILD FAILED for {inst} — stopping")
                            sys.exit(1)
                    else:
                        run_rebuild(con, inst, dry_run=True, trigger="CLI")
                else:
                    print(f"\n{inst}: up to date")

        elif args.resume:
            if not args.instrument:
                parser.error("--resume requires --instrument")
            success = run_rebuild(con, args.instrument, resume=True, trigger="CLI")
            sys.exit(0 if success else 1)
    finally:
        con.close()


if __name__ == "__main__":
    main()
```

**Step 2: Manual smoke test**

Run: `python scripts/tools/pipeline_status.py --status`
Expected: Status table for all 4 instruments showing current dates and staleness

Run: `python scripts/tools/pipeline_status.py --rebuild --instrument MGC --dry-run`
Expected: List of 13 steps that would execute

**Step 3: Commit**

```bash
git add scripts/tools/pipeline_status.py
git commit -m "feat: pipeline_status CLI — status, rebuild, resume, dry-run"
```

---

### Task 6: Drift Check — Staleness Gate

**Files:**
- Modify: `pipeline/check_drift.py`

**Step 1: Read current file to find insertion points**

Read `pipeline/check_drift.py` — find the CHECKS list (lines ~3055-3223) and identify the last check number (69).

**Step 2: Add check function**

Add before the CHECKS list:

```python
def check_pipeline_staleness() -> tuple[bool, str]:
    """Fail if any active instrument has orb_outcomes > 7 trading days behind daily_features."""
    try:
        con = _get_shared_con()
        if con is None:
            return True, "DB unavailable — skipped"
    except Exception:
        return True, "DB unavailable — skipped"

    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

    stale_instruments = []
    for inst in ACTIVE_ORB_INSTRUMENTS:
        df_max = con.execute(
            f"SELECT MAX(trading_day) FROM daily_features WHERE symbol = '{inst}' AND orb_minutes = 5"
        ).fetchone()[0]
        oo_max = con.execute(
            f"SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = '{inst}'"
        ).fetchone()[0]

        if df_max is None or oo_max is None:
            continue  # No data yet — not a staleness issue

        # Count trading days between
        gap = 0
        current = oo_max + datetime.timedelta(days=1)
        while current <= df_max:
            if current.weekday() < 5:
                gap += 1
            current += datetime.timedelta(days=1)

        if gap > 7:
            stale_instruments.append(f"{inst} ({gap}d behind)")

    if stale_instruments:
        return False, f"orb_outcomes stale: {', '.join(stale_instruments)}"
    return True, "orb_outcomes freshness OK for all instruments"
```

**Step 3: Add to CHECKS list**

Add tuple at the end of the CHECKS list (check #70):

```python
    (
        "Pipeline staleness: orb_outcomes not >7 trading days behind daily_features",
        lambda: check_pipeline_staleness(),
        False,  # blocking
        True,   # requires_db
    ),
```

**Step 4: Run drift check to verify**

Run: `python pipeline/check_drift.py`
Expected: New check #70 appears and passes (or correctly reports staleness)

**Step 5: Commit**

```bash
git add pipeline/check_drift.py
git commit -m "feat: drift check #70 — pipeline staleness gate (orb_outcomes vs daily_features)"
```

---

### Task 7: Health Check Integration

**Files:**
- Modify: `pipeline/health_check.py`

**Step 1: Read current file to find insertion point**

Read `pipeline/health_check.py` — find where fast_checks / slow_checks are defined.

**Step 2: Add staleness check function**

Add a new check function (before the check lists):

```python
def check_staleness():
    """Report pipeline staleness (advisory — doesn't block health check)."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.tools.pipeline_status import staleness_engine
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        stale = []
        for inst in ACTIVE_ORB_INSTRUMENTS:
            status = staleness_engine(con, inst)
            if status["stale_steps"]:
                stale.append(f"{inst}: {', '.join(status['stale_steps'])}")
        con.close()

        if stale:
            return True, f"STALENESS WARNING: {'; '.join(stale)}"
        return True, "All instruments up to date"
    except Exception as e:
        return True, f"Staleness check skipped: {e}"
```

**Step 3: Add to fast_checks list**

```python
fast_checks = [
    check_python_deps,
    check_database,
    check_staleness,  # <-- add here
    # ... existing checks
]
```

**Step 4: Run health check**

Run: `python pipeline/health_check.py`
Expected: New staleness check appears in output

**Step 5: Commit**

```bash
git add pipeline/health_check.py
git commit -m "feat: health check staleness advisory (non-blocking)"
```

---

### Task 8: Shell Script Manifest Writes

**Files:**
- Modify: `scripts/tools/run_rebuild_with_sync.sh`

**Step 1: Read current file**

**Step 2: Add manifest write after successful completion**

Add before the final "DONE" echo:

```bash
# Write rebuild manifest (success)
echo ""
echo "--- Writing rebuild manifest ---"
python scripts/tools/pipeline_status.py --write-manifest \
  --instrument "$INSTRUMENT" --status COMPLETED --trigger SHELL 2>/dev/null || true
```

And add a trap for failure:

```bash
# At the top of the script, after set -e:
trap 'python scripts/tools/pipeline_status.py --write-manifest --instrument "$INSTRUMENT" --status FAILED --trigger SHELL 2>/dev/null || true' ERR
```

**Step 3: Add `--write-manifest` flag to pipeline_status.py CLI**

Add to argparse:

```python
parser.add_argument("--write-manifest", action="store_true", help="Write a manifest record (used by shell scripts)")
parser.add_argument("--status-value", choices=["COMPLETED", "FAILED", "RUNNING"], help="Manifest status")
parser.add_argument("--trigger", default="CLI", help="Trigger type (CLI, SHELL, MANUAL)")
```

And in main():

```python
elif args.write_manifest:
    if not args.instrument or not args.status_value:
        parser.error("--write-manifest requires --instrument and --status-value")
    rid = str(_uuid.uuid4())
    write_manifest(con, rid, args.instrument, args.status_value, trigger=args.trigger or "CLI")
    print(f"Manifest written: {args.instrument} = {args.status_value}")
```

**Step 4: Commit**

```bash
git add scripts/tools/pipeline_status.py scripts/tools/run_rebuild_with_sync.sh
git commit -m "feat: shell script manifest writes on rebuild success/failure"
```

---

### Task 9: Final Integration + Ruff + Tests

**Files:**
- All modified files

**Step 1: Run ruff format**

```bash
ruff format scripts/tools/pipeline_status.py tests/test_pipeline/test_pipeline_status.py pipeline/check_drift.py pipeline/health_check.py pipeline/init_db.py
```

**Step 2: Run ruff check**

```bash
ruff check scripts/tools/pipeline_status.py tests/test_pipeline/test_pipeline_status.py pipeline/check_drift.py pipeline/health_check.py --fix
```

**Step 3: Run full test suite**

```bash
pytest tests/ -x -q
```

**Step 4: Run drift check**

```bash
python pipeline/check_drift.py
```

**Step 5: Run health check**

```bash
python pipeline/health_check.py
```

**Step 6: Smoke test the new tool**

```bash
python scripts/tools/pipeline_status.py --status
python scripts/tools/pipeline_status.py --rebuild --instrument MGC --dry-run
```

**Step 7: Final commit**

```bash
git add -A
git commit -m "feat: pipeline status tool — staleness detection, rebuild orchestration, manifest tracking"
```

---

### Task 10: Update CLAUDE.md + MEMORY.md

**Files:**
- Modify: `CLAUDE.md` (add pipeline_status.py to Key Commands)
- Modify: memory files

**Step 1: Add to CLAUDE.md Key Commands section**

```bash
# Pipeline Status (staleness + rebuild orchestration)
python scripts/tools/pipeline_status.py --status               # What's stale?
python scripts/tools/pipeline_status.py --rebuild --instrument MGC  # Rebuild one instrument
python scripts/tools/pipeline_status.py --rebuild-all           # Rebuild all stale
python scripts/tools/pipeline_status.py --resume --instrument MGC   # Resume failed rebuild
```

**Step 2: Update REPO_MAP**

```bash
python scripts/tools/gen_repo_map.py
```

**Step 3: Commit**

```bash
git add CLAUDE.md REPO_MAP.md
git commit -m "docs: add pipeline_status.py to key commands"
```

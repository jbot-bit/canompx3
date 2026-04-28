---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Infrastructure Hardening Design

**Date:** 2026-03-12
**Status:** Approved
**Architecture:** Modular hooks into existing orchestrator (Approach A)

## Problem

gold.db is a single file with no backup strategy, no audit trail, no post-rebuild data assertions,
and no write locking. The orchestrator (`pipeline_status.py`) is 90% complete but doesn't call
backup, logging, or assertion modules. Monitoring exists (CUSUM, fitness, Telegram) but is decoupled.

## Phased Delivery

### Phase 1 — Safety Net (protects data today) ✓ DONE
1. Automated DB backup + restore ✓
2. Immutable audit log table ✓
3. Advisory DB write locking ✓

### Phase 2 — Rebuild Quality (catches bad rebuilds) ✓ DONE
4. Post-rebuild assertion suite ✓
5. Orchestrator hooks (wires Phase 1 + 2 together) ✓

### Phase 3 — Observability (nice-to-have)
6. Monitoring integration (Telegram alerts on fitness changes)
7. Research script provenance
8. CI fixture DB

---

## Component Designs

### 1. Automated DB Backup + Restore

**File:** `scripts/infra/backup_db.py` (extend existing)

**Pre-rebuild backup:**
- Called BEFORE DuckDB connection opens (not after — `shutil.copy2()` on an open write connection is unsafe)
- `pipeline_status.py` `main()` restructured: backup → open connection → run_rebuild()
- If backup fails, abort rebuild (fail-closed)
- Uses existing `backup_db()` function

**Post-rebuild labeled backup:**
- After successful rebuild: `gold_REBUILD_{rebuild_id[:8]}_{stamp}.db`
- Separate from rolling backups (different retention: keep last 3 rebuilds)

**Restore command:**
- `python scripts/infra/backup_db.py --restore` → latest backup
- `python scripts/infra/backup_db.py --restore --file gold_20260312_140000.db` → specific
- Safety: snapshots current state as `gold_pre_restore_{stamp}.db` before overwriting
- Verify restored DB: open read-only, check `bars_1m` exists, check file size > 0

**Integrity verification:**
- After every backup: file size > 0, DuckDB opens read-only, `SELECT 1 FROM bars_1m LIMIT 0` succeeds
- Reject and delete corrupt backups immediately

**Not doing:** Compression, incremental backups, cloud sync. Full copy is fine (~500MB, <5s).

### 2. Immutable Audit Log

**New file:** `pipeline/audit_log.py`
**Schema change:** `pipeline/init_db.py` — new `pipeline_audit_log` table

```sql
CREATE TABLE IF NOT EXISTS pipeline_audit_log (
    log_id        TEXT        PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    operation     TEXT        NOT NULL,  -- 'INGEST', 'BUILD_5M', 'OUTCOME_BUILDER', etc.
    table_name    TEXT        NOT NULL,  -- 'bars_1m', 'orb_outcomes', etc.
    instrument    TEXT,
    date_start    DATE,
    date_end      DATE,
    rows_before   INTEGER,
    rows_after    INTEGER,
    duration_s    DOUBLE,
    git_sha       TEXT,
    rebuild_id    TEXT,                  -- links to rebuild_manifest (NULL if standalone)
    status        TEXT        NOT NULL   -- 'SUCCESS', 'FAILED', 'SKIPPED'
);
```

**API:**
```python
def log_operation(con, operation, table_name, instrument=None, ...) -> str:
    """Append one audit log row. Returns log_id. Never UPDATE or DELETE."""

def get_previous_counts(con, instrument, table_name) -> int | None:
    """Last rows_after for this instrument+table from most recent SUCCESS log."""
```

**Append-only rule:** `log_operation()` only does INSERT. No UPDATE, no DELETE exposed.
Drift check will enforce no DELETE/UPDATE on pipeline_audit_log in production code.

### 3. Advisory DB Write Lock

**New file:** `pipeline/db_lock.py`

**Mechanism:** PID-based `.lock` file (not `fcntl` — doesn't exist on Windows).
- Create `gold.db.lock` with JSON: `{"pid": 1234, "script": "outcome_builder.py", "started": "..."}`
- On acquire: check if lock file exists → if yes, check if PID is alive → if dead, steal lock (stale)
- On acquire: if PID is alive, wait up to 30s with 1s polling → if still held, abort with clear error
- On release: delete lock file
- Context manager: `with PipelineLock("outcome_builder"):`

**Integration:** Orchestrator wraps entire `run_rebuild()` in `PipelineLock`. Individual scripts
(outcome_builder, strategy_discovery, etc.) do NOT acquire locks — the orchestrator holds it for
the entire chain. Direct script invocation outside orchestrator is still allowed (no lock required
for development/debugging).

**Not doing:** Distributed locking, WAL mode changes, mandatory locking for all scripts.

### 4. Post-Rebuild Assertion Suite

**New file:** `scripts/tools/assert_rebuild.py`

**Assertions:**

| ID | Assertion | Severity | Logic |
|----|-----------|----------|-------|
| A1 | Row count decrease | WARNING | rows_after < rows_before for same instrument+table in audit log |
| A2 | Date continuity | FAIL | Gaps > 3 calendar days in bars_1m for any active instrument |
| A3 | Cross-table FK | FAIL | orb_outcomes rows without matching daily_features (symbol+trading_day+orb_minutes) |
| A4 | Strategy count drop | WARNING | validated_setups active count < 70% of previous rebuild (from audit log). Threshold configurable. |
| A5 | Outcome coverage | FAIL | Any enabled session missing outcomes for any aperture (5/15/30) |
| A6 | Schema alignment | FAIL | daily_features column count != expected (from init_db.py ORB_LABELS) |

**Severity model:**
- FAIL: Blocks. Operator must investigate. Orchestrator marks rebuild WARNING in manifest.
- WARNING: Logged to audit log + stdout. Rebuild continues.

**Standalone usage:** `python scripts/tools/assert_rebuild.py [--instrument MGC]`

**Context for thresholds:** A4's 70% threshold is configurable because major model changes
(like E0 purge) legitimately drop strategy counts. The assertion compares against the
PREVIOUS rebuild's audit log entry, not a hardcoded absolute.

### 5. Orchestrator Hooks

**File:** `scripts/tools/pipeline_status.py` (modify existing)

**Hook points in `run_rebuild()`:**

```
main() {
    1. backup_db()                          # Phase 1: pre-rebuild backup
    2. acquire PipelineLock                  # Phase 1: write lock
    3. open DuckDB connection
    4. write_manifest(RUNNING)
    5. for each step:
       a. preflight_check()                 # existing
       b. rows_before = count(table)        # Phase 2: for audit log
       c. subprocess.run(step_cmd)          # existing
       d. rows_after = count(table)
       e. log_operation(step, rows_before, rows_after, duration)  # Phase 1: audit
    6. run_assertions()                     # Phase 2: post-rebuild gate
    7. write_manifest(COMPLETED)
    8. labeled_backup(rebuild_id)           # Phase 1: post-rebuild backup
    9. send_telegram(digest)               # Phase 3: notification
    10. release PipelineLock
}
```

**Error handling:** If step fails, audit log records FAILED, manifest records FAILED,
lock releases, Telegram alerts. Pre-rebuild backup is already in place for rollback.

### 6. Monitoring Integration

**Files:** `scripts/tools/pipeline_status.py` + `pipeline/health_check.py`

**Post-rebuild fitness comparison:**
- After rebuild, query `get_strategy_fitness(summary_only=True)` equivalent
- Compare FIT/WATCH/DECAY counts to previous rebuild's audit log
- If any strategy flipped FIT → DECAY, include in Telegram digest
- If total FIT count dropped >20%, Telegram WARNING

**Health digest flag:**
- `python pipeline/health_check.py --digest` → one-line Telegram-friendly summary
- Example: `"MGC: 45 FIT, 3 WATCH | MNQ: 89 FIT | rebuild OK 2026-03-12"`

**Not doing:** ATR regime shift detection (premature — no baseline), CUSUM→rebuild auto-trigger
(need human judgment on why drift occurred), dashboards.

### 7. Research Script Provenance

**Schema change:** `pipeline/init_db.py` — new `research_runs` table

```sql
CREATE TABLE IF NOT EXISTS research_runs (
    run_id            TEXT        PRIMARY KEY,
    script_path       TEXT        NOT NULL,
    started_at        TIMESTAMPTZ NOT NULL,
    completed_at      TIMESTAMPTZ,
    git_sha           TEXT,
    entry_models      TEXT[],
    session_catalog   TEXT[],
    instruments       TEXT[],
    date_range_start  DATE,
    date_range_end    DATE,
    parameters        TEXT,       -- JSON blob of script args
    status            TEXT        NOT NULL
);
```

**API:** `pipeline/audit_log.py` (extend, not new file):
```python
def log_research_run(con, script_path, instruments, date_start, date_end, **params) -> str:
    """Log a research run. Called manually by research scripts. Returns run_id."""

def complete_research_run(con, run_id, status="SUCCESS"):
    """Mark a research run as complete."""
```

**NOT a decorator.** Research scripts are too varied for magic. Researcher calls
`log_research_run()` at start, `complete_research_run()` at end. Two lines of code.

**Auto-captured:** git SHA (from subprocess), entry models (from `trading_app.config`),
session catalog keys (from `pipeline.dst.SESSION_CATALOG`).

### 8. CI Fixture DB

**New files:**
- `scripts/tools/gen_ci_fixture.py` — generates synthetic but schema-correct test data
- `tests/fixtures/ci_test.db` — committed to repo (~2MB)
- `tests/test_data_integrity/test_ci_data.py` — data integrity tests against fixture

**Fixture generator:**
- 5 trading days of synthetic data per active instrument
- Deterministic (seeded RNG) — same output every run
- Schema from `init_db.py` (canonical source)
- Covers: bars_1m, bars_5m, daily_features, orb_outcomes
- NOT production data — synthetic prices, synthetic ORBs

**CI workflow addition:**
```yaml
- name: Data integrity tests
  run: uv run pytest tests/test_data_integrity/ -v
```

**Refresh cadence:** Re-generate when schema changes. Drift check can flag staleness
(compare fixture column count to init_db.py expected columns).

---

## Files Created / Modified Summary

### New Files
| File | Purpose |
|------|---------|
| `pipeline/audit_log.py` | Audit log writes + research run logging |
| `pipeline/db_lock.py` | PID-based advisory write lock |
| `scripts/tools/assert_rebuild.py` | Post-rebuild data assertions |
| `scripts/tools/gen_ci_fixture.py` | Synthetic CI fixture generator |
| `tests/test_data_integrity/test_ci_data.py` | Data integrity tests |

### Modified Files
| File | Changes |
|------|---------|
| `scripts/infra/backup_db.py` | Add restore, verify, labeled backup |
| `scripts/tools/pipeline_status.py` | Hook in backup, lock, audit log, assertions, alerts |
| `pipeline/init_db.py` | Add pipeline_audit_log + research_runs tables |
| `pipeline/health_check.py` | Add --digest flag |
| `.github/workflows/ci.yml` | Add data integrity test step |
| `tests/conftest.py` | Add ci_db fixture |

### New Drift Checks
- No DELETE/UPDATE on pipeline_audit_log in production code
- CI fixture column count matches init_db.py expected columns

---

## What We're NOT Building
- Cloud backup sync
- Incremental/differential backups
- Distributed write locking
- Event bus / pub-sub architecture
- Auto-revert on assertion failure (operator decides)
- CUSUM → auto-rebuild trigger
- ATR regime shift dashboards
- Research run decorator magic
- Production data in CI fixtures

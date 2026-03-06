# Update Pipeline — Design Document

**Date:** 2026-03-06
**Status:** Design approved (4TP flow)
**Problem:** Every rebuild step is manual. No staleness detection, no "what needs updating" diagnostic, no scheduling, no failure alerts. Strategies drift silently from reality between manual rebuilds.

---

## 1. Orient — Current State

### What Exists (Working)

| Step | Script | Idempotent | Manual |
|------|--------|------------|--------|
| 1. Ingest DBN → bars_1m | `pipeline/ingest_dbn.py` | Yes (checkpoint) | Yes |
| 2. bars_1m → bars_5m | `pipeline/build_bars_5m.py` | Yes (DELETE+INSERT) | Yes |
| 3. bars_5m → daily_features | `pipeline/build_daily_features.py` | Yes (DELETE+INSERT) | Yes |
| 4. daily_features → orb_outcomes | `trading_app/outcome_builder.py` | Yes (--force) | Yes |
| 5. outcomes → experimental_strategies | `trading_app/strategy_discovery.py` | Yes (INSERT OR REPLACE) | Yes |
| 6. experimental → validated_setups | `trading_app/strategy_validator.py` | Yes (INSERT OR REPLACE) | Yes |
| 7. Retire E3 promotions | `scripts/migrations/retire_e3_strategies.py` | Yes | Yes |
| 8. validated → edge_families | `scripts/tools/build_edge_families.py` | Yes (truncate+rebuild) | Yes |
| 9. Family RR locks | `scripts/tools/select_family_rr.py` | Yes | Yes |
| 10. Health check | `pipeline/health_check.py` | N/A | Yes |
| 11. Pinecone sync | `scripts/tools/sync_pinecone.py` | Yes (delta hash) | Yes |

### What's Missing

1. **Staleness detection** — No way to know "daily_features is 5 days behind bars_1m"
2. **"What needs updating" diagnostic** — No single command showing pipeline status
3. **Pre-flight validation** — Rebuild chain doesn't check prerequisites (e.g. daily_features exists for O15 before outcome_builder runs)
4. **Step-level resume** — If step 5 fails, must re-run from step 1
5. **Scheduling** — No cron/task scheduler. All rebuilds are manual.
6. **Failure alerts** — No notification when something's stale or broken
7. **Rebuild metadata** — No record of "when was each instrument last rebuilt"

---

## 2. Design — The Update Pipeline Manager

### Approach: Single orchestrator script with staleness engine

**NOT building:** A daemon, a web service, or a cron system.
**Building:** A CLI tool (`scripts/tools/pipeline_status.py`) that answers two questions:
1. **"What's stale?"** — Compare max dates across tables per instrument/aperture
2. **"Run what's needed"** — Execute only the stale steps, in dependency order, with pre-flight checks

### Architecture

```
pipeline_status.py
  ├── staleness_engine()     → query max dates, compute gaps
  ├── preflight_checks()     → verify prerequisites before each step
  ├── run_rebuild_chain()    → execute steps with checkpointing
  ├── rebuild_manifest       → JSON log of what ran, when, pass/fail
  └── report()               → human-readable status table
```

### Data Model: Staleness Matrix

For each (instrument, aperture) tuple, track:

```
bars_1m.max_date       → baseline (freshest raw data)
bars_5m.max_date       → should match bars_1m
daily_features.max_date → should match bars_5m
orb_outcomes.max_date  → should match daily_features
experimental.max_date  → should match orb_outcomes
validated.promoted_at  → last validation run timestamp
edge_families.created_at → last family build timestamp
```

**Stale = any downstream table is behind its upstream by > 1 trading day.**

### Rebuild Manifest

Each rebuild writes a JSON record to `gold.db:rebuild_manifest`:

```sql
CREATE TABLE rebuild_manifest (
    rebuild_id TEXT PRIMARY KEY,        -- UUID
    instrument TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL,               -- RUNNING, COMPLETED, FAILED
    failed_step TEXT,                   -- NULL if success, step name if failed
    steps_completed TEXT[],             -- list of step names
    trigger TEXT NOT NULL               -- MANUAL, SCHEDULED, CLI
);
```

### Pre-Flight Validation

Before each step, verify its prerequisites:

| Step | Pre-flight Check |
|------|-----------------|
| outcome_builder O15 | `daily_features` has rows for `orb_minutes=15` and instrument |
| strategy_discovery | `orb_outcomes` has rows for instrument + aperture |
| strategy_validator | `experimental_strategies` has rows for instrument |
| build_edge_families | `validated_setups` has ACTIVE rows for instrument |
| select_family_rr | `edge_families` has rows |

**Fail-closed:** If pre-flight fails, abort with clear error message ("daily_features missing for MGC O15 — run `python pipeline/build_daily_features.py --instrument MGC --orb-minutes 15` first").

### Step-Level Resume

Track completed steps in `rebuild_manifest.steps_completed[]`. On resume:
- Read last FAILED rebuild for instrument
- Skip steps already in `steps_completed`
- Re-run from `failed_step`

Since every step is idempotent, re-running a completed step is safe but wasteful. Resume avoids this.

---

## 3. Detail — Implementation Steps

### File: `scripts/tools/pipeline_status.py`

**CLI interface:**

```bash
# Status report — what's stale?
python scripts/tools/pipeline_status.py --status

# Status for one instrument
python scripts/tools/pipeline_status.py --status --instrument MGC

# Run rebuild for stale instruments only
python scripts/tools/pipeline_status.py --rebuild --instrument MGC

# Run rebuild for ALL stale instruments
python scripts/tools/pipeline_status.py --rebuild-all

# Resume a failed rebuild
python scripts/tools/pipeline_status.py --resume --instrument MGC

# Dry run (show what would execute)
python scripts/tools/pipeline_status.py --rebuild --instrument MGC --dry-run
```

**Status output example:**

```
Pipeline Status — 2026-03-06 18:00 Brisbane
=============================================

Instrument: MGC
  bars_1m:          2026-03-05  (1 day behind live)
  bars_5m:          2026-03-05  OK
  daily_features:   2026-03-05  OK (O5, O15, O30)
  orb_outcomes:     2026-02-19  STALE (14 days behind daily_features)
  experimental:     2026-02-19  STALE
  validated:        2026-02-19  STALE (promoted_at)
  edge_families:    2026-02-19  STALE
  family_rr_locks:  2026-02-19  STALE
  Last rebuild:     2026-02-19  FAILED at step 4 (outcome_builder O15)

  ACTION NEEDED: Run --rebuild --instrument MGC
                 (will execute: outcome_builder → discovery → validator → retire_e3 → families → rr_locks → health_check → sync)

Instrument: MNQ
  bars_1m:          2026-03-05  (1 day behind live)
  bars_5m:          2026-03-05  OK
  daily_features:   2026-03-05  OK
  orb_outcomes:     2026-03-05  OK
  experimental:     2026-03-05  OK
  validated:        2026-03-05  OK
  edge_families:    2026-03-05  OK
  family_rr_locks:  2026-03-05  OK
  Last rebuild:     2026-03-05  COMPLETED

  STATUS: UP TO DATE
```

### File: `pipeline/init_db.py` (schema addition)

Add `rebuild_manifest` table to schema.

### File: `scripts/tools/run_rebuild_with_sync.sh` (modify)

After rebuild completes, write a COMPLETED record to `rebuild_manifest`. On failure, write FAILED with `failed_step`.

### Integration: Drift Check Addition

Add drift check #58 (or next available):
- `check_pipeline_staleness()` — BLOCKING (not advisory)
- Fails if ANY active instrument has `orb_outcomes` > 7 trading days behind `daily_features`
- This makes staleness a CI-visible failure

### Integration: Health Check Addition

Add to `health_check.py`:
- Import and run `pipeline_status.staleness_engine()`
- Report staleness as WARNING (not blocking, since health_check runs during rebuilds)

---

## 4. Validate — Risks & Mitigations

### Risks

| Risk | Mitigation |
|------|-----------|
| `rebuild_manifest` table migration on existing gold.db | `init_db.py` uses CREATE TABLE IF NOT EXISTS — safe |
| Pipeline status queries slow on large DB | All queries are `MAX(trading_day)` with index — fast |
| Step-level resume skips a step that actually needs re-run | Every step is idempotent; worst case = stale data persists until next full rebuild |
| Staleness drift check too aggressive (fails during weekend) | Use trading days, not calendar days. Weekend = 0 trading day gap. |
| Multiple instruments rebuild concurrently hit DB lock | DuckDB single-writer. Rebuild one instrument at a time (sequential). |

### What This Does NOT Build

- **No daemon/scheduler** — User runs `--rebuild` manually or wires it into Windows Task Scheduler themselves
- **No Slack/email alerts** — Just CLI output. User can pipe to notification if desired.
- **No auto-promotion to live_config** — Validated strategies still require manual review before going live
- **No ML model retraining trigger** — ML retraining remains manual (separate concern)

### Tests Required

1. `test_staleness_engine` — Mock DB with known dates, verify correct staleness detection
2. `test_preflight_checks` — Missing daily_features → clear error
3. `test_rebuild_manifest` — Write/read/resume roundtrip
4. `test_resume_from_failed_step` — Verify skips completed steps
5. `test_weekend_not_stale` — Weekend gap doesn't trigger false positive

### Drift Checks Required

1. `check_pipeline_staleness` — orb_outcomes not > 7 trading days behind daily_features
2. `check_rebuild_manifest_exists` — table exists in schema

### Rebuild Requirements

None — this is additive. No existing tables modified. No existing logic changed.

### Rollback Plan

Delete `scripts/tools/pipeline_status.py`, remove `rebuild_manifest` from `init_db.py`, remove new drift check. Zero impact on existing pipeline.

---

## 5. Scope Summary

| Deliverable | New/Modified | Lines (est) |
|-------------|-------------|-------------|
| `scripts/tools/pipeline_status.py` | NEW | ~400 |
| `pipeline/init_db.py` | MODIFY (add table) | +15 |
| `pipeline/check_drift.py` | MODIFY (add check) | +40 |
| `pipeline/health_check.py` | MODIFY (add staleness) | +20 |
| `scripts/tools/run_rebuild_with_sync.sh` | MODIFY (manifest write) | +10 |
| `tests/test_pipeline/test_pipeline_status.py` | NEW | ~200 |
| Total | | ~685 |

---

## 6. Dependency Chain

```
1. init_db.py (rebuild_manifest table)
   └── 2. pipeline_status.py (staleness engine + CLI)
       ├── 3. check_drift.py (staleness drift check)
       ├── 4. health_check.py (staleness warning)
       ├── 5. run_rebuild_with_sync.sh (manifest writes)
       └── 6. tests
```

Steps 3-5 can be parallelized after step 2.

# PASS 2 Audit Fixes — Design Document

**Date:** 2026-02-28
**Origin:** "IS THIS REAL?" audit, PASS 1 findings (AMBER items)
**Task IDs:** #1 (Phase A), #2 (Phase B), #3 (Phase C), #4 (Phase D)

## Context

PASS 1 audit found 0 RED items and 7 AMBER items. User approved 4 minimal fixes,
implemented as sequential phases with verification gates between each.

## Phase A: E3 Soft Retirement

**Problem:** E3 (limit retrace) has 50 validated strategies, 0 FDR-significant. 90-91%
fill rate means late adverse fills are included. No timeout mechanism exists.

**Decision:** Soft retire only — flag rows, do NOT delete. E3 remains in the codebase
for future timeout research.

**Files:**
- `scripts/migrations/retire_e3_strategies.py` (NEW) — sets `status='RETIRED'`,
  `retirement_reason='PASS2: 0/50 FDR-sig, no timeout mechanism'` on all E3 rows
  in `validated_setups`
- `pipeline/check_drift.py` — new drift check: warn if any E3 rows have
  `status != 'RETIRED'` in validated_setups

**Verification:** Run migration, confirm 50 rows updated, run drift check, run pytest.

## Phase B: Walk-Forward Soft Gate Columns

**Problem:** Walk-forward results stored in JSONL only, not queryable in DB. No way to
filter validated_setups by WF pass/fail without parsing JSONL.

**Decision:** Soft gate — add columns for visibility, do NOT block promotion. Log
`SKIPPED` warning for MGC/MES when WF data missing.

**Files:**
- `trading_app/db_manager.py` — add 3 columns to `validated_setups`:
  `wf_tested` (BOOLEAN), `wf_passed` (BOOLEAN), `wf_windows` (INTEGER)
- `scripts/migrations/backfill_wf_columns.py` (NEW) — reads existing JSONL,
  populates columns for all existing validated_setups rows
- `trading_app/strategy_validator.py` — after validation, populate WF columns
  from walkforward results
- `pipeline/check_drift.py` — new drift check: warn if MGC/MES strategies have
  `wf_tested = FALSE` or `wf_passed = FALSE`

**Verification:** Run migration, verify column counts match JSONL, run drift check,
run pytest.

## Phase C: Data Years Drift Check

**Problem:** MNQ and M2K have only ~5 years of data. Strategies validated on short
histories may not survive regime changes.

**Decision:** Add drift check that warns when any instrument's validated strategies
have `years_tested < 7`. Advisory only, not blocking.

**Files:**
- `pipeline/check_drift.py` — new drift check: query validated_setups for
  instruments where min(years_tested) < 7, emit warning with instrument and count

**Verification:** Run drift check, confirm MNQ/M2K trigger warning, MGC/MES do not.

## Phase D: E2 Slippage Stress Research Script

**Problem:** E2 stop-market entry assumes ORB + slippage ticks. Real-world slippage
varies by volatility, time of day, and instrument.

**Decision:** Research script only — no schema changes, no production impact.

**Files:**
- `research/research_e2_slippage_stress.py` (NEW) — stress-tests E2 outcomes at
  1x, 1.5x, 2x, 3x slippage multiples. Reports Sharpe/winrate degradation per
  instrument/session. Uses `cost_model.stress_test_costs()`.

**Verification:** Script runs without error, produces output table showing degradation
curves.

## Execution Order

```
Phase A (E3 soft retire)
  → verify: 50 rows flagged, drift passes, pytest passes
    → Phase B (WF soft gate)
      → verify: columns populated, drift passes, pytest passes
        → Phase C (data years drift)
          → verify: drift check warns for MNQ/M2K only
            → Phase D (E2 slippage stress)
              → verify: script runs, output table produced
```

## What This Does NOT Do

- Does NOT delete E3 strategies or remove E3 from the codebase
- Does NOT hard-gate promotion on walk-forward (soft columns only)
- Does NOT block validation for short-data instruments
- Does NOT modify production trading logic or outcome_builder
- Does NOT touch cost model parameters

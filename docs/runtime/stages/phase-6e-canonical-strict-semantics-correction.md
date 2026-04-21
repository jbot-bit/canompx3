---
slug: phase-6e-canonical-strict-semantics-correction
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Phase 6e correction — enforce canonical strict-less-than per 2026-02-08 spec for Alerts 2 & 3
worktree: deploy/live-trading-buildout-v1 (C:/Users/joshd/canompx3-deploy-live)
---

# Stage: Phase 6e correction — canonical strict-less-than semantics

## What went wrong

Sub-step 2.c (circuit_break) shipped with `<=`. Sub-step 2.d (wr_drift)
shipped with inclusive drop_pp gate. Both diverge from the 2026-02-08
Phase 6 spec line 422-428, which uses STRICT `<` for all four threshold
alerts (Drawdown, Circuit Break, WR Drift, ExpR Drift). Institutional-
rigor Rule #4 violation (re-encoded semantics).

## Scope Lock

- trading_app/live/detectors/circuit_break.py
- trading_app/live/detectors/wr_drift.py
- tests/test_trading_app/test_detectors_circuit_break.py
- tests/test_trading_app/test_detectors_wr_drift.py

## Changes

- circuit_break.py: `<=` -> `<`; boundary flips from FIRE to NO_FIRE at
  exactly -5.00.
- wr_drift.py: inclusive `>=` -> strict `>`; boundary flips at exactly 10pp.
- Tests: rename and flip boundary assertions. Add one new test per module
  covering the just-beyond-boundary case.

## Acceptance criteria

1. All detector + alert_engine tests green.
2. `pipeline/check_drift.py` exit 0.
3. Pre-commit hook passes.
4. Self-review: every boundary in Alerts 1-4 uses strict `<` per spec.
5. Dead-code sweep clean.

## Non-goals

- No Alert 1 changes (correct already).
- No alert_engine.py rule changes.
- No new detectors.

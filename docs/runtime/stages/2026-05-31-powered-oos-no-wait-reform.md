---
task: |
  Powered-OOS no-wait reform (operator-commissioned 2026-05-31).
  Make a powered trade-fraction OOS split the MANDATORY, non-bypassable
  deployment/discovery gate inside validate_strategy(). The sacred
  2026-01-01 calendar holdout is PRESERVED as read-only forward monitoring /
  leakage sentinel ONLY — never a deployment blocker. No strategy may be
  written/promoted/treated as deployable without a powered-OOS verdict.
  Fail-closed: if powered-OOS cannot be computed -> BLOCKED/FAILED, never
  NULL/PASSED. Decision: NO schema change — repurpose c8_oos_status as the
  powered-OOS DEPLOY gate; calendar monitor status derived/notes-field.
  Root cause proven: validate_strategy Phases 1-6 return PASSED without
  calling C8; _run_phase_4 gates run C8 as a separate pass that mostly never
  fired -> 768 MNQ NULL-C8 deployable rows, which is why validated MNQ
  session breadth (CME_PRECLOSE 0.577 ExpR etc) is not live-allocated.
mode: IMPLEMENTATION
updated: 2026-05-31
agent: claude (joshd)
stage: 1
total_stages: 4

## Scope Lock

Stage 1 (THIS stage — upstream-first, zero callers):
- research/oos_holdout.py
- tests/test_research/test_oos_holdout.py

Later stages (NOT this scope):
- Stage 2: trading_app/strategy_validator.py (wire inline gate)
- Stage 3: trading_app/holdout_policy.py (monitor-only helpers) + pipeline/check_drift.py (5 checks)
- Stage 4: docs (append-only) + read-only backlog audit (PASS 3)

## Blast Radius

- research/oos_holdout.py — NEW file, zero callers at Stage 1. Pure function
  powered_oos_split() operating on an in-memory ordered trade list. No DB
  writes. Delegates power math to research/oos_power.py (canonical, no
  re-encode). Imports must be side-effect-free (no scan-on-import).
- tests/test_research/test_oos_holdout.py — NEW test file.
- Reads: none at runtime (pure function); test fixtures only.
- Writes: none.
- Canonical-source dependency: research.oos_power._n_for_power /
  one_sample_power / power_verdict / POWER_TIERS — DELEGATE, never re-encode
  (institutional-rigor.md §4).
- Mandatory rules honored: temporal split (not random); exclude 2026+ from
  selection; fail-closed on underpowered/unavailable; honest provenance flag.

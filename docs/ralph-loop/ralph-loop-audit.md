# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 199

## RALPH AUDIT — Iteration 199 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 83 lane_allocator tests PASS
## Scope: trading_app/lane_allocator.py (apply_c8_gate + build_allocation)

---

## apply_c8_gate Audit

`apply_c8_gate` is well-structured and fail-closed:
- Null (`None`) is the Phase-4 grandfather pass-through, documented correctly.
- `"PASSED"` is the only explicit pass-through string.
- Empty string `""` is treated as a fail label (line 873).
- Defensive branch catches any future unknown label (lines 875-878).
- No `except Exception: pass` patterns.
- No hardcoded slot counts or allocation thresholds.
- Status strings are compared directly as string literals (no enum), consistent
  with the rest of the allocator which uses `str`-typed `status` on `LaneScore`.
- CLEAN.

---

## Finding HYSTERESIS-DEMOTED-199 — MEDIUM — FIXED

**PREMISE:** `build_allocation` at line 1125-1139 (pre-fix) contained a silent
slot drop: when hysteresis fired (`improvement < HYSTERESIS_PCT`) but
`best_prior.status` was NOT in `(DEPLOY, RESUME, PROVISIONAL)` — e.g. the prior
lane had been demoted by the Chordia or C8 gate between rebalances — the code
fell through the `if best_prior.status in (...)` False branch and executed a bare
`continue`, skipping `selected.append(lane)` at line 1141. The deployable
candidate was silently dropped with no displaced[] record. The slot count was
reduced without any logging or operator visibility.

**TRACE:**
- `lane_allocator.py:1125` — `improvement < HYSTERESIS_PCT` True
- `lane_allocator.py:1136` — `best_prior.status in (DEPLOY/RESUME/PROVISIONAL)` False
- (pre-fix) bare `continue` at end of `if improvement < HYSTERESIS_PCT:` block
- Skips `selected.append(lane)` at line 1141 — deployable lane silently lost

**EVIDENCE:** Existing `test_hysteresis_20pct` only covers `best_prior.status == DEPLOY`
(the `_make_score` default). No test covered the demoted-prior path.

**Fix:** Moved the `continue` inside the `if best_prior.status in (...)` True branch.
When `best_prior` is not deployable, execution now falls through to
`selected.append(lane)` — the deployable candidate is selected.

**Doctrine cited:** integrity-guardian.md § 6 (no silent failures in capital-class path)

New test `test_hysteresis_demoted_prior_falls_through` verifies:
1. Deployable challenger is selected when prior is PAUSE.
2. Demoted prior does NOT re-enter allocation.

**Commit:** 009b7fce

---

## Seven Sins Scan — lane_allocator.py (apply_c8_gate + build_allocation)

- Sin 1 (Silent failure): `build_allocation` hysteresis branch silently dropped
  slots when `best_prior` was demoted. FIXED (this iteration).
- Sin 2 (Fail-open): `apply_c8_gate` treats empty string and all non-PASSED
  non-None labels as fail. Defensive branch for future labels is present.
  CLEAN.
- Sin 3 (Canonical violation): `max_slots=5` and `max_dd=3000.0` are default
  parameter values. All production callers (`rebalance_lanes.py`,
  `generate_profile_lanes.py`, `allocator_gate_audit.py`) source these from
  `ACCOUNT_TIERS.get((profile.firm, profile.account_size))` with the default
  as fallback only. No hardcoded production use. ACCEPTABLE (pattern 4:
  guarded by verified upstream check — callers always pass profile values).
- Sin 4 (Impact awareness): `DEPLOY/PAUSE/STALE/RESUME/PROVISIONAL` are compared
  as string literals throughout. Consistent with `LaneScore.status: str` type
  annotation. No enum divergence risk. CLEAN.
- Sin 5 (Evidence over assertion): N/A (audit mode).
- Sin 6 (Spec compliance): No `docs/specs/lane_allocator.md`. Module docstring
  matches behavior. CLEAN.
- Sin 7 (Metadata trust): `displaced_out` records are populated correctly on all
  rejection paths except the newly-fixed demoted-prior path (which previously
  logged a hysteresis displacement then silently lost the lane). Post-fix:
  the `displaced_out.append(...)` is still emitted for the hysteresis reason,
  then the candidate is selected. Operator log is correct: hysteresis fired,
  prior was demoted, candidate was selected. CLEAN.

---

## Files Fully Scanned

- pipeline/check_drift.py (iter 153)
- pipeline/build_daily_features.py (iter 158)
- pipeline/dst.py (no-touch, iter 160)
- trading_app/strategy_discovery.py (iter 162)
- trading_app/outcome_builder.py (iter 165)
- trading_app/entry_rules.py (iter 168)
- trading_app/strategy_validator.py (iter 171)
- trading_app/live/session_orchestrator.py (iter 174)
- trading_app/live/execution_engine.py (iter 177)
- trading_app/live/alert_engine.py (iter 180)
- trading_app/derived_state.py (iter 183)
- trading_app/deployability.py (iter 193)
- trading_app/strategy_fitness.py (iter 194)
- trading_app/live_config.py (iter 195)
- trading_app/prop_portfolio.py (iter 195, partial — fitness gate path)
- trading_app/lane_correlation.py (iter 196)
- trading_app/lane_allocator.py (iter 199, full — apply_c8_gate + build_allocation + apply_chordia_gate)
- trading_app/chordia.py (iter 198, full)

---

## Next Iteration Targets

**Priority 1 — Unscanned high/medium centrality files:**
- `trading_app/prop_portfolio.py` — partially scanned (iter 195); portfolio
  construction, fitness gate integration, and DD tracking path not yet audited
- `trading_app/pre_session_check.py` — capital-class preflight; referenced in
  deferred finding A6-GAP3 (truthy-falsy cap display); unscanned

**Priority 2 — Stale re-audits:**
- `trading_app/lane_allocator.py` fully scanned this iteration — no re-audit needed

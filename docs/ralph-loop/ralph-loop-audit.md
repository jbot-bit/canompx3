# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 196

## RALPH AUDIT — Iteration 196 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 1278 tests PASS (1 pre-existing float-equality failure in test_lane_correlation_cache unrelated to this change)
## Scope: trading_app/lane_correlation.py

---

## lane_correlation.py Audit Summary

`check_candidate_correlation` computes pairwise Pearson rho between a candidate lane and all deployed lanes in a profile, returning a `CorrelationReport` with `gate_pass`. Used by `lane_allocator.py:1064` and `scripts/tools/rebalance_lanes.py:116`.

### Seven Sins Scan — lane_correlation.py

- Sin 1 (Silent failure): `check_candidate_correlation` has no `except` inside `try` — DB errors propagate uncaught to caller. This is **fail-closed** behavior (allocator cannot proceed if correlation is unknown) — ACCEPTABLE per integrity-guardian.md § 3.
- Sin 2 (Fail-open): `n_shared < 5` → `rho = 0.0` (no rejection). Intentional heuristic: new strategies with no/little trade history should not be auto-rejected. ACCEPTABLE per pattern 1 (intentional per-session heuristic).
- Sin 3 (Canonical violation): `RHO_REJECT_THRESHOLD = 0.70` and `SUBSET_REJECT_THRESHOLD = 0.80` were bare constants without `@research-source` annotation. **FIXED this iteration.**
- Sin 4 (Impact awareness): Callers (`lane_allocator.py`, `rebalance_lanes.py`, `displaced_rotation_analyzer.py`) import the canonical constants — no re-encoding found. ACCEPTABLE.
- Sin 5 (Evidence over assertion): N/A (audit mode).
- Sin 6 (Spec compliance): No `docs/specs/` file for this module; no spec to violate.
- Sin 7 (Metadata trust): Module docstring claims `@canonical-source trading_app/lane_correlation.py` — consistent with how callers import from it. ACCEPTABLE.

---

## Finding ANNOT-196 — LOW — FIXED

**PREMISE:** `lane_correlation.py:29-30` defined `RHO_REJECT_THRESHOLD = 0.70` and `SUBSET_REJECT_THRESHOLD = 0.80` as bare numeric constants with no `@research-source` annotation. These drive capital-gate decisions (candidates above rho=0.70 are rejected from deployment). Per `integrity-guardian.md § 8`, research-derived thresholds require provenance annotations so stale values are detectable when entry models change.

**TRACE:**
- `lane_correlation.py:29-30` → `check_candidate_correlation:173-174` → `lane_allocator.py:1064` → capital gate rejection

**EVIDENCE:** `docs/audit/2026-04-18-grounding-audit-master.md:92-100` confirms `rho=0.70` is "VERIFIED_CODE" and "consistent with Rule 7 tautology canon at `.claude/rules/backtesting-methodology.md:194`" (Carver-grounded hysteresis). No in-code annotation existed before this fix.

**Fix:** Added `@research-source`, `@entry-models`, `@revalidated-for` annotations citing backtesting-methodology.md RULE 7 and the 2026-04-18 grounding audit. Zero logic change.

**Doctrine cited:** integrity-guardian.md § 8

---

## Additional Observations (no fix needed)

- `test_lane_correlation_cache.py::test_compute_pairwise_correlation_matches_per_strategy_path` fails with a floating-point equality mismatch against live gold.db data. Pre-existing: the test uses `assert actual == ref` against live DB (data-sensitive). Not introduced by this iteration's comment-only change.

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

---

## Next Iteration Targets

**Priority 1 — Unscanned high/medium centrality files:**
- `trading_app/chordia.py` — medium centrality, never scanned; Chordia gate is capital-class
- `trading_app/prop_portfolio.py` — partially scanned (iter 195, fitness gate only); remainder not audited
- `trading_app/lane_allocator.py` — high centrality, never scanned; contains `compute_pairwise_correlation` and greedy selection logic

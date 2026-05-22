# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 197

## RALPH AUDIT — Iteration 197 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 82 lane_allocator tests PASS
## Scope: trading_app/lane_allocator.py

---

## lane_allocator.py Audit Summary

`_classify_status` accepts 8 keyword-only parameters but the function body only
reads 3 of them (`trailing_expr`, `trailing_n`, `session_regime_expr`). The five
dead parameters (`actual_months`, `months_neg`, `months_pos_since`, `annual_r`,
`monthly`) are remnants of individual-strategy pause/resume logic that was
intentionally removed after backtest 2022-2025 proved regime-only gating (+630R)
outperforms individual pause/resume (-799R). The values are still computed by the
caller and stored on `LaneScore` for display/reporting purposes — they just no
longer influence the classification decision.

Pyright flagged lines 554-557 and 559 as "not accessed." This matched the
institutional-rigor.md § 5 "No dead code" rule — dead parameters are dead
parameters, not "future use."

---

## Finding DEAD-197 — LOW — FIXED

**PREMISE:** `_classify_status` signature declared 5 parameters that its body
never read. Computed values with no effect on the decision path are a dead-code
violation (institutional-rigor.md § 5).

**TRACE:**
- `lane_allocator.py:550-559` — signature with 5 unused params
- `lane_allocator.py:462-471` — caller passing the 5 dead params
- `research/garch_a4b_binding_budget_replay.py:447-456` — second caller

**EVIDENCE:** Function body (lines 574-591) uses only `trailing_expr`,
`trailing_n`, `session_regime_expr`. Pyright "not accessed" on lines 554-557, 559
confirmed. The parameters are listed in the docstring as "backtest proved not
needed."

**Fix:** Removed 5 dead parameters from `_classify_status` signature and all
3 production call sites. 8 test call sites also updated. Commit 03847687.

**Doctrine cited:** institutional-rigor.md § 5 (No dead code)

---

## Seven Sins Scan — lane_allocator.py (partial — entry/greedy path)

- Sin 1 (Silent failure): Greedy selection in `build_allocation` uses plain
  `for` iteration without try/except — correlation failures propagate to the
  caller. Fail-closed behavior. ACCEPTABLE.
- Sin 2 (Fail-open): `load_sr_state()` fails-open (returns `{}`) on stale/missing
  SR state file. Documented in docstring. ACCEPTABLE (per integrity-guardian § 3,
  fail-open documented at canonical source).
- Sin 3 (Canonical violation): `_compute_session_regime` at line 540 hardcodes
  `entry_model='E2'` AND `rr_target=1.0`, `confirm_bars=1`, `orb_minutes=5`.
  These are the canonical regime-proxy parameters — unfiltered E2 RR1.0 CB1 O5
  as defined in the module docstring ("Session regime (unfiltered E2 RR1.0 CB1,
  6-month window)"). This is an intentional architectural choice (regime proxy,
  not a lane-specific setting). Per iter 187, `compute_pairwise_correlation`
  hardcoded E2 was FIXED (commit 052403aa). This remaining hardcode is in the
  *regime proxy* function — different role. ACCEPTABLE per pattern 1 (intentional
  per-session heuristic).
- Sin 4 (Impact awareness): `LaneScore.months_negative` and
  `months_positive_since_last_neg_streak` still appear on the dataclass and are
  still computed by the caller — they are live reporting fields even though they
  don't influence `_classify_status`. No action needed.
- Sin 5 (Evidence over assertion): N/A (audit mode).
- Sin 6 (Spec compliance): No `docs/specs/` file for lane_allocator; no spec to violate.
- Sin 7 (Metadata trust): Module docstring accurately describes the two-layer
  architecture. ACCEPTABLE.

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
- trading_app/lane_allocator.py (iter 197, partial — _classify_status + greedy path)

---

## Next Iteration Targets

**Priority 1 — Unscanned high/medium centrality files:**
- `trading_app/chordia.py` — medium centrality, never scanned; Chordia gate is capital-class
- `trading_app/prop_portfolio.py` — partially scanned (iter 195); remainder not audited
- `trading_app/lane_allocator.py` — continue remainder of file (greedy selection,
  `apply_chordia_gate`, `build_allocation` silent-failure surface)

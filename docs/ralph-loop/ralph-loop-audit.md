# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 202

## RALPH AUDIT — Iteration 202 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 96/96 prop_profiles tests PASS
## Scope: trading_app/prop_profiles.py (capital-class — first full scan)

---

## Full-File Audit Results

### Finding ANNOT-202 — LOW — FIXED

**PREMISE:** `prop_profiles.py:1397-1401` (pre-fix) — `_P90_ORB_PTS` table drives
`validate_dd_budget()` (capital gate) but carried no `@research-source` or
`@revalidated-for` annotation. MES=30.0 was explicitly labeled "Estimated (not
in active lanes)" — no empirical basis, no audit trail. Per integrity-guardian.md § 8:
never inline research stats without annotation.

**TRACE:** `validate_dd_budget()` → `_P90_ORB_PTS.get(inst, 100.0)` → used as
`effective_orb` when `lane.max_orb_size_pts is None`. MES=30.0 is fallback for
future MES profiles without explicit caps.

**Fix:** `prop_profiles.py:1395-1404` — Added `@research-source adversarial audit
2026-03-29`, `@revalidated-for E2 (MNQ/MGC measured; MES estimated)`, update
trigger comment, and clarified MES=30.0 as "ESTIMATED — measure before adding
MES profile."

**Doctrine cited:** integrity-guardian.md § 8

**Commit:** f746f5ec

---

## Seven Sins Scan — prop_profiles.py (full)

- Sin 1 (Silent failure / Fail-open): CLEAN.
  - `_try_read` at line 1205 and `load_paused_strategy_ids` at line 1354 both
    catch `(json.JSONDecodeError, OSError)` and return None/empty — fail-closed.
  - `load_paused_strategy_ids` fails-open for non-profile accounts (empty frozenset)
    but ACCEPTABLE: session_orchestrator.py:451-478 wraps this with RuntimeError
    for capital-class profile accounts. Per integrity-guardian.md pattern 4.
- Sin 2 (Canonical violation): CLEAN.
  - `_PV` inline copy (`{"MNQ": 2.0, "MGC": 10.0, "MES": 5.0}`) has runtime
    parity check in `validate_dd_budget()` raising RuntimeError on drift.
    Circular import documented at line 1113-1121. ACCEPTABLE — documented exception.
  - No hardcoded DB paths, session names in enforcement logic, or instrument lists.
- Sin 3 (Fail-open on capital gate): CLEAN post-analysis.
  - `validate_dd_budget` returns violations list, does not raise — caller
    (pre_session_check) decides severity. Correct design for a reporting function.
- Sin 4 (Impact awareness): CLEAN.
  - `allowed_sessions` frozensets in profiles are routing heuristics, not
    canonical lists. ACCEPTABLE (pattern 1).
- Sin 5 (Evidence over assertion): ANNOT-202 FIXED this iteration.
- Sin 6 (Spec compliance): CLEAN — no spec docs for prop_profiles found.
- Sin 7 (Never trust metadata): CLEAN — `parse_strategy_id` delegates to
  canonical `trading_app.eligibility.builder.parse_strategy_id`.

**Ralph-specific extensions scan:**
- Async safety: No async code in prop_profiles.py. CLEAN.
- State persistence gap: Module-level data structures (PROP_FIRM_SPECS,
  ACCOUNT_TIERS, ACCOUNT_PROFILES) are frozen dataclasses and dicts — no
  mutable in-memory state. CLEAN.
- Contract drift: `resolve_execution_symbol` signature stable; `parse_strategy_id`
  delegates to canonical builder. CLEAN.

**`is_express_funded` declarations:** AST-verified — all ACCOUNT_PROFILES entries
declare `is_express_funded=` explicitly. Drift check `check_account_profiles_declare_is_express_funded` passes.

---

## Files Fully Scanned

- pipeline/check_drift.py (iter 186)
- pipeline/build_daily_features.py (iter 187)
- pipeline/outcome_builder.py (iter 188)
- trading_app/eligibility/builder.py (iter 189)
- trading_app/strategy_discovery.py (iter 190)
- trading_app/strategy_validator.py (iter 191)
- trading_app/live/session_orchestrator.py (iter 192)
- trading_app/live/risk_manager.py (iter 193)
- trading_app/scoring.py (iter 194)
- trading_app/chordia.py (iter 195)
- trading_app/lane_allocator.py (iter 196)
- trading_app/lane_correlation.py (iter 197)
- trading_app/live/order_router.py / projectx/ (iter 198)
- trading_app/live/tradovate/ (iter 199)
- trading_app/live/alert_engine.py (iter 200)
- trading_app/pre_session_check.py (iter 201)
- trading_app/prop_profiles.py (iter 202)

---

## Next Iteration Targets

**Priority 1 (unscanned critical/high files):**
- `trading_app/live/bot_state.py` — state persistence for live sessions (ALERT-CONTAM-N2 class)
- `trading_app/config.py` — no-touch zone; audit only
- `scripts/tools/rebalance_lanes.py` — allocator writer; medium centrality

**Priority 2 (open deferred):**
- ALERT-CONTAM-N2 (MEDIUM) — test writes to production runtime paths
- PR301-TRADO-IDEMPOTENCY (MEDIUM) — Tradovate idempotency gap

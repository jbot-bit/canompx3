# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 195

## RALPH AUDIT — Iteration 195 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 46 prop_portfolio tests PASS
## Scope: trading_app/live_config.py (primary scan) + trading_app/prop_portfolio.py (fix target — discovered via fitness gate chain)

---

## Iteration 195 — trading_app/live_config.py (full scan)

### Auto-Targeting
- Scope provided: `trading_app/live_config.py` — capital-class, calls compute_fitness on regime gate path, never scanned.

---

## live_config.py Audit Summary

`build_live_portfolio()` is DEPRECATED (explicit DeprecationWarning + log.warning at lines 591-600). `session_orchestrator.py:341-346` raises if called without explicit portfolio injection. The LIVE_PORTFOLIO specs resolve to 0 strategies (filter types missing from validated_setups). The actual live fitness gate is `prop_portfolio.py:287-292` via `ACCOUNT_PROFILES`.

### Seven Sins Scan — live_config.py

- Sin 1 (Silent failure): `_check_noise_floor:529` fail-closed on NULL. `_check_dollar_gate:549` fail-closed on NULL median_risk_points with logger.warning. ACCEPTABLE.
- Sin 2 (Fail-open): Regime gate `except (ValueError, duckdb.Error)` at line 888 sets weight=0.0 — fail-closed. Inside DEPRECATED `build_live_portfolio`. ACCEPTABLE.
- Sin 3 (Canonical violation): `"FIT"` string literal at line 883 — consistent across codebase (no canonical constant exists). Assessed ACCEPTABLE in iter 28/194.
- Sin 4 (Impact awareness): `import duckdb as _ddb` at line 617 is a redundant local import (module-level `import duckdb` at line 27). LOW, dormant DEPRECATED path. ACCEPTABLE per pattern 2.
- Sin 5 (Evidence over assertion): N/A (audit mode).
- Sin 6 (Spec compliance): No spec in docs/specs/ for this module.
- Sin 7 (Metadata trust): LIVE_PORTFOLIO comment header correctly marks specs as DEPRECATED.

**Overall live_config.py verdict: CLEAN** (all findings either ACCEPTABLE or already fixed in prior iterations).

---

## Finding SILENT-195 — LOW — FIXED (discovered via fitness gate chain from live_config.py)

**PREMISE:** `prop_portfolio.py:291` has `except (ValueError, duckdb.Error): pass` — silent swallow of compute_fitness exceptions on the real live fitness gate path. Operator sees "Fitness: UNKNOWN" HOLD with zero diagnostic context.

**TRACE:**
- `prop_portfolio.py:287` → `fitness_status = "UNKNOWN"`
- `prop_portfolio.py:289` → `compute_fitness(snap["strategy_id"], db_path=db_path)` → raises `ValueError` or `duckdb.Error`
- `prop_portfolio.py:291` → `except ... pass` (silent)
- `prop_portfolio.py:311` → `fitness_status not in lane.required_fitness` → HOLD (correctly fail-closed, but no operator-visible exception context)

**VERDICT:** SUPPORT — integrity-guardian.md § 6 (No silent failures — every except must record the exception)

**Fix:** Changed `pass` to `logger.warning("compute_fitness failed for %s — fitness_status=UNKNOWN (strategy held): %s", snap["strategy_id"], exc)`. Behavior unchanged.

**Doctrine cited:** integrity-guardian.md § 6

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

---

## Next Iteration Targets

**Priority 1 — Unscanned high/medium centrality files:**
- `trading_app/lane_correlation.py` — imports from strategy_fitness, medium centrality, never scanned
- `trading_app/chordia.py` — medium centrality, never scanned
- `trading_app/prop_portfolio.py` — partially scanned this iteration (fitness gate only); remainder not audited

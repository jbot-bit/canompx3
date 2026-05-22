# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 194

## RALPH AUDIT — Iteration 194 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 31 strategy_fitness tests PASS
## Scope: trading_app/strategy_fitness.py — magic-number defaults duplicating MIN_ROLLING_FIT

---

## Iteration 194 — trading_app/strategy_fitness.py (full scan)

### Auto-Targeting
- Scope provided: `trading_app/strategy_fitness.py` — medium centrality, never scanned.

---

## Finding CANON-194 — LOW — FIXED

**PREMISE:** `_compute_fitness_from_cache`, `_compute_fitness_with_con`, and `compute_fitness` all declare `min_rolling_trades: int = 15` as a magic-number default. The canonical constant `MIN_ROLLING_FIT = 15` (annotated `@research-source @sensitivity-tested @revalidated-for`) lives at `strategy_fitness.py:95`. If a research re-validation updates `MIN_ROLLING_FIT`, the three defaults silently diverge.

**TRACE:**
- `strategy_fitness.py:607` → `_compute_fitness_from_cache(... min_rolling_trades: int = 15 ...)`
- `strategy_fitness.py:720` → `_compute_fitness_with_con(... min_rolling_trades: int = 15 ...)`
- `strategy_fitness.py:815` → `compute_fitness(... min_rolling_trades: int = 15 ...)`
- Canonical constant: `strategy_fitness.py:95` → `MIN_ROLLING_FIT = 15` (research-annotated)

**VERDICT:** SUPPORT — canonical violation per integrity-guardian.md § 2 / institutional-rigor.md § 4

**Fix:** Changed all 3 defaults to `min_rolling_trades: int = MIN_ROLLING_FIT`.
Diff: 3 lines. Behavior: identical (value unchanged). No callers pass this kwarg.

**Doctrine cited:** integrity-guardian.md § 2 (canonical sources — never inline magic numbers), institutional-rigor.md § 4 (delegate to canonical sources)

---

## Seven Sins Scan — strategy_fitness.py

- Sin 1 (Silent failure): Fail-closed pattern at `_filter_outcomes_with_features:421-423` (unknown filter → warning + empty list). ACCEPTABLE.
- Sin 2 (Fail-open): `compute_portfolio_fitness:907` catches `(ValueError, duckdb.Error, KeyError)` and logs via `logger.exception` — not silent. ACCEPTABLE.
- Sin 3 (Canonical violation): FIXED this iteration (CANON-194).
- Sin 4 (Impact awareness): `fitness_status` literals "FIT"/"WATCH"/"DECAY"/"STALE" are string constants returned from a single function (`classify_fitness`) and consumed by callers. No enum currently exists; the pattern is consistent across the codebase. LOW risk — all callers compare against the same string literals returned from `classify_fitness`. Acceptable per pattern 1 (intentional per-session heuristic — monitoring only, not a canonical list).
- Sin 5 (Evidence over assertion): N/A (audit mode).
- Sin 6 (Spec compliance): No spec in docs/specs/ for this module.
- Sin 7 (Metadata trust): Not applicable.

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

---

## Next Iteration Targets

**Priority 1 — Unscanned high/medium centrality files:**
- `trading_app/lane_correlation.py` — imports from strategy_fitness, medium centrality, never scanned
- `trading_app/live_config.py` — capital-class, calls compute_fitness on regime gate path, never scanned
- `trading_app/chordia.py` — medium centrality, never scanned

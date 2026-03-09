# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 12 (Bloomey Deep Dive: Live Trading Critical Path)
## Date: 2026-03-09
## Bloomey Grade: B+
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest (companion)` | PASS | 135 passed (portfolio + strategy_fitness + rolling_portfolio) |
| `ruff check` | PASS | All checks passed |

## Target Files

Deep Bloomey review of 5 live trading critical-path files:
- `trading_app/risk_manager.py` — position limits, circuit breakers, correlation-weighted sizing
- `trading_app/portfolio.py` — portfolio construction, position sizing, diversification
- `pipeline/cost_model.py` — canonical cost specs, R-multiple calculations
- `trading_app/rolling_portfolio.py` — rolling window stability classification
- `trading_app/strategy_fitness.py` — 3-layer fitness assessment, decay diagnostics

---

### Finding F1 — Dormant orb_minutes=5 hardcode (ANNOTATED)
- Severity: MEDIUM (dormant)
- File: `trading_app/rolling_portfolio.py:304`
- Issue: `WHERE symbol = ? AND orb_minutes = 5` loads wrong daily_features for 15m/30m families
- Action: Added TODO comment noting multi-aperture extension needed. Currently dormant (rolling evaluation only runs 5m). Deferred to when rolling evaluation extends to multi-aperture.
- Status: DEFERRED (annotated)

---

### Finding F2 — Hardcoded SINGAPORE_OPEN exclusion (FIXED)
- Severity: MEDIUM
- File: `trading_app/portfolio.py:312,352`
- Issue: `AND vs.orb_label != 'SINGAPORE_OPEN'` hardcoded in both baseline and nested queries. Should use `config.EXCLUDED_FROM_FITNESS`.
- Action: Built exclusion clause from `EXCLUDED_FROM_FITNESS` constant (same pattern as strategy_fitness.py). Both baseline (line 288) and nested (line 329) queries updated. Import added.
- Status: FIXED

---

### Finding F3 — Unannotated magic numbers (PARTIALLY ANNOTATED)
- Severity: MEDIUM (batch)
- Locations annotated:
  - `strategy_fitness.py:89-90` — MIN_ROLLING_FIT=15, MIN_ROLLING_WATCH=10 ✓
  - `rolling_portfolio.py:35-39` — STABLE_THRESHOLD, TRANSITIONING_THRESHOLD, FULL_WEIGHT_SAMPLE ✓
- Locations remaining:
  - `portfolio.py:944` — 0.4 trades/strategy/day (estimate, not a gate — lower priority)
  - `strategy_fitness.py:120` — -0.1 Sharpe decline threshold
  - `cost_model.py:153-229` — SESSION_SLIPPAGE_MULT values (77 lines)
- Status: PARTIALLY DONE

---

### Finding F4 — SESSION_SLIPPAGE_MULT no provenance
- Severity: LOW
- File: `pipeline/cost_model.py:153-229`
- Issue: 77 lines of per-session slippage multipliers with no @research-source. Are these measured or estimated?
- Status: DEFERRED (LOW priority — affects live execution only, not backtesting)

---

### Finding F5 — Fail-open on unknown filter (FIXED)
- Severity: MEDIUM
- File: `trading_app/strategy_fitness.py:332-334`
- Issue: Unknown filter_type returned ALL outcomes (fail-open). Should fail-closed.
- Action: Changed to return [] with logger.warning. Also fixed divergent behavior in compute_portfolio_fitness inline batch path (line 479-482) to match. Population audit confirmed: all 32 active filter_types exist in ALL_FILTERS — no strategy will be affected by this change.
- Status: FIXED

---

### Deferred from Iteration 9 (3 LOW)
1. Fill price `or` pattern (falsy zero) — `order_router.py:136,140,202,206`
2. PRODUCT_MAP hardcodes instrument list — `contract_resolver.py:22-27`
3. Auth token refresh not logged — `auth.py:42-60`

---

## Summary
- Total findings: 5 NEW (2 FIXED, 1 partially annotated, 2 deferred)
- CRITICAL: 0, HIGH: 0, MEDIUM: 4, LOW: 1
- Position sizing: mechanically correct (floors with int(), conservative)
- Cost model: canonical and sound (friction increases risk, reduces reward)
- Fitness gate: protects capital via DECAY→weight=0 path; WATCH boundary noisy at N=15

## Next Targets
- Fix F1 properly when rolling evaluation extends to multi-aperture
- Complete F3 annotations (portfolio.py:944, strategy_fitness.py:120)
- Add F4 provenance to SESSION_SLIPPAGE_MULT
- Fix 3 deferred LOW findings from iter 9
- Audit `scripts/tools/build_edge_families.py` — family clustering (not yet covered)

# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 15 (trading_app/walkforward.py)
## Date: 2026-03-10
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest (target module)` | PASS | 26 walkforward tests, 0 failed (3.05s) |
| `ruff check` | PASS | All checks passed |

---

## Target Files This Iteration

- `trading_app/walkforward.py` (301 lines)

---

## Deferred Findings from Prior Iterations (Status Check)

### F1 — rolling_portfolio.py:304 orb_minutes=5 hardcode (STILL DEFERRED)
- Severity: MEDIUM (dormant)
- File: `trading_app/rolling_portfolio.py:304`
- Status: DEFERRED — annotated. Dormant until rolling evaluation extends to multi-aperture.

### F3 — Unannotated magic numbers (PARTIALLY DONE)
- Severity: MEDIUM (batch)
- Remaining locations NOT yet annotated:
  - `portfolio.py:944` — 0.4 trades/strategy/day estimate
  - `strategy_fitness.py:120` — -0.1 Sharpe decline threshold
  - `cost_model.py:153-229` — SESSION_SLIPPAGE_MULT values
- Status: PARTIALLY DONE (annotations added to strategy_fitness.py and rolling_portfolio.py in iter 12; build_edge_families + strategy_validator annotated in iter 14)

### N4 — HOT Tier Thresholds Missing @research-source (STILL DEFERRED)
- Severity: LOW
- File: `trading_app/live_config.py:54-57`
- Status: DEFERRED (HOT tier dormant)

### N5 — Live Portfolio Constructor Magic Numbers (STILL DEFERRED)
- Severity: LOW
- File: `trading_app/live_config.py:354-355,583-584`
- Status: DEFERRED (refactor scope — named constants in config.py needed)

### Iter 9 LOWs (STILL OPEN)
1. Fill price `or` pattern (falsy zero) — `order_router.py:136,140,202,206`
2. PRODUCT_MAP hardcodes instrument list — `contract_resolver.py:22-27`
3. Auth token refresh not logged — `auth.py:42-60`

---

## New Findings This Iteration

### Finding W1 — Cost Spec Guard (FALSE ALARM — CLEAN)
- Severity: N/A
- File: `trading_app/walkforward.py:133`
- Evidence: `if stop_multiplier != 1.0 and cost_spec is not None:`
- Analysis: Caller (`strategy_validator.py:579`) does `wf_cost_spec = get_cost_spec(instrument) if stop_multiplier != 1.0 else None`. When `stop_multiplier != 1.0`, cost_spec is ALWAYS fetched. The `and cost_spec is not None` guard is defensive but correct. Not a silent skip.
- Status: FALSE ALARM — no finding.

### Finding W2 — IS Minimum Sample Guard Missing @research-source
- Severity: LOW
- File: `trading_app/walkforward.py:162`
- Evidence:
  ```python
  is_metrics = compute_metrics(is_outcomes) if len(is_outcomes) >= 15 else None
  ```
- Root Cause: `15` is consistent with `wf_min_trades=15` in strategy_validator.py (annotated in iter 14 with Lopez de Prado AFML Ch.11 reference), but the local IS guard has no `@research-source`. This prevents computing IS metrics when the anchored expanding window hasn't accumulated enough in-sample trades — correct behavior, missing provenance.
- Fix Category: annotation

### Finding W3 — Window Imbalance Ratio Missing @research-source
- Severity: LOW
- File: `trading_app/walkforward.py:242`
- Evidence:
  ```python
  window_imbalanced = window_imbalance_ratio > 5.0
  ```
- Root Cause: The `5.0x` ratio flags when the largest OOS window has 5x+ more trades than the smallest. This detects regime concentration. No `@research-source` annotation links it to Pardo (2008) Ch.7 walk-forward balance or any calibration study. Imbalanced windows inflate aggregate OOS stats (one huge window dominates), so the flag is important — but unannotated.
- Fix Category: annotation

---

## Confirmed Clean

**walkforward.py:**
- Seven Sins: CLEAN. Fail-closed throughout: insufficient data → early return with empty windows at line 109-130. All pass conditions are AND-gated (all 4 required). `n_valid < min_valid_windows` → rejection. `pct_positive < min_pct_positive` → rejection. `agg_oos_exp_r <= 0` → rejection. `total_oos_trades < oos_trade_floor` → rejection. No phantom state, no look-ahead (IS only uses data BEFORE each OOS window via `outcomes[:lo]`).
- Canonical integrity: MOSTLY CLEAN. `min_trades_per_window` and `min_valid_windows` passed from caller. `min_pct_positive` passed from caller. Local `15` at line 162 and `5.0` at line 242 are the two unannotated magic numbers.
- Statistical: CLEAN. Anchored expanding IS, non-overlapping OOS windows. Direction: IS ExpR sign confirmed at line 175 (IS correlation check). No look-ahead in IS construction.

---

## Summary
- Total new findings: 2 (0 CRIT, 0 HIGH, 0 MEDIUM, 2 LOW)
- CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 2 (W2, W3)
- Deferred carry-forward: F1, F3 (partial), N4, N5, 3x iter-9 LOWs
- Infrastructure Gates: 4/4 PASS

**Top eligible fix: W2+W3** — batch annotation (comments only, blast radius = 0).

**Next iteration targets:**
- `trading_app/execution_engine.py` — live execution path not yet covered in this scope
- Resolve F3 remaining: portfolio.py:944, strategy_fitness.py:120
- Resolve N4/N5 (LOW): HOT tier + portfolio constructor constants

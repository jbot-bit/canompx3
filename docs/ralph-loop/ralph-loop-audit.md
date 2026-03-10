# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 20 (config.py, strategy_discovery.py, outcome_builder.py)
## Date: 2026-03-10
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest test_config + test_strategy_discovery + test_outcome_builder` | PASS | 141/141 passed |
| `ruff check` | PASS | All checks passed |

---

## Target Files This Iteration

- `trading_app/config.py` (1047 lines)
- `trading_app/strategy_discovery.py` (1343 lines)
- `trading_app/outcome_builder.py` (988 lines)

---

## Deferred Findings from Prior Iterations (Status Check)

### F1 — rolling_portfolio.py:304 orb_minutes=5 hardcode (STILL DEFERRED)
- Severity: MEDIUM (dormant)
- Status: DEFERRED — annotated. Dormant until rolling evaluation extends to multi-aperture.

### F3 — Unannotated magic numbers (PARTIALLY DONE)
- Severity: MEDIUM (batch)
- Remaining: `portfolio.py:944`, `strategy_fitness.py:120`, `cost_model.py:153-229`

### N4 — HOT Tier Thresholds Missing @research-source (STILL DEFERRED)
- Severity: LOW

### N5 — Live Portfolio Constructor Magic Numbers (STILL DEFERRED)
- Severity: LOW

### Iter 9 LOWs (STILL OPEN)
1. Fill price `or` pattern (falsy zero) — `order_router.py:136,140,202,206`
2. PRODUCT_MAP hardcodes instrument list — `contract_resolver.py:22-27`
3. Auth token refresh not logged — `auth.py:42-60`

### Iter 19 LOWs (STILL OPEN)
1. EE1: Conditional EXITED trade pruning — `execution_engine.py:1152-1154`
2. EE2: E3 stop-before-fill silent exit — `execution_engine.py:963-967`
3. EE3: IB start time hardcoded 23:00 UTC — `execution_engine.py:262`

---

## New Findings This Iteration

### Finding SD1 — `median_risk_dollars` / `avg_risk_dollars` include friction (MEDIUM)
- Severity: MEDIUM
- File: `trading_app/strategy_discovery.py:630,634`
- Evidence:
  ```python
  avg_risk_dollars = round(avg_risk * cost_spec.point_value + cost_spec.total_friction, 2)
  median_risk_dollars = round(median_risk * cost_spec.point_value + cost_spec.total_friction, 2)
  ```
- Root Cause: `total_friction` (spread + commission) added to risk_dollars. Risk = stop distance in dollars = `risk_pts * point_value`. Friction is a cost, not part of risk. Same error class as trade sheet T5 (fixed in iter 18).
- Cascade: `avg_win_dollars` (line 631) and `avg_loss_dollars` (line 632) inherit inflation since they're computed as `avg_win_r * avg_risk_dollars`.
- Impact: Stored values in `experimental_strategies` are informational only — no gate decision uses them. Dollar gate in `live_config.py` correctly computes `median_risk_pts * point_value` (no friction). But misleading to anyone reading the table.
- Fix: Remove `+ cost_spec.total_friction` from lines 630 and 634.

### Finding SD2 — Session fallback to ORB_LABELS (LOW)
- Severity: LOW
- Files: `outcome_builder.py:677-678`, `strategy_discovery.py:1022-1023`
- Evidence:
  ```python
  if not sessions:
      sessions = ORB_LABELS  # fallback: all sessions
  ```
- Root Cause: If `get_enabled_sessions()` returns empty, falls back to ALL known sessions. Could compute outcomes/strategies for sessions disabled for the instrument.
- Impact: Downstream gates (validator, live_config) filter to enabled sessions. Wasted compute only.
- Fix: Fail-closed: raise ValueError if no sessions enabled.

### Finding SD3 — CORE_MIN_SAMPLES / REGIME_MIN_SAMPLES missing @research-source (LOW)
- Severity: LOW
- File: `trading_app/config.py:961-962`
- Evidence: `CORE_MIN_SAMPLES = 100` / `REGIME_MIN_SAMPLES = 30` — FIX5 thresholds with inline comments but no @research-source annotation.
- Impact: Annotation debt only.

---

## Confirmed Clean

**config.py:**
- **Seven Sins: CLEAN.** No data processing (config only). All threshold clusters annotated with @research-source (EARLY_EXIT_MINUTES, E3_RETRACE_WINDOW). SESSION_EXIT_MODE complete for all 11 sessions. FIX5 classification correct. EXCLUDED_FROM_FITNESS per-instrument. Warning generation references config enums.

**strategy_discovery.py:**
- **Seven Sins: CLEAN** (except SD1). No look-ahead (filter_days computed per-day, outcomes loaded with holdout_date cap, relative volume uses prior bars only). BH FDR correctly computed at discovery (informational). DSR/FST with proper n_trials. Canonical dedup by filter specificity. DST regime split correct. E2/E3 correctly restricted to CB1. Tight stop n_trials not doubled (documented rationale — correlated streams).
- **Sharpe computation: CORRECT.** Per-trade → annualized via sqrt(trades_per_year). Lo (2002) autocorrelation adjustment. Haircut Sharpe via Mertens (2002) + BLP (2014). P-value from t-test with custom betainc (no scipy). All mathematical implementations verified.

**outcome_builder.py:**
- **Seven Sins: CLEAN.** Sequential bar processing — no look-ahead. Entry detection via `detect_break_touch` (E2) or `detect_entry_with_confirm_bars` (E1/E3). Post-entry bars filtered by entry_ts. Ambiguous bars = conservative loss. Fill bar checked first. Time-stop fires only when MTM < 0 at threshold. Batch insert via INSERT OR REPLACE (idempotent). Checkpoint/resume for incremental builds.
- **Cost model: CORRECT.** Per-trade `risk_dollars` via canonical `risk_in_dollars()`. P&L via `to_r_multiple()`. Both from `pipeline.cost_model`.
- **E2 slippage: CORRECT.** `_resolve_e2` applies `E2_SLIPPAGE_TICKS * cost_spec.tick_size`.

---

## Summary
- Total new findings: 3 (0 CRIT, 0 HIGH, 1 MEDIUM, 2 LOW)
- SD1 is actionable (MEDIUM) — same error class as iter 18 T5 fix
- SD2/SD3 are LOW — annotation debt / wasted compute
- Deferred carry-forward: F1, F3 (partial), N4, N5, 3x iter-9 LOWs, 3x iter-19 LOWs
- Infrastructure Gates: 4/4 PASS

**Next iteration targets:**
- Fix SD1 (risk_dollars friction inflation) — 2 lines changed
- Then: Resolve F3 remaining (portfolio.py:944, strategy_fitness.py:120)
- Then: Iter 9 LOWs (fill price `or` pattern, order_router.py)

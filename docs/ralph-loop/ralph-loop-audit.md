# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 13 (Tradebook/Pipeline: outcome_builder, strategy_discovery, strategy_validator, build_edge_families, live_config)
## Date: 2026-03-10
## Bloomey Grade: A-
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest (target modules)` | PASS | 163 passed (outcome_builder, strategy_discovery, strategy_validator, live_config, edge_families) |
| `ruff check` | PASS | All checks passed |

---

## Target Files This Iteration

- `trading_app/outcome_builder.py`
- `trading_app/strategy_discovery.py`
- `trading_app/strategy_validator.py`
- `scripts/tools/build_edge_families.py`
- `trading_app/live_config.py`

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
- Status: PARTIALLY DONE (annotations added to strategy_fitness.py and rolling_portfolio.py in iter 12)

### F4 — SESSION_SLIPPAGE_MULT no provenance (STILL DEFERRED)
- Severity: LOW
- File: `pipeline/cost_model.py:153-229`
- Status: DEFERRED

### Iter 9 LOWs (STILL OPEN)
1. Fill price `or` pattern (falsy zero) — `order_router.py:136,140,202,206`
2. PRODUCT_MAP hardcodes instrument list — `contract_resolver.py:22-27`
3. Auth token refresh not logged — `auth.py:42-60`

---

## New Findings This Iteration

### Finding N1 — Dollar Gate Fail-Open when median_risk_points is NULL
- Severity: MEDIUM
- File: `trading_app/live_config.py:330-332`
- Evidence:
  ```python
  median_risk_pts = variant.get("median_risk_points")
  if median_risk_pts is None:
      return True, "dollar gate skipped (no median_risk_points)"
  ```
- Root Cause: `median_risk_points` comes via LEFT JOIN from `experimental_strategies` in `_load_best_regime_variant()` (line 195-198). If a strategy is in `validated_setups` but absent from `experimental_strategies`, the LEFT JOIN returns NULL and the dollar gate silently passes. The dollar gate was added specifically because MNQ TOKYO ($2.93 = 1.07x) and MNQ BRISBANE ($2.83 = 1.03x) were too thin — exact strategies that might have NULL risk points in legacy data.
- Blast Radius: Any strategy bypassing the dollar gate could have expected profit below 1.3x round-trip transaction cost. Currently the gate passes silently with no log warning — operators do not know the gate was skipped. The logged note "dollar gate skipped (no median_risk_points)" appears in the notes list returned to callers, but is not emitted as a WARNING.
- Fix Category: validation

### Finding N2 — Edge Family Robustness Thresholds Missing @research-source
- Severity: LOW
- File: `scripts/tools/build_edge_families.py:31-38`
- Evidence:
  ```python
  # Robustness thresholds (Duke Protocol #3c)
  MIN_FAMILY_SIZE = 5
  WHITELIST_MIN_MEMBERS = 3
  WHITELIST_MIN_SHANN = 0.8
  WHITELIST_MAX_CV = 0.5
  WHITELIST_MIN_TRADES = 50
  SINGLETON_MIN_TRADES = 100
  SINGLETON_MIN_SHANN = 1.0
  ```
- Root Cause: "Duke Protocol #3c" is an internal label with no external definition. The concrete values (ShANN >= 0.8, CV <= 0.5, min_trades >= 50 for whitelist; >= 100 trades, ShANN >= 1.0 for singleton) have no `@research-source` annotation linking to the simulation or back-test that established them. Drift check #45 requires `@research-source` for research-derived config values that gate portfolio composition.
- Blast Radius: Thresholds determine which strategies survive the family robustness filter before entering the live portfolio. Too tight purges real edges; too loose admits overfit singletons. Cannot revalidate after entry model changes without provenance.
- Fix Category: annotation

### Finding N3 — Walk-Forward Gate Thresholds Missing @research-source
- Severity: LOW
- File: `trading_app/strategy_validator.py:654-656`
- Evidence:
  ```python
  wf_min_trades: int = 15,
  wf_min_windows: int = 3,
  wf_min_pct_positive: float = 0.60,
  ```
- Root Cause: These three hard WF gate values (15 trades/window, 3 valid windows, 60% positive) cause REJECTION if not met. Contrast: `min_years_positive_pct=0.75` at line 321 has `@research-source Fitschen "Building Reliable Trading Systems"`. The WF thresholds do not. No `@revalidated-for E1/E2` annotation.
- Blast Radius: Incorrect WF thresholds can flip large batches of strategies between PASSED and REJECTED. After entry model changes (e.g., E0 purge), these must be re-evaluated — without provenance, the revalidation cannot be traced.
- Fix Category: annotation

### Finding N4 — HOT Tier Thresholds Missing @research-source (Dormant Risk)
- Severity: LOW
- File: `trading_app/live_config.py:54-57`
- Evidence:
  ```python
  HOT_LOOKBACK_WINDOWS = 10   # no @research-source
  HOT_MIN_STABILITY = 0.6     # no @research-source
  ```
- Root Cause: HOT tier is currently dormant (no `LIVE_PORTFOLIO` entries use `tier="hot"`). But these unannotated thresholds control what experimental strategies enter live trading if HOT tier is re-activated. No `@research-source` or `@revalidated-for` per drift check #45.
- Blast Radius: Activating HOT tier without validating these thresholds would gate experimental (unvalidated) strategies into live trading using arbitrary stability cutoffs.
- Fix Category: annotation

### Finding N5 — Live Portfolio Constructor Magic Numbers Not in config.py
- Severity: LOW
- File: `trading_app/live_config.py:354-355,583-584`
- Evidence:
  ```python
  account_equity: float = 25000.0,
  risk_per_trade_pct: float = 2.0,
  ...
  max_concurrent_positions=3,
  max_daily_loss_r=5.0,
  ```
- Root Cause: Four portfolio risk parameters are hardcoded without named constants in config.py and without `@research-source`. `account_equity=25000.0` and `risk_per_trade_pct=2.0` are function defaults; `max_concurrent_positions=3` and `max_daily_loss_r=5.0` are hardcoded inline at Portfolio constructor call. These control actual dollar exposure and circuit-breaker logic.
- Blast Radius: Account equity and risk% control dollar position sizing. max_concurrent_positions limits simultaneous exposure. max_daily_loss_r is the circuit breaker. Any incorrect value is a live-trading risk error.
- Fix Category: refactor (extract to named constants in config.py) + annotation

---

## Confirmed Clean (No Findings)

**outcome_builder.py:**
- Seven Sins: CLEAN. No silent failures, fail-open paths, or look-ahead bias. Session fallback to `ORB_LABELS` at line 678 only fires when `get_enabled_sessions()` returns empty — acceptable. Cost model via `get_cost_spec()` throughout. No hardcoded instruments.
- Canonical integrity: CLEAN. `get_enabled_sessions()`, `get_cost_spec()`, `GOLD_DB_PATH`, `ENTRY_MODELS` — all canonical.
- Statistical: No quantitative claims — purely mechanical outcome computation.

**strategy_discovery.py:**
- Seven Sins: CLEAN. BH FDR applied at discovery line 1201 (q=0.05 across all combos). p-value computed per strategy via t-test. No look-ahead bias — `holdout_date` properly caps both features AND outcomes. Fail-closed volume filter (missing data → rejects). `_check_fill_bar_exit()` independent function preserved but superseded by inline vectorized path — no behavioral risk.
- Canonical integrity: CLEAN. `get_enabled_sessions()`, `get_cost_spec()`, `GOLD_DB_PATH`, `ENTRY_MODELS` from canonical sources.
- Statistical: CLEAN. BH FDR, p-value, DSR (haircut Sharpe), n_trials = total_combos.

**strategy_validator.py:**
- Seven Sins: CLEAN. Worker exception at line 863 logs and maps to REJECTED (fail-closed). `_walkforward_worker()` internal exception at line 633 captured in `result["error"]` → REJECTED at lines 898-900. BH FDR applied post-validation across all passed strategies. Phase 4c/4d demoted to informational correctly.
- Canonical integrity: CLEAN. `CORE_MIN_SAMPLES`, `REGIME_MIN_SAMPLES` from config. `get_cost_spec()`, `GOLD_DB_PATH` canonical.
- Statistical: CLEAN. BH FDR post-validation sound. `min_years_positive_pct=0.75` annotated (Fitschen). Phase 3 regime waiver requires at least 1 clean positive year (line 415-420) — correct.

**build_edge_families.py:**
- Seven Sins: CLEAN. `finally: con.close()` prevents connection leak. Fail gates at lines 385-395 abort before commit on mega-family or singleton-rate anomaly. PBO computed for families with 2+ members (Bailey et al. 2014).
- Canonical integrity: CLEAN. `ACTIVE_ORB_INSTRUMENTS`, `GOLD_DB_PATH`, `CORE_MIN_SAMPLES`, `REGIME_MIN_SAMPLES` canonical. Median head election (not max) avoids Winner's Curse.

**live_config.py:**
- Seven Sins: Mostly CLEAN. Dollar gate fails-closed on `get_cost_spec()` exception (line 346-347). Fitness gate fails-closed on exception (line 547-549, weight=0.0). HOT tier dollar gate intentionally omitted with documented rationale.
- One-way dependency: CLEAN. `live_config.py` imports from `pipeline/` but not the reverse.

---

## Summary
- Total findings: 5 NEW (1 MEDIUM, 4 LOW)
- CRITICAL: 0, HIGH: 0, MEDIUM: 1 (N1), LOW: 4 (N2, N3, N4, N5)
- Deferred carry-forward: F1, F3 (partial), F4, 3x iter-9 LOWs
- Infrastructure Gates: 4/4 PASS

**Top priority: N1** — dollar gate fail-open is a live portfolio safety gap. A strategy can enter live trading without cost-adequacy screening when `median_risk_points` is NULL. The condition is silent (no WARNING log). This should at minimum emit `logger.warning()` so operators know the gate was bypassed.

**Next iteration targets:**
- `trading_app/walkforward.py` — WF engine not yet covered; N3 thresholds originate here
- `trading_app/execution_engine.py` — live execution path not yet covered in this scope
- Resolve N2: document Duke Protocol #3c threshold derivation with @research-source
- Resolve N3: add @research-source for wf_min_trades=15, wf_min_windows=3, wf_min_pct_positive=0.60
- Complete F3 remaining: portfolio.py:944, strategy_fitness.py:120

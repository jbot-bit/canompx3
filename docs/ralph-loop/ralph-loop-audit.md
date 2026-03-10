# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 21 (order_router.py — both brokers)
## Date: 2026-03-10
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest test_order_router.py` | PASS | 8/8 passed (0.12s) |
| `ruff check` | PASS | All checks passed |

---

## Target Files This Iteration

- `trading_app/live/tradovate/order_router.py` (210 lines)
- `trading_app/live/projectx/order_router.py` (177 lines)

---

## Deferred Findings from Prior Iterations (Status Check)

### F1 — rolling_portfolio.py:304 orb_minutes=5 hardcode (STILL DEFERRED)
- Severity: MEDIUM (dormant)
- Status: DEFERRED — annotated. Dormant until rolling evaluation extends to multi-aperture.

### F3 — Unannotated magic numbers (PARTIALLY DONE)
- Severity: MEDIUM (batch)
- Remaining: `portfolio.py:953` (0.4 trade frequency), `strategy_fitness.py:124` (-0.1 Sharpe decay threshold), `cost_model.py:153-229`

### N4 — HOT Tier Thresholds Missing @research-source (STILL DEFERRED)
- Severity: LOW

### N5 — Live Portfolio Constructor Magic Numbers (STILL DEFERRED)
- Severity: LOW

### Iter 9 LOWs (PARTIALLY RESOLVED)
1. ~~Fill price `or` pattern (falsy zero) — `order_router.py`~~ **FIXED iter 21**
2. PRODUCT_MAP hardcodes instrument list — `contract_resolver.py:22-27`
3. Auth token refresh not logged — `auth.py:42-60`

### Iter 19 LOWs (STILL OPEN)
1. EE1: Conditional EXITED trade pruning — `execution_engine.py:1152-1154`
2. EE2: E3 stop-before-fill silent exit — `execution_engine.py:963-967`
3. EE3: IB start time hardcoded 23:00 UTC — `execution_engine.py:262`

### Iter 20 Findings (RESOLVED)
1. ~~SD1: risk_dollars friction inflation~~ **FIXED iter 20**
2. SD2: Session fallback to ORB_LABELS — LOW, deferred
3. SD3: CORE_MIN_SAMPLES missing @research-source — LOW, deferred

---

## New Findings This Iteration

### Finding OR1 — Fill price `or` pattern uses falsy check (LOW → FIXED)
- Severity: LOW (theoretical — futures never trade at $0.00)
- Files: `tradovate/order_router.py:136,140,202,206`, `projectx/order_router.py:74,88,171`
- Evidence (before fix):
  ```python
  fill_price = data.get("avgPx") or data.get("fillPrice")  # 0.0 skips to fallback
  fill_price=float(fill_price) if fill_price else None  # 0.0 → None
  ```
- Root Cause: Python truthiness check on numeric type — `0.0` is falsy. `or` between two `.get()` calls skips the first if it returns 0. `if fill_price` returns False for 0.0.
- Fix: Replace `or` with `if x is None: x = fallback`, replace `if x` with `if x is not None`. Also added `float()` cast to ProjectX `query_order_status` for consistency.
- Status: **FIXED** — 7 locations across 2 files.

### Finding OR2 — No fill_price parsing unit tests (LOW)
- Severity: LOW
- File: `tests/test_trading_app/test_order_router.py`
- Evidence: All 8 existing tests cover `build_order_spec` / `build_exit_spec` / auth guard. No test mocks HTTP responses to verify fill_price extraction from `submit()` or `query_order_status()`.
- Impact: If API response field names change, no test catches it.
- Fix: Add mock-based tests for submit/query_order_status response parsing.

---

## Confirmed Clean

**tradovate/order_router.py:**
- **Seven Sins: CLEAN** (after OR1 fix). No look-ahead. Fail-closed on auth (raises RuntimeError). Order ID validated (None or <= 0 → raise). Timeout=5s on all HTTP. `raise_for_status()` on all responses. Latency warning at 1000ms. E3 blocked live (ValueError).
- **Canonical integrity: CLEAN.** Imports broker_base. No hardcoded instruments. No magic numbers beyond API constants.

**projectx/order_router.py:**
- **Seven Sins: CLEAN** (after OR1 fix). Same patterns as Tradovate. Additional: `success` field checked on submit response (fail-closed). Native bracket support with OCO stop+limit. E3 blocked live.
- **Canonical integrity: CLEAN.** Imports from broker_base and auth. API constants (order types 1/2/4/5, sides 0/1) documented in module docstring.

---

## Summary
- Total new findings: 2 (0 CRIT, 0 HIGH, 0 MEDIUM, 2 LOW)
- OR1 FIXED this iteration (falsy zero pattern)
- OR2 deferred (test coverage gap — no HTTP mocking)
- Deferred carry-forward: F1, F3 (partial), N4, N5, 2x iter-9 LOWs, 3x iter-19 LOWs, SD2, SD3, OR2
- Infrastructure Gates: 4/4 PASS

**Next iteration targets:**
- F3 remaining: `portfolio.py:953`, `strategy_fitness.py:124` — annotation debt
- Then: Iter 9 LOW #2: PRODUCT_MAP hardcodes — `contract_resolver.py:22-27`
- Then: OR2 — fill_price parsing tests

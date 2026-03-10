# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 22 (contract_resolver.py, strategy_fitness.py, portfolio.py)
## Date: 2026-03-10
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest test_strategy_fitness + test_portfolio + test_live_config` | PASS | 119/119 passed |
| `ruff check` | PASS | All checks passed |

---

## Target Files This Iteration

- `trading_app/live/projectx/contract_resolver.py` (80 lines)
- `trading_app/strategy_fitness.py` (region: lines 85-130)
- `trading_app/portfolio.py` (region: lines 950-955)

---

## Deferred Findings from Prior Iterations (Status Check)

### F1 — rolling_portfolio.py:304 orb_minutes=5 hardcode (STILL DEFERRED)
- Severity: MEDIUM (dormant)
- Status: DEFERRED — annotated. Dormant until rolling evaluation extends to multi-aperture.

### F3 — Unannotated magic numbers (RESOLVED)
- ~~portfolio.py:953~~ **ANNOTATED iter 22** (inline @research-source)
- ~~strategy_fitness.py:124~~ **EXTRACTED iter 22** (SHARPE_DECAY_THRESHOLD constant)
- Remaining: `cost_model.py:153-229` (cost spec definitions — these ARE the canonical source, arguably self-documenting)

### N4 — HOT Tier Thresholds Missing @research-source (STILL DEFERRED)
- Severity: LOW

### N5 — Live Portfolio Constructor Magic Numbers (STILL DEFERRED)
- Severity: LOW

### Iter 9 LOWs (MOSTLY RESOLVED)
1. ~~Fill price `or` pattern~~ **FIXED iter 21**
2. ~~PRODUCT_MAP hardcodes instrument list~~ **CLOSED iter 22** — has fallback `.get(instrument, [instrument])`, not a gate
3. Auth token refresh not logged — `auth.py:42-60`

### Iter 19 LOWs (STILL OPEN)
1. EE1: Conditional EXITED trade pruning — `execution_engine.py:1152-1154`
2. EE2: E3 stop-before-fill silent exit — `execution_engine.py:963-967`
3. EE3: IB start time hardcoded 23:00 UTC — `execution_engine.py:262`

### Iter 20 Findings
1. ~~SD1: risk_dollars friction inflation~~ **FIXED iter 20**
2. SD2: Session fallback to ORB_LABELS — LOW, deferred
3. SD3: CORE_MIN_SAMPLES missing @research-source — LOW, deferred

### Iter 21 Findings
1. ~~OR1: Fill price falsy-zero~~ **FIXED iter 21**
2. OR2: No fill_price parsing unit tests — LOW, deferred

---

## New Findings This Iteration

### Finding CR1 — Account ID `or` falsy-zero (MEDIUM → FIXED)
- Severity: MEDIUM
- File: `trading_app/live/projectx/contract_resolver.py:40`
- Evidence (before fix):
  ```python
  acct_id = acct.get("id") or acct.get("accountId")
  ```
- Root Cause: Same Python antipattern as OR1 (iter 21). If `"id"` returns 0, `or` skips to `"accountId"` fallback. Account ID = 0 is unlikely in production but possible in test environments.
- Fix: Replace `or` with `if acct_id is None: acct_id = acct.get("accountId")`.
- Status: **FIXED**

### Finding CR2 — INSTRUMENT_SEARCH_TERMS is NOT a gate (CLOSED)
- Severity: CLOSED (was iter 9 LOW #2)
- File: `trading_app/live/projectx/contract_resolver.py:13-18,61`
- Evidence: Line 61: `INSTRUMENT_SEARCH_TERMS.get(instrument, [instrument])` — unknown instruments fall back to searching by instrument name directly.
- Status: **CLOSED** — not a bug. The fallback handles unknown instruments correctly.

### Finding F3a — Sharpe decay threshold extracted (LOW → FIXED)
- File: `trading_app/strategy_fitness.py:98,128`
- Action: Extracted inline `-0.1` to `SHARPE_DECAY_THRESHOLD` constant with @research-source annotation.
- Status: **FIXED**

### Finding F3b — Trade frequency estimate annotated (LOW → FIXED)
- File: `trading_app/portfolio.py:953`
- Action: Added @research-source annotation to inline `0.4` trade frequency estimate.
- Status: **FIXED**

---

## Confirmed Clean

**contract_resolver.py:**
- **Seven Sins: CLEAN** (after CR1 fix). Fail-closed on no accounts (raises RuntimeError). Account ID validated (None check). HTTP timeouts on all calls. `raise_for_status()` on all responses. Contract cache prevents redundant API calls. Search terms have safe fallback.

**strategy_fitness.py (classification region):**
- **Seven Sins: CLEAN.** All thresholds now extracted to named constants with @research-source. Classification logic correct: STALE → DECAY → WATCH → FIT ordering prevents false positives. Sharpe check gated on `is not None`.

**portfolio.py (capital estimation region):**
- **Seven Sins: CLEAN.** Risk dollars computed from stored median values with safe fallback to points. Empty portfolio handled (early return). No look-ahead — uses stored strategy attributes only.

---

## Summary
- Total findings: 4 (0 CRIT, 0 HIGH, 1 MEDIUM, 2 LOW, 1 CLOSED)
- CR1 FIXED (falsy-zero account ID)
- F3a/F3b FIXED (annotation debt)
- CR2 CLOSED (not a gate — has fallback)
- Deferred carry-forward: F1, F3 partial (cost_model only), N4, N5, 1x iter-9 LOW, 3x iter-19 LOWs, SD2, SD3, OR2
- Infrastructure Gates: 4/4 PASS

**Next iteration targets:**
- Iter 19 LOWs: EE1 (conditional EXITED prune) — `execution_engine.py:1152-1154`
- Then: Iter 9 LOW #3: auth token refresh not logged — `auth.py:42-60`
- Then: SD2 (session fallback to ORB_LABELS) — fail-closed fix

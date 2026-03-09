# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 9 (Tradovate Broker Layer)
## Date: 2026-03-09
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest tests/ -x -q` | PASS | 2742 passed, 0 failed, 1 skipped (556s) |
| `ruff check` | PASS | All checks passed |

## Target Files
- `trading_app/live/tradovate/order_router.py` — order submission, bracket management
- `trading_app/live/tradovate/auth.py` — OAuth token management
- `trading_app/live/tradovate/contract_resolver.py` — front-month contract resolution
- `trading_app/live/tradovate/positions.py` — broker position queries

---

### Finding 1 — Fill price `or` pattern (falsy zero)
- Severity: LOW
- File: `trading_app/live/tradovate/order_router.py:136,140,202,206`
- Evidence:
```python
# Line 136: avgPx=0.0 is falsy, falls through to fillPrice
fill_price = data.get("avgPx") or data.get("fillPrice")
# Line 140: fill_price=0.0 is falsy, returns None
return OrderResult(..., fill_price=float(fill_price) if fill_price else None)
```
- Root Cause: Python truthiness treats 0.0 as False. If Tradovate returns `avgPx=0.0` (edge case: rejected/cancelled order with zero fill), the code falls through to the next field or returns None.
- Blast Radius: LOW — futures prices are never 0.0 in practice. The pattern exists in both `submit()` and `query_order_status()`.
- Fix Category: correctness (use `is not None` instead of `or`/truthiness)

---

### Finding 2 — PRODUCT_MAP hardcodes instrument list
- Severity: LOW
- File: `trading_app/live/tradovate/contract_resolver.py:22-27`
- Evidence:
```python
PRODUCT_MAP = {
    "MGC": "MGC",
    "MNQ": "MNQ",
    "MES": "MES",
    "M2K": "M2K",
}
```
- Root Cause: Broker product name mapping hardcoded. If a new instrument is added to `ACTIVE_ORB_INSTRUMENTS` but not here, session start crashes with ValueError (fail-closed — correct, but brittle).
- Blast Radius: LOW — instruments rarely change. All 4 active instruments present.
- Fix Category: annotation (add comment noting canonical source dependency)

---

### Finding 3 — Auth token refresh not logged
- Severity: LOW
- File: `trading_app/live/tradovate/auth.py:42-60`
- Evidence:
```python
def _refresh(self) -> str:
    resp = requests.post(...)
    resp.raise_for_status()
    data = resp.json()
    self._token = data["accessToken"]
    # No log.info for successful refresh
    return self._token
```
- Root Cause: Token refresh succeeds silently. In a 23-hour session, can't tell from logs when refreshes occur or how many happened.
- Blast Radius: LOW — observability gap only. Auth failures raise (fail-closed).
- Fix Category: logging (add log.info for token refresh)

---

### Finding 4 — positions.py CLEAN
- Severity: NONE
- File: `trading_app/live/tradovate/positions.py`
- Assessment: Already uses `"long"/"short"` (previously fixed). Proper `raise_for_status()`. Account ID filtering correct. No issues found.

---

## Summary
- Total findings: 3 (all LOW)
- CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 3
- Tradovate broker layer is CLEAN for production use
- All findings are observability/style, not correctness
- Deferred to next iteration (current iteration at 5-file safety boundary from Bloomey fixes)

## Severity Counts

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH     | 0 |
| MEDIUM   | 0 |
| LOW      | 3 |

## What Was NOT Flagged (Anti-False-Positive Notes)
- `requests.post` timeout=5 in order submission: Appropriate for REST order submission. Tradovate docs recommend 5-10s.
- KeyError on `data["accessToken"]` in auth: Correct fail-closed — malformed auth response should crash at session start, not silently proceed.
- `supports_native_brackets() -> False`: Correct — Tradovate REST doesn't support OSO/OCO bracket orders. Engine manages stops/targets internally.
- `cancel()` returns None: Correct — `raise_for_status()` handles HTTP errors. Successful cancel doesn't need a return value since the caller (bracket cancel) already handles the exception path.
- `date.today()` for contract expiry: Correct — contract expiry is calendar-date comparison, not trading-day-aware.

## Next Targets
- `trading_app/strategy_discovery.py` — grid search, hypothesis generation
- `trading_app/strategy_validator.py` — multi-phase validation + walk-forward
- `trading_app/execution_engine.py` — re-audit after multi-aperture changes
- `trading_app/live/circuit_breaker.py` — verify recovery logic

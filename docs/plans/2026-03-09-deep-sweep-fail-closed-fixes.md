# Deep Sweep: Fail-Closed Fixes for Live Trading

**Date:** 2026-03-09
**Origin:** Bloomey 3x review cycle → code reviewer → deep sweep of entire codebase
**Grade impact:** Hardens live trading path from potential silent failures

## 7 Issues (All Isolated to trading_app/)

| # | File:Line | Severity | Current | Fix |
|---|-----------|----------|---------|-----|
| 1 | `live_config.py:343` | CRITICAL | `except: return True` (trade allowed) | `return False` (block) |
| 2 | `session_orchestrator.py:509,537` | HIGH | `or 10.0` magic fallback | Keep + `log.error` |
| 3 | `projectx/order_router.py:70` | HIGH | `orderId` default 0 | Validate like Tradovate |
| 4 | `market_state.py:338` | MEDIUM | bare `except: return` | `log.debug` |
| 5 | `market_state.py:360` | MEDIUM | bare `except: pass` | `log.debug` |
| 6 | `live_market_state.py:32` | HIGH | DST resolver silent fail | `log.warning` |
| 7 | `tradovate/data_feed.py:151` | MEDIUM | heartbeat silent break | `log.warning` |

## Design Decisions

- **Dollar gate (Fix 1):** The median_risk_points=None branch at line 328-329 correctly skips (can't compute without data). But the exception handler at 343 must BLOCK — if get_cost_spec() raises for a known instrument, the cost model is broken, and we must not allow trading.
- **Magic 10.0 (Fix 2):** Kept as last-resort fallback because _record_exit must not crash (would lose trade record) and _submit_bracket is supplementary (engine still manages exits). But log.error makes it immediately visible.
- **orderId (Fix 3):** Tradovate router validates orderId>0 (lines 122-125). ProjectX router must do the same.
- **Logging additions (Fixes 4-7):** market_state uses log.debug (regime is optional), live_market_state uses log.warning (missed trade), data_feed uses log.warning (imminent disconnect).

## Test Changes

- `test_exception_in_cost_spec_passes` → rename to `test_exception_in_cost_spec_blocks`, assert `passes is False`

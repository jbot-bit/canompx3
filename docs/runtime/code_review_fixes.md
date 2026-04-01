# Code Review Fixes — Lane Allocator

## Critical (fix before proceeding)

### 1. SM=0.75 adjustment diverges from canonical
- **File:** lane_allocator.py `_per_month_expr()` L207
- **Problem:** Uses `mae_r >= stop_multiplier` but canonical `apply_tight_stop` uses friction-adjusted math: `max_adv_pts = mae_r * risk_d / pv; killed = max_adv_pts >= sm * risk_pts`
- **Fix:** Add entry_price + stop_price to SQL query, use canonical friction-adjusted threshold
- **Status:** TODO

### 2. Hysteresis is a `pass` stub
- **File:** lane_allocator.py `build_allocation()` L534-539
- **Problem:** Spec requires 20% switching threshold. Code has `pass` placeholder.
- **Fix:** Compare each new candidate against prior_allocation. Only replace if >20% better.
- **Status:** TODO

### 3. No tests (16 required by spec)
- **File:** tests/test_trading_app/test_lane_allocator.py — MISSING
- **Fix:** Create all 16 tests from spec
- **Status:** TODO

## Important (fix before committing)

### 4. Private import `_P90_ORB_PTS`
- **File:** lane_allocator.py L526
- **Fix:** Rename to public `P90_ORB_PTS` in prop_profiles.py
- **Status:** TODO

### 5. Dead code `_parse_strategy_params`
- **File:** lane_allocator.py L73-117
- **Fix:** Remove (never called)
- **Status:** TODO

### 6. Dead constants STALENESS_*
- **File:** lane_allocator.py L46-47
- **Fix:** Move to a `check_staleness()` function or remove until Stage 2
- **Status:** TODO

### 7. trailing_wr is month-level not trade-level
- **File:** lane_allocator.py L315
- **Fix:** Compute actual trade WR from adjusted trades list, or rename to monthly_positive_pct
- **Status:** TODO

### 8. Hardcoded max_dd=2000 in CLI
- **File:** rebalance_lanes.py L87
- **Fix:** Read from ACCOUNT_TIERS via profile
- **Status:** TODO

---
mode: IMPLEMENTATION
task: Persistent dollar HWM tracker + prop firm compliance monitors
scope_lock:
  - trading_app/account_hwm_tracker.py
  - trading_app/consistency_tracker.py
  - trading_app/pre_session_check.py
  - trading_app/weekly_review.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/broker_base.py
  - trading_app/live/tradovate/positions.py
  - trading_app/live/projectx/positions.py
  - trading_app/live/tradovate/order_router.py
  - trading_app/live/projectx/order_router.py
  - tests/test_trading_app/test_account_hwm_tracker.py
  - tests/test_trading_app/test_consistency_tracker.py
  - tests/test_trading_app/test_price_collar.py
acceptance:
  - AccountHWMTracker persists HWM across sessions in dollars
  - check_halt() blocks trading when DD >= limit
  - Corrupt state file recovery works
  - pre_session_check shows HWM status
  - weekly_review shows account health section
  - consistency_tracker computes windfall % per firm rule
  - Price collar rejects entry orders >0.5% from market
  - All new tests pass
  - Existing 3112 tests unaffected
  - Drift checks 75/75 pass
---

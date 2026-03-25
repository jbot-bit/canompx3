---
mode: IMPLEMENTATION
task: Fresh-eyes audit — fix CRITICAL/HIGH safety findings (429 retry, HWM gate, dd ghost, isAutomated)
scope_lock:
  - trading_app/live/projectx/order_router.py
  - trading_app/live/session_orchestrator.py
  - trading_app/pre_session_check.py
  - trading_app/prop_profiles.py
  - trading_app/weekly_review.py
  - trading_app/log_trade.py
  - tests/test_trading_app/test_prop_profiles.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_e2e_sim_mocked.py
acceptance:
  - ProjectX order router has 429 retry with backoff+jitter
  - HWM halt check in entry gate chain before broker submit
  - dd_circuit_breaker ghost check removed or replaced with HWM check
  - 75/75 drift, 0 regressions
---

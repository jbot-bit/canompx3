---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Fix pre_session_check + get_lane_registry to use active profiles (not hardcoded apex_50k_manual)
pass: 2
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/pre_session_check.py
  - tests/test_trading_app/test_prop_profiles.py
blast_radius:
  - get_lane_registry() called by pre_session_check, log_trade, forward_monitor, slippage_scenario, sprt_monitor
  - pre_session_check is HARD GATE for trading
acceptance:
  - get_lane_registry() defaults to active Apex profile (not hardcoded apex_50k_manual)
  - pre_session_check loads lanes from active profile
  - All tests pass, drift clean
updated: 2026-03-30T18:00:00Z
---

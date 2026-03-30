---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Fix prop_profiles firm rules + design Apex 100K + Tradeify 5x config
pass: 2
scope_lock:
  - trading_app/prop_profiles.py
  - tests/test_trading_app/test_prop_profiles.py
blast_radius:
  - prop_profiles.py imported by pre_session_check, run_live_session, paper_trade_logger
  - DD budget validation catches misconfigs at import time
acceptance:
  - Apex rules match official docs (5:1 RR, 30% per-trade, scaling, safety net)
  - Tradeify notes clarify Group Trading broken + API-per-account required
  - Apex 100K profile activated with 5 lanes
  - Tradeify 50K profile has daily_lanes configured (CME_PRECLOSE priority)
  - validate_dd_budget passes for all active profiles
  - All tests pass, drift clean
updated: 2026-03-30T17:00:00Z
---

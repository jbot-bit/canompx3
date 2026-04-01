---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Pre-live audit fixes — S0.75 stat alignment + stale test fixes
updated: 2026-04-01T17:30:00Z
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/paper_trade_logger.py
  - trading_app/strategy_discovery.py
  - tests/test_trading_app/test_account_hwm_tracker.py
  - tests/test_trading_app/test_consistency_tracker.py
  - tests/test_trading_app/test_lane_ctl.py
  - tests/test_trading_app/test_paper_trade_logger.py
blast_radius:
  - prop_profiles.py: strategy_id references (_S075 alignment)
  - paper_trade_logger.py: lane strategy_ids must match prop_profiles
  - tests: pre-existing failures from stale DD/consistency/session values
acceptance:
  - prop_profiles references _S075 validated strategy_ids
  - paper_trade_logger LANES match prop_profiles
  - All tests pass (0 failures)
  - drift check clean
---

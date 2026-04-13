---
task: Wire allocator output into profile consumption path
mode: IMPLEMENTATION
stage: 1 of 4
scope_lock:
  - trading_app/lane_allocator.py
  - trading_app/prop_profiles.py
  - trading_app/prop_portfolio.py
  - trading_app/portfolio.py
  - trading_app/derived_state.py
  - trading_app/account_survival.py
  - trading_app/paper_trade_logger.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/multi_runner.py
  - scripts/run_live_session.py
  - scripts/tools/rebalance_lanes.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_prop_profiles.py
  - tests/test_trading_app/test_prop_portfolio.py
  - tests/test_trading_app/test_lane_allocator.py
blast_radius:
  - 48 files import prop_profiles (all go through resolve_daily_lanes — single wiring point)
  - 6 files import lane_allocator
  - pre_session_check already calls check_allocation_staleness — no change needed
acceptance:
  - load_allocation_lanes returns DailyLaneSpec tuples from JSON
  - load_allocation_lanes returns () on missing/corrupt/profile-mismatch (fail-closed)
  - resolve_daily_lanes loads from JSON when daily_lanes=()
  - drift check #95 validates JSON-sourced lanes
  - JSON includes avg_orb_pts and p90_orb_pts per lane
  - topstep_50k_mnq_auto has 7 lanes via allocator
  - all tests pass, drift clean
updated: 2026-04-13T14:35:00
---

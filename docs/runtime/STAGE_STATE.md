---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Adaptive Lane Allocator — monthly rebalancer (spec 2026-04-02)
updated: 2026-04-02T08:00:00Z
scope_lock:
  - trading_app/lane_allocator.py
  - scripts/tools/rebalance_lanes.py
  - scripts/tools/backtest_allocator.py
  - tests/test_trading_app/test_lane_allocator.py
  - trading_app/config.py
  - trading_app/strategy_validator.py
  - trading_app/paper_trade_logger.py
  - tests/test_trading_app/test_strategy_validator.py
blast_radius:
  - lane_allocator.py: NEW — reads gold.db (read-only), outputs lane_allocation.json
  - rebalance_lanes.py: NEW — CLI script, calls lane_allocator
  - test_lane_allocator.py: NEW — 16 tests from spec
  - Stage 2 (after proven): prop_profiles.py, paper_trade_logger.py, pre_session_check.py
acceptance:
  - compute_lane_scores returns correct trailing ExpR with SM adjust + filter
  - Pause/resume logic correct (2mo kill, 3mo resume, magnitude override)
  - DD budget respected (greedy selection within max_dd)
  - Zero look-ahead enforced (trading_day < rebalance_date)
  - Stateless (compute from data, not prior file)
  - 16 tests pass
  - Drift clean
  - Backtest 2022-2025 produces equity curve
---

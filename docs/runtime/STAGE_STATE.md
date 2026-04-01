---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: E2 honest entry — filter exclusion + execution engine parity + rebuild prep
updated: 2026-04-01T19:00:00Z
scope_lock:
  - trading_app/config.py
  - trading_app/strategy_discovery.py
  - trading_app/execution_engine.py
  - trading_app/prop_profiles.py
  - tests/test_trading_app/test_entry_rules.py
blast_radius:
  - config.py: E2_EXCLUDED_FILTERS for break-bar-derived filters
  - strategy_discovery.py: Skip excluded filters for E2
  - execution_engine.py: E2 enters on first ORB touch (not post-break)
  - prop_profiles.py: Deployed lanes switch to clean filters
acceptance:
  - E2 grid excludes CONT/FAST/VOL_RV/ATR70_VOL
  - execution_engine E2 matches outcome_builder honest entry
  - All deployed lanes use pre-entry-only filters
  - Tests pass, drift clean
---

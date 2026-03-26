---
mode: IMPLEMENTATION
task: Commit triage -- fairness audit + overnight range + waiver fix from Mar 26
scope_lock:
  - trading_app/config.py
  - trading_app/strategy_validator.py
  - trading_app/db_manager.py
  - pipeline/build_daily_features.py
  - tests/test_trading_app/test_strategy_validator.py
  - HANDOFF.md
acceptance:
  - OvernightRangeFilter docstring corrected (US→Asian, WR spread claim removed)
  - All 6 diffs reviewed and verified
  - Tests pass
---

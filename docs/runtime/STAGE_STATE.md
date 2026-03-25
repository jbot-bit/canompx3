---
mode: IMPLEMENTATION
task: Deploy US_DATA_1000 as lane 5 — code review fixes (paper_trade_logger, weekly_review, prop_profiles)
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/weekly_review.py
  - trading_app/paper_trade_logger.py
acceptance:
  - paper_trade_logger LANES matches prop_profiles (5 lanes)
  - weekly_review section_4 includes US_DATA_1000
  - _LANE_NAMES has US_DATA_1000 entry
  - apex_100k comment updated to 5 lanes
  - 75/75 drift, 0 regressions
---

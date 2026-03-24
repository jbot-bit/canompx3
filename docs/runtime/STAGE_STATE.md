---
mode: IMPLEMENTATION
task: Apex lane rebuild + TopStep MGC exclusion + edge families + drift fix
scope_lock:
  - trading_app/strategy_validator.py
  - trading_app/prop_profiles.py
  - pipeline/paths.py
  - pipeline/check_drift.py
acceptance: |
  4 Apex MNQ lanes updated to validated survivors.
  TopStep MGC lanes excluded with noise floor comment.
  Edge families rebuilt. All 82 drift checks pass.
---

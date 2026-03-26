---
mode: IMPLEMENTATION
task: Fix DD budget bypass in daily_lanes path (build_profile_portfolio)
scope_lock:
  - trading_app/portfolio.py
  - docs/runtime/STAGE_STATE.md
acceptance:
  - build_profile_portfolio() validates DD budget before returning Portfolio
  - Apex 50K profile (5 lanes × $935 = $4,675) fails-closed against $2K limit
  - No circular imports introduced
  - check_drift.py passes
  - pytest passes
---

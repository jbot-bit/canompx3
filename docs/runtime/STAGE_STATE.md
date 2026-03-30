---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Zero-command dashboard automation — data refresh + session launch from UI
scope_lock:
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - scripts/tools/refresh_data.py
  - pipeline/daily_backfill.py
  - tests/test_pipeline/test_daily_backfill.py
blast_radius:
  - bot_dashboard.py: additive endpoints only, no existing endpoint changes
  - bot_dashboard.html: new buttons + activity panel, existing UI untouched
  - refresh_data.py: add atr_20_pct patch call after build steps
  - No pipeline logic changes, no config changes, no schema changes
acceptance:
  - Dashboard launches standalone, shows data freshness
  - REFRESH DATA button downloads + rebuilds pipeline
  - START SESSION button launches signal-only session
  - PREFLIGHT and KILL buttons still work
  - 77/77 drift checks pass
  - All existing tests pass
---

Design: Zero-command automation for first automated trade. User opens dashboard, presses buttons, everything runs.

---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Add O5 outcome build to refresh_data.py + daily auto-refresh bat
updated: 2026-03-31T14:20:00Z
scope_lock:
  - scripts/tools/refresh_data.py
  - scripts/daily_refresh.bat
blast_radius:
  - refresh_data.py: standalone CLI, no callers. Adding sequential step after daily_features.
  - daily_refresh.bat: new file, Windows Task Scheduler entry point.
acceptance:
  - python -m scripts.tools.refresh_data --dry-run shows outcome_builder step
  - scripts/daily_refresh.bat runs end-to-end
  - python pipeline/check_drift.py passes
---

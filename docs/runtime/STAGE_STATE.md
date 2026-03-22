---
task: "Refresh MES data only"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Run refresh_data for MES only. Then resume Phase 2."
updated: 2026-03-22T16:30+10:00
terminal: main
scope_lock:
  - scripts/tools/refresh_data.py
acceptance:
  - "python scripts/tools/refresh_data.py --instrument MES runs end-to-end"
  - "MES bars_1m updated to current date"
proven:
  - "MGC already current"
  - "refresh_data.py PYTHONPATH fix already committed"
unproven: []
blockers: []
---

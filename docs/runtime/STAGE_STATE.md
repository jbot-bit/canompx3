---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Dashboard sim — fix bugs, fill gaps, improve UX
updated: 2026-03-31T15:00:00Z
scope_lock:
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
blast_radius:
  - bot_dashboard.py: FastAPI endpoints, no external callers except HTTP
  - bot_dashboard.html: Single-page UI, no imports
acceptance:
  - Refresh subprocess uses -m module syntax (not file path)
  - extractSession handles multi-word session names (US_DATA_830)
  - JS session times removed (use server-side countdown only)
  - Preflight button visible and functional
  - Trade table includes instrument column
---

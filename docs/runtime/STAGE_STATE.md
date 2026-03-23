---
task: "Guardian audit phase 2: type safety, dead code, stale docs"
mode: IMPLEMENTATION
stage: 3
stage_of: 4
stage_purpose: "Class B: Fix type safety gaps (slippage mult, DST dead code marker)"
updated: 2026-03-25T01:00+10:00
terminal: main
scope_lock:
  - pipeline/cost_model.py
  - pipeline/dst.py
acceptance:
  - "SESSION_SLIPPAGE_MULT: add missing MNQ COMEX_SETTLE, BRISBANE_1025, EUROPE_FLOW for active instruments"
  - "DST_AFFECTED_SESSIONS: add deprecation comment (empty dict, all consumers return CLEAN)"
blockers: []
---

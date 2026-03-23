---
task: "Guardian audit phase 2: fix remaining silent failures + type safety + stale docs + noise floor scripts"
mode: IMPLEMENTATION
stage: 2
stage_of: 4
stage_purpose: "Class D: Fix noise floor bootstrap filter bug + null_envelope connection leak"
updated: 2026-03-25T00:30+10:00
terminal: main
scope_lock:
  - scripts/tools/noise_floor_bootstrap.py
  - scripts/tools/null_envelope.py
acceptance:
  - "noise_floor_bootstrap: pass full daily_features row to filter (not just size column)"
  - "null_envelope: DuckDB connection leak fixed (use context manager)"
  - "null_envelope: redundant import re as re_mod removed"
blockers: []
---

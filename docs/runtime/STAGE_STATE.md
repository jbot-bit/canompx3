---
task: "Rebuild LIVE_PORTFOLIO specs from current validated truth"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Align LIVE_PORTFOLIO to current 404 validated strategies — drop dead specs, reclassify tiers"
updated: 2026-03-23T12:00+10:00
terminal: main
scope_lock:
  - trading_app/live_config.py
acceptance:
  - "8 dead specs dropped, 8 grounded specs remain"
  - "Live resolution: MNQ 8, MGC 0, MES 0"
  - "REGIME fitness-gate mismatch explicitly flagged in comment"
  - "No brittle snapshot numbers in header"
proven: []
unproven: []
blockers: []
---

---
task: "Correct REGIME gate misdiagnosis in live_config comment and HANDOFF"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Comment/doc correction only — no runtime logic changes"
updated: 2026-03-23T13:00+10:00
terminal: main
scope_lock:
  - trading_app/live_config.py
  - HANDOFF.md
acceptance:
  - "live_config.py comment corrected: REGIME gate is honest, CORE rolling eval is degraded"
  - "HANDOFF.md diagnosis corrected"
  - "Rolling portfolio rebuild logged as next data task"
  - "Zero runtime logic changes"
proven: []
unproven: []
blockers: []
---

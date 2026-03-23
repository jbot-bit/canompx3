---
task: "Fail-closed entry gate for rollover orphans — block new entries for orphaned strategy_ids"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "After rollover orphan detection, block entries for affected strategies. Fail-closed containment, not just observability."
updated: 2026-03-25T03:00+10:00
terminal: main
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
acceptance:
  - "Orphaned strategy_ids added to _blocked_strategies set after rollover"
  - "ENTRY events for blocked strategies rejected with log.critical + notify"
  - "Block persists until session end (no auto-clear — manual intervention required)"
  - "Non-orphaned strategies unaffected"
  - "All existing tests pass"
  - "New test: entry blocked after rollover orphan"
  - "New test: non-orphaned entry proceeds normally after rollover"
blockers: []
---

---
task: "Fix live execution recovery: PENDING_EXIT stuck + rollover orphan detection"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Add explicit logging on stuck PENDING_EXIT + orphan detection after rollover close failure"
updated: 2026-03-25T02:00+10:00
terminal: main
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_position_tracker.py
acceptance:
  - "Exit failure: position stays PENDING_EXIT, log.critical with strategy_id and state"
  - "Rollover: orphan check after close loop, log.critical + notify if positions survive"
  - "Existing test_position_tracker tests pass"
  - "Existing test_session_orchestrator tests pass"
  - "2 new tests cover the exact failure paths"
  - "No state machine behavior changes"
  - "No signature changes"
  - "Drift check 0 violations"
blockers: []
---

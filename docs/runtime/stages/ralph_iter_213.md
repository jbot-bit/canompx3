---
task: Ralph Loop iter 213 — _load_paused_lane_blocks fail-open → operator alert (SO-213-01)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
updated: 2026-05-30T02:30:00+00:00
agent: ralph
---

## Blast Radius
- trading_app/live/session_orchestrator.py — changes except handler at L1132: log.warning → log.critical + _notify
- tests/test_trading_app/test_session_orchestrator.py — adds 1 test for lifecycle-load-failure operator alert

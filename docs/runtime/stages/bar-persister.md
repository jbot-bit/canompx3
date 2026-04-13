---
task: Build broker-feed bar capture to eliminate Databento dependency
mode: IMPLEMENTATION
stage: 1 of 1
scope_lock:
  - trading_app/live/bar_persister.py
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_bar_persister.py
blast_radius:
  - session_orchestrator._on_bar gets one extra call (bar_persister.append)
  - session_orchestrator shutdown calls bar_persister.flush_to_db
  - No change to trading logic, order routing, or execution engine
  - Fail-open: persister failure does not block trading
acceptance:
  - BarPersister collects bars and batch-inserts to bars_1m at session end
  - Idempotent (DELETE+INSERT for time range)
  - Thread-safe append
  - Fail-open (errors logged, not raised)
  - SessionOrchestrator wires persister into _on_bar + shutdown
  - All tests pass, drift clean
updated: 2026-04-13T17:00:00
---

---
task: Ralph Loop iter 185 — F7 fill-poller PENDING timeout + halt-on-broker-stuck
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
  - pipeline/check_drift.py
blast_radius:
  - trading_app/live/session_orchestrator.py (add FILL_POLL_TIMEOUT_SECS, FILL_CANCEL_VERIFY_TIMEOUT_SECS constants + _handle_fill_timeout helper + timeout logic in _fill_poller)
  - tests/test_trading_app/test_session_orchestrator.py (7 new F7 test scenarios in TestFillPoller)
  - pipeline/check_drift.py (new drift check enforcing timeout constants + _handle_fill_timeout call pattern)
updated: 2026-04-25T00:00:00Z
agent: ralph
---

---
task: Ralph Loop iter 174 — F4 bracket submit failure leaves position naked (CRITICAL)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
blast_radius:
  - trading_app/live/session_orchestrator.py (_submit_bracket: 3 silent-failure sub-paths patched to _notify + _fire_kill_switch + _emergency_flatten)
  - tests/test_trading_app/test_session_orchestrator.py (TestF4BracketNakedPosition class added, mutation-proof)
updated: 2026-04-25T00:00:00+10:00
agent: ralph
---

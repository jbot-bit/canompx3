---
task: Close/fix all 8 open deferred-findings ledger items
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/session_orchestrator.py
  - trading_app/derived_state.py
  - docs/ralph-loop/deferred-findings.md
---

## Blast Radius

- trading_app/live/session_orchestrator.py — A6-GAP2 guard: adds invariant assertion before try block; fail-closed path unchanged; callers: session_orchestrator.__init__ only
- trading_app/derived_state.py — A6-GAP4: adds orb_minutes field to per-lane fingerprint dict; consumers: sr_monitor + live preflight fingerprint comparison; purely additive (fingerprints will change on next generation, no runtime crash)
- docs/ralph-loop/deferred-findings.md — ledger-only: close SR-L6 (displaced), close PR301 items (already fixed), HWM items with notes
- No pipeline/ touched; no schema change; no DB write

---
task: Close/fix all 8 open deferred-findings ledger items
mode: CLOSED
closed_date: 2026-05-25
closed_note: |
  Closed after fresh Codex verification. The named code fixes were already on
  main: A6-GAP2 predicate-divergence guard in `session_orchestrator.py` from
  `044a3ac0`, and A6-GAP4 `orb_minutes` fingerprint field in
  `derived_state.py` from `5dbd6b29`. `docs/ralph-loop/deferred-findings.md`
  has no unstruck active rows from this stage.

  Follow-up in this closeout added regression coverage to the
  `TestSafeguardExceptNarrowing` replay helper so the A6-GAP2 guard is tested
  rather than only present in production code. Evidence:
  `pytest -q tests/test_trading_app/test_session_orchestrator.py::TestSafeguardExceptNarrowing`
  => 12 passed.
original_mode: IMPLEMENTATION
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

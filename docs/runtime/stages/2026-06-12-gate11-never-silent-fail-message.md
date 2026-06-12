---
task: Gate-11 (strict live-readiness) never-silent FAIL message — consumer must name a reason (blockers ∪ launch-blocking warnings ∪ defensive fallback), never empty ()
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/preflight.py
  - tests/test_scripts/test_run_live_session_preflight.py
---

## Blast Radius

- trading_app/live/preflight.py — modifies `_check_live_readiness_report` (1 of 15
  preflight gates, called only from `scripts/run_live_session.py` gate [11]).
  ONLY the green=False message-construction path (lines 714-719) changes. NO
  pass/fail logic change — a FAIL stays a FAIL, an OK stays an OK; only the
  message string becomes legible. Lines 707-713 (incl. the defensive guard
  708-712) are untouched.
- tests/test_scripts/test_run_live_session_preflight.py — adds 5 tests mirroring
  the existing `_make_copy_trading_ctx` + monkeypatch pattern. Existing tests
  preserved: `test_live_readiness_report_blocking_warnings_block_live` (the
  defensive-guard regression anchor) and the two `strict_zero_warn green`
  OK-path asserts.
- Reads: none new. Writes: none. No schema change, no DB write, no broker/network
  call. `launch_blocking_strict_warnings` already imported at line 701 — no new
  import surface.
- Tier-B (live path) → severity LOW: message-only, fail-direction is safe-closed.
  No change to `scripts/tools/live_readiness_report.py` summary logic.

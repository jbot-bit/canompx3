---
task: Wire telemetry maturity gate into preflight (dashboard auto-surfaces via existing cache)
mode: IMPLEMENTATION
slug: telemetry-maturity-preflight-wiring
created: 2026-05-17
scope_lock:
  - scripts/run_live_session.py
  - tests/test_scripts/test_run_live_session_telemetry_maturity.py
  - tests/test_scripts/test_run_live_session_preflight.py
scope_expansion_note: |
  test_run_live_session_preflight.py pins the registry order and N-pass
  counts. Adding _check_telemetry_maturity to PREFLIGHT_CHECKS necessarily
  updates the pinned ordered list and forces the smoke fixtures to
  monkeypatch SIGNALS_DIR so the new check passes at the new total. This
  is a mechanical companion-test update, not behavior change.
---

## Task

Wire the existing trading_app.live.telemetry_maturity gate (committed in
7aaefcb6) into the preflight check list in scripts/run_live_session.py.
Dashboard auto-surfaces the count/threshold through the existing preflight
output parser at trading_app/live/bot_dashboard.py:598 (_parse_preflight_output).
No bot_dashboard.py edit required.

## Two-tier verdict (mode-dependent)

- --signal-only: passed=True, informational message. The whole point of a
  paper run is to ACCUMULATE distinct trading_days; blocking would prevent
  the gate from ever clearing. Message format example:
  "OK (signal-only: 11/30 distinct MNQ trading_days; auto-clears at 30)".
- --demo / --live: passed=False until n>=30. Capital-touching modes refuse
  to launch until the gate clears. Message: "FAILED: UNVERIFIED_INSUFFICIENT_TELEMETRY
  (11/30 distinct MNQ trading_days; run --signal-only until 30).".
- --instrument scope: when ctx.instrument is a single instrument, gate that
  one. When --all is used, gate evaluates per-instrument and reports the
  worst-case (any UNVERIFIED -> FAILED for non-signal-only; informational
  for signal-only).

## Separation from copy-trading gate

The copy-trading gate lives at _check_copy_trading_accounts (line 332+) and
operates on (profile.copies, account resolution). The maturity gate operates
on (signal-log distinct trading_days). The two are orthogonal -- maturity
checks measurement adequacy of the canonical signal-log telemetry, copy-set
checks broker-side account topology. New check name: _check_telemetry_maturity.
Distinct function, distinct grounding (Criterion 8 vs broker spec), distinct
ordering position in PREFLIGHT_CHECKS (placed AFTER _check_trade_journal,
BEFORE _check_copy_trading_accounts -- both adjacent measurement-side checks).

## Blast Radius

- scripts/run_live_session.py: add _check_telemetry_maturity (new function),
  add 1 entry to PREFLIGHT_CHECKS list. Auto-incremented [i/N] header via
  len(PREFLIGHT_CHECKS) (already implemented for additions).
- tests/scripts/test_run_live_session_preflight.py: new test file
  (preflight check coverage is presently sparse; create the file with the
  3 new tests scoped to _check_telemetry_maturity only).
- bot_dashboard.py: ZERO edits. Its _parse_preflight_output parses lines
  generically "[i/N] <title>... <message>"; the new check appears
  automatically.

## Reads / Writes

- Reads: signal log files at repo root via the maturity module (already
  the canonical source).
- Writes: none.
- Touches: zero allocator state, zero validated_setups, zero gate
  thresholds, zero filter logic.

## Acceptance

1. pytest tests/scripts/test_run_live_session_preflight.py -v passes
   (3 tests: signal_only informational pass, live below-floor FAILED,
   above-floor informational pass for both modes).
2. python pipeline/check_drift.py exit 0.
3. Manual run: python -c "from scripts.run_live_session import
   PREFLIGHT_CHECKS, _check_telemetry_maturity;
   assert _check_telemetry_maturity in PREFLIGHT_CHECKS;
   print(len(PREFLIGHT_CHECKS))" prints 8 (was 7).
4. Existing preflight tests continue to pass (companion sweep).

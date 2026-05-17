---
task: Harden bot_state.write_state against unittest.mock contamination
mode: IMPLEMENTATION
slug: bot-state-mock-guard
created: 2026-05-17
scope_lock:
  - trading_app/live/bot_state.py
  - tests/test_trading_app/test_bot_state_strict_types.py
---

## Task

trading_app/live/bot_state.write_state currently passes any value through
json.dumps with default=str. When a MagicMock (or any non-JSON-native
object) is in the payload, default=str silently coerces it to a string and
writes the polluted state to the canonical production path
data/bot_state.json. The 2026-05-17 live-throughput triage found this exact
contamination: fake TEST_STRAT_001 MGC lane state with six MagicMock
literal strings in the canonical state file (no real bot run had
overwritten it since the test fixture was written).

Fix: strict-type validator before serialization plus a strict default
callable. On contamination, REFUSE the write (log CRITICAL, return);
preserve fail-open behaviour for true disk/encoder errors so trade
execution never blocks on a dashboard write failure.

## Blast Radius

- trading_app/live/bot_state.py is a production module, NEVER_TRIVIAL.
- Callers: session_orchestrator.py writes bot state each bar and trade
  event; bot_dashboard.py reads via read_state (untouched).
- Tests touching this module (read-only API surface; behaviour unchanged
  for valid payloads): test_bot_dashboard.py, test_bot_dashboard_sse.py,
  test_bot_dashboard_holdtokill.py, test_session_orchestrator.py.
- Reads: none. Writes: data/bot_state.json via atomic tmp plus os.replace.
  Behaviour on clean payloads is identical to current.
- Any test that calls write_state with a MagicMock-laden dict without
  monkeypatching STATE_FILE will now refuse the write before the file
  touch. Existing tests in the project build payloads from real
  dataclasses via build_state_snapshot, not raw mocks (verified by grep).

## Acceptance

1. pytest tests/test_trading_app/test_bot_state_strict_types.py -v passes
   (3 tests).
2. pytest of test_session_orchestrator, test_bot_dashboard,
   test_bot_dashboard_sse, test_bot_dashboard_holdtokill shows no new
   failures vs main.
3. Mutation probe: inject a MagicMock into a known-good payload, call
   write_state, assert no file created and log records contain
   "bot_state contamination" plus the dotted path of the mock.
4. python pipeline/check_drift.py exit 0.

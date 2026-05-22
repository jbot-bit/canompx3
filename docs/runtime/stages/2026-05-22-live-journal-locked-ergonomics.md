---
task: TRIVIAL IMPLEMENTATION — live preflight ergonomics for locked TradeJournal. When a duplicate `run_live_session` or `bot_dashboard` already holds the DuckDB writer lock on `live_journal.db`, today's preflight [6/8] dumps a 30-line CRITICAL traceback and exits ambiguously. After this stage the journal init recognises the lock-collision case, suppresses the noisy traceback for THAT specific exception class only, exposes the holder PID via `journal.last_error`, and preflight prints a single recovery line naming `scripts/tools/stop_live.ps1`. Adds the stop_live helper script and one regression test. No production trading behavior changes — fail-open contract preserved.
mode: CLOSED
closed_date: 2026-05-22
closed_note: |
  Implemented. 31/31 tests pass (3 new + 28 prior). Drift: 157 PASSED, 0 violations.
  Pyright on touched python files: 0 errors, 0 warnings (also fixed 5 pre-existing
  errors in run_live_session.py + test_trade_journal.py per user "don't leave behind
  shit" directive).
original_mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/trade_journal.py
  - scripts/run_live_session.py
  - scripts/tools/stop_live.ps1
  - tests/test_trading_app/test_trade_journal.py
---

## Blast Radius

- `trading_app/live/trade_journal.py` — add `_extract_lock_holder_pid()` helper + `TradeJournalLockedError` sentinel + `last_error` attribute on `TradeJournal`. `__init__` keeps fail-open contract (`is_healthy=False` on any failure), but for the lock-collision case it logs ONE clean WARNING (not CRITICAL+traceback) and records the holder PID. All existing callers (`session_orchestrator.py` lines 203, 686-690, 938) only read `is_healthy` — behavior unchanged. New `last_error` is additive.
- `scripts/run_live_session.py` — `_check_trade_journal` reads `journal.last_error` if `not is_healthy`. When error type is the lock-collision case, returns a CheckResult message naming the PID + the recovery script path. Otherwise unchanged.
- `scripts/tools/stop_live.ps1` — NEW. Enumerates python processes whose CommandLine matches `run_live_session|bot_dashboard|webhook_server`, prints the table, prompts y/N before killing. Additive; does not run during tests or imports.
- `tests/test_trading_app/test_trade_journal.py` — adds 2 tests: (1) two journals against the same path in same process → second has `is_healthy=False` and `last_error` is a `TradeJournalLockedError`; (2) `_extract_lock_holder_pid` parses `"... PID 12345 ..."` → 12345, garbage → None.

## Non-goals (deferred)

- Real OS-level cross-process singleton lock for the live runner (separate stage; involves Windows + WSL + DuckDB semantics).
- Logging `FileHandler` for `logs/live/*.log` — carry-over (c-ii) from 2026-05-16 live debut; separate stage.
- `.env` parse-warning cleanup — separate stage.
- Auto-killing other people's processes without confirmation. `stop_live.ps1` always prompts.
- Touching `instance_lock.py`, `session_orchestrator.py`, broker code, or `pipeline/`.

## Done criteria

1. Existing `tests/test_trading_app/test_trade_journal.py` still passes (5 tests).
2. New tests pass (2 tests).
3. `python pipeline/check_drift.py` passes.
4. Manual sanity: starting a second journal against an already-open path produces ONE log line (no traceback) and `j.last_error` exposes the holder PID.

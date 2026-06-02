task: Improve journal-lock holder diagnostics in live preflight — report whether the DuckDB-writer-lock holder PID is alive (run stop_live.ps1) vs dead/stale (retry) vs unknown. Honest Stage 3 of the live-launch hardening baton; the baton's "self-heal a stale dead-PID lock" premise is impossible (DuckDB releases its writer lock on holder death — there is no lockfile to clear).
mode: IMPLEMENTATION
stage: (1/1)

## Scope Lock
- scripts/run_live_session.py
- tests/test_scripts/test_run_live_session_preflight.py

## Blast Radius
- scripts/run_live_session.py — `_check_trade_journal` only. On `TradeJournalLockedError`, probe `err.holder_pid` liveness via a small self-contained `os.kill`/Windows-`OpenProcess` check (mirrors canonical `worktree_guard._pid_is_alive`; NOT imported, to avoid coupling the live launcher to `scripts/tools/` lease internals + `filelock`). Report three variants: alive PID → run stop_live.ps1 -NoPrompt; dead/stale PID → retry, else stop_live.ps1 -NoPrompt; unknown PID → stop_live.ps1 -NoPrompt. Still returns `CheckResult(False, ...)` in every locked case — this is DIAGNOSTIC ONLY, it does NOT clear any lock and does NOT change pass/fail outcome. The healthy and non-locked-failure branches are untouched.
- tests/test_scripts/test_run_live_session_preflight.py — new focused tests for the three locked branches (alive/dead/unknown PID) via a fake TradeJournal + monkeypatched liveness probe. No real DB, no real process.
- Reads: none destructive. Writes: only the 2 files above. No gold.db writes. No live config / profile / allocation edits. No dashboard changes.
- Capital path: scripts/run_live_session.py IS capital-adjacent, but this change is read-only/diagnostic (a string + a liveness probe), never alters a launch decision beyond the message text. Minimal + tested per directive.

## Baton close-out (verified against code, not memory)
- Stage 2 (worktree lease PID-liveness): CLOSED_ALREADY_BUILT — `scripts/tools/worktree_guard.py:542` `_peer_is_live` already treats live holder PID as authoritative (with `expected_create_time` PID-reuse cross-check). No rebuild.
- Stage 3 (journal lock): DIAGNOSTIC_FIX_BUILT (this stage). Original "self-heal dead-PID lock" premise refuted.
- Stage 4 (dashboard kill): CLOSED_ALREADY_EXISTS — `trading_app/live/bot_dashboard.py:1875` `/api/action/kill` already present. Orphan-kill of a non-dashboard-owned PID is a separate higher-blast-radius task, DEFERRED.

---
task: BUG — `START_BOT.bat` opens 2 dashboard HTML tabs / 2 dashboard processes despite line 62-63 setting `CANOMPX3_DASHBOARD_ORIGIN=1` to suppress the orchestrator's auto-dashboard spawn. Reproduced twice 2026-05-26 by operator. Comment in .bat explicitly warns "without this env var, two dashboards + two browser tabs open and race for live_journal.db (fails preflight check 6)" — the suppression mechanism is in place but not firing. Need to trace: (a) where the orchestrator side reads `CANOMPX3_DASHBOARD_ORIGIN`, (b) why setting it via `set X=1 && python ...` in the .bat is not propagating to the python process, (c) whether the var name drifted on the consumer side. Likely candidates: `scripts/run_live_session.py`, `trading_app/live/session_orchestrator.py`, `trading_app/live/bot_dashboard.py`.
mode: IMPLEMENTATION
status: DEFERRED_POST_SMOKE_2026_05_26
priority: P2
deferred_reason: |
  Filed during Brisbane Mon 2026-05-26 live debut window (markets opening,
  user prioritized smoke test execution over investigation). Operator
  workaround: close the duplicate browser tab manually after launch.
  Investigate after tonight's session windows close.
scope_lock:
  - START_BOT.bat
  - scripts/run_live_session.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/bot_dashboard.py
agent: claude (opus 4.7)
---

## Blast Radius

- READS only initially: grep `CANOMPX3_DASHBOARD_ORIGIN` across the 3 candidate files to find the consumer. Once found, trace why suppression doesn't engage.
- WRITES (likely fix): single check in orchestrator dashboard-spawn path to honor the env var. Possibly also a corrected `set` syntax in `START_BOT.bat` if Windows env-var propagation through the chained `set X=Y && python ...` is the culprit (Windows cmd quirk: env-vars set inline this way don't always propagate to a subsequent python child — `setx` vs `set`, or moving `set` to its own line).
- LIVE-IMPACT: zero if investigation is read-only. If fix lands: prevents two dashboard processes from racing `live_journal.db` writer lock (the same class as the 2026-05-22 journal-locked ergonomics incident).

## Acceptance

1. Identify the var-name consumer in the orchestrator path.
2. Reproduce the bug with deterministic evidence (e.g., `START_BOT.bat` → 2 python `bot_dashboard` processes in tasklist).
3. Apply minimal fix.
4. Verify: single `START_BOT.bat` invocation → exactly 1 dashboard process + 1 browser tab.
5. Drift check passes; targeted dashboard/launcher tests pass.

## Related

- `feedback_duckdb_windows_lock_is_per_process.md` — the class that double-dashboard would re-trigger.
- 2026-05-22 live-journal-locked-ergonomics commit `53d25742` — operator-friendly error path for when this race does occur.
- HANDOFF 2026-05-22 § "Carry-overs from journal-locked session" — already lists dashboard/journal-lock interaction risks.

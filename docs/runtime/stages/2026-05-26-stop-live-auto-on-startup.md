---
task: |
  HARDEN — stale python process accumulation across sessions. Observed
  today (2026-05-26): 22 python.exe processes from sessions dated
  2026-05-22, 2026-05-24, and 2026-05-25 found at session start.
  `stop_live.ps1` exists and works (2026-05-22 ergonomics commit
  53d25742) but is manual.

  Integration target: invoke `stop_live.ps1` PROMPT-FREE in non-interactive
  mode from `START_BOT.bat` step 1, BEFORE the lock-file clear.

  Risk: `stop_live.ps1` today is INTERACTIVE (y/N prompt) by design — it
  enumerates matching processes and asks confirmation, NEVER auto-kills.
  This stage adds a `--auto-yes` flag for launcher-driven cleanup with
  guard: only kills processes whose StartTime is OLDER than this session's
  expected start AND match the orchestrator/dashboard pattern.

  WITHOUT this guard, a parallel-terminal session would be silently killed
  on START_BOT.bat launch. WITH the guard (StartTime > X hours ago), only
  truly stale processes die.

mode: IMPLEMENTATION
status: DESIGN_LOCKED_PENDING_POST_SESSION_2026_05_26
priority: P3
deferred_reason: |
  Editing process-kill semantics on the day of a live debut is high-risk.
  Land post-session with adversarial review of the StartTime guard.

scope_lock:
  - scripts/tools/stop_live.ps1
  - START_BOT.bat
  - tests/test_tools/test_stop_live_auto.py  # NEW — PowerShell-script-as-data tests

agent: claude (opus 4.7)
---

## Blast Radius

- WRITES (modify): `scripts/tools/stop_live.ps1` — add `[switch]$AutoYes` parameter and `[int]$StaleHours = 4` parameter. When `-AutoYes` is set, filter matching processes by `StartTime -lt (Get-Date).AddHours(-$StaleHours)` before killing. Print kill list to stdout for log audit.
- WRITES (modify): `START_BOT.bat` — add `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\tools\stop_live.ps1 -AutoYes -StaleHours 4` as new Step 1 (before lock-file clear). Skip on failure (don't block launch).
- WRITES NEW: test that exercises stop_live.ps1 with `-WhatIf`-style dry-run, fakes process StartTime via mock data.
- LIVE-IMPACT: nonzero — silent kill of stale procs. Mitigation: 4-hour StartTime floor (parallel sessions started within last 4h are preserved); --auto-yes ONLY enabled when called from launcher; manual stop_live.ps1 remains interactive.
- Idempotency: safe (kills are no-ops if processes already dead).
- Rollback: revert .ps1 + .bat changes.

## Acceptance

1. `stop_live.ps1 -AutoYes -StaleHours 4` with no stale procs → exits 0, logs "No stale procs found"
2. `stop_live.ps1 -AutoYes -StaleHours 4` with 2 stale procs → kills both, logs PIDs + StartTimes
3. `stop_live.ps1 -AutoYes -StaleHours 4` with a fresh proc (StartTime within 4h) → SKIPS it, logs "preserving fresh proc PID=X StartTime=Y"
4. Manual `stop_live.ps1` (no flag) still prompts y/N interactively, no regression
5. `START_BOT.bat` launches successfully with the new step

## Doctrine references

- 2026-05-22 commit `53d25742` — original stop_live.ps1 with interactive-prompt design
- `feedback_duckdb_windows_lock_is_per_process.md` (root class — stale procs hold DB locks)
- `institutional-rigor.md` § 6 (no silent failures — must log every kill)

## Sources

- PowerShell `Get-Process` StartTime: Microsoft Learn — Get-Process cmdlet reference
- PowerShell `Stop-Process -Force`: Microsoft Learn — Stop-Process cmdlet
- ExecutionPolicy Bypass per-invocation: Microsoft Learn — about_Execution_Policies

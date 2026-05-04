# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex
- **Date:** 2026-04-23
- **Summary:** Closed stale control-state and MES/MGC pipeline debt, implemented PP-167 per-(session,instrument) ORB-cap hardening, and verified the slice.

## Next Steps — Active
1. Resolve repo-wide health-check blockers: missing .dbn.zst inputs, health_check timeout around pipeline/check_drift.py, and full pytest exit -11.
2. Pick the next ranked cleanup item from the action queue only after pulse/verification stay green.

## Blockers / Warnings
- Global health check is still not green end-to-end: missing .dbn.zst data inputs, pipeline/check_drift.py can exceed the health_check timeout budget, and full pytest tests/ -x -q previously exited -11.
- Worktree remains intentionally dirty with unrelated in-flight threads; do not revert them blindly.

## Durable References
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
- `docs/plans/2026-04-19-pp167-per-session-instrument-cap.md`
- `docs/plans/2026-04-21-post-stale-lock-action-queue.md`
- `docs/handoffs/archived/2026-04-23-root-handoff-archive.md`

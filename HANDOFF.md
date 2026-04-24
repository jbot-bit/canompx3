# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-24
- **Commit:** 0be5d1a6 — docs(handoff): record F1-F8 landing and parked-branch triage
- **Files changed:** 1 files
  - `HANDOFF.md`

## Next Steps — Active
1. Resume work from `main` @ `bad97445`. No parked branches remain.
2. Remove stale `.worktrees/action-parked-branches/` directory once Windows releases the file lock (git no longer tracks it).

## Blockers / Warnings
- None. All three parked codex branches actioned (one landed, two discarded as whitespace-only).

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`

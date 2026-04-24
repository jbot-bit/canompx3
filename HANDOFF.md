# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex
- **Date:** 2026-04-24
- **Commit:** 45f50916 — fix(live): harden dashboard action coordination
- **Summary:** Landed the live-control hardening on `main`, preserved the unrelated follow-up threads on parked branches, and removed the extra parked worktree so there is one active working copy again.
- **Additional Summary:** Runtime changes now block conflicting dashboard `start`/`preflight`/`refresh` actions, add guided handoff state for session-mode switches, keep preflight non-invasive (no real `SessionOrchestrator`), and skip signal-only crash-recovery dedup when the trade journal is unavailable.

## Next Steps — Active
1. Resume work from `main`; no secondary worktrees are open.
2. Parked follow-up branches remain available:
   `codex/phase4-discovery-gates-parked` @ `a1385028`
   `codex/followup-system-brief-phase4-parked` @ `400e6c28`
3. If full repo-green is required, clear the current repo-wide drift blockers before claiming global clean.

## Blockers / Warnings
- Repo-wide drift still fails outside this task:
  - Check 4: `work_queue.py` schema parser false-positive (`table 'the'`)
  - Check 59: `MNQ` has 1 trading day with `!= 3` `daily_features` rows
- Parked branches exist intentionally, but no extra worktrees are open.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`

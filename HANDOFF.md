# HANDOFF.md - Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done - update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Merged `origin/main` into `codex/plugin-routing-grounding` and resolved the HANDOFF-only conflict. Main's live-readiness automation summary remains current; this branch adds cross-tool plugin/data routing, automatic 2P targeted grounding, `/resource` and `/lit` local-corpus grounding, research/fetch source separation, PDF/OCR/literature coverage checks, and matching Claude/Codex prompt hooks.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-30
- **Commit:** bfab5942 — fix(hooks): real worktree mutex — (session_id,ppid)+heartbeat, not a phantom subprocess lock
- **Files changed:** 6 files
  - `.claude/hooks/session-start.py`
  - `.claude/hooks/tests/test_worktree_guard_hook.py`
  - `.claude/hooks/worktree_guard.py`
  - `docs/runtime/stages/2026-05-30-worktree-lease-real-mutex.md`
  - `scripts/tools/worktree_guard.py`
  - `tests/test_tools/test_worktree_guard.py`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`

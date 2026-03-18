# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code. If another tool made commits since your last read, your mental model is wrong.

---

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-03-18
- **Commit:** d0460bb — feat: project pulse — 9-signal project awareness engine
- **Summary:** Merged project pulse from workstream branch. Fixed drift failures in worktrees (symlink gold.db + models, uv sync).

## Decisions Made
- HANDOFF.md is now auto-updated by post-commit hook — no manual "handover" needed
- Pre-commit hook auto-stages dirty HANDOFF.md so updates ride along with the next commit
- Post-commit hook skips rebase/merge/cherry-pick (no spam)
- Trade-count WF design approved (`docs/plans/2026-03-17-trade-count-wf-design.md`) — ready to build
- Project pulse BUILT and merged (`scripts/tools/project_pulse.py`) — 9 collectors, /orient skill, bat launcher menu item
- Worktree env fix: symlink gold.db + models/ from main repo, use `uv run` (not bare `python`) in worktrees

## Next Steps
- **Trade-count WF** — implement from design doc (4 files, no schema changes)
- Streamlit dashboard: prop portfolio view
- CUSUM-based fitness (MEMORY.md action queue item 11)
- ATR-normalized position sizing (item 12)

## Blockers / Warnings
- Pre-existing test failure: `test_pipeline_status.py` — MGC missing outcomes for CME_PRECLOSE/NYSE_CLOSE
- `prop_portfolio.py` and `prop_profiles.py` are NO-TOUCH zones for Ralph
- Windows uses `.venv/`, WSL uses `.venv-wsl/` — do not cross-wire

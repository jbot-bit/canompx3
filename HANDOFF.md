# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code. If another tool made commits since your last read, your mental model is wrong.

---

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-03-18
- **Commit:** a893e62 — [mechanical] chore: Ralph Loop iter 134 — update audit state files
- **Files changed:** 5 files
  - `HANDOFF.md`
  - `docs/ralph-loop/ralph-ledger.json`
  - `docs/ralph-loop/ralph-loop-audit.md`
  - `docs/ralph-loop/ralph-loop-history.md`
  - `docs/ralph-loop/ralph-loop-plan.md`

## Decisions Made
- HANDOFF.md is now auto-updated by post-commit hook — no manual "handover" needed
- Pre-commit hook auto-stages dirty HANDOFF.md so updates ride along with the next commit
- Post-commit hook skips rebase/merge/cherry-pick (no spam)
- Trade-count WF design approved (`docs/plans/2026-03-17-trade-count-wf-design.md`) — ready to build
- **Pulse v2 SHIPPED** — `scripts/tools/project_pulse.py` (1300 lines, 44 tests)
  - `--fast` serves cached drift/tests, `--deep` caches FIT/WATCH/DECAY
  - `/orient` skill narrates pulse JSON, supports `--full`
  - `[0] Orient me` in ai-workstreams.bat launcher
  - `--with-pulse` in session_preflight + claude-worktree.sh
  - Research innovations: skill suggestions, session continuity, time-since-green, conflict radar
- Worktree env fix: symlink gold.db + models/ from main repo, use `uv run` (not bare `python`) in worktrees
- **Workstream finish flow needs improvement** — [3] Finish only deletes, doesn't merge+push first. Noted in `workstream_finish_flow.md`.

## Next Steps
- **Trade-count WF** — implement from design doc (4 files, no schema changes)
- **Workstream finish flow** — enhance [3] Finish to merge+push before close (noted for after pulse)
- Streamlit dashboard: prop portfolio view
- CUSUM-based fitness (MEMORY.md action queue item 11)
- ATR-normalized position sizing (item 12)

## Blockers / Warnings
- Pre-existing test failure: `test_pipeline_status.py` — MGC missing outcomes for CME_PRECLOSE/NYSE_CLOSE
- `prop_portfolio.py` and `prop_profiles.py` are NO-TOUCH zones for Ralph
- Windows uses `.venv/`, WSL uses `.venv-wsl/` — do not cross-wire

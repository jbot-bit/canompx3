# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code. If another tool made commits since your last read, your mental model is wrong.

---

## Last Session
- **Tool:** Codex (GPT-5.4)
- **Date:** 2026-03-17
- **Summary:** Finished the cross-tool workstream layer. Restored `session_preflight.py`, wired WSL/Codex startup preflight, added managed Claude/Codex worktree wrappers, verified real `git worktree` create/reuse/close, then refactored the Windows UX into `ai-workstreams.bat` with purpose-first workstream flow and review-driven hardening.

## Decisions Made
- Apex: automation AND copy trading PROHIBITED (OFFICIAL RULE). Manual proof only, 1 account.
- Tradeify: PRIMARY MNQ scaling lane (5 accounts, overnight sessions, Tradovate API)
- TopStep: MGC morning lane (5 Express, CME_REOPEN + TOKYO_OPEN, ProjectX API)
- Self-funded: Phase 3, inactive until prop proof complete
- Prop ceiling ~$60K/year. $100K requires self-funded IBKR (Phase 3).
- Canonical playbook: `docs/plans/manual-trading-playbook.md` (V3)
- Trading surface authority:
  - `python -m trading_app.prop_portfolio --daily --profile apex_50k_manual` = manual canonical
  - `python -m trading_app.prop_portfolio --daily` = cross-account overview
  - `python scripts/tools/generate_trade_sheet.py` = live/automation canonical
  - `trade-sheet.bat` / `daily-sheet.bat` = double-click launchers
- DailyLaneSpec pins exact strategy IDs per manual profile. TRADE/HOLD/REVIEW/SKIP catches drift honestly.
- LONDON_METALS lane swapped to 0.75x strategy (S075) — no more stop mismatch.
- EUROPE_FLOW stays in plan. HOLD when fitness is WATCH — correct behavior.
- Parallel edit sessions should use isolated worktrees, not one shared mutable branch.
- Canonical parallel entrypoints:
  - `scripts/infra/claude-worktree.sh open <task-name>`
  - `scripts/infra/codex-worktree.sh open <task-name>`
- Shared stale-state guard: `scripts/tools/session_preflight.py`
- Human-friendly Windows front doors:
  - `ai-workstreams.bat` = primary front door
  - `ai-workspaces.bat` = compatibility alias
  - `agent-tasks.bat` = compatibility alias
  - `claude-task.bat`, `codex-task.bat`, `codex-search-task.bat` = direct shortcuts
- Workstream UX rules:
  - Human concept = `workstream`, not `task`
  - Primary launcher = `ai-workstreams.bat`
  - Start flow is intent-first: name -> purpose -> recommended agent
  - Continue/finish flows use numbered managed workstream selection
  - Reopening an existing workstream preserves its saved purpose instead of silently rewriting it

## Files Modified
- `trading_app/prop_profiles.py` — DailyLaneSpec, session/instrument routing, firm specs, daily lanes
- `trading_app/prop_portfolio.py` — daily lane resolver, cross-account daily, calendar gates, --daily/--verbose/--fitness/--date
- `tests/test_trading_app/test_prop_profiles.py` — 21 tests
- `tests/test_trading_app/test_prop_portfolio.py` — 38 tests (including 5 integration tests for daily lane resolver)
- `HANDOFF.md`, `AGENTS.md`, `CLAUDE.md`, `CODEX.md` — cross-tool coordination
- `scripts/infra/*.sh` — fail-fast env guards for .venv-wsl
- `trade-sheet.bat`, `daily-sheet.bat` — double-click launchers
- `MARKET_PLAYBOOK.md` — DELETED
- `docs/plans/TRADING_PLAN_BEGINNER.md` — DELETED
- `scripts/tools/beginner_tradebook.py` — DELETED
- `scripts/tools/gen_playbook.py` — DELETED
- `ROADMAP.md`, `TRADING_RULES.md`, `scripts/infra/check_root_hygiene.py` — references cleaned
- `scripts/tools/session_preflight.py` — restored shared startup/stale-state guard with claim verification
- `scripts/tools/worktree_manager.py` — managed git worktree create/list/prune/close helper
- `scripts/infra/codex-project.sh`, `codex-project-search.sh`, `codex-review.sh`, `wsl-env.sh` — root override + auto-preflight
- `scripts/infra/codex-worktree.sh`, `scripts/infra/claude-worktree.sh` — parallel session wrappers
- `scripts/infra/windows-agent-launch.ps1` — Windows workstream launcher/menu
- `ai-workstreams.bat`, `ai-workspaces.bat`, `agent-tasks.bat`, `claude-task.bat`, `codex-task.bat`, `codex-search-task.bat`, `task-list.bat`, `task-close.bat`, `task-prune.bat` — human-facing launcher layer
- `tests/test_tools/test_session_preflight.py`, `tests/test_tools/test_worktree_manager.py` — targeted coverage
- `AGENTS.md`, `CLAUDE.md`, `CODEX.md`, `.codex/STARTUP.md` — worktree/preflight routing guidance

## Next Steps
- Streamlit dashboard: add prop portfolio view (daily card + firm selector + DD bars) — plan with /4tp + /quant-tdd
- CUSUM-based fitness (MEMORY.md action queue item 11)
- ATR-normalized position sizing (item 12)
- MGC WF revalidation with trade-count windows
- If Claude native startup should also force preflight outside the worktree wrapper path, add a machine-local launcher or shell alias on the Windows side

## Blockers / Warnings
- `.venv-wsl/bin/python` exists in this checkout now. The old “empty `.venv-wsl/`” warning is stale.
- Windows uses `.venv/`, WSL uses `.venv-wsl/`. Do not cross-wire.
- Pre-existing test failure: `test_pipeline_status.py` — MGC missing outcomes for CME_PRECLOSE/NYSE_CLOSE. Not related to prop portfolio.
- A stale `HEAD` file in repo root was causing git desync — DELETED. If git commands fail with "ambiguous argument 'HEAD'", check for stale files in repo root.
- `prop_portfolio.py` and `prop_profiles.py` are NO-TOUCH zones for Ralph (actively developed).
- `session_preflight.py --verify-claim` will intentionally scream if another tool updates the same branch/HEAD claim. That is desired.
- Verified live: managed worktree create/reuse/close passed for `smoke-verify`.

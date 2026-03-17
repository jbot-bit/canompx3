# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code. If another tool made commits since your last read, your mental model is wrong.

---

## Last Session
- **Tool:** Claude Code (Opus 4.6)
- **Date:** 2026-03-17
- **Summary:** Built complete prop firm portfolio system end-to-end. Session→firm routing, DailyLaneSpec pinned manual lanes, cross-tool coordination, stale surface cleanup, double-click launchers, 5 rounds of code review. Fixed stale HEAD file that was causing git desync all session.

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

## Next Steps
- Streamlit dashboard: add prop portfolio view (daily card + firm selector + DD bars) — plan with /4tp + /quant-tdd
- CUSUM-based fitness (MEMORY.md action queue item 11)
- ATR-normalized position sizing (item 12)
- MGC WF revalidation with trade-count windows

## Blockers / Warnings
- `.venv-wsl/` is empty. Must `uv sync --frozen` in WSL before Codex scripts work.
- Windows uses `.venv/`, WSL uses `.venv-wsl/`. Do not cross-wire.
- Pre-existing test failure: `test_pipeline_status.py` — MGC missing outcomes for CME_PRECLOSE/NYSE_CLOSE. Not related to prop portfolio.
- A stale `HEAD` file in repo root was causing git desync — DELETED. If git commands fail with "ambiguous argument 'HEAD'", check for stale files in repo root.
- `prop_portfolio.py` and `prop_profiles.py` are NO-TOUCH zones for Ralph (actively developed).

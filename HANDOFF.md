# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

---

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-03-17
- **Summary:** Built prop portfolio session→firm routing. Canonicalized all prop firm info to playbook. Fixed 6+ stale files referencing dead "20 Apex copy-traded" plan. Added allowed_sessions/allowed_instruments to AccountProfile. 40/40 tests pass.

## Decisions Made
- Apex: automation AND copy trading PROHIBITED (OFFICIAL RULE). Manual proof only, 1 account.
- Tradeify: PRIMARY MNQ scaling lane (5 accounts, overnight sessions, Tradovate API)
- TopStep: MGC morning lane (5 Express accounts, CME_REOPEN + TOKYO_OPEN, ProjectX API)
- Self-funded: Phase 3, inactive until prop proof complete
- MFFU: deprioritized, not in plan
- Prop ceiling ~$60K/year. $100K requires self-funded IBKR (Phase 3).
- Canonical playbook: `docs/plans/manual-trading-playbook.md` (V3)

## Files Modified
- `trading_app/prop_profiles.py` — added session/instrument routing, fixed Apex to auto_trading="none"
- `trading_app/prop_portfolio.py` — added session/instrument filter steps
- `tests/test_trading_app/test_prop_profiles.py` — updated for new profiles + routing
- `tests/test_trading_app/test_prop_portfolio.py` — added 3 routing tests
- `docs/plans/2026-03-15-prop-portfolio-*.md` — marked HISTORICAL ARTIFACT
- `scripts/run_webhook_server.py` — fixed stale Apex checklist
- `memory/2026-03-16.md` — fixed stale Apex "semi-auto" reference
- Multiple memory files — aligned to playbook (prop_scaling_roadmap, trading_plan_sim, MEMORY.md, live_infra_todo)

## Next Steps
- Env hardening: fail-fast guards in WSL scripts for .venv-wsl
- Consider adding EUROPE_FLOW + LONDON_METALS to Tradeify overnight sessions (currently Apex-only manual)
- `python -m trading_app.prop_portfolio --all --summary` now works end-to-end — use it

## Blockers / Warnings
- `.venv-wsl/` is empty (just .gitignore). Must `uv sync --frozen` in WSL before Codex scripts work.
- `.venv/` is the working Windows env for Claude Code.
- Do not cross-wire: Windows shells use `.venv/`, WSL shells use `.venv-wsl/`.

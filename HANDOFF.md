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
- **Commit:** 1cc7f4a1 — fix(live): ralph iter 213 — lifecycle-block silent-fail + readiness effective-copies
- **Files changed:** 8 files
  - `scripts/run_live_session.py`
  - `scripts/tools/live_readiness_report.py`
  - `scripts/tools/refresh_control_state.py`
  - `tests/test_scripts/test_run_live_session_preflight.py`
  - `tests/test_tools/test_live_readiness_report.py`
  - `tests/test_tools/test_refresh_control_state.py`
  - `tests/test_trading_app/test_session_orchestrator.py`
  - `trading_app/live/session_orchestrator.py`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`

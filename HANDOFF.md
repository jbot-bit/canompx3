# HANDOFF.md - Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done - update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Merged `origin/main` into `codex/plugin-routing-grounding` and resolved the HANDOFF-only conflict. Main's live-readiness automation summary remains current; this branch adds cross-tool plugin/data routing, automatic 2P targeted grounding, `/resource` and `/lit` local-corpus grounding, research/fetch source separation, PDF/OCR/literature coverage checks, and matching Claude/Codex prompt hooks.

## Last Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Commit:** codex/plugin-routing-grounding - chore(codex): route plugins and grounding automatically
- **Files changed:** 18 files
  - `.claude/hooks/plugin-router.py`
  - `.claude/hooks/targeted-grounding-router.py`
  - `.claude/rules/auto-skill-routing.md`
  - `.claude/rules/plugin-routing.md`
  - `.claude/rules/targeted-grounding.md`
  - `.claude/settings.json`
  - `.codex/COMMANDS.md`
  - `.codex/INTEGRATIONS.md`
  - `.codex/PLUGIN_ROUTING.md`
  - `.codex/STARTUP.md`
  - `.codex/TARGETED_GROUNDING.md`
  - `.codex/WORKFLOWS.md`
  - `.codex/hooks/session_start.py`
  - `.codex/hooks/user_prompt_submit_grounding.py`
  - `HANDOFF.md`
  - `scripts/tools/check_literature_coverage.py`
  - `scripts/tools/check_pdf_tooling.py`
  - `tests/test_hooks/test_targeted_grounding_router.py`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`

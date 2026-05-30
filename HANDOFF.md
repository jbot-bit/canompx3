# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Added cross-tool plugin/data routing and automatic targeted-grounding behavior for the current Codex/Claude plugin set. New shared contract `.codex/PLUGIN_ROUTING.md` defines when GitHub, Browser/Playwright, Datadog, MarcoPolo, Supabase, Spreadsheets, Presentations, Slack, etc. are suitable and reiterates that repo truth remains code/DB/MCP/docs/GitHub/runtime artifacts. Codex prompt hook now auto-injects routing hints for Datadog/observability, MarcoPolo/external data, Supabase/Postgres, artifact plugins, personal-context plugins, CircleCI, and check/improve/implement/fix/review/plan prompts. Claude side now has `.claude/rules/plugin-routing.md`, `.claude/hooks/plugin-router.py`, `.claude/rules/targeted-grounding.md`, and `.claude/hooks/targeted-grounding-router.py` wired into `.claude/settings.json` so Claude gets the same compact prompt-time routing cues. `2P` now means second pass and triggers the same targeted-grounding route, alongside semantic equivalents like double check, fresh eyes, sanity check, red-team, critique, stress-test, take a look, is this good, what am I missing, poke holes, blind spots, sense check, will this work, flaws, gotchas, and risks. `/resource` and `/lit` now mean local grounding truth: run `python scripts/tools/check_pdf_tooling.py` and `python scripts/tools/check_literature_coverage.py`, open `resources/INDEX.md`, prefer mapped `docs/institutional/literature/` extracts only when covered, read raw resources directly only when present on the local PC, do not answer from memory/feel, and do not skim/guess PDF content. `resources/` raw PDFs are documented as local-PC assets, not guaranteed remote/CI state. Research/fetch prompts now inject source-separation guidance: official/primary sources first; user comments/issues/forums are unofficial cautionary signals unless corroborated. Added `tests/test_hooks/test_targeted_grounding_router.py` to keep Claude/Codex second-pass/resource/source-separation semantics covered. `.claude/rules/plugin-routing.md` also includes Claude plugin-alignment notes: infer missing capabilities from intent, check existing Claude tools first, and propose the smallest plugin/install only when a named task needs it. Hook JSON parsing hardened for Windows BOM-prefixed stdin. Verified with `py_compile`, `.claude/settings.json` JSON validation, hook simulations, and `git diff --check`.

## Last Session
- **Tool:** Unknown
- **Date:** 2026-05-30
- **Commit:** codex/plugin-routing-grounding — chore(codex): route plugins and grounding automatically
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
  - ... and 3 more

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`

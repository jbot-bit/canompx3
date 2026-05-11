# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-11
- **Commit:** fabae777 — feat(research): LLM hypothesis proposer (Track A) — literature-grounded prereg drafts
- **Files changed:** 18 files
  - `.claude/skills/propose-hypothesis/SKILL.md`
  - `docs/prompts/hypothesis-proposer-fewshot.md`
  - `docs/prompts/hypothesis-proposer-system.md`
  - `docs/runtime/stages/llm-hypothesis-proposer-track-a.md`
  - `scripts/research/lhp/__init__.py`
  - `scripts/research/lhp/adjacency.py`
  - `scripts/research/lhp/literature_index.py`
  - `scripts/research/lhp/llm_client.py`
  - `scripts/research/lhp/static_checks.py`
  - `scripts/research/lhp/yaml_emitter.py`
  - `scripts/research/llm_hypothesis_proposer.py`
  - `tests/fixtures/lhp/bad_banned_feature.yaml`
  - `tests/fixtures/lhp/bad_fabricated_citation.yaml`
  - `tests/fixtures/lhp/bad_minbtl_exceeded.yaml`
  - `tests/fixtures/lhp/bad_wrong_holdout.yaml`
  - ... and 3 more

## Next Steps — Active
1. Track D MNQ COMEX_SETTLE Gate 0 runner design — Design the Databento top-of-book table and bounded runner needed to execute the DESIGN_ONLY prereg.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`

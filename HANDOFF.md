# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Stage 1 — Routine-TBBO slippage registry refactor (2026-05-11, Claude Opus, branch `stage1/generalize-tbbo-slippage-inference`)

Landed: `refactor(deployability): registry-driven routine-TBBO slippage inference`.
- Replaced MNQ-only `_mnq_routine_tbbo_slippage_applies()` (deployability.py:349)
  with a registry-driven dispatcher; added `RoutineTbboPilot` dataclass +
  `ROUTINE_TBBO_SLIPPAGE_REGISTRY` populated from MNQ + MES pilot v1 evidence.
- New BLOCKING drift check `check_routine_tbbo_slippage_registry_coverage`
  parses `## Verdict: **PASS**` lines on `*slippage*pilot*v1*.md` and fails
  closed on under/over-coverage.
- Empirical: 2 MES COMEX_SETTLE rows drop the `slippage_missing` hard issue
  (verdict still `BLOCKED_FAMILY_FRAGILE` pending Stage 2 family_singleton
  policy decision).
- 124 drift checks pass, 33 deployability + 6 drift-check tests pass, 198 in
  Stage 1 scope, 4551 broader tests pass (1 pre-existing WSL-doctor failure
  unrelated to this diff).
- Capital safety verified: lane allocator keys off `s.status`, not the
  deployable flag — no MES capital deploys without Stage 3 profile +
  lane_allocation.json edit.
- Stage doc: `docs/runtime/stages/stage1-generalize-tbbo-slippage-inference.md`.
- Survey doc landed as evidence provenance:
  `docs/audit/results/2026-05-11-mes-profile-feasibility-readonly-survey.md`.

Next: Stage 2 (separate worktree from `origin/main`) — doctrine decision on
`family_singleton` policy. Without Stage 2, no MES verdict actually flips
to deployable.

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

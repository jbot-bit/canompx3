# CONVENTIONS.md

Aider-canonical project doctrine for the canompx3 repository.

This file is **pointers, not content**. Every rule below delegates to a single
source of truth elsewhere in the repo. Do not restate the rules here — read the
linked file. If the linked file disagrees with this one, the linked file wins.

## Authority hierarchy

- Code structure, guardrails, repo workflow → `CLAUDE.md`
- Trading logic, filters, sessions, entry models, costs → `TRADING_RULES.md`
- Research methodology, statistical thresholds, holdout policy → `RESEARCH_RULES.md`
- Strategy research routing, blueprint test sequence, NO-GO registry → `docs/STRATEGY_BLUEPRINT.md`
- Document role registry → `docs/governance/document_authority.md`
- Institutional literature grounding (cite from here, not training memory) → `docs/institutional/literature/`
- Locked validation thresholds (no post-hoc relaxation) → `docs/institutional/pre_registered_criteria.md`

## Banned patterns

- Hardcoded canonical values (instruments, sessions, costs, DB paths, ORB windows). Always import from the canonical source listed in `.claude/rules/integrity-guardian.md` § 2.
- Re-encoding logic that already exists in a canonical source. See `.claude/rules/institutional-rigor.md` § 4.
- Dead fields, dead enums, dead parameters. See `.claude/rules/institutional-rigor.md` § 5.
- Silent failures (`except Exception` without recording, fall-open without doctrine grounding). See `.claude/rules/institutional-rigor.md` § 6.
- Inline research stats (p-values, N counts) in code. Use `@research-source` + `@revalidated-for`. See `.claude/rules/research-truth-protocol.md`.
- Querying changing stats (strategy counts, session timings, cost specs) from memory or docs. Always query the live canonical source. See `CLAUDE.md` § Volatile Data Rule.

## Seven Sins of Quantitative Investing

The defensive surface against bias. Look-ahead, data snooping, overfitting,
survivorship, storytelling, outlier distortion, transaction-cost illusion.
See `.claude/rules/quant-agent-identity.md`.

## Verification gate (before claim of done)

All four are required, every time:

1. Tests pass (show output, do not summarize).
2. Dead code swept (`grep -r` for new symbols).
3. `python pipeline/check_drift.py` passes.
4. Self-review (code-review skill or equivalent) with line citations.

See `.claude/rules/institutional-rigor.md` § 8.

## Stage-gate protocol

Non-trivial work requires a stage file at `docs/runtime/stages/<slug>.md` with
`task`, `mode`, `scope_lock`, and `blast_radius` fields. The `stage-gate-guard.py`
hook enforces scope-lock; production-code edits outside the locked file set are
blocked. See `.claude/rules/stage-gate-protocol.md`.

## Pre-edit discovery marker

Edits to `pipeline/`, `trading_app/`, or `scripts/` require a discovery marker
or a `TRIVIAL:`/`REPRO:` declaration before the first edit. Hook-enforced
(fail-open). See `.claude/rules/institutional-rigor.md` § 9 and
`docs/plans/discovery-loop-hardening.md`.

## DeepSeek Coding Agent

The repo-native coding agent runs under the `deepseek_coding` profile in
`trading_app.ai.provider_registry`. Profile is fail-closed (`model=None`) until
the Phase 2.5 bake-off picks the winner. Every commit authored by the agent is
gated by a claude-side reviewer (Phase 3, pre-commit step 0d). See
`docs/runtime/stages/deepseek-coding-agent-v4.md` for the active stage.

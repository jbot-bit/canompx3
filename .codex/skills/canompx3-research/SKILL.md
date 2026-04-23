---
name: canompx3-research
description: Run research and analysis work under the repo's quant standards. Use when evaluating findings, interpreting strategy results, reviewing research claims, or planning/assessing new research against the canonical research rules.
---

# Canompx3 Research

Use this skill for research, statistical interpretation, and strategy-analysis work in this repo.

This skill is a Codex wrapper around the canonical research standards. It does not replace or reinterpret the Claude layer.

## Canonical Sources

Read these first:

- `RESEARCH_RULES.md`
- `.claude/rules/quant-agent-identity.md`
- `docs/specs/research_modes_and_lineage.md`
- `docs/institutional/research_pipeline_contract.md`
- `docs/institutional/conditional-edge-framework.md`
- `.claude/skills/audit/SKILL.md` (mode: prompts for guardian routing)
- `TRADING_RULES.md` when research conclusions touch live behavior
- `docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md` when the task is
  discovery, new-edge triage, or hypothesis shaping

## What This Skill Does

- Forces research work to follow the repo's statistical and market-structure standards
- Pushes mechanism-first interpretation instead of story-first interpretation
- Distinguishes validated findings from in-sample observations
- Keeps research claims honest, especially around sample size, multiple testing, and deployability

## Default Flow

1. Identify whether the task is discovery, validation, interpretation, or audit.
2. If it is discovery, route through
   `docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md` before any scan or
   implementation talk.
3. Apply `docs/institutional/research_pipeline_contract.md` so discovery,
   confirmation, validation, deployment, and execution are not collapsed.
4. Load the canonical research docs above.
5. Check sample size, time span, IS/OOS status, and multiple-testing risk.
6. Demand a mechanism for any claimed edge.
7. Report what survived, what failed, caveats, and next steps.

## Operator Rule

The user should not need to invoke Python scripts or shell wrappers directly.
When the user asks naturally to find, test, validate, classify, or audit a
research idea, Codex should run the repo tooling internally and return the
decision, destination, and verification.

For preregistered work, use the prereg front door internally to verify the
pipeline branch before execution:

- `standalone_edge` can write discovery candidates to `experimental_strategies`
  and may later promote through confirmation / validator to `validated_setups`.
- `conditional_role` goes to a bounded runner / result-doc flow and requires an
  explicit role decision. It should not be forced into `experimental_strategies`
  or `validated_setups` unless the prereg defines a complete standalone lane.

Do not describe live routing or `paper_trades` as required for research
validation. They are deployment / operations gates.

## Rules

- Do not present in-sample results as validated edges.
- Do not call noise a finding because it sounds plausible.
- Do not ignore BH/FDR or parameter-fragility risk after broad scans.
- Do not suggest weakening filters just to raise trade count.
- Prefer family-level and mechanism-aware conclusions over cherry-picked strategy anecdotes.
- Do not skip the discovery protocol when the real task is idea triage or role
  mapping.
- Do not force every finding into standalone validation.
- Do not require deployment before validation.
- If a conclusion affects live trading, cross-check against `TRADING_RULES.md`.

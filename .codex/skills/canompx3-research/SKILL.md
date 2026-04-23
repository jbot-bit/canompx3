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
3. Load the canonical research docs above.
4. Check sample size, time span, IS/OOS status, and multiple-testing risk.
5. Demand a mechanism for any claimed edge.
6. Report what survived, what failed, caveats, and next steps.

## Rules

- Do not present in-sample results as validated edges.
- Do not call noise a finding because it sounds plausible.
- Do not ignore BH/FDR or parameter-fragility risk after broad scans.
- Do not suggest weakening filters just to raise trade count.
- Prefer family-level and mechanism-aware conclusions over cherry-picked strategy anecdotes.
- Do not skip the discovery protocol when the real task is idea triage or role
  mapping.
- If a conclusion affects live trading, cross-check against `TRADING_RULES.md`.

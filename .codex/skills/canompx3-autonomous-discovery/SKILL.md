---
name: canompx3-autonomous-discovery
description: Run the MNQ board-driven autonomous discovery workflow in canompx3. Use when Codex should continue the bounded discovery hub that refreshes the board stack, chooses the next honest mechanism-family candidate, runs the cheap gate, and advances or parks the queue without reopening dead search space.
---

# Canompx3 Autonomous Discovery

Use this skill when Codex is acting as the discovery hub for the recovered
`wt-codex-mnq-hiroi-scan` workflow.

This skill carries the durable method. The runner should only schedule and
resume it.

## Canonical Sources

Read these first:

- `AGENTS.md`
- `CLAUDE.md`
- `CODEX.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `.claude/rules/quant-agent-identity.md`
- `docs/STRATEGY_BLUEPRINT.md`

Then load the recovered workflow surfaces:

- `docs/plans/2026-04-22-mnq-autonomous-discovery-automation.md`
- `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md`
- `docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md`
- `docs/plans/2026-04-22-mnq-geometry-transfer-workflow.md`
- `docs/plans/2026-04-22-mnq-usdata1000-geometry-family-register.md`
- `docs/audit/results/2026-04-22-mnq-layered-candidate-board-v1.md`
- `docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`
- `docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`
- `research/phase4_candidate_precheck.py`

## What This Skill Does

- rebuilds the live bounded queue from repo truth
- keeps the search wide across alive mechanism classes without random fishing
- chooses one bounded next move per iteration
- advances the queue only through honest prereg, precheck, discovery, validator,
  or explicit park/kill outputs

## Default Flow

1. Treat the tiered board stack as the read-only evidence surface.
2. Reconstruct which mechanisms are solved, paused, or still alive.
3. Pick the smallest honest next move:
   - prereg
   - cheap gate
   - narrow discovery/validator advancement
   - result note
4. Verify any durable change with repo-native checks.
5. Decide whether the loop should continue.

## Rules

- Do not reopen dead ML, dead stacked-pair, or dead reopen paths from
  `docs/STRATEGY_BLUEPRINT.md`.
- Do not confuse broad route-map evidence with direct promotion evidence.
- Do not treat read-only board observations as validated strategies.
- Do not mine solved lanes again just because a local exact cell still looks
  tempting.
- Do not advance more than one exact bridge candidate in a single iteration.
- Do not skip the cheap gate before a real discovery write.
- Do not leave queue decisions only in chat output when they should live in repo
  docs.

## Good Outputs

- a refreshed shortlist grounded in the board stack
- a widened shortlist that considers more than one alive mechanism class
- one narrow prereg for the best alive candidate
- one cheap-gate decision on an exact hypothesis
- one promoted candidate with validator evidence
- one explicit park/kill note that removes bias

## Bad Outputs

- giant prompt-sprawl that duplicates repo rules
- a broad rediscovery sweep over arbitrary feature pairs
- holdout shopping after weak OOS
- multiple simultaneous branch advances from one loop
- tunnel vision on one local exact cell while broader alive mechanisms exist

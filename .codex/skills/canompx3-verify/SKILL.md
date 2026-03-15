---
name: canompx3-verify
description: Run the standard canompx3 verification flow after code changes. Use when verifying edits, checking whether work is actually done, or running the repo's expected gates before sign-off.
---

# Canompx3 Verify

Use this skill for post-edit verification in this repo.

This skill is a Codex wrapper around the canonical Claude verification workflow. It does not replace or reinterpret the Claude layer.

## Canonical Sources

Read these first:

- `.claude/commands/quant-verify.md`
- `.claude/commands/health-check.md`
- `.claude/agents/verify-complete.md`
- `CLAUDE.md`

## What This Skill Does

- Chooses the right verification path for the current task
- Prefers targeted verification before broad verification
- Uses the repo's existing drift, integrity, behavioral, and test gates
- Reports evidence instead of making unsupported claims

## Default Flow

1. Identify what changed.
2. Load the canonical verification docs above.
3. Run the smallest correct verification set first.
4. Escalate to broader checks only when needed.
5. Report pass/fail clearly with real command evidence.

## Rules

- Do not claim work is done without running checks.
- Do not hardcode drift-check counts or stale repo stats.
- Prefer targeted tests before broad tests.
- If a gate fails, stop and report the failure before claiming completion.
- Keep verification aligned with Claude workflow and repo guardrails.

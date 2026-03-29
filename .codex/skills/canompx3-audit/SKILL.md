---
name: canompx3-audit
description: Run a structured audit pass for canompx3. Use when auditing the repo, reviewing risks, checking project health, or doing a quick/system audit against the canonical Claude audit workflows.
---

# Canompx3 Audit

Use this skill for repo audits in this project.

This skill is a Codex wrapper around the canonical Claude audit workflows. It does not replace the Claude layer.

## Canonical Sources

Read the smallest relevant set:

- `.claude/skills/audit/SKILL.md` (modes: full, quick, phase N, hypothesis)
- `CLAUDE.md`

## What This Skill Does

- Routes audit work to the correct canonical workflow
- Helps distinguish quick audits from deeper multi-phase audits
- Keeps findings evidence-based and ordered by importance
- Avoids vague "looks fine" summaries

## Default Flow

1. Determine the audit scope:
   - quick check
   - current-change review
   - deeper system audit
2. Load the matching canonical Claude audit docs.
3. Run the appropriate checks and inspections.
4. Report findings first, then residual risks, then brief summary.

## Rules

- Findings come before summaries.
- Prefer concrete evidence over inference.
- Separate code risk, live-trading risk, and documentation drift.
- Do not silently invent a new audit standard when Claude already defines one.

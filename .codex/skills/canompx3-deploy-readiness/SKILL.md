---
name: canompx3-deploy-readiness
description: Evaluate whether a canompx3 change, strategy, profile, or runtime path is actually ready to promote or deploy. Use when deciding if something is production-safe, promotion-ready, or blocked by verification, provenance, or live-control gaps.
---

# Canompx3 Deploy Readiness

Use this skill when the real question is "can this be promoted or deployed?"
rather than "does this look promising?"

This skill is a Codex wrapper around the canonical Claude verification,
research, and live-control rules. It does not replace or reinterpret the Claude
layer.

## Canonical Sources

Read the smallest relevant set:

- `CLAUDE.md`
- `.claude/skills/verify/SKILL.md`
- `.claude/skills/audit/SKILL.md`
- `TRADING_RULES.md`
  - for live-trading behavior and execution constraints
- `RESEARCH_RULES.md`
  - when strategy readiness depends on research validity
- `trading_app/prop_profiles.py`
  - for live profile routing and deployment authority

## What This Skill Does

- Distinguishes "implemented" from "deployable"
- Forces a real evidence chain across tests, drift, audits, provenance, and
  runtime authority
- Checks whether the target is blocked by missing controls, stale docs, dirty
  env assumptions, or unvalidated research claims
- Produces a clear go/no-go or blocked verdict with concrete blockers

## Default Flow

1. Identify the deployable unit:
   - code change
   - live runtime path
   - account/profile routing change
   - strategy or validated lane promotion
2. Load the governing canonical sources.
3. Check the evidence chain:
   - verification results
   - drift / audit status
   - provenance / prereg / validation status where relevant
   - live profile and runtime authority alignment
4. Separate hard blockers from advisories.
5. Output one of:
   - GO
   - NO_GO
   - BLOCKED_PENDING_X

## Rules

- Do not call something deployable without fresh evidence.
- Do not promote in-sample or under-validated strategy claims.
- Do not treat deprecated compatibility surfaces as deployment authority.
- Do not ignore monitor, gate, or kill-switch requirements just because the
  code path exists.
- Prefer an honest `BLOCKED_PENDING_X` over a soft, vague "probably ready."

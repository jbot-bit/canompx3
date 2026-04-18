---
name: canompx3-live-audit
description: Audit live-trading and runtime-control surfaces for canompx3. Use when reviewing session orchestration, broker adapters, webhook safety, monitoring, account-routing behavior, or other live-path risks against the canonical Claude rules.
---

# Canompx3 Live Audit

Use this skill for live-trading and runtime-control audits in this repo.

This skill is a Codex wrapper around the canonical Claude live-audit and
runtime guardrails. It does not replace or reinterpret the Claude layer.

## Canonical Sources

Read the smallest relevant set first:

- `.claude/skills/audit/SKILL.md`
  - use `phase 7` for live-trading audits
- `CLAUDE.md`
- `TRADING_RULES.md` when the audit touches live trading behavior
- `trading_app/prop_profiles.py`
  - live profile authority
- `trading_app/live/session_orchestrator.py`
  - live orchestration authority
- `trading_app/live/webhook_server.py`
  - webhook and alert-entry authority when relevant

## What This Skill Does

- Routes live audits to the canonical authority surfaces
- Focuses review on fail-closed controls, routing correctness, and runtime risk
- Distinguishes code defects from operator/runbook gaps and env drift
- Keeps findings anchored to the real live authorities, not deprecated surfaces

## Default Flow

1. Identify the live surface under review:
   - session orchestration
   - broker adapter
   - webhook/alert entry
   - monitor/state/kill-switch behavior
   - account/profile routing
2. Load the matching canonical sources above.
3. Check fail-closed behavior, auth/rate-limit/dedup controls, and authority
   usage.
4. Verify the code is reading the canonical live profile and runtime surfaces.
5. Report findings first, then residual operator risk, then brief next steps.

## Rules

- `trading_app/prop_profiles.py` is the live profile authority.
- Treat `trading_app/live_config.py` as compatibility or deprecated unless the
  current code proves otherwise.
- Webhook auth, dedup, rate limiting, account resolution, and flatten/kill
  behavior are safety-critical, not optional niceties.
- Separate runtime/env problems from code regressions.
- Do not claim a live path is safe because it "worked once" in paper or local
  testing.

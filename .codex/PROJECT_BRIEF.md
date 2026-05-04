# Codex Project Brief

Use this as the fast orientation layer for the repo. It is a summary only. Claude docs and project docs remain canonical.

## What This Project Is

- Multi-instrument futures research and trading platform centered on ORB breakout systems.
- Canonical database: `gold.db`.
- Canonical code paths:
  - `pipeline/` for ingestion, aggregation, sessions, and feature building
  - `trading_app/` for outcomes, discovery, validation, portfolio, and the in-progress live trading stack
  - `scripts/` for audits, rebuild helpers, and operator tooling

## Canonical Sources

- Project workflow and guardrails: `CLAUDE.md`
- Trading truth: `TRADING_RULES.md`
- Research and statistical discipline: `RESEARCH_RULES.md`
- Current shared session state and near-term decisions: `HANDOFF.md`
- Architecture and command reference: `docs/ARCHITECTURE.md`
- Planned but not yet built work: `ROADMAP.md`
- Codex adapter front door: `CODEX.md`

## Trading Model In One Pass

- ORB breakout is the confirmed edge family.
- ORB size is the primary edge across instruments.
- Sessions are event-based and resolved per day from `pipeline/dst.py`.
- E1 and E2 are the live entry models worth caring about for most current work.
- E0 is purged. E3 is not the default path and should be treated as specialist or retired unless current canonical docs say otherwise.
- Do not memorize active instruments from this file. Confirm the current universe from canonical docs and current code, and if they disagree, report the contradiction instead of normalizing it inside `.codex/`.

## Core Systems

- `bars_1m` -> `bars_5m` -> `daily_features`
- `daily_features` -> `orb_outcomes`
- `orb_outcomes` -> `experimental_strategies`
- `experimental_strategies` -> `validated_setups`
- `validated_setups` -> `edge_families`
- live deployment profiles -> `trading_app/prop_profiles.py`
- Rolling/HOT real-time activation is not fully wired yet and should not be described as finished live trading.

## Current Operating Philosophy

- Claude Code is the canonical authority for project workflow and guardrails.
- Codex is a thin client on the same project.
- Do not create a second project state, second database workflow, or competing rule layer.
- Prefer read-first exploration, validation-heavy implementation, and thin
  startup context.

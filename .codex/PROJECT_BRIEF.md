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
- Architecture and command reference: `docs/ARCHITECTURE.md`
- Planned but not yet built work: `ROADMAP.md`

## Trading Model In One Pass

- ORB breakout is the confirmed edge family.
- ORB size is the primary edge across instruments.
- Sessions are event-based and resolved per day from `pipeline/dst.py`.
- E1 is the honest conservative entry.
- E2 is the honest aggressive production entry and is now dominant in the declared live portfolio config.
- E0 is purged.
- E3 is soft-retired and should not be treated as the default direction.

## Active Instruments

- Active ORB instruments: MGC, MNQ, MES, M2K.
- Dead for ORB breakout: MCL, SIL, M6E, MBT.

## Core Systems

- `bars_1m` -> `bars_5m` -> `daily_features`
- `daily_features` -> `orb_outcomes`
- `orb_outcomes` -> `experimental_strategies`
- `experimental_strategies` -> `validated_setups`
- `validated_setups` -> `edge_families`
- `validated_setups` + fitness logic -> declared live portfolio in `trading_app/live_config.py`
- Rolling/HOT real-time activation is not fully wired yet and should not be described as finished live trading.

## Current Operating Philosophy

- Claude Code is the canonical authority for project workflow and guardrails.
- Codex is a thin client on the same project.
- Do not create a second project state, second database workflow, or competing rule layer.
- Prefer read-first exploration and validation-heavy implementation.

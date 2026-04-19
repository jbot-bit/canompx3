# Level Interaction V1

**Date:** 2026-04-19
**Status:** Research-only specification

## Purpose

Define a thin, reusable research layer for level-aware intraday event studies
without creating a second truth system.

This layer exists to support the first two hypothesis families only:

1. level pass/fail
2. sweep/reclaim

It is explicitly **not** a live-trading feature spec, not a pipeline schema
change, and not an ICT/FVG ontology.

## Existing overlap

This repo already has canonical pre-trade and break-time primitives:

- `prev_day_high`, `prev_day_low`, `prev_day_close`
- `overnight_high`, `overnight_low`
- `session_asia_high/low`, `session_london_high/low`, `session_ny_high/low`
- `orb_{session}_high/low`
- `orb_{session}_break_delay_min`
- `orb_{session}_break_bar_continues`
- `orb_{session}_double_break`

Source of truth:

- `pipeline/build_daily_features.py`
- `pipeline/session_guard.py`

The missing piece is a single reusable contract for **level interactions** so
research scripts stop re-encoding touch / fail / reclaim semantics ad hoc.

## Boundary

### In scope

- research-only helper layer
- canonical level resolution from `daily_features`
- chronology-safe access via `pipeline.session_guard`
- first-interaction classification for:
  - `touch_only`
  - `wick_fail`
  - `close_through`
- sweep flag on the interaction bar
- reclaim detection after a swept close-through

### Out of scope

- changes to `pipeline/build_daily_features.py`
- changes to `orb_outcomes`
- live execution code
- entry/exit automation
- market profile / `POC` / `VAH` / `VAL`
- FVG / IFVG / order-block object models
- generic state-machine or DSL infrastructure

## Canonical level set (v1)

Supported ex-ante levels:

- `prev_day_high`
- `prev_day_low`
- `pivot`
- `overnight_high`
- `overnight_low`
- `session_asia_high`
- `session_asia_low`
- `session_london_high`
- `session_london_low`
- `session_ny_high`
- `session_ny_low`

`pivot` is derived only from:

- `prev_day_high`
- `prev_day_low`
- `prev_day_close`

No other synthetic levels are allowed in v1.

## Chronology rules

The layer must be fail-closed.

- A level may only be resolved if it is available for the target session under
  `pipeline.session_guard.is_feature_safe(...)`.
- If a level is not chronologically safe for the target session, the layer must
  return unavailable rather than infer or backfill.
- Session-complete highs/lows are never legal before their safe-after session.

Examples:

- `prev_day_high` is safe for all sessions.
- `overnight_high` is **not** safe for `TOKYO_OPEN`.
- `session_london_high` is **not** safe before `NYSE_OPEN`.

## Interaction semantics

All interactions are directional and require an explicit `reference_side`:

- `below`: price approaches the level from below
- `above`: price approaches the level from above

The first eligible interaction bar is the first bar where:

- the pre-bar price is still on the `reference_side`
- the bar range touches the level

Pre-bar price:

- previous bar close when available
- otherwise the current bar open

### `touch_only`

The bar touches the level but does not close through it and does not wick back
after a breach.

Examples:

- `reference_side=below`, bar high reaches the level, close remains at or below
  the level without a clear breach-and-return.
- `reference_side=above`, bar low reaches the level, close remains at or above
  the level without a clear breach-and-return.

### `wick_fail`

The bar breaches beyond the level intrabar from the `reference_side` and closes
 back on the `reference_side`.

This is the v1 mechanical form of a failed breakout / failed sweep.

### `close_through`

The first interaction bar closes on the opposite side of the level after
touching or breaching it.

This is the v1 mechanical form of a pass / break-through.

## Sweep flag

`swept = True` when the interaction bar breaches beyond the level by at least
`sweep_epsilon` in the direction away from the `reference_side`.

This is a flag on the interaction bar, not a separate event class.

Examples:

- `reference_side=below`: `bar.high - level >= sweep_epsilon`
- `reference_side=above`: `level - bar.low >= sweep_epsilon`

No volume, imbalance, or candle-story language is part of the definition.

## Reclaim semantics

`reclaim` is defined only after a swept `close_through`.

Within `reclaim_lookahead_bars`, a later bar must close back on the original
`reference_side`.

Same-bar close-back is **not** reclaim. That is `wick_fail`.

## Fail-closed behavior

The layer must return an unavailable reason instead of guessing when:

- the level name is unsupported
- the level is not safe for the target session
- the level value is missing
- required bar columns are missing
- the bar frame is empty
- `reference_side` is invalid

## Intended use

Use this layer to build small, falsifiable research studies such as:

- pass vs fail at `prev_day_high`
- sweep/reclaim at `prev_day_low`
- session-conditioned response at `pivot`

Do **not** use it to justify broad family scans until:

- scope is pre-registered
- honest K is locked
- chronology is audited

## Implementation surface (v1)

- `research/lib/level_interactions.py`
- `tests/test_research/test_level_interactions.py`

Promotion to canonical pipeline features is explicitly deferred until this layer
proves useful under pre-registered research.

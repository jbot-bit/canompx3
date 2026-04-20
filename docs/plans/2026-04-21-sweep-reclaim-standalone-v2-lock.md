# Sweep Reclaim Standalone V2 — Lock Decision

**Date:** 2026-04-21  
**Status:** LOCKED_SCOPE  
**Purpose:** reduce the earlier `v2-starter` into one honest standalone family
that can actually be run without bloating K or importing ungrounded ontology.

## Grounding

Local authority only:

- `docs/prompts/LIQUIDITY_DISPLACEMENT_TRANSLATOR.md`
- `docs/specs/level_interaction_v1.md`
- `docs/institutional/mechanism_priors.md`
- `docs/institutional/pre_registered_criteria.md`
- `research/lib/level_interactions.py`
- `pipeline/session_guard.py`
- `docs/audit/results/2026-04-19-sweep-reclaim-v1.md`

Canonical DB checks from `gold.db`:

- `daily_features` exists with `symbol`, `trading_day`, `orb_minutes`,
  `prev_day_high`, `prev_day_low`
- `bars_1m` exists with `symbol`, `ts_utc`, `open`, `high`, `low`, `close`
- `orb_outcomes` exists and confirms pre-2026 `MNQ` / `MES` coverage for
  `EUROPE_FLOW` and `NYSE_OPEN` at `orb_minutes=5`

## What changed from the starter

The starter was deliberately an upper-bound draft. It left too much open:

- two entry modes
- two target modes
- four levels
- explicit long/short axis

That was not yet a real family. It was a scope envelope.

V2 lock reduces that to:

- instruments: `MNQ`, `MES`
- sessions: `EUROPE_FLOW`, `NYSE_OPEN`
- levels: `prev_day_high`, `prev_day_low`
- role: standalone reversal
- entry: `reclaim_close`
- stop: beyond sweep extreme
- target: fixed `1.5R`
- direction implied by level side

## Why these reductions are the right ones

### 1. Drop overnight levels

`overnight_high/low` are chronology-safe for `EUROPE_FLOW` and `NYSE_OPEN`, but
they are not necessary for the first locked family.

Reason:

- `prev_day_high/low` are cleaner ex-ante anchors
- `sweep-reclaim-v1` reconnaissance was already on `PDH/PDL`
- removing overnight levels halves K immediately

This is a K-control decision, not a claim that overnight levels are invalid.

### 2. Keep one entry mode only

Choose `reclaim_close`.

Reason:

- `first_retest_of_reclaimed_level` adds extra execution ambiguity and another
  search axis
- `reclaim_close` is the cleanest direct translation of the mechanical event
- first-touch fill assumptions should not be smuggled into the first lock

### 3. Keep one target mode only

Choose fixed `1.5R`.

Reason:

- `mechanism_priors.md` explicitly marks target-at-next-liquidity as not yet
  locally grounded
- fixed RR is already native to the repo and keeps the first test honest

### 4. Direction is implied, not searched

For this family:

- reclaim below `prev_day_high` implies short reversal
- reclaim above `prev_day_low` implies long reversal

Treating direction as a separate axis would fake additional K without adding a
real design degree of freedom.

## Locked K

Final family burden:

- `2 instruments x 2 sessions x 2 levels = K 8`

MinBTL implications:

- strict `E[max_N]=1.0`: `2*ln(8)/1.0^2 = 4.16` years
- relaxed `E[max_N]=1.2`: `2*ln(8)/1.2^2 = 2.89` years

Pre-2026 clean history from `2019-05-06` to `2025-12-31` exceeds both.

## What this lock does not claim

- It does **not** claim the family is alive.
- It does **not** reopen dead ORB retest continuation work.
- It does **not** validate any FVG / IFVG / displacement ontology.
- It does **not** justify moving to confluence or allocator roles.

It only defines the next honest standalone family.

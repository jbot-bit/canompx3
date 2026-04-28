---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Active Shelf Era Hardening

**Date:** 2026-04-11
**Status:** PARTIALLY IMPLEMENTED
**Scope:** active-shelf honesty for micro-only filters

## What is now enforced

Two active-shelf drift guards now exist in `pipeline/check_drift.py`:

1. `check_active_micro_only_filters_on_real_micros()`
   - Uses canonical filter metadata:
     - `trading_app.config.ALL_FILTERS[filter_type].requires_micro_data`
     - `pipeline.data_era.is_micro(instrument)`
   - Prevents promotion/deployment of micro-only filters on parent or proxy lanes.

2. `check_active_micro_only_filters_after_micro_launch()`
   - Recomputes first traded day for active `requires_micro_data=True` rows from
     canonical computed facts:
     - `daily_features`
     - `orb_outcomes`
     - canonical filter logic in `trading_app.strategy_discovery`
     - `pipeline.data_era.micro_launch_day(instrument)`
   - Prevents active micro-only strategies from surviving if their first traded
     day predates the real micro launch.

## Why this design is honest

The strict launch-date check does **not** trust `strategy_trade_days`.

That table is documented stale for active-shelf audit purposes. Using it as the
source of truth for era discipline would create fake rigor. Recomputing from
`daily_features` + `orb_outcomes` is slower but honest and canonical.

Golden nugget:
Strictness is only worth having if it is grounded in canonical facts. A
"fail-closed" check built on stale metadata is just theater.

## What remains unresolved

### 1. Promotion-time provenance is still weak

The current active-shelf guards prove era discipline by recomputation at audit
time. They do **not** record a durable promotion-time proof object. That means:

- the shelf can be audited honestly today
- but the promotion record itself still lacks explicit date-range lineage

### 2. `validated_setups` still does not carry date-window provenance

There is no explicit:

- `first_trade_day`
- `last_trade_day`
- `dataset_snapshot_id`
- `rebuild_id`

on active validated rows.

So if the project wants promotion-time, query-free attestations later, that
requires schema work.

### 3. Full drift verification can be blocked by DB access

The new checks are covered by targeted DB tests. Full real-DB drift verification
can still be blocked by environmental file access issues on `gold.db`. That is
an operator/environment problem, not a logic gap in the new checks.

## Recommended next steps

### Low blast radius

1. Keep the current drift checks as the canonical active-shelf honesty gate.
2. Add targeted tests whenever a new `requires_micro_data=True` filter family is
   introduced.
3. Keep `requires_micro_data` ownership on the filter itself. Do not move that
   knowledge into ad hoc drift allowlists.

### Medium blast radius

1. Add promotion-time provenance fields to `validated_setups`:
   - `first_trade_day`
   - `last_trade_day`
   - `dataset_snapshot_id`
2. Populate them in `strategy_validator.py` at promotion time from canonical
   computed facts.
3. Then teach drift to cross-check stored provenance against recomputed truth.

### High blast radius

1. Introduce full dataset snapshot lineage across discovery and validation.
2. Backfill provenance for historical active rows.
3. Re-run validation for any rows whose stored lineage is missing or unverifiable.

## Decision rule

Do **not** keep adding drift heuristics to compensate for missing lineage.

If the next desired guarantee is "the shelf row itself carries immutable proof
of the era window it was validated on", that is a schema/provenance project, not
another string-matching or metadata workaround.

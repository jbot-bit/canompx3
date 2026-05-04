---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Validated Shelf Lifecycle Hardening

Date: 2026-04-11

## Problem

`validated_setups` had honest active-shelf cleanup, but deployable-shelf
semantics were still duplicated all over the repo as raw `status='active'`
predicates. That left two recurrence risks:

1. Another writer path could mutate `validated_setups` outside the canonical
   validator / maintenance surface.
2. Another downstream reader could quietly treat every active row as
   deployable even after more non-deployable row classes are introduced.

## Decision

Harden the deployable shelf in low/medium blast-radius layers instead of a
large table split:

1. Add explicit `validated_setups.deployment_scope`
   - `deployable`
   - `non_deployable`
2. Keep `status` as lifecycle state
   - `active` still means currently on the shelf
   - `retired` still means no longer on the shelf
3. Treat deployable shelf membership as:
   - `LOWER(status) = 'active'`
   - and `deployment_scope = 'deployable'`
4. Centralize that rule in `trading_app/validated_shelf.py`
5. Move critical production readers to the canonical helper
6. Add drift checks for:
   - validated-shelf writer allowlist
   - critical reader deployable-shelf semantics

## Why this shape

- Lower blast radius than splitting `validated_setups` into separate deployable
  and research tables.
- Explicit enough to stop semantic drift.
- Backward compatible with older minimal test schemas through helper fallback.
- Leaves room for future non-deployable classes without rewriting every query.

## Canonical semantics

- `validated_shelf_lifecycle(instrument)` decides promotion-time status/scope.
- Instruments in `ACTIVE_ORB_INSTRUMENTS`
  - `status='active'`
  - `deployment_scope='deployable'`
- Instruments outside `ACTIVE_ORB_INSTRUMENTS`
  - `status='retired'`
  - `deployment_scope='non_deployable'`
  - retirement reason stays explicit in-row

## Writer boundary

Allowed mutators of `validated_setups` are now explicitly treated as:

- `trading_app/strategy_validator.py`
- `trading_app/edge_families.py`
- `trading_app/db_manager.py`
- `scripts/migrations/*`
- `scripts/infra/parallel_rebuild.py`
- `scripts/infra/revalidate_null_seeds.py`
- selected maintenance/backfill tools already in use

Anything outside that set should trip drift.

## Critical reader set

The following are treated as deployable-shelf-critical and should not regress
to ad hoc `status='active'` semantics:

- `trading_app/live_config.py`
- `trading_app/prop_portfolio.py`
- `trading_app/lane_allocator.py`
- `trading_app/strategy_fitness.py`
- `trading_app/sr_monitor.py`
- `trading_app/sprt_monitor.py`
- `scripts/tools/generate_trade_sheet.py`
- `scripts/tools/project_pulse.py`
- `trading_app/ai/sql_adapter.py`

## Blast radius

Low:
- `validated_shelf.py`
- drift checks
- migration/backfill for `deployment_scope`

Medium:
- any consumer query that now uses deployable-shelf semantics instead of raw
  `status='active'`

Not done in this slice:
- separate research shelf table
- `validated_setups_archive` lifecycle enrichment
- repo-wide replacement of every historical `status='active'` query

## Verification bar used

- targeted schema / validator / drift / adapter tests
- downstream reader tests (`live_config`, `prop_portfolio`, trade sheet, SR monitor)
- `audit_integrity.py`
- `audit_behavioral.py`
- `pipeline/check_drift.py`

## Next steps

1. Audit non-critical readers that still query `validated_setups` directly and
   decide whether they should use deployable-shelf semantics or raw lifecycle
   semantics.
2. Decide whether `validated_setups_archive` should also carry
   `deployment_scope`.
3. If research-only validated rows expand materially, revisit a dedicated
   research shelf instead of overloading one table further.

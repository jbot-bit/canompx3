# PR48 MGC shadow-only overlay contract

**Status:** implemented Phase 1
**Date:** 2026-04-23  
**Purpose:** implement the smallest honest carrier for frozen `MGC:cont_exec` as a `shadow_only` profile-local conditional overlay.

## Why this stage exists

The prior bounded translation stage closed `REDESIGN`:

- `MGC:cont_exec` is alive
- the current runtime has no honest carrier for it
- the existing execution-time `size_multiplier` path is not valid for this branch

The design audit chose the smallest next move:

- one checked-in overlay spec
- one daily derived-state shadow envelope
- read-only operator/runtime visibility
- no execution sizing

## Inputs

- `docs/audit/results/2026-04-23-pr48-mgc-cont-exec-bounded-translation.md`
- `docs/audit/results/2026-04-23-pr48-mgc-shadow-only-overlay-design.md`
- `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
- `docs/audit/hypotheses/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.yaml`
- `research/output/pr48_mes_mgc_sizer_rule_breakpoints_v1.csv`
- `trading_app/derived_state.py`
- `trading_app/pre_session_check.py`
- `trading_app/live/bot_dashboard.py`

## Exact question

Can the repo express `MGC:cont_exec` as a **shadow-only**, profile-local,
fail-closed conditional overlay object without:

- touching live execution size
- mutating `validated_setups`
- abusing `paper_trades`
- or pretending the overlay is a standalone lane?

## Required outputs

1. Static overlay registry surface
   - one canonical spec object
   - one owning profile
   - frozen artifact references only

2. Daily derived-state envelope
   - validated with current profile/DB/code fingerprints
   - explicit `ready` / `unscored` / `invalid` states

3. Read-only runtime/operator integration
   - pre-session visibility
   - dashboard visibility
   - no execution mutation

4. Verification
   - drift clean
   - fail-closed behavior demonstrated
   - no execution-path behavior change

## Implemented surface

- `trading_app/conditional_overlays.py` owns the static `pr48_mgc_cont_exec_v1` shadow-only overlay spec and derived-state helpers.
- `trading_app/lifecycle_state.py` exposes conditional overlay state through the shared lifecycle reader.
- `trading_app/pre_session_check.py` reports overlay readiness / invalidity as a non-blocking operator check.
- `trading_app/live/bot_dashboard.py` exposes overlay state in the operator payload and dashboard checks.
- Tests cover auto-refresh, invalid-artifact degradation, non-finite feature handling, lifecycle propagation, pre-session messaging, and dashboard visibility.

## Rules

- Do not write to `validated_setups`
- Do not modify `lane_allocator.py`
- Do not modify execution sizing
- Do not write fake shadow trades to `paper_trades`
- Do not retune frozen breakpoints or the size map
- Preserve `2026-01-01` holdout freeze

## Blast radius ceiling

Allowed:

- one new overlay-spec module or adjacent registry surface
- one builder / loader for the derived-state envelope
- `pre_session_check.py`
- `live/bot_dashboard.py`

Not allowed in this stage:

- `execution_engine.py`
- `risk_manager.py`
- DuckDB schema changes
- live routing / allocator changes

## Final verdict options

- `IMPLEMENT` — selected for Phase 1
- `PARK`
- `REDESIGN`

## Non-goals

- no fresh discovery
- no live sizing
- no broad conditional-role framework rebuild
- no MES / DUO / MNQ reopening in the same step

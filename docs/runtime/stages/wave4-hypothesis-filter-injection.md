---
mode: IMPLEMENTATION
task: Hypothesis-mode filter injection in discovery grid
classification: CORE_MODIFICATION
created: 2026-04-11T01:45:00+10:00
updated: 2026-04-11T01:45:00+10:00
---

# Stage: Hypothesis-Mode Filter Injection

## Purpose
Allow pre-registered hypothesis files to test filter types that aren't in the legacy discovery grid. Currently, GAP_R015 and DOW composites (ORB_G5_NOFRI etc.) are in ALL_FILTERS but NOT in get_filters_for_grid() for MNQ/MES at most sessions. The 4 remaining wave 4 hypothesis files (mnq-gap, mes-gap, mnq-dow, mes-dow) cannot run because the grid never generates their filter types.

The fix: when running in Phase 4 hypothesis mode (scope_predicate set), inject the declared filter types from ALL_FILTERS into the discovery grid for that run only. Legacy mode grid is unchanged.

## Scope Lock
- trading_app/strategy_discovery.py
- tests/test_trading_app/test_strategy_discovery.py

## Blast Radius
Discovery: enumeration loop gains hypothesis-declared filters in hypothesis mode. Legacy mode unaffected (no scope_predicate). Validator: no change. Config: no change. Drift check: no change. Filter day set pre-computation must include injected filters — update `all_grid_filters` before `_build_filter_day_sets`. DOW alignment safety: skip injected DOW composites for NYSE_OPEN (DOW_MISALIGNED_SESSIONS guard).

## Approach
1. Import ALL_FILTERS, CompositeFilter, DayOfWeekSkipFilter from trading_app.config and DOW_MISALIGNED_SESSIONS from pipeline.dst
2. After building all_grid_filters from get_filters_for_grid(), if scope_predicate is set:
   - For each declared filter type not in all_grid_filters but in ALL_FILTERS:
     - Build per-session injection map (skip DOW-misaligned sessions for DOW composites)
     - Add to all_grid_filters for pre-computation
3. In per-session loop, merge injection map into session_filters
4. Log the injected filters for audit trail

## Acceptance Criteria
1. Discovery with `mnq-gap.yaml` produces >0 combos accepted (was 0 before)
2. Discovery with `mnq-dow.yaml` produces >0 combos for NYSE_OPEN-excluded sessions
3. Legacy mode discovery (no --hypothesis-file) produces SAME combo count as before
4. Test: `test_hypothesis_filter_injection_expands_grid` passes
5. Test: `test_hypothesis_filter_injection_respects_dow_misalignment` passes
6. No changes to validated_setups from prior runs
7. Drift check passes

## Kill Criteria
- If injection causes legacy mode to produce different combo counts → BUG, revert
- If a hypothesis file with invalid filter type causes crash (not clean error) → BUG, fix

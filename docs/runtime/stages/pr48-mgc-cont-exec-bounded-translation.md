# PR48 MGC Continuous-Exec Bounded Translation

**Status:** active bounded stage  
**Date:** 2026-04-23  
**Purpose:** translate the single strongest recovered PR48 arm, `MGC:cont_exec`, into an honest repo object without pretending it is a standalone validated lane.

## Why this stage exists

The PR48 translation audit closed the role mapping:

- `MGC:cont_exec` is the only arm stamped `READY_FOR_BOUNDED_TRANSLATION`
- `MES:q45_exec` still needs a larger bridge
- `DUO` and `MNQ:shadow_addon` stay shadow-only

This stage isolates the smallest safe next move.

## Inputs

- `docs/audit/results/2026-04-23-pr48-conditional-role-validation-translation.md`
- `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
- `docs/institutional/conditional-edge-framework.md`
- `trading_app/prop_profiles.py`
- `trading_app/execution_engine.py`
- `trading_app/risk_manager.py`

## Exact question

What is the smallest honest bridge that can carry the frozen `MGC:cont_exec` sizing map into a profile-local, non-standalone runtime surface?

## Rules

- Do not promote `MGC:cont_exec` into `validated_setups`.
- Do not route it through `lane_allocator.py` as if it were a standalone lane.
- Do not widen scope to MES, DUO, or MNQ in the same step.
- Do not retune breakpoints or the size map.
- Preserve the sacred `2026-01-01` split.

## Required outputs

1. Exact target surface:
   - report-only profile note
   - shadow runtime object
   - or one new profile-local conditional config surface
2. Exact blast radius:
   - files
   - schemas
   - runtime readers
3. Exact fail-closed behavior:
   - what happens if the conditional surface is missing or invalid
4. Final verdict:
   - `IMPLEMENT`
   - `PARK`
   - or `REDESIGN`

## Non-goals

- No fresh discovery
- No broad conditional schema overhaul
- No live deployment recommendation from this stage alone

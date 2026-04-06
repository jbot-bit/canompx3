---
mode: IMPLEMENTATION
slug: eligibility-phase-0-1
task: Build eligibility context foundation (types + decomposition registry + builder + tests)
created: 2026-04-07
updated: 2026-04-07
stage: 1
of: 3
scope_lock:
  - trading_app/eligibility/__init__.py
  - trading_app/eligibility/types.py
  - trading_app/eligibility/decomposition.py
  - trading_app/eligibility/builder.py
  - tests/test_trading_app/test_eligibility_types.py
  - tests/test_trading_app/test_eligibility_decomposition.py
  - tests/test_trading_app/test_eligibility_builder.py
blast_radius: New trading_app/eligibility/ package with four modules (types, decomposition, builder, __init__) plus three test files. Zero production code modified. Read-only imports from pipeline.dst, pipeline.paths, pipeline.cost_model, trading_app.config (ALL_FILTERS + get_filters_for_grid), trading_app.calendar_overlay. No reverse imports until Phase 2 trade sheet integration. No schema changes. No pipeline rebuild. Fixture-based tests; no live DB dependency.
---

# Stage: Eligibility Context Foundation (Phase 0 + Phase 1)

## Purpose

Build the foundation for operational filter visibility: immutable data types, decomposition registry, and eligibility builder. Produces an `EligibilityReport` for any deployed lane or validated strategy, surfacing explicit statuses instead of silent defaults.

Nine explicit `ConditionStatus` values replace v1's PASS/FAIL/PENDING trichotomy:
PASS, FAIL, PENDING, DATA_MISSING, NOT_APPLICABLE_INSTRUMENT, NOT_APPLICABLE_SESSION, NOT_APPLICABLE_DIRECTION, RULES_NOT_LOADED, STALE_VALIDATION.

This stage adds NEW files only. Zero production code modifications. Next stages (Phase 2 Trade Sheet, Phase 3 Dashboard) will integrate it.

## Scope Rationale

The scope is deliberately constrained to NEW files only. No modifications to:
- trading_app/config.py
- trading_app/live/ (bot_state, session_orchestrator, bot_dashboard)
- trading_app/execution_engine.py
- scripts/tools/generate_trade_sheet.py
- pipeline/* (including init_db.py, check_drift.py)

This isolation means the foundation can be tested independently and reviewed before any consumer touches it. If the design is wrong, reverting is trivial (delete the directory).

## Decomposition Coverage (initial)

Registry covers filter types used by currently-deployed lanes plus their session overlays:

- NO_FILTER (no atoms — always PASS)
- ORB_G2/G3/G4/G5/G6/G8 (INTRA_SESSION, ORB size threshold)
- COST_LT08/10/12/15 (INTRA_SESSION, cost ratio threshold)
- OVNRNG_10/25/50/100 (PRE_SESSION for US/EU sessions only)
- PIT_MIN (PRE_SESSION, CME_REOPEN only)
- GAP_R005/R015 (PRE_SESSION, MGC CME_REOPEN only)
- PDR_R080/R105/R125 (PRE_SESSION, validated sessions only)
- X_MES_ATR60/70, X_MGC_ATR70 (PRE_SESSION, MNQ US sessions only)
- ATR_P30/P50/P70 (PRE_SESSION)
- ATR70_VOL (hybrid: PRE for ATR component, INTRA for rel_vol)
- FAST5/FAST10 (INTRA_SESSION, per-instrument validity)
- CONT (INTRA_SESSION, E1-only due to E2 look-ahead)
- DIR_LONG/DIR_SHORT (DIRECTIONAL)
- NOMON (PRE_SESSION, DOW)
- Composites decomposed atomically (e.g., ORB_G5_FAST5_CONT → 3 atoms)

Overlays: Calendar (PASS/FAIL/HALF/RULES_NOT_LOADED), ATR velocity (PASS/FAIL/DATA_MISSING/NOT_APPLICABLE_INSTRUMENT), Bull-day short avoidance (DIRECTIONAL, deferred until NYSE_OPEN lanes exist).

## Acceptance Criteria

1. File existence: All 7 files in scope_lock exist on disk.
2. Import cleanliness: `python -c "from trading_app.eligibility import build_eligibility_report"` runs without error.
3. Test suite passes: `PYTHONPATH=. python -m pytest tests/test_trading_app/test_eligibility_*.py -v` → all tests pass.
4. Status coverage: Tests exist for EACH of the nine ConditionStatus values.
5. Composite decomposition: Test proves ORB_G5_FAST5_CONT decomposes into 3+ atomic conditions.
6. Per-instrument validity: Test proves FAST5 on MGC returns NOT_APPLICABLE_INSTRUMENT.
7. DATA_MISSING explicit: Test proves NULL pit_range_atr returns DATA_MISSING for PIT_MIN atom, NOT FAIL.
8. STALE freshness: Test proves as_of > 1 day old returns freshness_status=STALE.
9. RULES_NOT_LOADED: Test proves missing calendar rules returns RULES_NOT_LOADED for calendar atom.
10. Drift check passes: `PYTHONPATH=. python pipeline/check_drift.py` → 0 failures.
11. No production code modified: `git diff HEAD -- trading_app/config.py trading_app/live/ trading_app/execution_engine.py scripts/tools/ pipeline/` → empty.

## Out of Scope

- Phase 2: Trade sheet integration (next stage)
- Phase 3: Dashboard integration (stage 3)
- Phase 4: Filter routing expansion (deferred)
- Drift check enforcing 1:1 ALL_FILTERS ↔ decomposition mapping (Phase 2)
- Literature extraction from local PDFs (optional pre-implementation hardening)

## Commit Message Template

```
feat: eligibility context foundation — types, decomposition, builder (Phase 0+1)

Adds trading_app/eligibility/ with:
- ConditionRecord + EligibilityReport immutable data types (9 explicit statuses)
- Decomposition registry for deployed-lane filters and overlays
- build_eligibility_report() with explicit DATA_MISSING / STALE / NOT_APPLICABLE handling
- Fixture-based tests covering all status enum values

No production code modified. Foundation for Phase 2 (trade sheet) and Phase 3 (dashboard).

Design: docs/plans/2026-04-07-eligibility-context-design.md
```

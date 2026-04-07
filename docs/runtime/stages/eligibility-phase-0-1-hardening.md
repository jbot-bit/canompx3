---
mode: IMPLEMENTATION
slug: eligibility-phase-0-1-hardening
task: Fix review findings from commit 046e80b (HALF_SIZE mapping, TypeError crashes, label lies, dead code)
created: 2026-04-07
updated: 2026-04-07
stage: 1
of: 1
scope_lock:
  - trading_app/eligibility/types.py
  - trading_app/eligibility/builder.py
  - trading_app/eligibility/decomposition.py
  - tests/test_trading_app/test_eligibility_types.py
  - tests/test_trading_app/test_eligibility_builder.py
  - tests/test_trading_app/test_eligibility_decomposition.py
blast_radius: Modifies the eligibility-package files from commit 046e80b. No consumers yet (Phase 2/3 haven't started), so caller discipline is not a concern. Adds ConditionStatus.NOT_APPLICABLE_ENTRY_MODEL, removes NOT_APPLICABLE_SESSION (dead enum), adds size_multiplier field to ConditionRecord for HALF_SIZE handling, adds TypeError guards to derived-value resolvers, wires build_errors to capture silenced exceptions, and delegates ATR velocity logic to canonical ATRVelocityFilter.matches_row() to eliminate re-encoded filter semantics.
---

# Stage: Eligibility Foundation Hardening

## Purpose

Fix the six review findings from the self-code-review of commit 046e80b:

1. **HALF_SIZE silently becomes INELIGIBLE** — calendar HALF_SIZE is a sizing action, not a block. Need to separate "does this condition gate the trade" from "does this condition modify sizing."
2. **Hard crash on bad feature row types** — `_resolve_pdr` / `_resolve_gap` do arithmetic on raw dict values without type guards. A string in `prev_day_range` crashes the whole report.
3. **Silent TypeError in `_compare`** — returns False on type mismatch, which becomes FAIL, which becomes INELIGIBLE. Schema drift looks identical to legitimate trading signal.
4. **`NOT_APPLICABLE_INSTRUMENT` label lie** — used for CONT+E2 case with comment "close-enough semantically." The real reason is entry model.
5. **Dead `NOT_APPLICABLE_SESSION` enum value** — defined in types.py but no code path produces it.
6. **Dead `build_errors` field** — initialized and returned but never appended to.

## Scope Rationale

Same four files from Phase 0+1 foundation. No Phase 2/3 consumers exist yet, so these changes are purely internal — no caller discipline needed.

Deferred to a later stage (item 8 from the review):
- Deriving `validated_for` lists from `get_filters_for_grid()` iteration — requires drift check #N to enforce parity, which itself requires a test harness that iterates all (instrument, session, filter_type) combinations. Significant test infrastructure. Defer.

Included in this stage (item 7 added after "ensure proper fixes always"):
- Delegate ATR velocity logic to canonical `ATRVelocityFilter.matches_row()` — eliminates re-encoded filter semantics and ensures schema drift cannot create divergence.

## Acceptance Criteria

1. `ConditionRecord` gains a `size_multiplier: float = 1.0` field.
2. Calendar HALF_SIZE produces `status=PASS, size_multiplier=0.5` — not FAIL.
3. Test proves `overall_status == ELIGIBLE` for a strategy whose only non-NEUTRAL condition is calendar HALF_SIZE.
4. `ConditionStatus` enum adds `NOT_APPLICABLE_ENTRY_MODEL`.
5. `NOT_APPLICABLE_SESSION` removed from enum (dead value).
6. CONT+E2 case uses `NOT_APPLICABLE_ENTRY_MODEL`, no "close-enough" comment.
7. `_resolve_pdr` and `_resolve_gap` return `(None, True)` on TypeError/ValueError instead of crashing.
8. Test proves `build_eligibility_report` does not raise when `prev_day_range='bad'`.
9. `_compare()` returns `(result: bool | None, type_error: bool)` tuple. Call site maps `type_error=True` to DATA_MISSING.
10. `build_errors` field is populated when:
    - `_fetch_feature_row` catches an exception (records exception type + message)
    - Calendar lookup raises
    - `_compare` returns type_error
11. Test proves `build_errors` is non-empty when DB fetch fails (via an invalid path).
12. All 78 existing tests still pass + new tests for items 3, 8, 10, 11.
13. `PYTHONPATH=. python pipeline/check_drift.py` → 0 failures.
14. ATR velocity condition delegates to canonical `trading_app.config.ATRVelocityFilter.matches_row()` rather than re-encoding the logic. Test proves the builder's result matches the canonical filter for all permutations of (Contracting, Expanding, Stable) × (Compressed, Expanded, Neutral, unknown).

## Out of Scope

- Phase 2 trade sheet integration
- Phase 3 dashboard integration
- ATR velocity delegation to canonical filter (item 7)
- Derived validated_for lists (item 8)
- Adding drift check #N for ALL_FILTERS ↔ decomposition registry parity

## Commit Message Template

```
fix: eligibility foundation hardening — close six review findings (046e80b)

Fixes all action items from self-code-review of Phase 0+1:
- Calendar HALF_SIZE no longer misclassified as FAIL (adds size_multiplier field)
- _resolve_pdr / _resolve_gap now return DATA_MISSING on bad data types
  (was: hard TypeError crash)
- _compare() distinguishes type errors from legitimate FAIL
- CONT+E2 case uses new NOT_APPLICABLE_ENTRY_MODEL status (was: lying
  NOT_APPLICABLE_INSTRUMENT label)
- Removes dead NOT_APPLICABLE_SESSION enum value
- Wires build_errors field to capture silenced exceptions

All 78 original tests still pass. Adds 6 new tests covering the fixed paths.
Design: docs/plans/2026-04-07-eligibility-context-design.md
Review: self-code-review C grade → targets B+ with these fixes.
```

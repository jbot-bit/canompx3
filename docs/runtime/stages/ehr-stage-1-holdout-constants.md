---
task: "EHR PASS 2 Stage 1 — add EARLY_HOLDOUT_BOUNDARY constant + enforce_early_holdout_date helper + is_ehr_mode predicate (NO behavior change to Mode A)"
mode: IMPLEMENTATION
slug: ehr-stage-1-holdout-constants
scope_lock:
  - trading_app/holdout_policy.py
  - tests/test_trading_app/test_holdout_policy_ehr.py
  - docs/runtime/stages/ehr-stage-1-holdout-constants.md
---

## Plan Reference

`EARLY_HOLDOUT_REDISCOVERY — Narrow Guarded PASS 2 Plan` (2026-05-17). This stage = Stage 1 of 8.
Plan invariants 1, 2, 3, 4, 5, 6 apply (see top-of-plan).

## Purpose (What and Why)

Add an additive, behavior-neutral set of EHR constants to the canonical holdout module so subsequent stages (validator hard-fail, allocator assertion, drift checks) have a single source of truth to import from. Mode A's `HOLDOUT_SACRED_FROM = date(2026, 1, 1)` and `enforce_holdout_date()` are byte-unchanged. The new EHR symbols are not imported by any production caller in this stage — they are unused after Stage 1 lands and become live only when Stages 3-6 consume them.

This staging order is mandatory: assertions in later stages reference symbols that must exist first. Landing constants in their own commit means a revert of Stages 3+ (the RED-path scenario) is a clean revert of consumers without leaving the canonical module in a degraded state.

## Blast Radius

- **Edits:** `trading_app/holdout_policy.py` — additive. New symbols: `EARLY_HOLDOUT_BOUNDARY: date`, `is_ehr_mode(mode: str) -> bool`, `enforce_early_holdout_date(holdout_date: date | None) -> date`. Existing `HOLDOUT_SACRED_FROM`, `HOLDOUT_GRANDFATHER_CUTOFF`, `PHASE_4_1_SHIP_DATE`, `HOLDOUT_OVERRIDE_TOKEN`, `enforce_holdout_date` byte-unchanged. Module docstring gains one paragraph stating "EHR boundary is a separate probe-mode constant; never aliases Mode A."
- **New file:** `tests/test_trading_app/test_holdout_policy_ehr.py` — three tests:
  1. `test_ehr_boundary_constant` asserts `EARLY_HOLDOUT_BOUNDARY == date(2025, 1, 1)`.
  2. `test_mode_a_sacred_unchanged` asserts `HOLDOUT_SACRED_FROM == date(2026, 1, 1)` (regression guard — Stage 1 must never alter Mode A).
  3. `test_enforce_early_holdout_rejects_post_2025` asserts `enforce_early_holdout_date(date(2025, 6, 1))` raises `ValueError` citing EHR; `enforce_early_holdout_date(date(2024, 12, 31))` returns the date unchanged; `enforce_early_holdout_date(None)` returns `EARLY_HOLDOUT_BOUNDARY`.
  4. `test_is_ehr_mode_predicate` asserts `is_ehr_mode("EARLY_HOLDOUT_REDISCOVERY") is True`, `is_ehr_mode("STANDARD") is False`, `is_ehr_mode(None) is False`.
- **Callers affected:** zero. New symbols not yet imported by `pipeline/check_drift.py`, `trading_app/strategy_discovery.py`, `trading_app/strategy_validator.py`, `trading_app/lane_allocator.py`, or any research script. They become live in Stages 3-6.
- **Reads:** module reads only `datetime.date` from stdlib and `pipeline.log.get_logger`. No DB writes, no schema change, no allocator touch.
- **Reversibility:** Stage 1 is fully revertible by `git revert <commit>`. No persistent state is created.
- **CI surface:** `pytest tests/test_trading_app/test_holdout_policy_ehr.py` (new ~30-line file). `pipeline/check_drift.py` must continue to pass — it imports `HOLDOUT_SACRED_FROM` (unchanged) and `HOLDOUT_GRANDFATHER_CUTOFF` (unchanged).

## Acceptance

1. `EARLY_HOLDOUT_BOUNDARY == date(2025, 1, 1)` exported from `trading_app.holdout_policy`.
2. `HOLDOUT_SACRED_FROM` byte-identical to the pre-stage value (`date(2026, 1, 1)`). Grep confirmation: `git diff HEAD -- trading_app/holdout_policy.py | grep HOLDOUT_SACRED_FROM` returns no removals.
3. `enforce_early_holdout_date()` rejects dates `>= EARLY_HOLDOUT_BOUNDARY` with `ValueError` whose message cites EHR mode and points to plan invariants.
4. `is_ehr_mode()` returns `True` only for the exact literal `"EARLY_HOLDOUT_REDISCOVERY"`; all other inputs (including `None`, `""`, lower-case variants) return `False`.
5. `pytest tests/test_trading_app/test_holdout_policy_ehr.py -v` — 4 tests pass.
6. `python pipeline/check_drift.py` passes with count self-reported (no new drift checks in this stage; existing checks must remain green).
7. Module docstring includes one paragraph stating the EHR boundary is a SEPARATE constant that never aliases or modifies Mode A.
8. `__all__` extended with the 3 new public names.
9. Self-review confirms: no in-production caller imports the new symbols in this stage (verified by `git grep "EARLY_HOLDOUT_BOUNDARY\|enforce_early_holdout_date\|is_ehr_mode"` showing only `holdout_policy.py` + the new test file).

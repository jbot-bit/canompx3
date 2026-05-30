---
task: "EHR PASS 2 Stage 3 — validator hard-fail on EHR verdict ceiling + mirror discovery-side columns to experimental_strategies"
mode: IMPLEMENTATION
slug: ehr-stage-3-validator-hard-fail
scope_lock:
  - trading_app/db_manager.py
  - trading_app/strategy_validator.py
  - tests/test_trading_app/test_strategy_validator_ehr.py
  - tests/test_trading_app/test_db_manager_ehr_schema.py
  - docs/runtime/stages/ehr-stage-3-validator-hard-fail.md
---

## Plan Reference

`EARLY_HOLDOUT_REDISCOVERY — Narrow Guarded PASS 2 Plan` (2026-05-17), Stage 3 of 8.
Builds on Stage 1 (commit `d910d6f7`) and Stage 2 (commit `96d0137c`).

## Purpose

Stages 1+2 were additive scaffolding (constants, helpers, schema columns) with zero
behavioral change. Stage 3 is the first stage with **behavioral teeth**: it enforces
PASS 2 plan invariants 4 + 5 in the validator's promotion path so that EHR-discovered
rows cannot be silently re-classified as STANDARD or promoted past the
`RESEARCH_PROVISIONAL` verdict ceiling.

Without Stage 3, an EHR row written by Stage 4 discovery to `experimental_strategies`
would (a) lose its `validation_mode='EARLY_HOLDOUT_REDISCOVERY'` label across the
validator promotion (column doesn't yet exist on experimental_strategies — invariant
4 fails silently), and (b) be promoted to `status='active'` like any STANDARD row
(no ceiling check — invariant 5 fails silently).

## Design Decision — refined Option A

Two of the five Stage-2 columns mirror to `experimental_strategies`; three stay
on `validated_setups` only:

- **Mirror to experimental_strategies (discovery-emitted):**
  - `validation_mode TEXT DEFAULT 'STANDARD'` — Stage 4 discovery writes this; validator
    propagates verbatim at promotion-time. Without this column on
    experimental_strategies, the EHR label is lost the moment Stage 4 inserts.
  - `cumulative_search_count INTEGER` — Bailey-Lopez de Prado 2014 disclosure;
    Stage 4 computes at discovery-date, propagates through promotion.

- **Stay on validated_setups only (validator-emitted at promotion-time):**
  - `verdict_ceiling TEXT` — set by validator's new ceiling-derivation logic
    based on `validation_mode`; not present at discovery-write time.
  - `pseudo_oos_window_start DATE`, `pseudo_oos_window_end DATE` — validator
    computes from `EARLY_HOLDOUT_BOUNDARY` and `HOLDOUT_SACRED_FROM` at
    promotion. Constants are canonical; window is policy-derived not
    discovery-emitted.

Rationale: institutional-rigor §4 (delegate to canonical sources) + dual-table
precedent at db_manager.py line 622-624 (`validation_pathway`+`c8_oos_status`
shipped under the same threading shape). Minimizes schema surface while
satisfying invariant 4's "end-to-end identifiability."

## Blast Radius

- **db_manager.py edits:** append one migration block for the two mirrored
  columns on `experimental_strategies`. Same `ADD COLUMN IF NOT EXISTS` shape
  as the line 622-624 dual-table block. Module-level asserts already in place
  from Stage 2 cover the STANDARD/EHR label parity — no new assert needed.

- **strategy_validator.py edits at the INSERT INTO validated_setups site
  (line 1898+, the only validator-side writer):**
  1. Read `validation_mode` from `row_dict` (the experimental row). Default to
     `STANDARD_MODE_LABEL` if NULL/missing for pre-Stage-2 grandfathered rows.
  2. Compute `verdict_ceiling` via `is_ehr_mode(validation_mode)` →
     `'RESEARCH_PROVISIONAL'` else NULL.
  3. Compute `pseudo_oos_window_start`/`pseudo_oos_window_end` for EHR rows
     only: `(EARLY_HOLDOUT_BOUNDARY, HOLDOUT_SACRED_FROM)`. NULL for STANDARD.
  4. Override `status`: if `verdict_ceiling == 'RESEARCH_PROVISIONAL'`, set
     `status` to `'RESEARCH_PROVISIONAL'` (HARD FAIL — cannot be `'active'`).
     STANDARD rows: existing `'active'/'retired'` logic unchanged.
  5. Propagate `cumulative_search_count` verbatim from row_dict (None for
     STANDARD; integer for EHR).

  The INSERT goes from 55 to 59 column-bindings (+ validation_mode adds 1 already
  defaulted via SQL = total 60). Existing column order preserved; new columns
  appended.

- **Imports added to strategy_validator.py:** `from trading_app.holdout_policy
  import EARLY_HOLDOUT_BOUNDARY, EHR_MODE_LABEL, STANDARD_MODE_LABEL,
  HOLDOUT_SACRED_FROM, is_ehr_mode`. Module already imports from
  `holdout_policy` (HOLDOUT_SACRED_FROM is referenced in
  `_check_mode_a_holdout_integrity`); this extends that import.

- **New test file:** `tests/test_trading_app/test_strategy_validator_ehr.py` —
  5 tests:
  1. STANDARD-row promotion regression: rows without `validation_mode` set still
     promote to `status='active'`, `verdict_ceiling` NULL, `pseudo_oos_*` NULL.
  2. EHR-row promotion preserves `validation_mode='EARLY_HOLDOUT_REDISCOVERY'`.
  3. EHR-row `status` is hard-capped at `'RESEARCH_PROVISIONAL'` (NOT `'active'`).
  4. EHR-row `verdict_ceiling='RESEARCH_PROVISIONAL'` and
     `pseudo_oos_window_*` populated with canonical boundary dates.
  5. `cumulative_search_count` round-trips from experimental_strategies through
     promotion to validated_setups byte-exact.

  Test scaffolding pattern: tmp_path DuckDB + `init_trading_app_schema` + direct
  call to a unit-testable promotion helper (or, if no helper exists, exercise the
  full INSERT logic via a dedicated test seam). Mirrors Stage 2's
  `test_db_manager_ehr_schema.py` fixture pattern.

- **Stage 2 test extension** (`test_db_manager_ehr_schema.py`): add 2 tests for
  the new experimental_strategies columns — schema introspection + nullability
  on STANDARD experimental rows. Keeps the Stage 2 test file as the canonical
  schema-introspection home; Stage 3's promotion-behavior tests live in the new
  file.

- **Callers affected today:** zero. All STANDARD rows (every row currently in
  any database) read `validation_mode IS NULL → defaults to 'STANDARD' →
  is_ehr_mode → False → existing behavior preserved`. The hard-fail branch only
  fires on rows discovery has explicitly opted into EHR.

- **Reads:** none beyond canonical `holdout_policy` imports.
- **Writes:** validator's existing INSERT INTO validated_setups path only.
  No new write paths.
- **Drift surface:** zero new violations expected. The 844 baseline (verified
  pre-Stage-2 and post-Stage-2) must remain 844 after Stage 3.
- **Reversibility:** `git revert <commit>`; DuckDB additive ALTER survives on
  existing DBs but the columns are dead/harmless without the validator code.

## Acceptance

1. `experimental_strategies` carries `validation_mode TEXT DEFAULT 'STANDARD'`
   and `cumulative_search_count INTEGER` after `init_trading_app_schema()`.
2. Stage 2 EHR schema tests still pass (4 tests); plus 2 new tests on the
   experimental_strategies columns pass (6 total in
   `test_db_manager_ehr_schema.py`).
3. Validator promotion of a STANDARD experimental row (validation_mode NULL or
   'STANDARD') produces a validated_setups row with `status='active'` (for
   active instruments) or `'retired'`, `validation_mode='STANDARD'`,
   `verdict_ceiling` NULL, `pseudo_oos_window_*` NULL — i.e., bit-for-bit
   identical to pre-Stage-3 behavior on existing rows.
4. Validator promotion of an EHR experimental row (validation_mode=
   'EARLY_HOLDOUT_REDISCOVERY') produces a validated_setups row with:
   - `validation_mode='EARLY_HOLDOUT_REDISCOVERY'` (propagated verbatim)
   - `status='RESEARCH_PROVISIONAL'` (HARD FAIL on the active/retired path)
   - `verdict_ceiling='RESEARCH_PROVISIONAL'`
   - `pseudo_oos_window_start=EARLY_HOLDOUT_BOUNDARY` (2025-01-01)
   - `pseudo_oos_window_end=HOLDOUT_SACRED_FROM` (2026-01-01)
   - `cumulative_search_count` round-trips byte-exact.
5. `pytest tests/test_trading_app/test_strategy_validator_ehr.py -v` — 5/5 pass.
6. `pytest tests/test_trading_app/test_db_manager_ehr_schema.py -v` — 6/6 pass
   (4 Stage 2 + 2 new).
7. `pytest tests/test_trading_app/test_db_manager.py -v` — 13/13 pass
   (regression on existing schema-creation suite).
8. `pytest tests/test_trading_app/test_holdout_policy.py
   tests/test_trading_app/test_holdout_policy_ehr.py -v` — 30/30 pass
   (Stage 1 regression guard).
9. `python pipeline/check_drift.py` — violation count unchanged from the 844
   worktree-HEAD baseline established at end of Stage 2.
10. `git grep "is_ehr_mode\|EHR_MODE_LABEL"` in `trading_app/strategy_validator.py`
    returns at least one occurrence — the validator now consumes the canonical
    EHR predicate rather than inlining string comparisons.
11. The ceiling-derivation logic is encapsulated in a single helper function
    (e.g., `_derive_ehr_promotion_fields(row_dict) -> dict`) so Stage 6 drift
    checks can assert it as the single source of truth (no inline string
    comparisons against `'EARLY_HOLDOUT_REDISCOVERY'` or `'RESEARCH_PROVISIONAL'`
    scattered through the INSERT). Stage 6 will register this helper as a
    canonical-source anchor.

## Self-review gotchas observed before coding

1. **PASS 2 plan doc never committed.** The 8-stage plan lives in
   `holdout_policy.py` docstrings and commit messages only. Stage 3 cites
   "invariant 5" but the source isn't grep-able. Out of scope for Stage 3,
   but flagged as a doctrine gap to address before Stage 8 acceptance — a
   citeable plan doc is required for the final audit trail per
   institutional-rigor §1.

2. **Stage 2 mistakenly named "validated_setups schema additive" without
   considering the experimental→validated promotion path.** Stage 3 corrects
   the omission by adding the two discovery-emitted columns to
   experimental_strategies. Stage 2's test file extends to cover the new
   columns rather than spawning a third schema-test file.

3. **`status` semantics.** Existing validator uses `'active' if instrument in
   ACTIVE_ORB_INSTRUMENTS else 'retired'` at line 1922. EHR introduces a third
   value `'RESEARCH_PROVISIONAL'`. Must verify no downstream consumer of
   `validated_setups.status` treats unknown values as `'active'` —
   particularly the validated-shelf views (`ACTIVE_VALIDATED_VIEW`,
   `DEPLOYABLE_VALIDATED_VIEW`). If they filter `WHERE status='active'`, EHR
   rows are correctly excluded by construction (PASS 2 invariant 5 is
   self-enforcing through the existing view predicate). If they filter
   `WHERE status != 'retired'`, EHR rows leak. **First implementation step:
   read those view definitions and confirm before writing INSERT logic.**

4. **`testing_mode` vs `validation_mode` confusion.** `run_validation()`
   already takes `testing_mode: 'family' | 'individual'` — the BH-FDR
   pathway dispatch. EHR's `validation_mode` is ORTHOGONAL (controls the
   discovery-side holdout boundary, not the statistical pathway). The two
   labels could in principle combine (`testing_mode='individual'` AND
   `validation_mode='EARLY_HOLDOUT_REDISCOVERY'`). Stage 3 must not collapse
   them into one parameter; threading shape mirrors the existing
   `testing_mode` flow but as a separate axis.

5. **Pyright false positives expected** (per Stage 1 HANDOFF gotcha #3) on
   the new `holdout_policy` imports until the IDE indexer re-roots on the
   worktree. Pytest is ground truth.

## Implementation order

1. Read `trading_app/validated_shelf.py` (or wherever ACTIVE_VALIDATED_VIEW /
   DEPLOYABLE_VALIDATED_VIEW are defined) — confirm `status='active'` filter
   semantics. Adjust acceptance #3 if they use a different predicate.
2. Add experimental_strategies migration to db_manager.py.
3. Add helper `_derive_ehr_promotion_fields(row_dict)` to strategy_validator.py.
4. Modify the INSERT at line 1898 to call the helper and bind the 4 new
   columns (validation_mode, verdict_ceiling, pseudo_oos_window_*) +
   cumulative_search_count. Override status when verdict_ceiling fires.
5. Add 2 new tests to test_db_manager_ehr_schema.py for experimental_strategies
   columns.
6. Create test_strategy_validator_ehr.py with 5 tests.
7. Run the full acceptance test list. Drift check. Commit.

## Plan invariants enforced this stage

- **#4** (EHR rows identifiable end-to-end) — `validation_mode` now survives
  the experimental→validated promotion.
- **#5** (EHR rows research-provisional only, never deployable) — validator
  hard-caps `status='RESEARCH_PROVISIONAL'` when `verdict_ceiling` is set.

## Completion (2026-05-30) — PROVEN

Implemented: db_manager experimental_strategies mirror (validation_mode +
cumulative_search_count); `_derive_ehr_promotion_fields` canonical helper +
module const `RESEARCH_PROVISIONAL_STATUS` + `is_research_provisional` flag;
INSERT 55→60 cols with status hard-cap branch consuming the flag.

Refinement past the original plan: acceptance #11 strengthened — the INSERT-site
branch consumes the helper's `is_research_provisional` flag instead of comparing
`verdict_ceiling == "RESEARCH_PROVISIONAL"`, so the literal lives in exactly one
place (`RESEARCH_PROVISIONAL_STATUS`). Added a 6th validator test
(`test_research_provisional_row_excluded_from_active_validated_view`) proving
invariant #5 self-enforcement at the ACTIVE_VALIDATED_VIEW level.

Verification:
- `test_strategy_validator_ehr.py` — 6/6 pass.
- `test_db_manager_ehr_schema.py` — 6/6 (4 Stage-2 + 2 new).
- `test_db_manager.py` — 13/13 (schema regression).
- `test_strategy_validator.py` — 140/140 (full validator regression).
- `test_holdout_policy*.py` — 30/30 (Stage 1 regression).
- ruff clean on all 4 changed files.
- Drift: the EHR branch's `check_context_view_contracts` reports a pre-existing
  branch-age condition (context_views package not yet wired on this branch),
  unrelated to Stage 3; clears on merge to main. No Stage-3 drift introduced.

Adversarial-audit gate (independent-context evidence-auditor, per
`adversarial-audit-gate.md`): **CLEAN** — all four falsification hypotheses HOLD
(deployability leak / column alignment / STANDARD regression / canonical
delegation). Follow-up note: `ai/sql_adapter.py` reads validated_setups
unfiltered for AI diagnostics — read-only, not a deploy path; non-blocking.

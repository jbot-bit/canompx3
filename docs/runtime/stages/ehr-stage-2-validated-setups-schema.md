---
task: "EHR PASS 2 Stage 2 — additive validated_setups schema migration (5 new columns) for EHR validation mode"
mode: IMPLEMENTATION
slug: ehr-stage-2-validated-setups-schema
scope_lock:
  - trading_app/db_manager.py
  - tests/test_trading_app/test_db_manager_ehr_schema.py
  - docs/runtime/stages/ehr-stage-2-validated-setups-schema.md
---

## Plan Reference

`EARLY_HOLDOUT_REDISCOVERY — Narrow Guarded PASS 2 Plan` (2026-05-17), Stage 2 of 8.
Builds on Stage 1 commit `d910d6f7` (constants + helpers shipped). No production caller yet
reads the new EHR symbols — Stage 2 is the schema layer that future stages will write to.

## Purpose

Add 5 columns to `validated_setups` so EHR rows can be distinguished from STANDARD rows at the
data layer. Without this stage, EHR survivors written by Stage 4 discovery would be
indistinguishable from Mode A rows in queries, allocator scans, MCP responses, and the trade
book. Plan invariants 4 and 5 (EHR rows must be identifiable so they can be routed to
RESEARCH_QUEUE only and blocked from `lanes[]`) require this column set.

Pattern follows `db_manager.py` lines 372-624: additive `ALTER TABLE ... ADD COLUMN IF NOT
EXISTS` migrations. STANDARD default for `validation_mode` means every pre-existing row is
implicitly Mode A; EHR rows are explicitly opted in by the Stage 4 discovery writer.

## Blast Radius

- **Edits:** `trading_app/db_manager.py` — append one migration block (5 ADD COLUMN
  IF NOT EXISTS statements) immediately after the validation_pathway/c8_oos_status
  block (line 624). Imports `EHR_MODE_LABEL` and `STANDARD_MODE_LABEL` from
  `trading_app.holdout_policy` for default-value derivation (canonical-source delegation
  per `integrity-guardian.md` § 2 — never inline `"STANDARD"` magic string).
- **Columns added to `validated_setups`:**
  - `validation_mode TEXT DEFAULT 'STANDARD'` — `STANDARD` or `EARLY_HOLDOUT_REDISCOVERY`
  - `pseudo_oos_window_start DATE` — first day of the PSEUDO-OOS window (EHR rows only;
    NULL on STANDARD)
  - `pseudo_oos_window_end DATE` — exclusive upper bound of PSEUDO-OOS window (EHR only)
  - `verdict_ceiling TEXT` — hard ceiling on verdict for this row; `RESEARCH_PROVISIONAL`
    on EHR rows, NULL on STANDARD (Stage 3 validator enforces)
  - `cumulative_search_count INTEGER` — total trials across prior Mode A discovery +
    this EHR run, summed at discovery-date (Bailey-Lopez de Prado 2014 disclosure)
- **New file:** `tests/test_trading_app/test_db_manager_ehr_schema.py` — 4 tests:
  1. `test_validated_setups_has_ehr_columns` — schema introspection finds all 5 columns
     with correct types (DuckDB info_schema query).
  2. `test_validation_mode_default_is_standard` — inserting a minimal row without
     `validation_mode` produces `'STANDARD'` (default).
  3. `test_ehr_columns_nullable_on_standard_row` — STANDARD rows accept NULL for the
     4 EHR-specific columns.
  4. `test_round_trip_ehr_row_preserves_columns` — insert+select an EHR row;
     all 5 column values round-trip exactly (no truncation, no implicit cast).
- **Reads:** module reads `trading_app.holdout_policy.{EHR_MODE_LABEL, STANDARD_MODE_LABEL}`
  to ground default-value strings in the canonical source. Test file uses an in-memory
  DuckDB connection seeded by `init_trading_app_schema()` — no touches to `gold.db`.
- **Writes:** none to gold.db. All test DB writes are temp paths via tmp_path fixture.
- **Callers affected today:** zero. New columns are NULL/default for all existing rows.
  Stage 4 (discovery), Stage 5 (allocator + trade_book), and Stage 6 (drift checks) are
  the future consumers.
- **Reversibility:** `ALTER TABLE DROP COLUMN` in DuckDB; or `git revert <commit>` restores
  the pre-Stage-2 schema on the next `init_trading_app_schema()` call against a fresh DB.
  Existing DBs would still carry the columns (additive-only design — no migration-down
  required for the PASS 2 RED-path revert; columns become dead but harmless).
- **Drift surface:** `pipeline/check_drift.py` does not currently introspect
  `validated_setups` for column shape. Stage 6 will add the EHR-isolation drift check
  that queries these columns. Stage 2 alone introduces zero new drift violations.

## Acceptance

1. `validated_setups` table has 5 new columns after `init_trading_app_schema()` runs against
   a fresh DB: `validation_mode`, `pseudo_oos_window_start`, `pseudo_oos_window_end`,
   `verdict_ceiling`, `cumulative_search_count`.
2. `validation_mode` defaults to `'STANDARD'` for any row INSERTed without specifying it.
   The default value string is sourced from `trading_app.holdout_policy.STANDARD_MODE_LABEL`,
   not inlined.
3. All 4 EHR-specific columns are nullable; STANDARD rows are unchanged.
4. Round-trip insert+select of an EHR row preserves all 5 column values byte-exact (text
   labels, date ISO strings, integer counts).
5. `pytest tests/test_trading_app/test_db_manager_ehr_schema.py -v` — 4 tests pass.
6. `pytest tests/test_trading_app/test_holdout_policy.py tests/test_trading_app/test_holdout_policy_ehr.py -v` — 30 tests still pass (Stage 1 regression guard).
7. `python pipeline/check_drift.py` — violation count unchanged from the 296 baseline.
8. `git grep "'STANDARD'"` in `trading_app/db_manager.py` returns the existing pre-EHR
   default values plus exactly one new occurrence wrapped in a Python expression that
   asserts equality with `STANDARD_MODE_LABEL` (defense-in-depth: the DEFAULT must be the
   literal SQL string, but a runtime assert pins it to the canonical constant).
9. Module docstring or block comment at the new migration site cites the PASS 2 plan and
   plan invariants 4 + 5.

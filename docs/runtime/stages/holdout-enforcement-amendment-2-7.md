---
mode: IMPLEMENTATION
slug: holdout-enforcement-amendment-2-7
task: Implement the 4 deferred enforcement items from Amendment 2.7 properly — create canonical `trading_app/holdout_policy.py` single-source-of-truth, refactor `check_drift.py` to import from it, add `strategy_discovery.py` runtime enforcement (require --holdout-date, reject post-sacred values), add `strategy_validator.py` validation gate, add new declaration-consistency drift check. Proper institutional refactor, not band-aid.
created: 2026-04-08
updated: 2026-04-08
stage: 1
of: 6
scope_lock:
  - trading_app/holdout_policy.py
  - pipeline/check_drift.py
  - trading_app/strategy_discovery.py
  - trading_app/strategy_validator.py
  - tests/test_trading_app/test_holdout_policy.py
  - tests/test_pipeline/test_check_drift.py
  - .claude/rules/integrity-guardian.md
blast_radius: New canonical source module at `trading_app/holdout_policy.py` exports `HOLDOUT_SACRED_FROM`, `HOLDOUT_GRANDFATHER_CUTOFF`, `enforce_holdout_date()`. Three downstream consumers import from it — `pipeline/check_drift.py` (refactor existing inlined dict), `trading_app/strategy_discovery.py` (add runtime validation in main), `trading_app/strategy_validator.py` (validation gate at promote-time). New drift check asserts docs + code agree on the canonical values. Zero database writes. The existing grandfather behavior in check_drift.py is preserved byte-for-byte (same 71,901 rows remain advisory, same 0 new violations). integrity-guardian.md canonical-sources table gets one new row.
---

# Stage: Holdout Enforcement (Amendment 2.7)

## Purpose

Implement the 4 deferred enforcement items from `docs/institutional/pre_registered_criteria.md` Amendment 2.7 (2026-04-08) as a proper institutional refactor with a single canonical source, not as scattered band-aids. Per user directive *"go proper institutional"*.

## Why a canonical source

Current state: `pipeline/check_drift.py` has `HOLDOUT_DECLARATIONS` inlined inside `check_holdout_contamination()` as a function-local dict. The date `2026-04-08` is a magic number with no importable reference. Any downstream consumer (strategy_discovery, strategy_validator, new drift checks) would either duplicate the constant (violating canonical-sources rule 2) or reach into `check_drift.py` internals.

Per `.claude/rules/integrity-guardian.md` rule 2: *"Import from the single source of truth. Never inline lists or magic numbers."*

Per `.claude/rules/institutional-rigor.md` rule 4: *"Delegate to canonical sources — never re-encode"*.

Proper fix: create `trading_app/holdout_policy.py` as the canonical source. All consumers import from it.

## Stages

### Stage 1 — Canonical source (this stage)

Create `trading_app/holdout_policy.py` with:
- `HOLDOUT_SACRED_FROM: date` — the sacred window start (2026-01-01 per Amendment 2.7)
- `HOLDOUT_GRANDFATHER_CUTOFF: datetime` — the Amendment 2.7 enforcement moment (2026-04-08 UTC)
- `enforce_holdout_date(arg: date | None) -> date` — CLI argument validator; defaults None to sacred_from, rejects post-sacred values with clear error message citing Amendment 2.7
- Docstring citing Amendment 2.7 + RESEARCH_RULES.md

Add `tests/test_trading_app/test_holdout_policy.py` with:
- Constant values match Amendment 2.7
- `enforce_holdout_date(None)` returns `HOLDOUT_SACRED_FROM`
- `enforce_holdout_date(2025-12-31)` returns the passed date (unchanged)
- `enforce_holdout_date(2026-01-01)` returns sacred_from (boundary case)
- `enforce_holdout_date(2026-01-02)` raises ValueError
- `enforce_holdout_date(2026-07-01)` raises ValueError with "Mode A" in message

### Stage 2 — check_drift.py imports canonical source

Replace the inlined `HOLDOUT_DECLARATIONS` dict with import from `trading_app.holdout_policy`. The grandfather cutoff becomes `HOLDOUT_GRANDFATHER_CUTOFF`. Semantics unchanged, source canonicalized.

Add new drift check (Check #91 or next available): `check_holdout_policy_declaration_consistency()` — asserts that `RESEARCH_RULES.md` mentions the canonical sacred_from date, `pre_registered_criteria.md` Amendment 2.7 exists, and `strategy_discovery.py` imports from the canonical source. Catches future doc-code drift.

### Stage 3 — strategy_discovery.py runtime enforcement

In `main()` (argparse section), call `enforce_holdout_date(args.holdout_date)` before passing to `run_discovery()`. If None, defaults to `HOLDOUT_SACRED_FROM`. If post-sacred, raises and prints a clear error citing Amendment 2.7. Required-ish — not blocking None (defaults to sacred), but blocking violations.

### Stage 4 — strategy_validator.py validation gate

Before promoting a strategy to `validated_setups`, verify its source discovery respected the holdout. Mechanism: check that the strategy's `experimental_strategies` row's `yearly_results` does NOT contain data for years ≥ `HOLDOUT_SACRED_FROM.year` UNLESS the row is grandfathered (created ≤ `HOLDOUT_GRANDFATHER_CUTOFF`). Raises on violation.

### Stage 5 — Integrity guardian canonical table

Add `HOLDOUT_SACRED_FROM / HOLDOUT_GRANDFATHER_CUTOFF` to the canonical sources table in `.claude/rules/integrity-guardian.md`. Points at `trading_app.holdout_policy`.

### Stage 6 — Full verification

Run `python pipeline/check_drift.py` — expect 84 passes (83 current + 1 new declaration-consistency check), 0 violations.
Run `python -m pytest tests/test_trading_app/test_holdout_policy.py tests/test_pipeline/test_check_drift.py` — expect all pass.
Run `python -m ruff check pipeline/ trading_app/` — expect ≤1 error (only the pre-existing UP037).
Run `python scripts/tools/audit_behavioral.py` — expect clean.
Self-review each stage's commit.

## Acceptance

1. `trading_app/holdout_policy.py` exists and exports the 3 names
2. `tests/test_trading_app/test_holdout_policy.py` passes (5+ tests)
3. `pipeline/check_drift.py` imports from the canonical source; no inlined dict
4. `check_holdout_contamination()` behavior byte-identical (same grandfathered count, same 0 new violations)
5. New drift check `check_holdout_policy_declaration_consistency()` passes
6. `trading_app/strategy_discovery.py` calls `enforce_holdout_date()` in main; rejects post-sacred values with a clear error
7. `trading_app/strategy_validator.py` has a validation gate that uses the canonical source
8. `.claude/rules/integrity-guardian.md` canonical-sources table has the new row
9. `python pipeline/check_drift.py` → NO DRIFT (all checks pass)
10. `python -m pytest tests/test_trading_app/test_holdout_policy.py tests/test_pipeline/test_check_drift.py tests/test_trading_app/test_strategy_discovery.py` → all pass (or skip if test files don't exist)
11. Ruff unchanged (still 1 pre-existing error)
12. Each of the 6 stages is its own commit with clear rationale

## Out of scope

- Phase 4 rediscovery to restore the clean 117-strategy baseline (separate stage, needs DB writes + long runtime)
- Drift Check 57 fix (MGC 2026-04-06 partial daily_features — separate stage, needs data rebuild)
- Phase 2 Databento redownload (separate stage, needs user approval on scope)
- Changing the 5 deployed lanes' operational status (they remain research-provisional per Amendment 2.4)
- Memory file housekeeping beyond what's already been done

---
mode: IMPLEMENTATION
slug: phase-4-1-discovery-hypothesis-file
task: Phase 4 Stage 4.1 wire hypothesis-file into discovery, add scope predicate, SHA stamping, and integrity drift check 94
started: 2026-04-08T12:47:51Z
updated: 2026-04-08T12:47:51Z
parent_plan: docs/plans/2026-04-08-phase-4-clean-rediscovery-design.md
pre_flight_audit: docs/plans/2026-04-08-phase-4-clean-rediscovery-design.md Stage 4.1
blast_radius: trading_app/strategy_discovery.py + strategy_validator.py + hypothesis_loader.py + holdout_policy.py + phase_4_discovery_gates.py (new) + pipeline/check_drift.py + hypothesis_registry_template.md + 4 test files extended + 1 new test file; downstream verification drift count 86 to 87, test suite zero regressions, behavioral audit 7/7, ruff clean on 12 files
scope_lock:
  - trading_app/strategy_discovery.py
  - trading_app/strategy_validator.py
  - trading_app/hypothesis_loader.py
  - trading_app/holdout_policy.py
  - trading_app/phase_4_discovery_gates.py
  - pipeline/check_drift.py
  - docs/institutional/hypothesis_registry_template.md
  - tests/test_trading_app/test_strategy_discovery.py
  - tests/test_trading_app/test_strategy_validator.py
  - tests/test_trading_app/test_hypothesis_loader.py
  - tests/test_trading_app/test_phase_4_discovery_gates.py
  - tests/test_pipeline/test_check_drift.py
---

# Phase 4 Stage 4.1 — Discovery Hypothesis-File Integration

## Purpose

Wire the pre-registered hypothesis file as a **pre-facto enumeration constraint** on the discovery routine. Stage 4.0 shipped the read-side (validator gates for criteria 1, 2, 8, 9; pure-YAML loader; `experimental_strategies.hypothesis_file_sha` column). Stage 4.1 wires the write-side: CLI arg, scope predicate, single-use enforcement, git-cleanliness, SHA stamping, integrity drift check.

## Design authority

- `docs/plans/2026-04-08-phase-4-clean-rediscovery-design.md` § Stage 4.1
- `docs/institutional/pre_registered_criteria.md` (Criteria 1, 2; Amendments 2.1, 2.2, 2.4, 2.7)
- `docs/audit/hypotheses/README.md` (registry workflow)
- `.claude/rules/institutional-rigor.md` (rule 4: delegate to canonical sources; rule 6: no silent failures)

## Pre-implementation gates (all verified via blast-radius scout 2026-04-08)

1. `_check_criterion_2_minbtl` exists at `strategy_validator.py:853-887`, signature `(meta, on_proxy_data=False)`, hardcoded bounds 300/2000 at line 874 — refactorable via 2-line delegation preserving function name and `on_proxy_data` passthrough
2. `StrategyFilter.filter_type` exists as base dataclass field (`config.py:334`); `.threshold` does NOT exist generically — predicate matches on `filter_type` string only
3. Direct `run_discovery()` callers: main() + test_strategy_discovery.py only. Regime/nested have their own discovery functions
4. Subprocess CLI callers: `run_full_pipeline.py:102`, `parallel_rebuild.py:127`, `test_synthetic_null.py:780` — NOT touched in this stage; legacy-mode default preserves them
5. `HOLDOUT_SACRED_FROM: date`, `HOLDOUT_GRANDFATHER_CUTOFF: datetime(tzinfo=UTC)`, `PHASE_4_1_SHIP_DATE` absent (safe to add)
6. `test_check_drift.py` has no existing DB fixture pattern for experimental_strategies — follow monkeypatch pattern for check #94 tests
7. `_check_criterion_1_hypothesis_file` uses `load_hypothesis_by_sha`; drift check #94 does NOT need the SHA lookup — simple SQL suffices

## Locked decisions

| # | Decision | Rationale |
|---|---|---|
| D-1 | `--hypothesis-file` soft-optional; None = legacy mode | Preserves 3 subprocess callers; architecturally consistent with validator's `_is_phase_4_grandfathered` |
| D-2 | Drift check #94 = INTEGRITY only (stamped SHA must reference real file) | Avoids regime/nested/legacy ambiguity; surgical invariant |
| D-3 | `run_discovery(hypothesis_file: Path \| None = None)` default | Preserves all existing test_strategy_discovery.py tests |
| D-4 | Single-use + git checks fire BEFORE `with duckdb.connect()` | Must run before DELETE+INSERT idempotent wipe at line 1316 |
| D-5 | `_check_criterion_2_minbtl` delegation preserves name + `on_proxy_data` passthrough | Drift check #93 asserts function name; Stage 4.1 activates `on_proxy_data` flag |
| D-6 | Predicate skip placement: between line 1191 and 1195 | After `combo_idx += 1`, after `matching_day_set` check — preserves trial counting semantics |
| D-7 | `_BATCH_COLUMNS` length assertion added to `_flush_batch_df` | Defensive guard against silent column misalignment (Risk 1) |
| D-8 | Subprocess CLI callers (`run_full_pipeline.py`, `parallel_rebuild.py`, `test_synthetic_null.py`) NOT touched in Stage 4.1 | Legacy mode default keeps them working; dedicated follow-up when Stage 4.2 hypothesis files ship |
| D-9 | MinBTL mode: default 300; exceed to 2000 requires `metadata.data_source_mode == "proxy"` AND non-empty `metadata.data_source_disclosure` | Explicit opt-in disclosure per Criterion 2 locked text |
| D-10 | ScopePredicate stores primitive (frozenset of filter_type strings, frozenset of sessions, etc.) not StrategyFilter objects | Loader stays pure-YAML; call site extracts `strategy_filter.filter_type` |

## Scope

### Files modified (production)

1. `trading_app/holdout_policy.py` — add `PHASE_4_1_SHIP_DATE: datetime` constant + `__all__` export. Set to `datetime(2026, 4, 9, 0, 0, 0, tzinfo=UTC)` (tomorrow midnight UTC = operational grace period)
2. `trading_app/hypothesis_loader.py` — add `enforce_minbtl_bound(meta, on_proxy_data)` pure function, `ScopePredicate` frozen dataclass, `extract_scope_predicate(meta, instrument)` pure function, `check_mode_a_consistency(meta)` pure function. Update `__all__`
3. `trading_app/phase_4_discovery_gates.py` — **NEW** module with `check_git_cleanliness(path)` (subprocess git) and `check_single_use(sha, con)` (DB query). Not exported from loader to preserve loader purity
4. `trading_app/strategy_validator.py` — refactor `_check_criterion_2_minbtl` body to delegate to `loader.enforce_minbtl_bound(meta, on_proxy_data)`. Preserve function name + signature + call site at line 1121
5. `trading_app/strategy_discovery.py` — add `--hypothesis-file` CLI arg, `hypothesis_file` kwarg on `run_discovery` (default None), Phase 4 enforcement block (load → git → MinBTL → Mode A → single-use → extract predicate → instrument-consistency check), scope predicate skip in enumeration loop (between lines 1191-1195), safety-net raw-count assertion, SHA stamping in `_BATCH_COLUMNS` + `_flush_batch_df` INSERT + batch assembly loop, `_BATCH_COLUMNS` length assertion in `_flush_batch_df`
6. `pipeline/check_drift.py` — add `check_phase_4_sha_integrity` function as check #94 (queries `experimental_strategies WHERE hypothesis_file_sha IS NOT NULL AND created_at >= PHASE_4_1_SHIP_DATE`, asserts every SHA references a real file in `docs/audit/hypotheses/`). Register in CHECKS list. Count 86 → 87

### Files modified (docs)

7. `docs/institutional/hypothesis_registry_template.md` — document new optional metadata fields `data_source_mode` and `data_source_disclosure` under the metadata block; add example showing proxy-mode use

### Test files

8. `tests/test_trading_app/test_hypothesis_loader.py` — +15 tests (scope predicate, MinBTL bound, Mode A consistency, boundary conditions)
9. `tests/test_trading_app/test_phase_4_discovery_gates.py` — **NEW** file, +6 tests (git cleanliness with subprocess mocks + 1 real temp-repo integration, single-use fresh/used/edge)
10. `tests/test_trading_app/test_strategy_discovery.py` — +8 tests (CLI arg soft-optional, hypothesis-file enforcement mode, scope predicate limits enumeration, safety net raises on budget overshoot, SHA stamping on all rows, instrument-consistency fails loud, early-exit predicate equivalence)
11. `tests/test_trading_app/test_strategy_validator.py` — +1 test (MinBTL delegation shares constants with loader)
12. `tests/test_pipeline/test_check_drift.py` — +4 tests for check #94 (empty DB, post-ship row with valid SHA, post-ship row with orphan SHA, pre-ship row with orphan SHA is ignored)

### Files explicitly NOT touched

- `trading_app/db_manager.py` — `hypothesis_file_sha` column shipped in Stage 4.0
- `trading_app/config.py` — StrategyFilter exposes `.filter_type` already
- `pipeline/run_full_pipeline.py` — legacy mode preserves; update deferred to follow-up stage
- `scripts/infra/parallel_rebuild.py` — legacy mode preserves; same
- `scripts/tests/test_synthetic_null.py` — legacy mode preserves; same
- `trading_app/regime/discovery.py` + `trading_app/nested/discovery.py` — known gap, not in Stage 4.1 scope (drift check #94 scoped to integrity only avoids flagging their rows)

## Blast Radius

Production code edits span two trading_app/ modules (one NEVER_TRIVIAL: strategy_discovery.py), one new helper module, one pipeline/ drift check, and one holdout_policy constant addition. All changes preserve 1-way dependency (pipeline/ → trading_app/ unchanged). Zero schema migrations. Zero canonical-source re-encoding (MinBTL logic delegated to loader as canonical source). Four test files extended + one new test file.

Downstream verification targets: drift check count 86 → 87, test suite 3,456 + ~34 new = zero regressions, behavioral audit 7/7, ruff clean on all 12 touched files, Stage 4.0's `_check_criterion_2_minbtl` test still passes after delegation refactor, drift check #93 still fires correctly on function name presence.

## Sequencing (executed in order)

1. `holdout_policy.py` — add PHASE_4_1_SHIP_DATE constant + __all__. Verify baseline tests pass.
2. `hypothesis_loader.py` — add enforce_minbtl_bound + ScopePredicate + extract_scope_predicate + check_mode_a_consistency + __all__. Add tests in test_hypothesis_loader.py.
3. `strategy_validator.py` C-2 refactor — delegate `_check_criterion_2_minbtl` to loader. Verify Stage 4.0 test suite stays green.
4. `phase_4_discovery_gates.py` NEW — check_git_cleanliness + check_single_use. Add test_phase_4_discovery_gates.py tests.
5. `strategy_discovery.py` — add CLI arg, kwarg, Phase 4 enforcement block, predicate skip, safety net, SHA stamping. Add test_strategy_discovery.py tests.
6. `check_drift.py` — add check #94. Add test_check_drift.py tests. Verify count 86 → 87.
7. `hypothesis_registry_template.md` — document new metadata fields.
8. Self-audit pass: run drift + tests + behavioral audit + ruff.
9. Code-reviewer subagent pass on loader + discovery changes.
10. Fix any findings, re-verify, commit.

## Done criteria

- [ ] All 12 files in scope_lock edited as designed
- [ ] All 34 new tests pass (15 loader + 6 gates + 8 discovery + 1 validator + 4 drift)
- [ ] Full test suite passes with zero regressions (baseline: 3,456 pass)
- [ ] `python pipeline/check_drift.py` shows 87 passing / 0 failing / 7 advisory
- [ ] Behavioral audit 7/7 clean (`scripts/tools/audit_behavioral.py`)
- [ ] Ruff clean on all 12 touched files
- [ ] Dead code swept (`grep -r TODO\|FIXME\|XXX` on touched files)
- [ ] Code-reviewer subagent produces PASS verdict on loader + discovery changes
- [ ] Self-audit pass after implementation identifies zero unfixed findings
- [ ] Stage 4.0's `test_strategy_validator.py` Phase 4 test classes still pass after MinBTL delegation refactor
- [ ] Drift check #93 still passes (function name `_check_criterion_2_minbtl` preserved)
- [ ] Commit message captures audit trail, blast-radius gates, and all 10 locked decisions

## Rollback plan

Per-step reversibility — each sequencing step is a pure add OR a small refactor, reversible via `git revert <commit>`. No schema migrations means no DB rollback needed. Single-use check failures during testing can be cleared by `DELETE FROM experimental_strategies WHERE hypothesis_file_sha IS NOT NULL` if test fixtures leak rows into gold.db (they should not; tests use isolated fixture DBs).

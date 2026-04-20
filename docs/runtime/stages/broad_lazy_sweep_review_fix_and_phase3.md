---
task: A+ review fix (honest comment in strategy_discovery) + Phase 3 (pipeline/build_daily_features lazy-load) — the binding transitive-chain constraint unlocking Phase 1 + 1b deferrals
mode: IMPLEMENTATION
scope_lock:
  - trading_app/strategy_discovery.py
  - pipeline/build_daily_features.py
  - docs/plans/2026-04-20-broad-lazy-import-sweep.md
  - docs/runtime/stages/broad_lazy_sweep_review_fix_and_phase3.md
blast_radius: strategy_discovery.py comment-only edit (zero runtime impact). pipeline/build_daily_features.py — 1811 LOC, feature-builder hot path. Heavy deps at module-top (pandas 16 sites, numpy 19 sites, duckdb 7 sites). No module-scope pd/np/duckdb usage verified. Annotations use pd.DataFrame / pd.Timestamp / np.ndarray / duckdb.DuckDBPyConnection — PEP 563 makes them strings. Importers grep'd (run_pipeline, check_drift, trading_app.portfolio, trading_app.outcome_builder, trading_app.nested.audit_outcomes, research/cme_fx_futures_orb_pilot, scripts/backfill_htf_levels, ~17 research/archive/*). All use named symbols — zero module-attribute access to pd/np/duckdb. Companion tests — tests/test_pipeline/test_build_daily_features.py (if exists), plus any pipeline-level integration tests. CLI entry python -m pipeline.build_daily_features or via run_pipeline.py. Zero schema changes.
acceptance:
  - strategy_discovery.py comment replaced with honest wording
  - pipeline/build_daily_features.py uses `from __future__ import annotations` + TYPE_CHECKING guard + function-body lazy imports
  - Companion tests pass (test_build_daily_features.py + any integration tests that exercise the feature-builder path)
  - python -m pipeline.check_drift (isolated) = 0 violations
  - warm-import 5-run median pre/post on current OS-cache state reported honestly
  - strategy_discovery cold re-measured to show chain-unlock delta (if any)
  - two commits: (1) fix stale comment, (2) Phase 3 lazy-load
agent: claude
---

# Stage — broad-lazy-sweep review fix + Phase 3

Closes review finding (B+ → A+): stale "~5s import" comment in strategy_discovery.py:24 replaced with range that covers both warm and cold OS-cache states.

Then Phase 3 on pipeline/build_daily_features.py using the same pattern as outcome_builder (PEP 563 + TYPE_CHECKING + per-function lazy imports).

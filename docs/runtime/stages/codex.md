---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Fix Codex sweep finding 7 (stale scratch DB docstrings) + add orb_active drift guard
pass: 2
scope_lock:
  - pipeline/run_full_pipeline.py
  - pipeline/check_drift.py
  - scripts/tools/explore.py
  - scripts/tools/orb_size_deep_dive.py
  - scripts/tools/backfill_dollar_columns.py
  - scripts/tools/audit_ib_single_break.py
  - scripts/tools/backtest_atr_regime.py
  - scripts/tools/build_edge_families.py
  - scripts/tools/detect_volume_spikes.py
  - scripts/tools/find_pf_strategy.py
  - scripts/tools/ml_license_diagnostic.py
  - scripts/tools/parity_check.py
  - scripts/tools/profile_1000_runners.py
  - scripts/tools/stress_test.py
  - scripts/tools/volume_session_analysis.py
  - scripts/infra/rolling_eval_parallel.py
  - scripts/infra/run_parallel_ingest.py
  - scripts/ingestion/ingest_mnq.py
  - scripts/migrations/backfill_atr20.py
  - scripts/migrations/backfill_sharpe_ann.py
  - scripts/migrations/backfill_strategy_trade_days.py
  - scripts/migrations/backfill_wf_columns.py
  - scripts/migrations/migrate_add_dynamic_columns.py
  - scripts/migrations/retire_e3_strategies.py
  - scripts/reports/report_edge_portfolio.py
  - trading_app/live_config.py
  - docs/prompts/SYSTEM_AUDIT.md
  - scripts/run_live_session.py
  - trading_app/live/session_orchestrator.py
blast_radius:
  - All edits are docstring-only (usage examples) — ZERO runtime impact
  - 2 new drift checks in check_drift.py — additive, no existing check modified
acceptance:
  - All 27 stale C:/db/gold.db docstring references removed
  - Drift check 83 (orb_active) passes
  - Drift check 84 (scratch DB docstrings) passes
  - All 84 drift checks pass
updated: 2026-03-29T15:00:00Z
---

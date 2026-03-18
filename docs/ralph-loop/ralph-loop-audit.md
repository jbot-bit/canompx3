# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 131

## RALPH AUDIT — Iteration 131
## Date: 2026-03-18
## Infrastructure Gates: PASS (drift 71 PASS + 6 advisory [14 env-only violations: missing databento/requests in headless env, pre-existing]; behavioral audit 6/6 clean; ruff clean via uv run; 36/36 test_predict_live.py pass)

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS (env caveat) | 71 checks PASS, 14 env-only violations (missing databento/requests packages), 6 advisory |
| `audit_behavioral.py` | PASS | all 6 checks clean |
| `ruff check` | PASS | clean (via `uv run`) |
| pytest targeted | PASS | 36/36 (test_predict_live.py) |

---

## Scope: trading_app/ml/predict_live.py

---

## Seven Sins Scan

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | Line 307: `logger.debug()` for aggressive RR mismatch (trade skip, take=False) — invisible at INFO log level. Aperture mismatch (take=True) correctly logged at INFO. Inconsistent severity ordering. | MEDIUM | FIXED (d0fe929) |
| Silent failure | Line 196: `except Exception` in `_load_models` — logged at WARNING with exc_info=True. Intentional fail-open design. | — | ACCEPTABLE |
| Silent failure | Line 410: `except Exception` in `predict()` — logged at WARNING with exc_info=True, fail_open_count tracked. Intentional. | — | ACCEPTABLE |
| Fail-open | Module docstring explicitly declares fail-open as design principle (de Prado AIFML). All fail-open paths logged at WARNING/INFO and counter-tracked. | — | ACCEPTABLE |
| Look-ahead bias | Features read from pre-computed daily_features table. No LAG() or double_break. 60/20/20 split is in training only (meta_label.py). | — | CLEAN |
| Cost illusion | No P&L computation — ML prediction only outputs P(win). | — | CLEAN |
| Canonical violation | ACTIVE_INSTRUMENTS from trading_app.ml.config (derived from ACTIVE_ORB_INSTRUMENTS). GLOBAL_FEATURES from ml.config. No hardcoded lists. | — | CLEAN |
| Orphan risk | No dead code, no unused imports. | — | CLEAN |
| Volatile data | No hardcoded counts. | — | CLEAN |

---

## Summary
- trading_app/ml/predict_live.py: 1 MEDIUM finding — FIXED
- Action: fix committed (d0fe929)

---

## Files Fully Scanned

> Cumulative list — 194 files fully scanned (1 new file added this iteration: trading_app/ml/predict_live.py).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
- trading_app/ml/config.py — added iter 129
- trading_app/ml/meta_label.py — added iter 130
- trading_app/ml/predict_live.py — added iter 131
- trading_app/outcome_builder.py — added iter 115
- trading_app/strategy_discovery.py — added iter 116
- trading_app/strategy_validator.py — added iter 117
- trading_app/portfolio.py — added iter 118
- trading_app/live_config.py — added iter 119
- trading_app/db_manager.py — added iter 120
- trading_app/live/projectx/auth.py — added iter 121
- trading_app/live/projectx/order_router.py — added iter 122
- trading_app/live/projectx/data_feed.py — added iter 123
- trading_app/live/broker_base.py — added iter 123
- trading_app/live/tradovate/data_feed.py — added iter 124
- trading_app/live/session_orchestrator.py — added iter 124
- trading_app/live/bar_aggregator.py — added iter 125
- trading_app/live/tradovate/order_router.py — added iter 126
- trading_app/live/tradovate/auth.py — added iter 126
- trading_app/live/tradovate/contract_resolver.py — added iter 126
- trading_app/live/tradovate/positions.py — added iter 126
- trading_app/live/broker_factory.py — added iter 127
- trading_app/live/tradovate/__init__.py — added iter 127
- trading_app/live/circuit_breaker.py — added iter 128
- trading_app/live/cusum_monitor.py — added iter 128
- trading_app/live/projectx/__init__.py — added iter 128
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 19 files (iters 101-111): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105); research_zt_event_viability.py (iter 106); research_vol_regime_switching.py (iter 107); research_edge_structure.py, research_1015_vs_1000.py (iters 108-109); research_overlap_analysis.py (iter 110); research_aperture_scan.py (iter 111)
- research/ additional (iters 112-113): research_alt_stops.py, research_direction_asymmetry.py, research_signal_stack.py (iter 112); discover.py, research_wf_stress_keepers.py, research_trend_day_mfe.py scanned but already clean (iter 113)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 194 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `trading_app/walkforward.py` — LOW tier, has recently modified (M) in git; worth scanning for introduced issues
- `trading_app/ml/evaluate.py` — LOW tier, unscanned ML evaluation path (1 importer)
- `trading_app/ml/evaluate_validated.py` — LOW tier, unscanned (0 importers — standalone eval script)

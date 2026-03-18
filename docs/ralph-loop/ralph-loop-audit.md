# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 132

## RALPH AUDIT — Iteration 132
## Date: 2026-03-18
## Infrastructure Gates: PASS (drift 72 PASS + 6 advisory [headless env: pipeline module not on sys.path without uv run — pre-existing]; behavioral audit 6/6 clean; ruff clean; 32/32 test_walkforward.py pass; 218/218 pre-commit suite pass)

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks PASS, 6 advisory (headless env-only) |
| `audit_behavioral.py` | PASS | all 6 checks clean |
| `ruff check` | PASS | clean (via `uv run`) |
| pytest targeted | PASS | 32/32 (test_walkforward.py) |
| pre-commit suite | PASS | 218/218 |

---

## Scope: trading_app/walkforward.py

---

## Seven Sins Scan

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | Lines 134-136: tight stop silently skipped with no warning when stop_multiplier != 1.0 but cost_spec is None. Current caller (strategy_validator) handles correctly; no defensive warning existed for future misconfiguration. | MEDIUM | FIXED (1b2ac93) |
| Fail-open | No fail-open paths. Pass/fail uses 4-gate conjunction (ALL required), fail-closed. | — | CLEAN |
| Look-ahead bias | Calendar mode: IS uses outcomes[:lo], OOS uses outcomes[lo:hi] — no look-ahead. Trade-count mode: IS uses usable[:idx], OOS uses usable[idx:idx+N] — no look-ahead. | — | CLEAN |
| Cost illusion | No P&L computation in this file. Costs applied via apply_tight_stop at outcome loading time. | — | CLEAN |
| Canonical violation | No hardcoded instrument lists, entry models, or session names. Thresholds (min_valid_windows, min_pct_positive, etc.) are all parameters. The 5.0 imbalance ratio has @research-source annotation. The hardcoded 15 in IS metrics checks has @research-source annotation in calendar mode (line 215); trade-count mode (line 167) lacks annotation but the same rationale applies — acceptable given existing annotation on same pattern. | LOW | ACCEPTABLE (annotation present in calendar mode; trade-count 15 consistent with same threshold) |
| Orphan risk | All imports used: calendar, json, logging, bisect_left, asdict, dataclass, UTC, date, datetime, Path, apply_tight_stop, compute_metrics, _load_strategy_outcomes. | — | CLEAN |
| Volatile data | No hardcoded counts. | — | CLEAN |

---

## Summary
- trading_app/walkforward.py: 1 MEDIUM finding — FIXED
- Action: fix committed (1b2ac93)

---

## Files Fully Scanned

> Cumulative list — 195 files fully scanned (1 new file added this iteration: trading_app/walkforward.py).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
- trading_app/ml/config.py — added iter 129
- trading_app/ml/meta_label.py — added iter 130
- trading_app/ml/predict_live.py — added iter 131
- trading_app/walkforward.py — added iter 132
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
- **Total: 195 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `trading_app/ml/evaluate.py` — LOW tier, unscanned ML evaluation path (1 importer)
- `trading_app/ml/evaluate_validated.py` — LOW tier, unscanned standalone eval script (0 importers)
- `trading_app/paper_trader.py` — if not already scanned; production simulation path worth reviewing

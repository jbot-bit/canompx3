# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 127

## RALPH AUDIT — Iteration 127
## Date: 2026-03-17
## Infrastructure Gates: PASS (drift 72 PASS + 6 advisory; behavioral audit 6/6 clean; ruff clean; 9/9 targeted + 218/218 pre-commit suite)

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks PASS, 0 failed, 6 advisory |
| `audit_behavioral.py` | PASS | all 6 checks clean |
| `ruff check` | PASS | clean |
| pytest targeted | PASS | 9/9 (test_broker_factory.py) |

---

## Scope: trading_app/live/broker_factory.py + trading_app/live/tradovate/__init__.py

**Diminishing Returns check:**
- `consecutive_low_only: 2` in ledger (last HIGH+ finding iter 124)
- Scope: broker_factory.py = LOW centrality (1 importer per import_centrality.json, but hub-like: 11 imports)
- Proceed: YES (consecutive_low_only < 3)

---

## Seven Sins Scan

### broker_factory.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | None — early guard raises ValueError for invalid brokers; fail-closed | — | CLEAN |
| Fail-open | None — early guard + AssertionError defensive tail | — | CLEAN |
| Look-ahead bias | N/A — live trading module | — | CLEAN |
| Cost illusion | N/A — no P&L computation | — | CLEAN |
| Canonical violation | VALID_BROKERS declared as "canonical source" but only used in error message string — if/elif chain was the actual guard. These could drift. | LOW | FIXED — added early `if broker not in VALID_BROKERS` guard; removed redundant else clause |
| Orphan risk | None | — | CLEAN |
| Volatile data | None | — | CLEAN |

### tradovate/__init__.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| All sins | None — pure re-export module; 5 classes match broker_factory.py usage exactly | — | CLEAN |

---

## Summary
- broker_factory.py: 1 finding (canonical_violation LOW — FIXED)
- tradovate/__init__.py: 0 findings — clean
- Commit: 2f039df

---

## Files Fully Scanned

> Cumulative list — 188 files fully scanned (2 new files added this iteration: trading_app/live/broker_factory.py, trading_app/live/tradovate/__init__.py).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
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
- **Total: 188 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `trading_app/live/projectx/__init__.py` — unscanned re-export module (check for stale exports vs what's actually in the package)
- `trading_app/live/circuit_breaker.py` — unscanned (low centrality but safety-critical: circuit breaking in live trading)
- `trading_app/live/cusum_monitor.py` — unscanned (low centrality, but CUSUM is referenced in action queue as future work)

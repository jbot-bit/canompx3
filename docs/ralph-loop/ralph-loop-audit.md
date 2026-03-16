# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 122

## RALPH AUDIT — Iteration 122
## Date: 2026-03-16
## Infrastructure Gates: PASS (ruff clean; drift 72/72 PASS + 6 advisory; 22 tests pass)

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72/72 checks passed, 0 skipped, 6 advisory (pre-existing: staleness, ML config hash, WF coverage, data years, data continuity, uncovered strategies) |
| `audit_behavioral.py` | PASS | all 6 checks clean |
| `ruff check` (target + all dirs) | PASS | order_router.py + 3 UP038 fixes; all dirs clean |
| pytest test_projectx_router.py -x -q | PASS | 22 passed in 0.13s |

---

## Scope: trading_app/live/projectx/order_router.py (high centrality, unscanned, sibling of auth.py)

**Diminishing Returns check:**
- `consecutive_low_only: 1` in ledger (last HIGH+ finding iter 120)
- Scope file has centrality tier `high`
- Proceed: YES

---

## Seven Sins Scan

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | None | — | CLEAN |
| Fail-open | `cancel()` — no `data.get("success")` check after `raise_for_status()`. ProjectX API returns `{"success": false, "errorMessage": "..."}` on HTTP 200. `submit()` already has this guard; `cancel()` did not. | MEDIUM | FIXED (commits d8e0f67 + 4db63d2) |
| Look-ahead bias | N/A — order routing module, no data queries | — | CLEAN |
| Cost illusion | N/A — order routing module | — | CLEAN |
| Canonical violation | `build_order_spec()` matches on hardcoded `"E1"`/`"E2"` strings — routing logic, not a canonical list claim; raises ValueError for anything else | LOW | ACCEPTABLE (routing guard pattern, not a list definition) |
| Orphan risk | No unused imports or dead code paths | — | CLEAN |
| Volatile data | No hardcoded counts | — | CLEAN |

---

## Summary
- 1 finding — MEDIUM (cancel() fail-open, no success check on response body)
- 3 pre-existing UP038 lint errors fixed (gen_repo_map.py, orb_size_deep_dive.py, strategy_fitness.py)
- Verdict: FIXED
- Commits: d8e0f67 (cancel success check already present), 4db63d2 (UP038 lint fixes)

---

## Files Fully Scanned

> Cumulative list — 177 files fully scanned (1 new file added this iteration: trading_app/live/projectx/order_router.py).

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
- **Total: 177 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `trading_app/live/projectx/data_feed.py` — high centrality, unscanned, sibling of auth.py/order_router.py (blast radius context loaded for projectx package)
- `trading_app/live/broker_base.py` — high centrality, unscanned, base class for all order routers

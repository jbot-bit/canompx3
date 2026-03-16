# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 114

## RALPH AUDIT — Iteration 114
## Date: 2026-03-16
## Infrastructure Gates: 2/2 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory; 5 violations = env-only (missing joblib/websockets) |
| `ruff check` (target) | PASS | trading_app/ml/features.py — clean |

---

## Scope: trading_app/ml/features.py (critical centrality, 13 importers)

**Diminishing Returns check:**
- `consecutive_low_only: 7` in ledger BUT scope file has centrality tier `critical`
- Rule: DIMINISHING_RETURNS only triggers when scope file is `low` or `medium`
- Override: NOT needed — critical tier bypasses the check automatically
- Proceed: YES

---

## Seven Sins Scan

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | `_backfill_global_features:84` — `GLOBAL_FEATURES[0]` ("atr_20") as sole proxy for post-backfill NaN count; pre-backfill check iterates ALL features (asymmetry) | LOW | **FIXED** (FE-01) |
| Silent failure | Unknown `filter_type` in filter application (lines 638, 893) → silent skip with warning | LOW | **ACCEPTABLE** — guarded by drift check #29 ("all validated filter_types registered in ALL_FILTERS"); warning logged |
| Fail-open | None found | — | CLEAN |
| Look-ahead bias | LOOKAHEAD_BLACKLIST guard active (line 414-417); cross-session features use pre-entry information only | — | CLEAN |
| Cost illusion | Not applicable (ML feature extraction, no P&L) | — | CLEAN |
| Canonical violation | SESSION_CHRONOLOGICAL_ORDER from config; ALL_FILTERS from trading_app.config; GLOBAL_FEATURES from ml.config | — | CLEAN |
| Orphan risk | `configs_df.iterrows()` at line 869 — logging only (O(sessions) ≈ 12 rows), not filter application; drift check #77 passes | — | ACCEPTABLE |
| Volatile data | None found | — | CLEAN |

---

## Summary
- 1 finding fixed: FE-01 (LOW — post-backfill NaN count asymmetry)
- 1 finding ACCEPTABLE: silent skip for unknown filter_type (guarded by drift check 29)
- 1 pattern ACCEPTABLE: iterrows for logging path (not performance-critical)
- Verdict: ACCEPT
- Commit: 2c7e419

---

## Files Fully Scanned

> Cumulative list — 169 files fully scanned (1 new file added this iteration: trading_app/ml/features.py).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
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
- **Total: 169 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `trading_app/ml/meta_label.py` — low centrality but related ML module (consider skipping to higher value)
- `trading_app/outcome_builder.py` — critical centrality (11 importers) — **recommended next** (no-touch zone for SQL query logic, but can scan for other sins)
- `trading_app/strategy_discovery.py` — high centrality (8 importers) — **recommended next** (SQL logic no-touch but other patterns auditable)
- `trading_app/nested/builder.py` — low centrality, lower priority

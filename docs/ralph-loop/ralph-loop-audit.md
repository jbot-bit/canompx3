# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 135

## RALPH AUDIT — Iteration 135
## Date: 2026-04-01
## Infrastructure Gates: PASS (67 drift checks PASS, 14 skipped DB busy, 3 advisory; behavioral audit 7/7 clean; ruff clean; pre-commit suite PASS)

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 67 checks PASS, 14 skipped (DB busy), 3 advisory (headless env-only) |
| `audit_behavioral.py` | PASS | all 7 checks clean |
| `ruff check` | PASS | clean |
| pytest targeted | N/A | No dedicated test file for these CLI scripts |

---

## Scope: scripts/databento_backfill.py + scripts/tools/refresh_data.py

---

## Seven Sins Scan

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | `load_manifest()` had no JSONDecodeError handler — corrupt manifest from partial write crashes entire `run_download()` | MEDIUM | FIXED (7e70c22) |
| Silent failure | `except Exception: pass` on `out_file.unlink()` in `refresh_data.py:248` — cleanup failure silently swallowed, orphaned file no visibility | LOW | FIXED (7e70c22) |
| Fail-open | `download_dbn()` calls `db.Historical()` with no API key argument — relies on env var; `main()` guards this before calling so fail-closed at CLI entry. download_dbn() as a callable has no guard, but it's only called from refresh_instrument() which is only called from main(). | LOW | ACCEPTABLE (pattern 4: guarded by verified upstream check in main()) |
| Look-ahead bias | None | — | CLEAN |
| Cost illusion | None — scripts download raw files, no P&L computation | — | CLEAN |
| Canonical violation | `DOWNLOAD_SYMBOLS` in refresh_data.py is a local mapping dict (not a canonical list) containing M2K, 2YY, ZT. M2K is dead but the dict is only consulted per-instrument. `ACTIVE_ORB_INSTRUMENTS` is used to determine the auto-refresh list (line 286). M2K would only be downloaded if explicitly requested via `--instrument M2K`. | LOW | ACCEPTABLE (pattern 1: intentional per-instrument mapping, not a canonical list; M2K entry is a dead-code artifact but harmless) |
| Orphan risk | None — scripts are actively used CLI tools | — | CLEAN |
| Volatile data | None | — | CLEAN |

---

## Config Consistency (databento_config.yaml)

- All `stored_as` values map to recognized instruments (MGC, MNQ, MES) — active only
- All end dates set to "2026-04-01" (today) — appropriate for a config written today; will need updating on next use
- `cost_confirm_threshold: 5.00` — appropriate guard against accidental paid downloads
- `chunk_months: 6` — reasonable for the download sizes described
- Tier 2 items labeled "SUBSCRIPTION-INCLUDED" rely on plan-level access — this is documented in comments but not enforced in code. The cost estimation API call will return 0 for included items, so the confirmation gate handles this correctly.
- No dead instruments (M2K, MCL, SIL, etc.) appear in the YAML config — clean

---

## Additional Notes

- `validate_download` reports `valid=False` for 0-row responses (e.g., holiday/gap). This is deliberately conservative — an empty-but-valid download is treated as failed validation requiring investigation. ACCEPTABLE.
- `save_manifest` has no atomic write (writes directly to the target file). A crash during write produces a corrupt manifest. The new JSONDecodeError handler in `load_manifest` now recovers from this. Atomic rename pattern would be cleaner but is a separate improvement beyond this iteration's scope.

---

## Summary

- scripts/databento_backfill.py: 1 MEDIUM finding (manifest JSONDecodeError crash) — FIXED
- scripts/tools/refresh_data.py: 1 LOW finding (silent unlink exception) — FIXED
- 2 LOW findings ACCEPTABLE (db.Historical() API key, DOWNLOAD_SYMBOLS dict with M2K)
- Action: fix (judgment — behavior change: exceptions now surfaced instead of silently crashing/swallowed)

---

## Files Fully Scanned

> Cumulative list — 199 files fully scanned (2 new files added this iteration: scripts/databento_backfill.py, scripts/tools/refresh_data.py).

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
- pipeline/calendar_filters.py — added iter 133
- pipeline/stats.py — added iter 134
- scripts/tools/ — 51 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100); refresh_data.py fully re-scanned iter 135
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- scripts/databento_backfill.py — added iter 135
- research/ — 19 files (iters 101-111): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105); research_zt_event_viability.py (iter 106); research_vol_regime_switching.py (iter 107); research_edge_structure.py, research_1015_vs_1000.py (iters 108-109); research_overlap_analysis.py (iter 110); research_aperture_scan.py (iter 111)
- research/ additional (iters 112-113): research_alt_stops.py, research_direction_asymmetry.py, research_signal_stack.py (iter 112); discover.py, research_wf_stress_keepers.py, research_trend_day_mfe.py scanned but already clean (iter 113)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 199 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `pipeline/audit_log.py` — MEDIUM tier, unscanned pipeline audit logging (2 importers)
- `pipeline/ingest_dbn_mgc.py` — MEDIUM tier, unscanned instrument ingestion (3 importers)
- `trading_app/ml/evaluate.py` — LOW tier, unscanned ML evaluation path (1 importer)
- `trading_app/ml/evaluate_validated.py` — LOW tier, unscanned ML evaluation validation path (1 importer)

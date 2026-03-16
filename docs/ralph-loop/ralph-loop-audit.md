# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 108

## RALPH AUDIT — Iteration 108 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean after fixes |

---

## Files Audited This Iteration

### research/research_edge_structure.py — FIXED (ES-01 through ES-05)

Seven Sins scan:
- Silent failure: No bare `except: pass`. scipy `ImportError` guarded with `HAS_SCIPY` flag — not silent. CLEAN.
- Fail-open: No exception handlers that return success. CLEAN.
- Look-ahead bias: Standalone bar scanner from raw 1m bars; no DB outcome lookups; no future data. CLEAN.
- Cost illusion: Research script — overlap/structure analysis, no P&L vs cost model required. CLEAN.
- Canonical violation: `INSTRUMENTS = ["MGC", "MNQ", "MES"]` hardcoded (line 98) — missing M2K. However this script is a standalone research tool analysing session overlap structure for three instruments; M2K was not part of the overlap study scope. ACCEPTABLE per WF-05 pattern (one-off diagnostic, returning 0 rows if instrument absent is not dangerous).
- Orphan risk: `csv` import present and used (CSV output). `_opens` intentionally discarded via naming. CLEAN.
- Volatile data: No hardcoded counts. CLEAN.
- DB path: Uses `pipeline.paths.GOLD_DB_PATH` with graceful `ImportError` fallback. CLEAN.
- **ES-01 (FIXED, LOW/MECHANICAL):** I001 import sort.
- **ES-02 (FIXED, LOW/MECHANICAL):** 30x F541 extraneous f-string prefixes on print statements.
- **ES-03 (FIXED, LOW/MECHANICAL):** F841 unused `drop` variable.
- **ES-04 (FIXED, LOW/MECHANICAL):** B007 unused loop variables.
- **ES-05 (FIXED, LOW/MECHANICAL):** B023 `band_stats` closes over `regime_days` loop variable — fixed by adding `days` parameter and passing `regime_days` explicitly at call sites.

### research/research_1015_vs_1000.py — FIXED (ES-06 through ES-07, batched)

Seven Sins scan:
- Silent failure: No bare `except: pass`. CLEAN.
- Fail-open: No handlers returning success on error. CLEAN.
- Look-ahead bias: Standalone bar scanner; no DB outcomes used; no future data leak. CLEAN.
- Cost illusion: Research script — aperture comparison, no cost model needed. CLEAN.
- Canonical violation: `INSTRUMENTS = ["MNQ", "MES", "MGC"]` hardcoded — missing M2K. ACCEPTABLE per WF-05 pattern (1000 vs 1015 aperture comparison is Japan-timezone CLEAN sessions; M2K not relevant to this specific study).
- Orphan risk: `argparse` imported and used. CLEAN.
- Volatile data: No hardcoded counts. CLEAN.
- DB path: Uses `pipeline.paths.GOLD_DB_PATH` with graceful `ImportError` fallback. CLEAN.
- **ES-06 (FIXED, LOW/MECHANICAL):** I001 import sort, 10x F541 extraneous f-strings, F841 unused `n` and `window_mins` assignments, 6x E702 semicolons split to two-line form. Auto-fixed + manual removal.
- **ES-07 (FIXED, LOW/MECHANICAL):** B023 `bar_stats` closes over `volumes`, `highs`, `lows`, `closes`, `opens`, `m`, `start_1000` loop variables — fixed by moving `bar_stats` outside the loop body and passing all closed-over names as explicit parameters. Also resolved incidental E741 (ambiguous `l` renamed to `lv_bar`).

---

## Summary
- 2 targets reviewed (batched): research/research_edge_structure.py + research/research_1015_vs_1000.py
- 7 findings fixed (ES-01 through ES-07): all LOW/mechanical ruff cleanup
- 0 findings deferred
- 2 ACCEPTABLE canonical violations (WF-05 pattern — one-off diagnostic scripts, M2K not in study scope)
- Infrastructure Gates: 3/3 PASS

**Codebase steady state maintained for major violation classes:**
- Hardcoded DB paths: ELIMINATED (0 in production code; research/ now consistent)
- Hardcoded apertures [5,15,30]: ELIMINATED from production (7th fix applied)
- Hardcoded entry model IN clauses: ELIMINATED (research_vol_regime_switching.py both SQL sites now fixed)
- Hardcoded instrument lists: ELIMINATED (1 acceptable with assertion guard in production; research scripts with scope-limited studies ACCEPTABLE per WF-05)
- SESSION_ORDER coverage: COMPLETE across all scripts (12/12 sessions in all SESSION_ORDER lists)

**Remaining LOW-priority items:**
- ~22 CLI scripts with connection leaks (process exit closes them)
- DF-04 deferred: rolling_portfolio.py dormant orb_minutes=5 (annotated TODO, design work required)

---

## Files Fully Scanned

> Cumulative list — 163 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 17 files (iters 101-108): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105); research_zt_event_viability.py (iter 106); research_vol_regime_switching.py (iter 107); research_edge_structure.py, research_1015_vs_1000.py (iter 108)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 163 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_overlap_analysis.py` — unscanned; likely has similar ruff violations; full Seven Sins scan
- `research/research_session_clustering.py` — unscanned; full Seven Sins scan
- DF-04 remains open but annotated — do not re-investigate unless rolling portfolio is extended to multi-aperture

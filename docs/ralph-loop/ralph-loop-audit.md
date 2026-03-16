# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 109

## RALPH AUDIT — Iteration 109 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean (both target files) |
| `pyright` (targeted) | PASS | 0 errors, 0 warnings on both target files |

---

## Files Audited This Iteration

### research/research_edge_structure.py + research/research_1015_vs_1000.py — FIXED (PE-01 through PE-07)

Seven Sins scan: carried forward from iter 108 (both files fully scanned then). No new sins introduced by these fixes.

Note: iter 108 scan incorrectly noted `csv` import as used for CSV output in research_edge_structure.py — PE-04 corrects this; `csv.` was never referenced in the file body. The fix removes the dead import.

**PE-01 (FIXED, LOW/MECHANICAL):** `research_edge_structure.py:59` — `dt.utcoffset()` returns `timedelta | None`; `.total_seconds()` called directly. Fixed: extract to `offset` local, assert not None before call.

**PE-02 (FIXED, LOW/MECHANICAL):** `research_edge_structure.py:66` — same `utcoffset()` None pattern in `is_uk_dst()`. Same assert-guard fix applied.

**PE-03 (FIXED, LOW/MECHANICAL):** `research_edge_structure.py:720-721` — `pearsonr()` return type typed as `PearsonRResult` whose tuple elements carry `_T_co@tuple | float`, causing numpy ufunc argument mismatch when passed to `np.isnan()`. Fixed: `_res = pearsonr(...); orb_r, orb_p = float(_res[0]), float(_res[1])`.

**PE-04 (FIXED, LOW/MECHANICAL):** `research_edge_structure.py:28` — `import csv` present but `csv.` never referenced. Removed.

**PE-05 (FIXED, LOW/MECHANICAL):** `research_edge_structure.py:400` — `all_days` unpacked but not used in Q1 loop body (only `highs`/`lows`/`closes` consumed). Renamed to `_all_days`. Other unpack sites (lines 484, 650) that do use `all_days` are untouched.

**PE-06 (FIXED, LOW/MECHANICAL):** `research_1015_vs_1000.py:44` — same `utcoffset()` None pattern in `is_us_dst()`. Same assert-guard fix applied.

**PE-07 (FIXED, LOW/MECHANICAL):** `research_1015_vs_1000.py` — `all_days` unpacked but unused at 3 sites: q2 (line 312), q3 (line 398), q4 (line 482). Renamed to `_all_days` at all three. q1 at line 261 legitimately uses `all_days` in a `len()` print — left unchanged.

---

## Summary
- 2 targets patched (batched): research/research_edge_structure.py + research/research_1015_vs_1000.py
- 7 findings fixed (PE-01 through PE-07): all LOW/mechanical Pyright type-guard fixes
- 0 findings deferred
- Infrastructure Gates: 4/4 PASS (0 errors, 0 warnings from pyright on both files)

**Codebase steady state maintained for major violation classes:**
- Hardcoded DB paths: ELIMINATED (0 in production code; research/ consistent)
- Hardcoded apertures [5,15,30]: ELIMINATED from production
- Hardcoded entry model IN clauses: ELIMINATED
- Hardcoded instrument lists: ELIMINATED (research scripts with scope-limited studies ACCEPTABLE per WF-05)
- SESSION_ORDER coverage: COMPLETE (12/12 sessions)
- Pyright errors in research/: ELIMINATED for all scanned files

**Remaining LOW-priority items:**
- ~22 CLI scripts with connection leaks (process exit closes them)
- DF-04 deferred: rolling_portfolio.py dormant orb_minutes=5 (annotated TODO, design work required)

---

## Files Fully Scanned

> Cumulative list — 163 files fully scanned (no new files added this iteration — re-fix of previously scanned files).

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 17 files (iters 101-109): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105); research_zt_event_viability.py (iter 106); research_vol_regime_switching.py (iter 107); research_edge_structure.py, research_1015_vs_1000.py (iters 108-109)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 163 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_overlap_analysis.py` — unscanned; likely has similar ruff/Pyright violations; full Seven Sins scan
- `research/research_session_clustering.py` — unscanned; full Seven Sins scan
- DF-04 remains open but annotated — do not re-investigate unless rolling portfolio is extended to multi-aperture

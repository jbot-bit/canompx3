# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 111

## RALPH AUDIT — Iteration 111 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean after auto-fix (17 F541 + 1 I001 fixed) |
| `check_drift.py` post-fix | PASS | 72/72 PASS |

---

## Files Audited This Iteration

### research/research_aperture_scan.py — FIXED (AS-01, AS-02)

Seven Sins scan:
- Silent failure: None — `con.close()` in `finally` block; empty data returns early with message
- Fail-open: None — no exception swallowing; result=None treated as empty
- Look-ahead bias: None — standalone vectorized scan, break detection uses first close AFTER ORB ends (break_start = start_min + aperture_min), outcome window does not reference future data beyond the outcome window
- Cost illusion: N/A — no P&L in R-units beyond raw avg_r (research only, no cost deduction needed for screening scan)
- Canonical violation: `INSTRUMENTS = ["MGC", "MNQ", "MES", "MCL"]` line 77 includes dead instrument MCL — ACCEPTABLE per WF-05 pattern (read-only investigation; MCL returns 0 bars → prints "No data, skipping"; no correctness impact). Sessions dict uses fixed Brisbane clock times (intentional for this research — testing raw time slots, not production sessions).
- Orphan risk: `opens` array allocated but never passed to scan_session_aperture (only highs/lows/closes used). Line 661: `all_days, opens, highs, lows, closes = build_day_arrays(...)`. `opens` is built but unused. Deleted via `del opens, highs, lows, closes` at line 739. LOW/ACCEPTABLE — orphan array, no correctness impact, memory is released.
- Volatile data: `n_combos` computed dynamically from len(INSTRUMENTS)*len(SESSIONS)*len(APERTURES) — correct.

**AS-01 (FIXED, LOW/MECHANICAL):** `research_aperture_scan.py:32` — I001 unsorted import block. Fixed: `ruff check --fix`.

**AS-02 (FIXED, LOW/MECHANICAL):** `research_aperture_scan.py:513,547,559,572,587-592,603-608,636` — 17 F541 bare f-strings without any placeholders in `print_honest_summary()` and `main()`. Fixed: `ruff check --fix` removed extraneous `f` prefix.

**AS-03 (ACCEPTABLE):** `research_aperture_scan.py:77` — `INSTRUMENTS = ["MGC", "MNQ", "MES", "MCL"]` includes dead instrument MCL. Per WF-05 pattern: read-only investigation script; MCL has no bars data → `load_bars` returns empty df → script prints "No data for MCL, skipping". No correctness impact.

**AS-04 (ACCEPTABLE):** `research_aperture_scan.py:661` — `opens` array allocated in `build_day_arrays` return but never passed to `scan_session_aperture`. Array is released via `del` at line 739. Style difference with no correctness impact — scan only needs highs/lows/closes.

Note: `research_session_stats.py` does not exist — skip.

---

## Summary
- 1 target patched: research/research_aperture_scan.py
- 2 finding groups fixed (AS-01 + AS-02): all LOW/mechanical ruff auto-fixes (1 I001 + 17 F541)
- 2 findings ACCEPTABLE (AS-03 + AS-04): dead instrument skip, unused array (no correctness impact)
- 0 findings deferred
- Infrastructure Gates: 4/4 PASS (ruff clean, check_drift 72/72 PASS)

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

> Cumulative list — 165 files fully scanned (1 new file added this iteration).

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 19 files (iters 101-111): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105); research_zt_event_viability.py (iter 106); research_vol_regime_switching.py (iter 107); research_edge_structure.py, research_1015_vs_1000.py (iters 108-109); research_overlap_analysis.py (iter 110); research_aperture_scan.py (iter 111)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 165 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- Remaining unscanned research/ files: glob research/ to find any not yet in the scanned list
- DF-04 remains open but annotated — do not re-investigate unless rolling portfolio is extended to multi-aperture

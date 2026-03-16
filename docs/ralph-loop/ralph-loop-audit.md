# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 106

## RALPH AUDIT — Iteration 106 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean (target file) |

---

## Files Audited This Iteration

### research/research_zt_event_viability.py — FIXED (ZT-01)

Seven Sins scan:
- Canonical: No gold.db access — reads raw DBN files via `get_asset_config("ZT")["dbn_path"]` ✓
- `"ZT"` hardcoded throughout: intentional — this is a ZT-specific viability study for a prospective instrument. ZT is not in the active ORB instruments list (MGC/MNQ/MES/M2K). ACCEPTABLE.
- `_FOMC_DATES_RAW`, `build_cpi_set()`, `is_nfp_day()` from `pipeline.calendar_filters`: canonical source. CLEAN.
- Silent failure: no bare exception handlers. Script raises `SystemExit` on fatal conditions (empty dir). Returns `None` for missing windows and checks before use. CLEAN.
- Fail-open: no exception handlers returning success. CLEAN.
- Look-ahead bias: sequential windows only (pre → shock → follow). No ORB aperture mixing (no gold.db usage). CLEAN.
- Cost illusion: tick-based economics study, not a P&L backtest. No cost model needed. CLEAN.
- Orphan risk: all imports used. CLEAN.
- `usable == True` comparisons (lines 264, 345, 346): flagged `# noqa: E712` intentionally (pandas comparison quirk). CLEAN.
- **ZT-01 (FIXED, LOW):** Line 358 — hardcoded `"18"` variation count in markdown report template. Count is derived from `len(EVENT_FAMILIES) × len(follow_windows) × 2 models` = 18, but was a static string literal. If EVENT_FAMILIES structure changes, the report would silently diverge from the actual tested count. Fixed by computing `variation_count` dynamically and using f-string interpolation.

---

## Summary
- 1 target reviewed: research/research_zt_event_viability.py
- 1 finding fixed (ZT-01: volatile data count in report template)
- Infrastructure Gates: 3/3 PASS

**Codebase steady state maintained for major violation classes:**
- Hardcoded DB paths: ELIMINATED (0 in production code; research/ now consistent)
- Hardcoded apertures [5,15,30]: ELIMINATED from production (7th fix applied)
- Hardcoded entry model IN clauses: ELIMINATED
- Hardcoded instrument lists: ELIMINATED (1 acceptable with assertion guard in production; 1 acceptable in read-only diagnostic)
- SESSION_ORDER coverage: COMPLETE across all scripts (12/12 sessions in all SESSION_ORDER lists)

**Remaining LOW-priority items:**
- ~22 CLI scripts with connection leaks (process exit closes them)
- DF-04 deferred: rolling_portfolio.py dormant orb_minutes=5 (annotated TODO, design work required)

---

## Files Fully Scanned

> Cumulative list — 160 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 14 files (iters 101-106): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105); research_zt_event_viability.py (iter 106)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 160 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_vol_regime_switching.py` — highest ruff violation count (45) in non-archive research files; full Seven Sins scan
- Batch-triage: `research_edge_structure.py` (37 violations), `research_1015_vs_1000.py` (28 violations)
- DF-04 remains open but annotated — do not re-investigate unless rolling portfolio is extended to multi-aperture

# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 107

## RALPH AUDIT — Iteration 107 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean after fixes |

---

## Files Audited This Iteration

### research/research_vol_regime_switching.py — FIXED (VS-01 through VS-05)

Seven Sins scan:
- Canonical: DB path via `GOLD_DB_PATH` from `pipeline.paths`. Instruments via `ACTIVE_ORB_INSTRUMENTS`. Entry models via local `ENTRY_MODELS` (now used in both SQL IN clauses). CLEAN after fix.
- Silent failure: `except Exception` in `get_validated_sessions()` prints WARNING and falls back to enabled sessions. Not silent. ACCEPTABLE for pre-scan fallback in research context.
- Fail-open: `except Exception` in kruskal block (line 424) → sets `kw_p = float("nan")` → filtered from kw_valid downstream. Not fail-open — NaN propagated correctly. ACCEPTABLE.
- Look-ahead bias: `shift(1)` on expanding percentile rank with 60-day warmup. Spot-check verification included. CLEAN.
- Cost illusion: regime switching study, no P&L backtest against cost model needed. CLEAN.
- Orphan risk: `import datetime` added in prior interrupted session but date fix was incomplete — both resolved this iteration.
- Volatile data: hardcoded `2026-03-01` in output — fixed to `datetime.date.today().isoformat()`.
- **VS-01–VS-04 (FIXED, LOW/MECHANICAL):** 45 ruff violations — import sort (I001), unused `os` import, 4x unused loop control variables (B007), 40x extraneous f-string prefixes (F541). Auto-fixed via `ruff --fix` + manual B007 renames. Applied in prior interrupted session; verified clean this iteration.
- **VS-05 (FIXED, MEDIUM):** Line 192 in `load_data()` — `AND o.entry_model IN ('E1', 'E2')` hardcoded. Second IN clause not updated when `get_validated_sessions` was fixed. Fixed to use ENTRY_MODELS f-string expansion: `{", ".join(f"'{m}'" for m in ENTRY_MODELS)}`.
- **VS-04 (FIXED, LOW):** Line 784 — `"**Date:** 2026-03-01\n"` hardcoded. Fixed to `f"**Date:** {datetime.date.today().isoformat()}\n"`, also resolving the unused `datetime` import.
- **VS-03 (ACCEPTABLE):** Line 424 — bare `except Exception` on `stats.kruskal()`. Sets `kw_p = float("nan")`, filtered downstream. Research script, not fail-open. Matches ACCEPTABLE rule #1 (research context, not production path).

---

## Summary
- 1 target reviewed: research/research_vol_regime_switching.py
- 5 findings fixed (VS-01 through VS-05: ruff cleanup, canonical ENTRY_MODELS, dynamic date)
- 1 finding ACCEPTABLE (VS-03: kruskal except → NaN, research context)
- Infrastructure Gates: 3/3 PASS

**Codebase steady state maintained for major violation classes:**
- Hardcoded DB paths: ELIMINATED (0 in production code; research/ now consistent)
- Hardcoded apertures [5,15,30]: ELIMINATED from production (7th fix applied)
- Hardcoded entry model IN clauses: ELIMINATED (research_vol_regime_switching.py both SQL sites now fixed)
- Hardcoded instrument lists: ELIMINATED (1 acceptable with assertion guard in production; 1 acceptable in read-only diagnostic)
- SESSION_ORDER coverage: COMPLETE across all scripts (12/12 sessions in all SESSION_ORDER lists)

**Remaining LOW-priority items:**
- ~22 CLI scripts with connection leaks (process exit closes them)
- DF-04 deferred: rolling_portfolio.py dormant orb_minutes=5 (annotated TODO, design work required)

---

## Files Fully Scanned

> Cumulative list — 161 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 15 files (iters 101-107): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105); research_zt_event_viability.py (iter 106); research_vol_regime_switching.py (iter 107)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 161 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_edge_structure.py` — 37 ruff violations; full Seven Sins scan
- `research/research_1015_vs_1000.py` — 28 ruff violations; full Seven Sins scan
- DF-04 remains open but annotated — do not re-investigate unless rolling portfolio is extended to multi-aperture

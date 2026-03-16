# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 104

## RALPH AUDIT — Iteration 104 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean (target file) |

---

## Files Audited This Iteration

### research/research_mgc_mnq_correlation.py — FIXED (RC-01, RC-02)

Seven Sins scan:
- Canonical: DB path uses `GOLD_DB_PATH` from `pipeline.paths` ✓
- `SHARED_SESSIONS` hardcoded list (8 sessions): intentional research scope — this script studies cross-instrument concordance for these two specific instruments (MGC/MNQ) only. ACCEPTABLE.
- `WHERE symbol IN ('MGC', 'MNQ')`: intentional single-pair research scope. ACCEPTABLE.
- `orb_minutes = 5` hardcoded in queries: single-aperture research design (fair cross-instrument comparison). Documented in docstring. ACCEPTABLE.
- Silent failure: no bare exception handlers. Functions return empty/skip on insufficient data, explicitly logged. CLEAN.
- Fail-open: no exception handlers returning success. `con.close()` in finally block. CLEAN.
- Look-ahead bias: N/A — research analysis script, not a strategy. No future data as predictor. CLEAN.
- Cost illusion: N/A — uses pre-computed `pnl_r` from `orb_outcomes`. CLEAN.
- Volatile data: no hardcoded counts. Dynamic. CLEAN.
- **F541 (FIXED, RC-01):** `print(f"  Interpretation: ", end="")` at line 158 — f-string without any placeholders. Removed extraneous `f` prefix.
- **B905 x2 (FIXED, RC-02):** `zip(valid, p_adj)` and `zip(results, p_adj)` at lines 309, 433 without `strict=`. Both cases: `p_adj = bh_fdr(p_vals)` where `p_vals` is built from the same list — always equal length. Added `strict=False` to make intent explicit.

**Finding RC-01 (FIXED):** F541 — extraneous f-prefix on print string. Mechanical. 1-line fix.
**Finding RC-02 (FIXED):** B905 x2 — zip without strict= in BH FDR result-merge loops. Mechanical. 2-line fix.

---

## Deferred Findings — Status After Iter 104

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — annotated TODO, fix requires design work on orb_label→aperture mapping

---

## Summary
- 1 target reviewed: research_mgc_mnq_correlation.py (3 ruff violations fixed)
- 2 findings fixed (RC-01: F541 f-string; RC-02: B905 zip strict=False x2)
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

> Cumulative list — 157 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 11 files (iters 101-104): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 157 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_atr_velocity_gate.py` — unscanned (run `ruff check` to triage first)
- `research/research_mgc_regime_shift.py` — unscanned
- Batch-triage remaining unscanned research/ files with `ruff check research/` to prioritize by violation count
- DF-04 remains open but annotated — do not re-investigate unless rolling portfolio is extended to multi-aperture

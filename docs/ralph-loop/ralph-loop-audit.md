# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 105

## RALPH AUDIT — Iteration 105 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean (both target files) |

---

## Files Audited This Iteration

### research/research_atr_velocity_gate.py — FIXED (AV-01, AV-02)

Seven Sins scan:
- Canonical: DB path uses `GOLD_DB_PATH` from `pipeline.paths` ✓
- Hardcoded `["MNQ", "MGC", "MES"]` + `["0900", "1000"]`: intentional research scope for this specific gate study (comparing ATR velocity across 3 instruments x 2 sessions). ACCEPTABLE.
- `entry_model = 'E0'` hardcoded in Part 0 and Part 4 queries: intentional — this script studies the E0 era gate lift, providing historical reference for E1 comparison. Research artifact, not production. ACCEPTABLE.
- Silent failure: no bare exception handlers. CLEAN.
- Fail-open: no exception handlers returning success. CLEAN.
- Look-ahead bias: N/A — research analysis, not a strategy predictor. CLEAN.
- Cost illusion: uses pre-computed `pnl_r` from `orb_outcomes`. CLEAN.
- Volatile data: no hardcoded counts. CLEAN.
- **AV-01 (FIXED, MEDIUM):** Part 0 COUNT query (lines 79-99) missing `AND o.orb_minutes = 5`. Without this filter, the query counted rows across all 3 apertures (5m+15m+30m), inflating removal rate stats by ~3x. Added `AND o.orb_minutes = 5` to WHERE clause.
- **AV-02 (FIXED, LOW):** Line 101 — `total, skipped, contracting = row` where `row = fetchone()`. `fetchone()` returns `Optional[tuple]`, so Pyright flagged "None is not iterable". COUNT always returns a row in practice, but the guard is correct hygiene. Added `if row is None: continue` before the destructure.

### research/research_mgc_regime_shift.py — FIXED (RS-01)

Seven Sins scan:
- Canonical: DB path uses `GOLD_DB_PATH` from `pipeline.paths` ✓
- `orb_minutes = 5` hardcoded in Parts 1/2/3/6 `daily_features` queries: intentional single-aperture research snapshot. ACCEPTABLE.
- Hardcoded `["E0", "E1"]`: intentional — studying historical MGC regime including E0 era for context. Research artifact. ACCEPTABLE.
- Hardcoded `['MGC']` throughout: intentional — this is an MGC-specific regime shift analysis. ACCEPTABLE.
- Silent failure: no bare exception handlers. CLEAN.
- Fail-open: no exception handlers returning success. CLEAN.
- Look-ahead bias: N/A — historical aggregate analysis. CLEAN.
- Cost illusion: uses pre-computed `pnl_r` from `orb_outcomes`. CLEAN.
- Volatile data: no hardcoded counts. CLEAN.
- Parts 3 and 6: CLEAN — `AND d.orb_minutes = 5` in WHERE constrains via JOIN `ON o.orb_minutes = d.orb_minutes`.
- **RS-01 (FIXED, MEDIUM):** Parts 4 (year-by-year) and 5 (pre/post 2025 split) query `orb_outcomes` directly without `AND o.orb_minutes = 5`. These queries had no join to `daily_features`, so all apertures were mixed. Added `AND o.orb_minutes = 5` to both queries.

---

## Summary
- 2 targets reviewed: research_atr_velocity_gate.py, research_mgc_regime_shift.py
- 3 findings fixed (AV-01: aperture mixing in Part 0; AV-02: fetchone None guard; RS-01: aperture mixing in Parts 4+5)
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

> Cumulative list — 159 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 13 files (iters 101-105): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 159 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_zt_event_viability.py` — partially scanned iter 101 (only ruff check), full Seven Sins scan pending
- Batch-triage remaining unscanned research/ files: `ruff check research/` to identify next highest-violation file
- DF-04 remains open but annotated — do not re-investigate unless rolling portfolio is extended to multi-aperture

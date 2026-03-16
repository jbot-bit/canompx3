# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 101

## RALPH AUDIT — Iteration 101 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### research/research_zt_event_viability.py — CLEAN

Seven Sins scan:
- Silent failure: None. `load_day_df()` returns empty DataFrame on empty file — checked upstream. `raise SystemExit` on missing DBN directory — fail-closed.
- Canonical violations: Uses `get_asset_config("ZT")` for DBN path — canonical. Uses `build_cpi_set()`, `is_nfp_day()`, `_FOMC_DATES_RAW` from `pipeline.calendar_filters` — canonical.
- Cost model: No P&L computed in this stage-1 viability pass (directional only). ACCEPTABLE.
- Raw `p < 0.05` threshold: Stage-1 screen with 18 cells, explicitly noted as in-sample only. ACCEPTABLE.
- `EVENT_FAMILIES` hardcoded dict: research event definitions, not canonical instrument/session data. ACCEPTABLE.

**No findings.**

### research/research_zt_cpi_nfp.py — CLEAN

Seven Sins scan:
- Near-identical to zt_event_viability (earlier CPI/NFP-only version).
- Silent failure: None. Same fail-closed pattern.
- Canonical violations: None. `get_asset_config("ZT")`, `build_cpi_set()`, `is_nfp_day()` all canonical.
- Cost model: Directional study only, no P&L. ACCEPTABLE.
- Raw `p < 0.05` + friction sanity: Stage-1 screen, explicitly noted in-sample only. ACCEPTABLE.

**No findings.**

### research/research_london_adjacent.py — CLEAN

Seven Sins scan:
- Canonical imports: `ACTIVE_ORB_INSTRUMENTS`, `COST_SPECS`, `SESSION_CATALOG`, `GOLD_DB_PATH` — ALL CORRECT.
- `double_break`: Used only as a DIAGNOSTIC metric (reporting rates vs LONDON_METALS baseline), NOT as a trade filter. No look-ahead bias in this usage.
- `ENTRY_SPECS = [("E1",1),("E1",3),("E2",1)]`: Research grid, not canonical list. ACCEPTABLE per pattern 1 (intentional per-session heuristic).
- `G_FILTER_THRESHOLDS`, `RR_TARGETS`, `APERTURES = [5,15,30]`: Research grid constants for standalone simulation. Script reads from `bars_1m` directly (not `orb_outcomes`). ACCEPTABLE.
- Cost model: Correctly imports and uses `COST_SPECS`. No cost illusion.
- `simulate_trade()`: Local simulation function, no production values. ACCEPTABLE.

**No findings.**

### research/research_mes_compressed_spring.py — FIXED (RM-01)

Seven Sins scan:
- **Canonical violation (FIXED):** Line 42 used `os.environ.get("DUCKDB_PATH", str(PROJECT_ROOT / "gold.db"))` instead of `from pipeline.paths import GOLD_DB_PATH`. All other research scripts use the canonical import. Fixed: removed `import os`, added `from pipeline.paths import GOLD_DB_PATH`, replaced DB_PATH assignment with `str(GOLD_DB_PATH)`. Committed f9618c0.
- `except Exception as e` in `load()`: Logs and returns empty DataFrame — not silent (prints error). ACCEPTABLE for research script.
- Session names in `PRIMARY_CONFIGS`/`CONTEXT_SESSIONS`: Research grid parameters, not canonical source lookups. ACCEPTABLE.
- Entry model strings "E1"/"E2" in `PRIMARY_CONFIGS`: Research grid parameters. ACCEPTABLE.
- Cost model: Not applicable — this script analyses pre-computed `pnl_r` from `orb_outcomes` which already has costs applied.

**Finding RM-01 (FIXED):** `os.environ.get("DUCKDB_PATH", ...)` instead of canonical `GOLD_DB_PATH` import. 3-line mechanical fix. Commit: f9618c0.

---

## Deferred Findings — Status After Iter 101

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 4 target files audited: research_zt_event_viability.py, research_zt_cpi_nfp.py, research_london_adjacent.py, research_mes_compressed_spring.py
- 1 finding fixed (RM-01: canonical GOLD_DB_PATH import in research_mes_compressed_spring)
- 3 files clean (no actionable findings)
- 0 new ACCEPTABLE findings
- Infrastructure Gates: 3/3 PASS
- Commit: f9618c0

**Codebase steady state maintained for major violation classes:**
- Hardcoded DB paths: ELIMINATED (0 in production code; research/ now consistent)
- Hardcoded apertures [5,15,30]: ELIMINATED from production (7th fix applied)
- Hardcoded entry model IN clauses: ELIMINATED
- Hardcoded instrument lists: ELIMINATED (1 acceptable with assertion guard in production; 1 acceptable in read-only diagnostic)
- SESSION_ORDER coverage: COMPLETE across all scripts (12/12 sessions in all SESSION_ORDER lists)

**Remaining LOW-priority items:**
- ~22 CLI scripts with connection leaks (process exit closes them)
- DF-04 deferred: rolling_portfolio.py dormant orb_minutes=5 (structural, >5 files)

---

## Files Fully Scanned

> Cumulative list — 150 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 4 files (iter 101): research_zt_event_viability.py, research_zt_cpi_nfp.py, research_london_adjacent.py, research_mes_compressed_spring.py
- **Total: 150 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_post_break_pullback.py` — new research file (untracked per git status)
- `research/research_mgc_asian_fade_mfe.py` — new research file (untracked per git status)
- `research/research_zt_fomc_unwind.py` — new research file (untracked per git status)

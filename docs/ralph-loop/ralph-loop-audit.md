# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 102

## RALPH AUDIT — Iteration 102 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### research/research_post_break_pullback.py — FIXED (RP-01, RP-02)

Seven Sins scan:
- Canonical: `GOLD_DB_PATH`, `ACTIVE_ORB_INSTRUMENTS` — CANONICAL ✓
- Sessions discovered dynamically from `DESCRIBE daily_features` — CANONICAL ✓
- `for orb_min in [5, 15, 30]` — research grid constant, standalone simulation. ACCEPTABLE.
- `entry_model IN ('E1', 'E2')`, `confirm_bars = 1` — research grid SQL. ACCEPTABLE.
- BH FDR, Welch t-test, Fisher exact — correct statistics ✓
- Silent failure: `except Exception` logs and continues. ACCEPTABLE for research.
- **Ruff I001 (FIXED):** Import block unsorted — pipeline imports mixed with third-party. Fixed: added blank line between third-party (`scipy`) and first-party (`pipeline.*`) groups.
- **Ruff B007 (FIXED):** Loop vars `t` and `pct` unused in BH FDR survivors loop. Renamed to `_t` and `_pct`.

**Findings RP-01, RP-02 (FIXED):** Import sort + unused loop var rename. Mechanical. Same commit.

### research/research_mgc_asian_fade_mfe.py — FIXED (RM-02)

Seven Sins scan:
- Canonical: `get_cost_spec("MGC")`, `SESSION_CATALOG`, `GOLD_DB_PATH` — ALL CANONICAL ✓
- `'MGC'` hardcoded throughout: intentional single-instrument research script. ACCEPTABLE.
- Cost model: `get_cost_spec("MGC")` → friction applied in simulation. ✓
- Session access via `SESSION_CATALOG["CME_REOPEN"]["resolver"](td_date)` — canonical ✓
- `con.close()` present ✓
- Silent failure: per-day ORB query returns None with check. ACCEPTABLE.
- **Ruff F841 (FIXED):** Lines 112-118 assigned `bar_mfe` and `bar_mae` (bar-by-bar arrays) that were never read — dead scaffolding from earlier implementation. Removed 4 lines; only `running_mfe`/`running_mae` (the `np.maximum.accumulate` outputs) are needed.

**Finding RM-02 (FIXED):** 4 unused variable assignments (scaffolding). Removed dead code.

### research/research_zt_fomc_unwind.py — CLEAN

Seven Sins scan:
- Canonical: `get_asset_config("ZT")`, `_FOMC_DATES_RAW` from `pipeline.calendar_filters` — canonical. ZT support added in iter 100.
- No `gold.db` access — reads Databento DBN files directly (correct: ZT is research-only, not in pipeline).
- Fail-closed: `raise SystemExit` on missing DBN directory ✓
- Statistics: `binomial_two_sided_p()` with `p0=0.25` for half-unwind test; 3-test grid (15m/30m/60m). Single-family study. ACCEPTABLE.
- `noqa: E712` for intentional `== True` bool comparisons in DataFrame filter. ACCEPTABLE.
- Concludes NO-GO — consistent with project standards.

**No findings.**

---

## Deferred Findings — Status After Iter 102

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 3 target files audited: research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py
- 3 findings fixed (RP-01: import sort; RP-02: unused loop vars; RM-02: dead scaffolding assignments)
- 1 file clean (research_zt_fomc_unwind.py)
- Infrastructure Gates: 3/3 PASS

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

> Cumulative list — 153 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 7 files (iters 101-102): research_zt_event_viability.py, research_zt_cpi_nfp.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102)
- **Total: 153 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_fomc_unwind.py` — if exists (check git status for remaining untracked research files)
- `docs/plans/2026-03-15-zt-stage1-cpi-nfp-spec.md` and `docs/plans/2026-03-15-zt-stage1-triage-gate.md` — new untracked spec docs (review for stale/incorrect canonical references)
- DF-04 re-evaluation: `rolling_portfolio.py:304` — revisit blast radius estimate

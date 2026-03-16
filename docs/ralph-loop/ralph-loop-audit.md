# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 103

## RALPH AUDIT — Iteration 103 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### research/research_zt_cpi_nfp.py — FIXED (RZ-01)

Seven Sins scan:
- Canonical: `get_asset_config("ZT")` for dbn_path, `build_cpi_set`/`is_nfp_day` from `pipeline.calendar_filters` — CANONICAL ✓
- `"ZT"` hardcoded as instrument string: intentional single-instrument research script. ACCEPTABLE.
- Fail-closed: `raise SystemExit` on missing DBN directory ✓
- Silent failure: explicit exclusion tracking per event (`missing_daily_file`, `empty_day`, `missing_window_data`, `zero_shock`). CLEAN.
- Fail-open: no exception handlers that return success. CLEAN.
- Look-ahead bias: not applicable — event study, not ORB strategy. CLEAN.
- Cost illusion: friction sanity uses inferred tick size heuristic (avg_ticks > 2.0). Spec explicitly documents this is not a full cost model. ACCEPTABLE.
- Volatile data: no hardcoded counts. CLEAN.
- **Ruff B905 (FIXED):** `zip(uniq, uniq[1:])` without `strict=` in `infer_tick_size`. Lists are deliberately unequal length (consecutive-diff pattern). Added `strict=False` to make intent explicit.

**Finding RZ-01 (FIXED):** zip() without strict= in infer_tick_size. Mechanical. 1-line fix.

### docs/plans/2026-03-15-zt-stage1-cpi-nfp-spec.md — CLEAN (doc review)

- Pure planning/research document. No production code. No canonical violations.
- One markdown hyperlink uses an absolute path (`/mnt/c/users/joshd/...`) — cosmetic doc artifact, no code impact.
- No findings.

### docs/plans/2026-03-15-zt-stage1-triage-gate.md — CLEAN (doc review)

- Pure planning/research document. No production code. No canonical violations.
- Contains example bash commands with `find`/`sed` — doc examples only, no impact.
- No findings.

### research_fomc_unwind.py — DOES NOT EXIST

Target from previous iteration. File not present in repo. Skipped.

### DF-04 Re-evaluation: rolling_portfolio.py:304

`compute_day_of_week_stats` at line 315-326 hardcodes `orb_minutes=5` in `daily_features` query.
- TODO annotation already present and accurate: "When rolling evaluation is extended to 15m/30m ORBs, this must load per family's actual orb_minutes"
- `FamilyResult.orb_label` is a session name, not an aperture minutes value — fix requires architectural decision on how to map session families to their aperture minutes
- Caller: only internal to `rolling_portfolio.py` (line 591), 0 external callers
- Current rolling evaluation only uses 5m families — genuinely dormant
- **REMAINS DEFERRED.** TODO annotation is accurate. Fix requires semantic design work beyond a mechanical change.

---

## Deferred Findings — Status After Iter 103

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — annotated TODO, fix requires design work on orb_label→aperture mapping

---

## Summary
- 4 targets reviewed: research_zt_cpi_nfp.py (fixed), 2 spec docs (clean), DF-04 re-evaluation (still deferred)
- 1 finding fixed (RZ-01: zip strict=False)
- research_fomc_unwind.py: does not exist
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

> Cumulative list — 156 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 10 files (iters 101-103): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 156 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `research/research_mgc_mnq_correlation.py` — unscanned, recent MGC/MNQ correlation work (520 lines)
- Any remaining unscanned research/ files — run `ruff check research/` to triage batch
- DF-04 remains open but annotated — do not re-investigate unless rolling portfolio is extended to multi-aperture

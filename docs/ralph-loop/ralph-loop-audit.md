# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 97

## RALPH AUDIT — Iteration 97 (audit-only)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### scripts/tools/backtest_1100_early_exit.py — CLEAN

Seven Sins scan:
- Line 27: `from pipeline.cost_model import get_cost_spec, to_r_multiple` — canonical cost model. CLEAN.
- Line 28: `from pipeline.paths import GOLD_DB_PATH` — canonical DB path. CLEAN.
- Lines 34, 59-65: Hardcoded `'MGC'`, `'1100'`, `'E1'` in SQL — intentional investigation scope (entire script is scoped to MGC 1100 E1 backtest). Read-only. ACCEPTABLE.
- Line 35: `duckdb.connect(str(db_path), read_only=True)` — read-only connection. CLEAN.
- No silent failures; early exit paths use `continue`, not silent pass. CLEAN.

**No findings.**

### scripts/tools/backtest_atr_regime.py — CLEAN

Seven Sins scan:
- Lines 45-54: `FAMILIES = [...]` hardcoded list — intentional per-family research configuration. The list includes session/em/filter tuples as the explicit scope of investigation. ACCEPTABLE (intentional research scope, not canonical list).
- Line 35: `from pipeline.paths import GOLD_DB_PATH` — canonical DB path. CLEAN.
- Lines 36-42: `ALL_FILTERS`, `ENTRY_MODELS`, `CONFIRM_BARS_OPTIONS`, `RR_TARGETS` — all canonical imports. CLEAN.
- Lines 51-52: `{"session": "LONDON_METALS", "em": "E3", ...}` — E3 soft-retired but present in FAMILIES as research artifact. Read-only research script, no writes. ACCEPTABLE.
- Lines 276-278: `finally: con.close()` — proper connection cleanup. CLEAN.
- No silent failures; empty/insufficient data paths print message and continue. CLEAN.

**No findings.**

### scripts/tools/beginner_tradebook.py — CLEAN

Seven Sins scan:
- Lines 32-44: `TIER1_SESSIONS` and `TIER2_SESSIONS` hardcoded lists — intentional hand-picked curated tradebook sessions. Not a canonical instrument list — it's a personal daily reference. Comment at line 27 explicitly says "Update this list after each rebuild chain." ACCEPTABLE.
- Line 22: `from pipeline.cost_model import COST_SPECS` — canonical. CLEAN.
- Line 23: `from pipeline.paths import GOLD_DB_PATH` — canonical. CLEAN.
- Lines 147, 213, 223: Three DuckDB connections opened in sequence (con, _con, _con2). First closed at 194, second at 219, third at 229. Read-only CLI script — process exit would close any leaks. Redundant connections are LOW/ACCEPTABLE (process-exit cleanup pattern). ACCEPTABLE.
- No silent failures in `get_session_best` — returns None on no rows. CLEAN.

**No findings.**

### scripts/tools/find_pf_strategy.py — CLEAN

Seven Sins scan:
- Lines 36-39: `TARGET_SESSIONS`, `START_DATE`, `END_DATE`, `YEARS_SPAN` hardcoded — intentional 2-year research window configuration. ACCEPTABLE (research scope, not canonical list).
- Line 25: `from pipeline.paths import GOLD_DB_PATH` — canonical. CLEAN.
- Lines 26-33: `ALL_FILTERS`, `ENTRY_MODELS`, `CONFIRM_BARS_OPTIONS`, `RR_TARGETS` — all canonical. CLEAN.
- Lines 85-250: Read-only `try/finally` with `con.close()`. CLEAN.
- No silent failures. CLEAN.

**No findings.**

### scripts/tools/rank_slots.py — CLEAN

Seven Sins scan:
- Lines 19-28: `slots` hardcoded list with specific strategy IDs — intentional research portfolio diagnostic. Read-only. ACCEPTABLE (per-investigation scope, not canonical list).
- Line 14: `from pipeline.cost_model import get_cost_spec` — canonical. CLEAN.
- Line 15: `from pipeline.paths import GOLD_DB_PATH` — canonical. CLEAN.
- Line 17: `con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)` — canonical DB path, module-level connection, read-only. CLEAN.
- Line 196: `con.close()` — proper cleanup. CLEAN.
- Lines 71: Sharpe computed with `np.sqrt(252)` — daily Sharpe annualization, appropriate for portfolio-level daily P&L simulation. Not used for strategy selection. CLEAN.
- No silent failures. CLEAN.

**No findings.**

---

## Deferred Findings — Status After Iter 97

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 5 target files audited: backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py
- 0 findings fixed (audit-only iteration — all files clean)
- 0 new ACCEPTABLE findings (all canonical violations are intentional research scope, no new additions to Won't Fix)
- Infrastructure Gates: 3/3 PASS
- Commit: NONE

**Codebase steady state maintained for major violation classes:**
- Hardcoded DB paths: ELIMINATED (0 in production code)
- Hardcoded apertures [5,15,30]: ELIMINATED from production (7th fix applied)
- Hardcoded entry model IN clauses: ELIMINATED
- Hardcoded instrument lists: ELIMINATED (1 acceptable with assertion guard in production; 1 acceptable in read-only diagnostic)
- SESSION_ORDER coverage: COMPLETE across all scripts (12/12 sessions in all SESSION_ORDER lists)

**Remaining LOW-priority items:**
- ~22 CLI scripts with connection leaks (process exit closes them)
- DF-04 deferred: rolling_portfolio.py dormant orb_minutes=5 (structural, >5 files)

---

## Files Fully Scanned

> Cumulative list — 134 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 38 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- **Total: 134 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `scripts/tools/backtest_compressed_spring.py` — backtest script, orb_1100 session reference worth examining
- `scripts/tools/backtest_early_exit_mfe.py` — backtest script, early exit research
- `scripts/tools/backtest_mgc_stop_opt.py` — stop optimization backtest
- `scripts/tools/backtest_t80_profile.py` — T80 profile backtest
- Any remaining scripts/tools/ backtest_*.py files not yet scanned

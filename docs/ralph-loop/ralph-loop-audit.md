# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 98

## RALPH AUDIT — Iteration 98 (audit-only)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### scripts/tools/parity_check.py — CLEAN

Seven Sins scan:
- Line 31: `from pipeline.cost_model import get_cost_spec` — canonical. CLEAN.
- Line 32: `from pipeline.paths import GOLD_DB_PATH` — canonical. CLEAN.
- Line 33: `from trading_app.config import TRADEABLE_INSTRUMENTS` — canonical instrument list. CLEAN.
- Lines 343-361: try/finally with `con.close()` — proper cleanup. CLEAN.
- Line 373: `checks_per_inst = 4` — local computation count, not a drift check count. CLEAN.
- No silent failures; all failures collected and reported, sys.exit(1) on any failure. CLEAN.

**No findings.**

### scripts/tools/build_outcomes_fast.py — CLEAN

Seven Sins scan:
- Line 29: `from pipeline.cost_model import get_cost_spec` — canonical. CLEAN.
- Line 30: `from pipeline.init_db import ORB_LABELS` — canonical. CLEAN.
- Line 31: `from pipeline.paths import GOLD_DB_PATH` — canonical. CLEAN.
- Line 32: `from trading_app.config import ENTRY_MODELS` — canonical. CLEAN.
- Line 40: `DB_PATH = GOLD_DB_PATH` — canonical alias. CLEAN.
- Line 86: `if em == "E3" and cb > 1: continue` — hardcoded "E3" skip matches SKIP_ENTRY_MODELS logic in production outcome_builder. Intentional per-entry-model heuristic. ACCEPTABLE.
- Lines 235-268: Write connection not in try/finally — if INSERT loop throws, connection leaks. LOW, same as ~22 CLI script pattern already catalogued as acceptable (process exit closes it). ACCEPTABLE.
- Lines 50-126: Worker connection in try/finally. CLEAN.
- No silent failures; exceptions propagate from worker futures. CLEAN.

**No findings.**

### scripts/tools/build_mes_outcomes_fast.py — CLEAN

Seven Sins scan:
- Line 29: `from pipeline.cost_model import get_cost_spec` — canonical. CLEAN.
- Line 30: `from pipeline.init_db import ORB_LABELS` — canonical. CLEAN.
- Line 31: `from pipeline.paths import GOLD_DB_PATH` — canonical. CLEAN.
- Line 32: `from trading_app.config import ENTRY_MODELS` — canonical. CLEAN.
- Line 41: `DB_PATH = GOLD_DB_PATH` — canonical alias. CLEAN.
- Lines 40-43: `INSTRUMENT = "MES"`, `START_DATE`, `END_DATE` — intentional MES-specific script scope. ACCEPTABLE.
- Line 73: `if em == "E3" and cb > 1: continue` — same as build_outcomes_fast.py above. ACCEPTABLE.
- Lines 234-268: Write connection not in try/finally — same CLI connection leak pattern. ACCEPTABLE.
- Line 137: Read connection in try/finally block (implicitly — actually no try/finally but read-only, process-exit safe). LOW.

**No findings.**

### scripts/tools/prospective_tracker.py — CLEAN

Seven Sins scan:
- Line 20: `from pipeline.paths import GOLD_DB_PATH` — canonical. CLEAN.
- Line 71: `WHERE orb_minutes = 5` in SQL — intentional 5m-aperture scoping for this prospective tracker. ACCEPTABLE (per-investigation scope, not a canonical list).
- Lines 30-41: `SIGNALS` dict with hardcoded MGC/CME_REOPEN config — intentional frozen hypothesis definition. ACCEPTABLE.
- Lines 117-155: DELETE+INSERT pattern — canonical write pattern. CLEAN.
- Line 156: `con.commit()` called after inserts. CLEAN.
- Line 300-306: `con = duckdb.connect(args.db_path)` not in try/finally — CLI script, process exit closes. ACCEPTABLE.
- No silent failures; all errors propagate. CLEAN.

**No findings.**

### scripts/tools/profile_1000_runners.py — CLEAN

Seven Sins scan:
- Lines 42-43: `spec = get_cost_spec("MGC")` — hardcoded MGC, but this is an MGC-specific research script. ACCEPTABLE.
- Lines 46-63: Hardcoded `'MGC'`, `'1000'`, `'E1'`, `RR=2.0`, `CB=2` in SQL — intentional research scope. Read-only. ACCEPTABLE.
- Line 31: `from pipeline.cost_model import get_cost_spec, to_r_multiple` — canonical. CLEAN.
- Line 32: `from pipeline.paths import GOLD_DB_PATH` — canonical. CLEAN.
- Lines 43-80: Read-only connection, `con.close()` at line 80 after all queries. CLEAN.
- Line 84: `for _, row in df.iterrows()` — iterrows in a research script, not production pipeline. Not caught by drift check #77 (pipeline-only). ACCEPTABLE.
- No silent failures; continues on empty bars (explicit check at line 87). CLEAN.

**No findings.**

---

## Deferred Findings — Status After Iter 98

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 5 target files audited: parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py
- 0 findings fixed (audit-only iteration — all files clean)
- 0 new ACCEPTABLE findings added to Won't Fix (connection leaks match existing catalogued pattern)
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

> Cumulative list — 139 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 43 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- **Total: 139 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `scripts/tools/m25_nightly.py` — M25 nightly runner, production-adjacent
- `scripts/tools/m25_audit.py` — M25 audit script
- `scripts/tools/m25_auto_audit.py` — M25 auto audit
- `scripts/tools/m25_preflight.py` — M25 preflight checks
- `scripts/tools/m25_run_grounded_system.py` — M25 grounded system runner

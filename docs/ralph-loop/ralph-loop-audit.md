# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 94

## RALPH AUDIT — Iteration 94 (fix)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### scripts/tools/gen_playbook.py — 1 FINDING FIXED (GP-94)

Seven Sins scan:
- Lines 18-31: `SESSION_ORDER` — hardcoded ordered list of sessions with Brisbane times. This is intentional (display order for Quick Reference table must match column header — cannot use sorted()). Comment on line 32-34 explains this. However, EUROPE_FLOW (added Mar 11 2026) was **missing** from the list. SESSION_CATALOG has 12 sessions; SESSION_ORDER only had 11. **FIXED** — added EUROPE_FLOW tuple between SINGAPORE_OPEN and LONDON_METALS.
- Line 35: `INSTRUMENTS = ["MGC", "MNQ", "MES", "M2K"]` — hardcoded but guarded by assert on line 36 against `ACTIVE_ORB_INSTRUMENTS`. ACCEPTABLE (assertion guard present, whitelisted in audit_behavioral.py line 34).
- Line 42: `MIN_EXPR = 0.05` — playbook quality floor heuristic. Not a trading parameter. ACCEPTABLE.
- Canonical imports: `ACTIVE_ORB_INSTRUMENTS`, `is_uk_dst`, `is_us_dst`. CLEAN.
- No hardcoded DB paths, no silent failures, no fail-open paths.

**Finding GP-94: FIXED — EUROPE_FLOW added to SESSION_ORDER (1 line, [mechanical])**

### scripts/tools/ml_audit.py — CLEAN

Seven Sins scan:
- Lines 12-14: imports from `trading_app.ml.*` — canonical ML config/features. CLEAN.
- Lines 20-24: model file discovery with explicit "NO MODEL FOUND" print and empty return — fail-open only for absent model file (acceptable in audit tool). No silent pass.
- Line 311: `--db-path gold.db` default in argparse — CLI tool script, not production path. Covered by `pipeline.paths.GOLD_DB_PATH` in production code. ACCEPTABLE.
- No hardcoded instrument lists (uses `ACTIVE_INSTRUMENTS` from `trading_app.ml.config`).
- No hardcoded session names in logic paths — sessions discovered dynamically from data.
- Lines 124-126: `std > 0` guard before Sharpe division — no silent divide-by-zero.

**Finding: CLEAN — no actionable findings**

### scripts/tools/audit_integrity.py — CLEAN

Seven Sins scan:
- Line 17-19: `ACTIVE_INSTRUMENTS = ACTIVE_ORB_INSTRUMENTS`, `_SQL_IN` built dynamically. CLEAN — canonical source.
- Lines 22-23: `_connect()` returns fresh read-only connection; callers manage lifecycle. CLEAN.
- Lines 178-192: `CHECKS` registry — dynamic count used in output (line 301: `len(CHECKS)`). No hardcoded check count. CLEAN.
- Lines 195-267: `_print_informational()` — all SQL uses `_SQL_IN` from canonical source. CLEAN.
- Lines 270-303: `main()` — `try/finally con.close()` pattern. CLEAN.
- No hardcoded DB paths (uses `GOLD_DB_PATH`), no hardcoded instruments, no silent failures.
- Exit code pattern: 0 = pass, 1 = fail. Correctly fail-closed (sys.exit(1) on violations).

**Finding: CLEAN — no actionable findings**

---

## Deferred Findings — Status After Iter 94

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 3 files audited: gen_playbook.py (1 finding FIXED), ml_audit.py (clean), audit_integrity.py (clean)
- 1 finding fixed: GP-94 — EUROPE_FLOW missing from SESSION_ORDER in gen_playbook.py (1 line, [mechanical])
- Infrastructure Gates: 3/3 PASS
- Commit: 69ac9ac

**Codebase steady state maintained for major violation classes:**
- Hardcoded DB paths: ELIMINATED (0 in production code)
- Hardcoded apertures [5,15,30]: ELIMINATED from production (7th fix applied)
- Hardcoded entry model IN clauses: ELIMINATED
- Hardcoded instrument lists: ELIMINATED (1 acceptable with assertion guard)
- SESSION_ORDER coverage: NOW COMPLETE (12/12 sessions)

**Remaining LOW-priority items:**
- ~22 CLI scripts with connection leaks (process exit closes them)
- DF-04 deferred: rolling_portfolio.py dormant orb_minutes=5 (structural, >5 files)

---

## Files Fully Scanned

> Cumulative list — 120 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 24 files (iters 18-72, 89, 90, 91, 92, 93, 94): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- **Total: 120 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `scripts/tools/ml_cross_session_experiment.py` — has its own SESSION_ORDER (may have same gap)
- `scripts/tools/ml_hybrid_experiment.py` — has its own SESSION_ORDER (may have same gap)
- `scripts/tools/ml_instrument_deep_dive.py` — has its own SESSION_ORDER (may have same gap)

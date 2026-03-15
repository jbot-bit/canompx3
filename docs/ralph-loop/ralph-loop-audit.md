# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 93

## RALPH AUDIT — Iteration 93 (fix)
## Date: 2026-03-15
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### scripts/tools/rr_selection_analysis.py — CLEAN

Seven Sins scan:
- Line 18: `active_models = [em for em in ENTRY_MODELS if em not in SKIP_ENTRY_MODELS]` — canonical dynamic filter using `ENTRY_MODELS` + `SKIP_ENTRY_MODELS`. CLEAN.
- Lines 15-16: canonical imports (`GOLD_DB_PATH`, `ENTRY_MODELS`, `SKIP_ENTRY_MODELS`). CLEAN.
- Lines 22-31: `try/finally con.close()` — clean connection management.
- Line 67: `rho = 0.7` — assumed inter-RR correlation for Jobson-Korkie SE formula. Statistical analysis heuristic, not a trading parameter. ACCEPTABLE (Intentional per-session or per-instrument heuristic).
- Lines 135-136: hardcoded DD ceiling brackets — display bucketing for analysis output only, not trading logic. ACCEPTABLE.
- Lines 146: hardcoded account sizes — illustrative sizing for analysis display, not production trading config. ACCEPTABLE.
- No hardcoded instruments, sessions, DB paths. No silent failures on trading paths. No fail-open execution paths.

**Finding: CLEAN — no actionable findings**

### scripts/tools/sensitivity_analysis.py — 1 FINDING FIXED (SA-01)

Seven Sins scan:
- Line 40 (pre-fix): `RR_STEPS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]` — hardcoded duplicate of `RR_TARGETS` from `trading_app/outcome_builder.py`. **FIXED** — replaced with `from trading_app.outcome_builder import RR_TARGETS` and `RR_STEPS = RR_TARGETS`.
- Line 43: `CB_STEPS = [1, 2, 3, 4, 5]` — confirm bar sweep steps. These are a complete ordered sweep (1 through 5), not a subset of canonical. No canonical source for this list. ACCEPTABLE.
- Line 46: `G_LADDER = ["NO_FILTER", "ORB_G4", "ORB_G5", "ORB_G6", "ORB_G8"]` — adjacency sweep ladder for G-filters. These are the base G-filter names used to navigate adjacency, guarded by `if G_LADDER[idx-1] in ALL_FILTERS` (line 118-120). ACCEPTABLE.
- Line 49: `STABILITY_THRESHOLD = 0.50` — analysis stability threshold heuristic. Not a trading parameter. ACCEPTABLE.
- Line 161: `orb_minutes = 5` hardcoded in `query_strategy_outcomes` — deliberate: filter columns (rel_vol etc.) only populated at 5m aperture. Comment explains reasoning at lines 152-155. ACCEPTABLE (matches strategy_discovery.py pattern).
- Lines 169-172: `for _, row in df_rows.iterrows()` — iterrows in tool script. Drift check #77 applies to pipeline only. ACCEPTABLE.
- Canonical imports: `VALID_ORB_MINUTES`, `get_cost_spec`, `GOLD_DB_PATH`, `ALL_FILTERS`, `TRADEABLE_INSTRUMENTS`, `LIVE_PORTFOLIO`. CLEAN.
- No silent failures, no hardcoded instruments/sessions/DB paths.

**Finding SA-01: FIXED — RR_STEPS hardcoded list replaced with canonical RR_TARGETS import (3 lines, [mechanical])**

---

## Deferred Findings — Status After Iter 93

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 2 files audited: rr_selection_analysis.py (clean) + sensitivity_analysis.py (1 LOW finding FIXED)
- 1 finding fixed: SA-01 — RR_STEPS hardcoded list replaced with canonical RR_TARGETS import (3 lines, [mechanical])
- Infrastructure Gates: 3/3 PASS
- Commit: b7804ef

**Codebase steady state maintained for major violation classes:**
- Hardcoded DB paths: ELIMINATED (0 in production code)
- Hardcoded apertures [5,15,30]: ELIMINATED from production (7th fix applied)
- Hardcoded entry model IN clauses: ELIMINATED
- Hardcoded instrument lists: ELIMINATED (1 acceptable with assertion guard)

**Remaining LOW-priority items:**
- ~22 CLI scripts with connection leaks (process exit closes them)
- DF-04 deferred: rolling_portfolio.py dormant orb_minutes=5 (structural, >5 files)

---

## Files Fully Scanned

> Cumulative list — 117 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 21 files (iters 18-72, 89, 90, 91, 92, 93): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- **Total: 117 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `scripts/tools/gen_playbook.py` — playbook generator, not yet scanned
- `scripts/tools/ml_audit.py` — ML audit tool, not yet scanned
- `scripts/tools/audit_integrity.py` — integrity audit tool, not yet scanned

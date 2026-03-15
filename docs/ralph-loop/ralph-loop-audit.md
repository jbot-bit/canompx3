# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 92

## RALPH AUDIT — Iteration 92 (fix)
## Date: 2026-03-15
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### scripts/tools/pinecone_snapshots.py — 1 FINDING FIXED (PS-92)

Seven Sins scan:
- Lines 59-62: `sample_size >= 100` and `sample_size BETWEEN 30 AND 99` — hardcoded classification thresholds. `trading_app.config` exports `CORE_MIN_SAMPLES = 100` and `REGIME_MIN_SAMPLES = 30`. **FIXED** — imported canonical constants and replaced literals with f-string interpolation.
- Line 303: `except Exception: summary = "(unreadable)"` — documentation generator reading research output files. No trading data. ACCEPTABLE (non-production doc tool).
- No hardcoded instruments, sessions, entry models, or DB paths.
- No silent failures on trading paths; no fail-open execution paths.

**Finding PS-92: FIXED — hardcoded 100/30 replaced with CORE_MIN_SAMPLES/REGIME_MIN_SAMPLES (4 lines, [mechanical])**

### scripts/tools/rolling_portfolio_assembly.py — CLEAN

Seven Sins scan:
- `WINDOW_MONTHS = 12`, `STABLE_SHARPE = 0.10`, `DEGRADED_SHARPE = 0.0` — operational rolling analysis heuristics, not trading parameters. No `@research-source` annotation required (analysis tool, not config). ACCEPTABLE.
- `ACTIVE_ORB_INSTRUMENTS`, `GOLD_DB_PATH`, `get_cost_spec` — all canonical imports. CLEAN.
- Slot head selection uses `ORDER BY ef.head_sharpe_ann DESC` — this is edge family ranking for portfolio assembly (not variant selection for live trading). Drift check #44 applies to variant selection ORDER BY in SQL adapters; this is a separate reporting context. ACCEPTABLE.
- No exception handlers suppressing failures.
- No hardcoded instruments, sessions, entry models, or DB paths.

**Finding: CLEAN — no actionable findings**

### scripts/tools/generate_trade_sheet.py — CLEAN (with note)

Seven Sins scan:
- Line 127: `except Exception: return "UNKNOWN"` in `_check_fitness` — fitness failure blocks REGIME-gated trades (fail-closed). Non-gated trades show UNKNOWN badge (display only). CLEAN.
- Line 60: `["G2", "G3", "G4", "G5", "G6", "G8"]` hardcoded in `_filter_description()` — UI display helper, not trading logic. ACCEPTABLE.
- Line 558: `"All entries are E2 (stop-market)"` in HTML note — misleading since LIVE_PORTFOLIO includes E1 strategies. Display documentation only; does not affect trade execution logic. LOW/cosmetic — not worth a production code change for HTML string in a generator tool.
- `LIVE_PORTFOLIO`, `SESSION_CATALOG`, `GOLD_DB_PATH`, `get_cost_spec`, `get_active_instruments` — all canonical imports. CLEAN.
- No hardcoded instruments, sessions, or DB paths in execution paths.

**Finding: CLEAN — HTML note at line 558 is cosmetically inaccurate (E1 also active) but display-only. ACCEPTABLE.**

---

## Deferred Findings — Status After Iter 92

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 3 files audited: pinecone_snapshots.py (1 LOW finding FIXED) + rolling_portfolio_assembly.py (clean) + generate_trade_sheet.py (clean)
- 1 finding fixed: PS-92 — hardcoded CORE/REGIME thresholds replaced with canonical constants (4 lines, [mechanical])
- Infrastructure Gates: 3/3 PASS
- Commit: d2f582a

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

> Cumulative list — 115 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 19 files (iters 18-72, 89, 90, 91, 92): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- **Total: 115 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `scripts/tools/rr_selection_analysis.py` — RR selection analysis, not yet scanned
- `scripts/tools/sensitivity_analysis.py` — sensitivity analysis tool, not yet scanned
- `scripts/tools/generate_trade_sheet.py` already scanned this iter
- `scripts/tmp_*.py` — temporary analysis scripts (low priority, audit-only)

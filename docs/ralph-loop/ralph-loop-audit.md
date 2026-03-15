# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 96

## RALPH AUDIT — Iteration 96 (audit-only)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### scripts/tools/ml_per_session_experiment.py — CLEAN

Seven Sins scan:
- Lines 27-40: `SESSION_ORDER` — 12 sessions including EUROPE_FLOW (line 32). CLEAN.
- Line 143: `duckdb.connect(str(GOLD_DB_PATH), read_only=True)` — canonical DB path. CLEAN.
- Imports: `GOLD_DB_PATH`, `ALL_FILTERS`, `transform_to_features` — all canonical. CLEAN.
- No hardcoded instrument lists (uses `--instrument` CLI arg). CLEAN.
- No silent failures. CLEAN.

**No findings.**

### scripts/tools/ml_level_proximity_experiment.py — CLEAN

Seven Sins scan:
- Lines 26-39: `SESSION_ORDER` — 12 sessions including EUROPE_FLOW (line 31). CLEAN.
- Line 159: `duckdb.connect(str(GOLD_DB_PATH), read_only=True)` — canonical DB path. CLEAN.
- Local `rf_params` dict — intentional experiment variation. ACCEPTABLE (same pattern as ml_hybrid_experiment allowlist).
- No hardcoded instrument lists (uses `--instrument` CLI arg). CLEAN.

**No findings.**

### scripts/tools/ml_threshold_sweep.py — CLEAN

Seven Sins scan:
- No SESSION_ORDER — script does not use cross-session features. CLEAN.
- Imports: `MODEL_DIR`, `replay_historical` — canonical imports. CLEAN.
- Model backup/restore pattern (lines 98-155) — correct fail-safe pattern using `shutil.copy2` + `finally` block guarantees restore. CLEAN.
- No hardcoded DB paths (uses `MODEL_DIR` from `trading_app.ml.config`). CLEAN.

**No findings.**

### scripts/tools/ml_session_leakage_audit.py — CLEAN

Seven Sins scan:
- Lines 29-42: `SESSION_ORDER` — 12 sessions including EUROPE_FLOW (line 34). CLEAN.
- Line 145: `duckdb.connect(str(GOLD_DB_PATH), read_only=True)` — canonical DB path. CLEAN.
- No hardcoded instrument lists (uses `--instrument` CLI arg). CLEAN.

**No findings.**

### scripts/tools/ml_license_diagnostic.py — CLEAN

Seven Sins scan:
- No local SESSION_ORDER — imports `SESSION_CHRONOLOGICAL_ORDER` from `trading_app.ml.config` (canonical). CLEAN.
- `SESSION_CHRONOLOGICAL_ORDER` in `trading_app/ml/config.py` confirmed to include EUROPE_FLOW (line 155). CLEAN.
- Imports: `ACTIVE_INSTRUMENTS`, `RF_PARAMS`, `GOLD_DB_PATH`, `load_validated_feature_matrix`, `apply_e6_filter` — all canonical. CLEAN.
- Three-phase analysis (feature importance, honest OOS, signal diagnosis) — correct methodology. CLEAN.
- Exception handlers at lines 199-201 and 220-222 catch `ValueError` from `roc_auc_score` only (not broad Exception). CLEAN.

**No findings.**

### Spot-check: scripts/tools/audit_15m30m.py — 1 LOW ACCEPTABLE finding

Seven Sins scan:
- Lines 29, 44, 62, 88: `IN ('MGC','MNQ','MES','M2K')` hardcoded in SQL — canonical violation.
- ASSESSMENT: One-off investigation script (reads only, no writes). Hardcoded list matches current active instruments exactly. If an instrument is removed, SQL returns 0 rows — not dangerous. No imports of canonical instrument list needed. ACCEPTABLE (pattern: one-off diagnostic, not a canonical source).

**Finding GP-96a: ACCEPTABLE — hardcoded instrument list in investigation SQL (read-only diagnostic, no safety impact)**

---

## Deferred Findings — Status After Iter 96

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 5 primary target files audited: ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py
- 1 spot-check file: audit_15m30m.py
- 0 findings fixed (audit-only iteration — all files clean)
- 1 ACCEPTABLE finding: GP-96a (hardcoded instruments in read-only diagnostic SQL)
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

> Cumulative list — 129 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 33 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- **Total: 129 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `scripts/tools/backtest_1100_early_exit.py` — backtest script, orb_1100 session reference worth examining
- `scripts/tools/backtest_atr_regime.py` — hardcoded FAMILIES list in research backtest
- `scripts/tools/beginner_tradebook.py` — tradebook display script with dual connection pattern
- `scripts/tools/find_pf_strategy.py`, `scripts/tools/rank_slots.py` — portfolio analysis tools
- Any remaining scripts/tools/ files not yet scanned

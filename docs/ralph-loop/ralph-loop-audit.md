# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 99

## RALPH AUDIT — Iteration 99 (audit-only)
## Date: 2026-03-16
## Infrastructure Gates: 3/3 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 72 checks passed, 0 skipped, 6 advisory |
| `audit_behavioral.py` | PASS | 6/6 clean |
| `ruff check` | PASS | Clean |

---

## Files Audited This Iteration

### scripts/tools/m25_nightly.py — CLEAN

Seven Sins scan:
- Lines 59-62: `except SystemExit` on `load_api_key()` failure → prints warning, returns. Deliberate graceful degradation for nightly batch. ACCEPTABLE.
- Lines 98-103: `except Exception as e:` → prints ERROR, continues to next file. Not silent. ACCEPTABLE.
- No hardcoded DB paths, no canonical violations, no volatile data.
- `len(NIGHTLY_TARGETS)` used dynamically throughout — no hardcoded counts. CLEAN.
- File list (`NIGHTLY_TARGETS`) is a curated batch configuration, not a canonical instrument/session list. ACCEPTABLE.

**No findings.**

### scripts/tools/m25_audit.py — CLEAN

Seven Sins scan:
- Lines 83-84: `except Exception: data = {}` in budget counter read — silent on corrupt/missing JSON budget file. LOW/ACCEPTABLE: budget tracking is advisory only, not a safety-critical path.
- Lines 93-94: `except Exception: pass` in budget file write — silent on write failure. Same reasoning. ACCEPTABLE.
- Lines 504-505, 528-529, 574-575, 614-615, 669-670: `except Exception: pass` in `gather_runtime_context` / `build_diff_content` — context enrichment helpers that feed M2.5 pre-context. Silently skipping context gathering is correct design (advisory tool; missing context = less accurate M2.5, not a broken system). ACCEPTABLE per `.claude/rules/m25-audit.md`.
- Line 704: bare `pass` in `triage_output` — follows `skip_section = False` assignment. Intentional no-op (no body needed after the flag set). CLEAN.
- Line 454: `db_path = project_root / "gold.db"` in `gather_runtime_context` — uses dynamic project_root, not hardcoded absolute path. CLEAN.
- Lines 487-495, 514-518, 538-560: subprocess calls use canonical imports (`ACTIVE_ORB_INSTRUMENTS`, `COST_SPECS`, `SESSION_CATALOG`, `ENTRY_MODELS`). CLEAN.
- No hardcoded instrument lists, no hardcoded entry model strings in non-subprocess context.

**No findings.**

### scripts/tools/m25_auto_audit.py — CLEAN

Seven Sins scan:
- Lines 141-142: `except SystemExit: return 0` on missing API key — deliberate no-op. Documented in docstring. ACCEPTABLE.
- Lines 167-169: `except Exception as e: print(...)` → logs `type(e).__name__`, continues. Not silent. ACCEPTABLE.
- `AUDIT_DIRS = ("pipeline/", "trading_app/")` — curated scope filter for what M2.5 should auto-audit. Not a canonical instrument/session list. ACCEPTABLE.
- `MODE_MAP` hardcodes file → mode mappings — intentional configuration, not canonical trading data. ACCEPTABLE.
- No DB path references, no canonical violations.

**No findings.**

### scripts/tools/m25_preflight.py — CLEAN

Seven Sins scan:
- Lines 73-76: `except Exception as e: print(f"WARNING..."); return 0` — fail-open on M2.5 API error. This is intentional: if the advisory scanner crashes, don't block the researcher. The `.claude/rules/m25-audit.md` explicitly classifies M2.5 as advisory only. ACCEPTABLE.
- Return code 2 for "no API key" (skipped) vs return 0 for "API error" (fail-open) vs return 1 for "FAIL verdict" — distinct codes allow callers to distinguish states. CLEAN.
- No hardcoded DB paths, no canonical violations.

**No findings.**

### scripts/tools/m25_run_grounded_system.py — CLEAN

Seven Sins scan:
- Line 17: `api_key = load_api_key()` at module level — raises SystemExit if key missing. Intentional: this is a one-shot research script, not a library. ACCEPTABLE.
- Line 20: `NB_DIR = Path(r"C:\Users\joshd\OneDrive\Desktop\Organisation\nb resources")` — hardcoded absolute path to personal OneDrive. LOCAL RESEARCH SCRIPT ONLY. Files skipped gracefully if not found (line 127: `if not fpath.exists(): continue`). ACCEPTABLE (per-machine research script, not production).
- `KEY_FILES` list (lines 154-170) enumerates production files to read — not a canonical instrument/session list. Intentional research scope. ACCEPTABLE.
- No hardcoded DB paths, no hardcoded entry model strings in SQL, no canonical violations.

**No findings.**

---

## Deferred Findings — Status After Iter 99

### STILL DEFERRED (carried forward)
- **DF-04** — `rolling_portfolio.py:304` dormant `orb_minutes=5` in rolling DOW stats — structural multi-file fix, blast radius >5 files

---

## Summary
- 5 target files audited: m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py
- 0 findings fixed (audit-only iteration — all files clean)
- 0 new ACCEPTABLE findings (patterns match existing catalogued patterns)
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

> Cumulative list — 144 files fully scanned.

- trading_app/ — 44 files (iters 4-61)
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 48 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- **Total: 144 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `scripts/tools/refresh_data.py` — data refresh utility (has uncommitted changes per git status)
- `scripts/tools/m25_ml_audit.py` — M25 ML-specific audit (if exists)
- `scripts/tools/codex_review.py` — codex review tooling (if exists)
- `research/research_zt_event_viability.py` — new research file (untracked per git status)
- `research/research_zt_cpi_nfp.py` — new research file (untracked per git status)

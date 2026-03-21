# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

## Current Session
- **Tool:** Claude Code (Pipeline Methodology Audit + Canon Lock)
- **Date:** 2026-03-22
- **Branch:** `main`
- **Status:** Policy plumbing committed. NO rebuild run. NO validation rerun. NO live_config update.

### What was done this session

#### 1. Full Pipeline Truth Audit
- Non-ML pipeline audited from ground up (DB, code, config, git history, seed artifacts)
- Found: DB state is artifact of multiple partial rebuilds with inconsistent config
- Found: min_sample=50 was used (non-canonical), noise floor=0.22/0.32 applied cross-instrument
- Found: MGC 0 validated, MES 0 validated, MNQ 11 validated (current DB)
- Found: family_rr_locks 3 days stale vs discovery, 5/11 strategies orphaned

#### 2. Noise Floor Methodology Audit
- Proved: noise floor is NOT White's RC or Hansen's SPA — it's a heuristic minimum-effect-size gate
- Proved: WF+FDR alone are insufficient (38,755 MNQ noise strategies pass both on random walk)
- Proved: noise floor IS non-redundant but was miscalibrated (MGC sigma 2.54x real, cross-instrument reuse)
- Proved: mean+2std aggregation has no literature basis, conflates noise volume with magnitude
- Found: MNQ and MES null test seeds already exist (100 MNQ, 94 MES) — were not surfaced in prior sessions

#### 3. Canon Lock — D1 (Noise Floor) + D2 (Min Sample)
- **D1 locked:** Per-instrument p95 of pooled null survivor ExpR. Gate moved downstream of WF/FDR. Flag, not hard rejection.
- **D2 locked:** min_sample=30 (REGIME_MIN_SAMPLES). Code default restored.
- Interim floors (p95, Gaussian seeds): MGC E2=0.21, MES E2=0.29, MNQ E2=0.21

#### 4. Implementation (commit `f0086d7`)
- Phase 2b hard gate REMOVED from strategy_validator.py
- `noise_risk` BOOLEAN column added to validated_setups (NULL stub — not populated yet)
- `NOISE_FLOOR_BY_INSTRUMENT` added to config.py (interim p95 values)
- min_sample=30 in 4 wrapper locations (was 50)
- Drift check 80 converted to no-op
- Tests updated: 114 pass, 0 regressions
- 2x code review caught + fixed missed `pipeline_status.py:317` (second --min-sample 50 occurrence)

### What was NOT done (explicit scope boundaries)
- **No rebuild run.** DB still has stale validation state from Mar 19.
- **No validation rerun.** Current validated_setups (11 MNQ) built with old gates.
- **No live_config update.** `_check_noise_floor` in live_config.py still imports zeroed `NOISE_EXPR_FLOOR` — passes everything. Separate design decision.
- **No noise_risk population.** Column exists but is NULL everywhere. Computation is next stage.
- **No skill/shell script updates.** 6 files still hardcode `--min-sample 50` (.claude/skills/, shell scripts).

### Truth State
- **Validator logic:** Phase 2b removed, min_sample=30. Next validation run will produce different population than current DB.
- **DB:** Stale. Built with old gates (noise floor=0.22/0.32, min_sample=50). Does NOT match current code.
- **NOISE_FLOOR_BY_INSTRUMENT:** Interim. Derived from Gaussian null seeds with known sigma overshoot. p95 aggregation. NOT final truth — block bootstrap calibration deferred.
- **MGC/MES bars:** 14+ days stale (Mar 6/7). MNQ current (Mar 20).
- **ML:** Still frozen. V2 gate active. 6 bugs identified, 0 fixed.

### Next Steps (for incoming session)
Pick ONE:
1. **Populate noise_risk flag** — compute from NOISE_FLOOR_BY_INSTRUMENT on current validated_setups
2. **Rebuild truth path** — ingest fresh bars → discovery → validation (with new gates) → downstream

Do not combine. One stage at a time.

### Known workflow issues
- Post-edit drift hook fails on `ModuleNotFoundError: No module named 'pipeline'` — import path issue in hook runner. Not blocking but masks real drift detection.

---

## Prior Session
- **Tool:** Claude Code (ML Audit + Fix Planning Terminal)
- **Date:** 2026-03-21 (night)
- **Summary:** ML audit complete. Fix plan designed. Live portfolio found EMPTY (42 specs no match). Version gate deployed. Implementation paused for pipeline rebuild.

## Prior Session
- **Tool:** Claude Code
- **Date:** 2026-03-21 (earlier)
- **Summary:** Multi-RR portfolio built. ML audit found 4 FAILs. Bootstrap 5K code committed. Confluence design started. Session crashed mid-brainstorm.

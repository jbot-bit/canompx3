# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

## Current Session
- **Tool:** Claude Code (Canonical Refresh + FDR Fix + Pipeline Status Fix + Live Spec Rebuild)
- **Date:** 2026-03-23
- **Branch:** `main`
- **Status:** Full canonical refresh complete. 404 validated. 8 specs. 8 MNQ live-resolvable (MGC 0, MES 0). FDR K snapshot fix applied. pipeline_status per-aperture fix applied.

### What was done this session

#### 1. Full Pipeline Truth Audit
- Non-ML pipeline audited from ground up (DB, code, config, git history, seed artifacts)
- Found: DB state was artifact of multiple partial rebuilds with inconsistent config
- Found: min_sample=50 was used (non-canonical), noise floor=0.22/0.32 applied cross-instrument
- Found: MGC 0 validated, MES 0 validated, MNQ 11 validated (prior DB)

#### 2. Noise Floor Methodology Audit
- Proved: noise floor is NOT White's RC or Hansen's SPA — heuristic minimum-effect-size gate
- Proved: WF+FDR alone are insufficient (38,755 MNQ noise strategies pass both on random walk)
- Proved: noise floor IS non-redundant but was miscalibrated (cross-instrument reuse, wrong aggregation)
- Found: MNQ and MES null test seeds already exist (100 MNQ, 94 MES)
- Locked: p95 of pooled null survivor ExpR as canonical aggregation

#### 3. Canon Lock + Implementation
- Commits: `f0086d7` (Phase 2b removal, noise_risk stub, min_sample=30)
- Commit: `475ec12` (min_sample=30 in all remaining entrypoints)
- Phase 2b hard gate REMOVED. Noise floor is now post-validation flag (not yet computed).
- min_sample=30 locked in ALL active entrypoints (Python, shell, skills, rules, docs).
- 114 tests pass, 0 regressions.

#### 4. Interim Validation Rebuild (Mar 22, 00:46-01:12 AEST)
- Cleared stale validation state (experimental_strategies reset, validated_setups/edge_families/family_rr_locks cleared)
- Ran validation for MGC, MES, MNQ with canonical gates (min_sample=30, no Phase 2b, WF+FDR)
- Results: **MGC 20, MES 81, MNQ 731 = 832 total validated**
- Global FDR K=105,612 (total canonical strategies across all instruments)

#### 5. Downstream Rebuild (01:38-01:42 AEST)
- family_rr_locks: 255 rows (MGC 10, MES 40, MNQ 205)
- edge_families: 248 rows (MGC 10, MES 40, MNQ 198)
- Edge family robustness: MGC 2 ROBUST, MES 2 ROBUST + 8 WHITELISTED, MNQ 61 ROBUST + 45 WHITELISTED

#### 6. Live_config Redesign (commit `21dca30`)
- LIVE_PORTFOLIO rewritten: 46 dead specs → 16 ground-truth specs (10 CORE, 6 REGIME)
- MGC: 2 strategies (TOKYO_OPEN ORB_G4 + ORB_G4_CONT, both ROBUST)
- MES: 4 strategies (CME_PRECLOSE ATR70_VOL + ORB_G5 + ORB_G4, NYSE_CLOSE ORB_G6)
- MNQ: 11 strategies (CME_PRECLOSE 5 filters, COMEX_SETTLE, CME_REOPEN, BRISBANE_1025, NYSE_CLOSE, SINGAPORE_OPEN)
- PURGED/SINGLETON families placed in REGIME tier with fitness gating (D1 lock)
- No instrument exclusions (all FDR-significant at K=105,612)
- 2x code review: CLEAN (all specs valid, all callers generic, zero downstream breakage)

### What was done this session (continued — Mar 22 evening)

#### 7. noise_risk Implementation (commit `324b3a5`)
- Added `oos_exp_r` (DOUBLE) + `noise_risk` (BOOLEAN) to validated_setups
- Computed from WF `agg_oos_exp_r` vs per-instrument p95 null floor
- live_config `_check_noise_floor` reads pre-computed flag (fail-closed on NULL)
- Rolling-sourced strategies bypass noise check (own quality gates)

#### 8. Full Revalidation (all instruments)
- MES: 81/81 passed. 78 noise_risk=True, 3 clean (CME_PRECLOSE ATR70_VOL cluster)
- MNQ: 731/731 passed. 671 noise_risk=True, 60 clean
- MGC: 9/20 passed (11 rejected Phase 3 — yearly robustness with fresh data). 9/9 noise_risk=True (all below 0.21 floor)

#### 9. Layer 1-6 Audit (full retroactive)
- Layer 1 (Data): SURVIVED. Zero duplicates, no lookahead, correct TZ handling
- Layer 2 (Outcomes): SURVIVED. Cost model verified from raw arithmetic. Scratch conservative bias noted (+0.40R avg net excursion)
- Layer 3 (Discovery): SURVIVED. No redefinitions, full grid written, FDR correct
- Layer 4 (Validation): SURVIVED. WF anchored expanding, no leakage, FDR hard gate at global K=105,612
- Layer 5 (noise_risk): Implemented and verified
- Layer 6 (Portfolio): FROZEN. 11 active strategies (MES 1, MNQ 10, MGC 0)

#### 10. Layer 7 (Execution Realism): FROZEN
- No code bugs found
- 4 policy constraints documented in TRADING_RULES.md "Execution Realism Constraints"
- Signal collision priority, scratch gap, manual session focus, correlation guard

### What was done this session (Mar 23)

#### 1. Full Canonical Refresh
- Fixed MNQ O15/O30 outcomes (2 weeks stale: max was 2026-03-06, rebuilt to 2026-03-20)
- Full discovery rerun (3 instruments x 3 apertures = 9 runs)
- Full validation rerun with --fdr-k 105640 (fixed K snapshot)
- Edge families + RR locks rebuilt
- Result: 404 validated (MGC 6, MES 9, MNQ 389), down from 821

#### 2. FDR K Fixed-Snapshot Fix
- strategy_validator.py: added --fdr-k parameter for BH consistency across instrument runs
- Fail-closed: RuntimeError on K mismatch, ValueError on fdr_k <= 0
- Backward compatible: omitting --fdr-k computes from DB at runtime

#### 3. pipeline_status.py Per-Aperture Fix
- orb_outcomes staleness now checked per aperture (matching daily_features pattern)
- Root cause of missed MNQ O15/O30 staleness: MAX(trading_day) across all apertures masked per-aperture gaps

#### 4. LIVE_PORTFOLIO Spec Rebuild
- 16 specs -> 8 specs (8 dead specs dropped, 0 added)
- 2 CORE (CME_PRECLOSE ATR70_VOL, COMEX_SETTLE ATR70_VOL)
- 6 REGIME (CME_PRECLOSE x4 filters, CME_REOPEN ATR70_VOL, CME_PRECLOSE ORB_G8)
- Resolves 8 MNQ strategies. MGC/MES do not qualify under current live gates.

### Truth State (verified Mar 23 2026)
- **validated_setups:** 404 rows (MGC 6, MES 9, MNQ 389). All wf_passed=True, FDR-corrected at K=105,640.
- **family_rr_locks:** 166 rows (MGC 2, MES 7, MNQ 157).
- **edge_families:** 162 rows (MGC 2, MES 7, MNQ 153).
- **LIVE_PORTFOLIO:** 8 specs (2 CORE, 6 REGIME). Resolves 8 MNQ strategies (MGC 0, MES 0).
- **noise_risk:** Fully populated. Zero NULLs.
- **Bars:** All instruments current to 2026-03-21. daily_features/outcomes to 2026-03-20.
- **ML:** Still frozen. V2 gate active.

### Next Steps (for incoming session)
1. **REGIME fitness-gate audit** — rolling portfolio session-name mismatch (numeric vs event-based) means regime gate always passes through. Separate fix needed.
2. **Layer 8** — forward test / paper trade the 8-strategy MNQ portfolio
3. **MGC/MES edge investigation** — MGC 6 validated but all below 0.22 ExpR at locked RR. MES 9 validated but all below 0.08 ExpR. Neither qualifies under current live rules.

### Known issues
- Post-edit drift hook fails on module import (PYTHONPATH not set) — masks drift detection during edits
- REGIME fitness gate has session-name mismatch — always resolves FIT (not gating). Flagged in live_config.py comment.
- MGC/MES: 0 live strategies under current rules

---

## Prior Session
- **Tool:** Claude Code (ML Audit + Fix Planning Terminal)
- **Date:** 2026-03-21 (night)
- **Summary:** ML audit complete. Fix plan designed. Live portfolio found EMPTY (42 specs no match). Version gate deployed. Implementation paused for pipeline rebuild.

## Prior Session
- **Tool:** Claude Code
- **Date:** 2026-03-21 (earlier)
- **Summary:** Multi-RR portfolio built. ML audit found 4 FAILs. Bootstrap 5K code committed. Confluence design started. Session crashed mid-brainstorm.

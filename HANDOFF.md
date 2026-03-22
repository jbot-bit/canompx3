# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

## Current Session
- **Tool:** Claude Code (Pipeline Audit → Canon Lock → Interim Rebuild)
- **Date:** 2026-03-22
- **Branch:** `main`
- **Status:** Full audit Layers 1-7 FROZEN. 821 validated. 16 specs. 11 strategies live-resolvable (MGC 0, MES 1, MNQ 10). noise_risk fully populated.

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

### Truth State (verified Mar 22 2026)
- **validated_setups:** 821 rows (MGC 9, MES 81, MNQ 731). All wf_passed=True, all FDR-corrected at K=105,612.
- **family_rr_locks:** 248 rows (MGC 3, MES 40, MNQ 205).
- **edge_families:** 241 rows (MGC 3, MES 40, MNQ 198).
- **LIVE_PORTFOLIO:** 16 specs (10 CORE, 6 REGIME). Resolves 11 strategies (MGC 0, MES 1, MNQ 10).
- **noise_risk:** Fully populated for all instruments. Zero NULLs.
- **Bars:** All instruments current to 2026-03-21. MGC daily_features refreshed to 2026-03-20.
- **ML:** Still frozen. V2 gate active. 6 bugs identified, 0 fixed.

### Next Steps (for incoming session)
1. **Layer 8** — forward test / paper trade the resolved 11-strategy portfolio
2. **MGC edge investigation** — all 9 survivors below noise floor. MGC may be dead for ORB.
3. **LIVE_PORTFOLIO spec review** — 5 specs resolve nothing for any instrument (dead specs)

### Known issues
- Post-edit drift hook fails on module import — masks drift detection during edits
- MGC: 0 live strategies (all 9 validated below noise floor 0.21)
- MES: only 1 live strategy (CME_PRECLOSE ATR70_VOL). Weak instrument for ORB.
- MGC live resolution is 0 because MGC validated set (TOKYO_OPEN ORB_G4) doesn't match any live spec

---

## Prior Session
- **Tool:** Claude Code (ML Audit + Fix Planning Terminal)
- **Date:** 2026-03-21 (night)
- **Summary:** ML audit complete. Fix plan designed. Live portfolio found EMPTY (42 specs no match). Version gate deployed. Implementation paused for pipeline rebuild.

## Prior Session
- **Tool:** Claude Code
- **Date:** 2026-03-21 (earlier)
- **Summary:** Multi-RR portfolio built. ML audit found 4 FAILs. Bootstrap 5K code committed. Confluence design started. Session crashed mid-brainstorm.

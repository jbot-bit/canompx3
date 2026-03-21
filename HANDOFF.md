# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

## Current Session
- **Tool:** Claude Code (Pipeline Audit → Canon Lock → Interim Rebuild)
- **Date:** 2026-03-22
- **Branch:** `main`
- **Status:** Interim validation rebuild COMPLETE. 832 strategies validated. Downstream tables rebuilt. Live_config partially resolves (9/46 specs → 9 strategies).

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

#### 6. Live_config Resolution Test
- MGC: 0 strategies resolved (MGC validated strategies are TOKYO_OPEN ORB_G4, but specs ask for ORB_G5_CONT/ORB_G5)
- MES: 2 strategies (CME_PRECLOSE ORB_G5 + ATR70_VOL)
- MNQ: 7 strategies (CME_PRECLOSE, COMEX_SETTLE, SINGAPORE_OPEN, NYSE_CLOSE)
- 43/46 specs have matching validated strategies in DB; resolution blocked by spec-filter mismatch, not missing data
- live_config LIVE_PORTFOLIO specs need redesign to match the actual validated inventory

### What this is NOT
- **Not final canon.** NOISE_EXPR_FLOOR is zeroed (interim). noise_risk not computed. MGC/MES bars 16 days stale.
- **Not a full ground-up rebuild.** Reused existing experimental_strategies from Mar 19. Only validation re-ran.
- **Not live-ready.** live_config specs don't match new inventory. Specs were designed for prior state.

### Truth State
- **validated_setups:** 832 rows (MGC 20, MES 81, MNQ 731). All wf_passed=True, all FDR-corrected at K=105,612.
- **family_rr_locks:** 255 rows. Fresh (rebuilt Mar 22 from current experimental_strategies).
- **edge_families:** 248 rows. Fresh. MGC 2 ROBUST, MES 2+8 ROBUST/WHITELISTED, MNQ 61+45.
- **noise_risk:** NULL everywhere. Column exists but not computed.
- **Bars:** MGC/MES stale (Mar 6/7). MNQ current (Mar 20).
- **ML:** Still frozen. V2 gate active. 6 bugs identified, 0 fixed.
- **live_config:** Resolves 9/46 specs (2 MES + 7 MNQ). MGC=0 (spec-filter mismatch). LIVE_PORTFOLIO needs redesign.

### Next Steps (for incoming session)
Two paths, pick based on priority:

**Path A: Make live_config work (trading-ready)**
1. Redesign LIVE_PORTFOLIO specs to match actual 832-set inventory
2. Populate noise_risk flag
3. Verify live resolution → paper trade

**Path B: Full upstream refresh (data completeness)**
1. Ingest fresh MGC/MES bars (16 days stale)
2. Rebuild 5m bars → features → outcomes
3. Re-run discovery + validation on fresh data
4. Then Path A

Path A is faster and gets the book live sooner. Path B is more thorough. Either is valid.

### Known issues
- Post-edit drift hook fails on module import — masks drift detection during edits
- 3 live specs have 0 matching validated strategies (dead specs: CME_REOPEN ORB_G4_FAST10, CME_REOPEN ORB_G5, TOKYO_OPEN E1 ORB_G5_FAST10)
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

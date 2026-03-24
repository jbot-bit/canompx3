# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

## Current Session
- **Tool:** Claude Code (FDR Remediation + Truth Protocol + Forward Eval Pack)
- **Date:** 2026-03-24 (session 2)
- **Branch:** `main`
- **Status:** FDR remediation complete. Forward eval pack frozen. Paper book ready to run.

### What was done this session (Mar 24 — session 2)

#### 1. MNQ FDR Remediation (DB-only, no code changes)
- Applied BH FDR to 757 MNQ validated_setups (all had fdr_significant=NULL)
- Used canonical `benjamini_hochberg()` with global K=105,627
- Result: 494 TRUE, 263 FALSE, 0 NULL remaining
- Rebuilt MNQ edge families (228 families: 55 ROBUST, 54 WHITELISTED, 24 SINGLETON, 95 PURGED)
- Refreshed orb_outcomes: MES +432 rows (3 days), MGC +360 rows (3 days), all through 2026-03-23

#### 2. Research Truth Protocol — K-Rule Addition
- RESEARCH_RULES.md: BH K selection rule added (report both global K and instrument K; use instrument K for promotion, global K for headlines; never swap post-hoc)
- .claude/rules/research-truth-protocol.md: mirrored in claim requirements
- Governance lane is now FROZEN. No more protocol/doc changes unless real conflict appears.

#### 3. Forward Eval Pack (pre-registered)
- `docs/plans/2026-03-24-mnq-core5-forward-eval-pack.md`
- 5 CORE MNQ sessions: CME_PRECLOSE, COMEX_SETTLE, NYSE_OPEN, US_DATA_1000, EUROPE_FLOW
- Daily checklist, weekly risk memo template, scoreboard fields
- 8-gate promotion card (fail-closed, human sign-off required)
- Kill criteria: 3 consecutive months negative, cumulative <= -10R, slippage > 2x on 20+ fills
- Monitoring only — no promotion, no portfolio edits, no discovery

#### 4. Decisions
- MGC TOKYO_OPEN: intentionally excluded from LIVE_PORTFOLIO (in PAPER_TRADE_CANDIDATES, pending forward evidence)
- CME_REOPEN: no new unfiltered path; existing live spec NOT removed (separate audit)
- 3 lanes separated: governance (FROZEN), truth of edge (MNQ unfiltered baseline real), live deployment (NOT YET)

### Next actions
1. **Run paper book daily** per forward eval pack checklist
2. **Do NOT reopen governance** — protocol is frozen
3. **Do NOT edit LIVE_PORTFOLIO** — current live specs stay until separate audit
4. **C1 null rerun** still paused at chunk 2 of 10 (lower priority than paper monitoring)

### MNQ Unfiltered Baseline — Paper-Trade Control Note (2026-03-24)

**Finding:** MNQ E2, 5m ORB, RR1.0, CB1, NO_FILTER is positive across 6 sessions.
Verified by: source-of-truth audit, anti-bias audit, no-lookahead audit, refreshed data (1475 days).

**5 CORE sessions** (BH FDR PASS at K=105,627, WF PASS, all years positive):
| Session | N | ExpR | p | WFE | OOS ExpR | OOS %Pos |
|---------|---|------|---|-----|----------|----------|
| CME_PRECLOSE | 1108 | +0.200 | <1e-8 | 0.77 | +0.191 | 89% |
| US_DATA_1000 | 1310 | +0.137 | 2e-7 | 1.00 | +0.138 | 100% |
| COMEX_SETTLE | 1272 | +0.124 | 1e-6 | 1.50 | +0.141 | 100% |
| NYSE_OPEN | 1305 | +0.117 | 1e-5 | 1.61 | +0.131 | 100% |
| EUROPE_FLOW | 1324 | +0.101 | 3e-5 | 1.15 | +0.109 | 100% |

**1 REGIME session** (marginal — WFE=0.53, p=0.010, 67% WF positive, 2026 marginal):
| TOKYO_OPEN | 1325 | +0.062 | 0.010 | 0.53 | +0.062 | 67% |

**Paper-trade specs:** `PAPER_TRADE_CANDIDATES` in `trading_app/live_config.py`. NOT in LIVE_PORTFOLIO.
**Kill criteria:** 3 consecutive months negative OR cumulative -10R.
**Forward baseline target:** +0.12R/day (5-session equal-weight portfolio mean).
**2026 holdout:** SACRED. Not used for discovery. Same 6 sessions selected from pre-2026 data.
**Doc changes:** TRADING_RULES.md ORB table updated, config.py NO_FILTER comment updated.

### Project Truth Protocol (enforced 2026-03-24)

**CANONICAL layers (safe for discovery):** `bars_1m`, `daily_features`, `orb_outcomes`.
**DERIVED layers (banned for discovery):** `validated_setups` (757 MNQ fdr=NULL), `edge_families`, `live_config`, docs/comments.
**Rule:** If derived layer contradicts canonical query, canonical wins. Mark derived layer STALE.
**Written into:** `CLAUDE.md` (Guardrails > Project Truth Protocol), `RESEARCH_RULES.md` (Discovery Layer Discipline).
**PASS 4 verified:** monitor script output matches raw SQL exactly (14/14 metrics, 10/10 trades).

### What was done this session (Mar 24 — C1 null rerun + sync)

#### 1. C1 Time-Varying Null Rerun (IN PROGRESS)
- Previous run crashed (orphan workers, 62 processes). 3 seeds salvaged (7, 18, 19) with 14 total survivors.
- Max null OOS ExpR provisionally 0.2369 — may challenge current 0.21 floor.
- 3 script changes to `run_null_batch.py`: per-seed manifest save, enriched checkpoint (date_range, max_oos_expr, output_path), fail-closed date range guard.
- Running 10 chunks of 10 seeds, parallel 6, IS-only 2020-01-01 to 2025-12-31.
- Chunk 1 DONE: 10/10 complete, 0 new survivors, all date ranges verified.
- Chunk 2 RUNNING.

#### 2. Canonical Sync
- `chatgpt-project-kit/TRADING_RULES.md` synced (was missing Layer 7 execution realism section)
- `chatgpt-project-kit/PROJECT_INSTRUCTIONS.md` updated: ML institutional audit, live portfolio count, C1 status, orphan gate
- `chatgpt-project-kit/PROJECT_REFERENCE.md` updated: strategic direction Mar 24, ML audit finding

### What was done prior session (Mar 24 — guardian audit)

#### 1. Live Execution Bug Audit (Pass 1 — audit only)
- Full source-of-truth chain traced: PositionTracker owns state, SessionOrchestrator owns containment
- Bug A (PENDING_EXIT stuck): PENDING_EXIT is HONEST broker state, not a bug. Added visibility (log.critical + _notify) so overnight failures are seen.
- Bug B (rollover orphan): Rollover close failure leaves positions the engine forgets on reset. Without containment, engine re-enters same strategy = doubling up at broker.
- State transition table, unaffected path proof, blast radius, diff budget, stop conditions — all documented before writing code.

#### 2. Implementation (3 commits)
- `761308e` — PENDING_EXIT visibility (log.critical + _notify) + rollover orphan detection
- `7763b59` — _notify on exit failure path (review finding: log alone invisible overnight)
- `6f81cfc` — fail-closed orphan containment gate: `_blocked_strategies` set blocks entries for orphaned strategies. No auto-clear. Manual resolution required.

#### 3. Documentation + Coverage Pass
- `3dfac08` — Fixed stale recovery comment (referenced overwrite path now blocked by gate), documented intentional double-notify, added partial rollover test (2 strategies, 1 fails = only failed one blocked)

#### 4. Code Review (2 independent passes)
- Pass 1: pr-review-toolkit:code-reviewer — all 9 edge cases verified, all checklist items PASS
- Pass 2: second pr-review-toolkit run — SHIP verdict, 0 Critical/Important findings
- 92/92 orchestrator tests, 20/20 position tracker tests, 75/75 drift checks

### What was done prior session (Mar 23-24 — review terminal)


### What was done this session (Review Terminal, Mar 23-24)

#### 1. 2-Day Code Review (4 parallel review agents)
- 12 production bugs found and fixed across 9 files
- C1: Portfolio ATR gate crash (TypeError), C3: BH total_tests param + IndexError fix
- C4+C5: 5 test regressions fixed (pipeline_status per-aperture + seasonal gate)
- H1: holdout check fail-closed, H3: seasonal gate in REGIME/HOT, H4: schema verification 14 cols
- M1: holdout JSON extraction, M3: dead instruments docs, P1: post-edit hook PYTHONPATH
- Drift checks: 18 violations → 3 (all ML #61, frozen)
- Post-edit hook now fully operational (was broken by PYTHONPATH)

#### 2. MGC Institutional Truth Audit
- Full raw-verified audit (all metadata matched raw recomputation)
- Sigma overshoot: 2.82x trimmed but trade-weighted match 1.03x in 2025 regime
- Sigma-invariance PROVEN: R-multiple floors don't depend on sigma (MGC vs MNQ test)
- Real Gaussian defects: fat tails (kurtosis 374-8950), vol clustering (autocorr 0.64-0.91), session structure
- Noise floor is approximately correct for tradeable regime, not artificially strict
- RR lock: HONEST BUT CLIFFY (flat Sharpe surface, CV=5.2%). Not binding — noise_risk blocks all RR levels
- Verdict: MGC 0-live stands. Shadow only.

#### 3. Shadow Tracker Activated
- `docs/pre-registrations/mgc-tokyo-shadow-ACTIVE.md` — frozen pre-registration
- `docs/pre-registrations/shadow-logs/mgc-tokyo-shadow.csv` — empty, header-only
- Candidate: MGC_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G4_S075 (verified from raw)
- Forward-only start: 2026-03-21. Zero capital. Kill-switches defined.
- No production impact. Not in LIVE_PORTFOLIO. Not CORE.

#### 4. Block Bootstrap Audit
- Verdict: NO CHANGE. Bootstrap is methodologically better but would produce EQUAL OR HIGHER floors
- MES/MNQ: floors go UP (stricter). MGC: direction UNCERTAIN but blocked regardless
- 225 hours compute for no tradability change. Not justified.
- Other terminal running sign-randomization bootstrap (faster alternative)

#### 5. Tooling Fixes
- null_envelope.py: fixed MAX→P95 floor computation (matched production)
- websockets installed, spa_test configure_connection added
- ML model check #51 → advisory (ML frozen)
- Dead instruments regex: fixed cross-function false positive

### Prior Session (same page, earlier Mar 23)
- **Tool:** Claude Code (Canonical Refresh terminal)
- See sections below for full canonical refresh details (FDR K fix, pipeline_status, live_config redesign, etc.)

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

#### 5. REGIME Fitness-Gate Audit (misdiagnosis corrected)
- **REGIME gate works correctly.** `compute_fitness()` loads outcomes directly from
  `orb_outcomes` — independent of rolling portfolio tables. All 6 REGIME strategies
  are genuinely FIT (rolling ExpR 0.11-0.35, N 40-228, recent Sharpe all positive).

#### 6. Rolling Portfolio Rebuild (event-based session names)
- **Root cause:** Rolling tables (`regime_strategies`, `regime_validated`) had stale
  numeric session names (0030/0900/1000/1100/1800/2300) from before Feb 2026 rename.
  HOT-path `_check_rolling_stability()` always returned "family not found" (0.0) because
  it queried with event-based names that never matched stale numeric labels.
- **Bug found during rebuild:** `regime.discovery` lines 66-67 deleted by `run_label`
  without `AND instrument = ?`, so sequential multi-instrument runs wiped each other.
  Fixed: scoped delete to `(run_label, instrument)`.
- **Rebuild:** All 3 instruments (MNQ, MGC, MES), 12m+18m windows, 2024-07 to 2026-03.
  regime_strategies: 500,748 rows (MNQ 190,512, MES 181,044, MGC 129,192).
  regime_validated: 8,795 rows (MNQ 6,275, MES 1,414, MGC 1,106).
- **Live portfolio impact:** MES gained 1 strategy (`MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR70_VOL`,
  source=rolling, ExpR=+0.262, stability 1.000 = 10/10 windows). This is a CORE spec
  with `regime_gate=None` — it was blocked by missing variant data, not by a stability gate.
- **Verification (all PASS):** orb_labels event-based only, validated_setups/edge_families/
  family_rr_locks unchanged (404/162/166), strategy_fitness identical, MGC/MNQ counts unchanged.

### Truth State (verified Mar 23 2026)
- **validated_setups:** 404 rows (MGC 6, MES 9, MNQ 389). All wf_passed=True, FDR-corrected at K=105,640.
- **family_rr_locks:** 166 rows (MGC 2, MES 7, MNQ 157).
- **edge_families:** 162 rows (MGC 2, MES 7, MNQ 153).
- **LIVE_PORTFOLIO:** 8 specs (2 CORE, 6 REGIME). Resolves MNQ 8, MES 1, MGC 0 (9 total).
- **noise_risk:** Fully populated. Zero NULLs.
- **Bars:** All instruments current to 2026-03-21. daily_features/outcomes to 2026-03-20.
- **ML:** Still frozen. V2 gate active.

#### 7. Threshold-Grounding Audit (commit `a9c40cb`, `75abdbf`)
- All 9 rolling/regime thresholds sensitivity-tested +-20% from live DB.
- 8/9 ROBUST (no live cliff). 1 SENSITIVE: DOUBLE_BREAK_THRESHOLD=0.67.
- BUG FIXED: `compute_double_break_pct` had no instrument filter (blended 7 instruments incl 4 dead).
  Per-instrument margins: MNQ 7.6pp, MES 5.0pp, MGC 1.7pp.
- All thresholds annotated with @research-source, @sensitivity-tested, @heuristic where applicable.
- Proximity warning added (DOUBLE_BREAK_PROXIMITY_WARN=0.05).

#### 8. MGC Research Truth Audit (CORRECTED Mar 24 — friction debunked)
- **57% friction claim DEBUNKED.** Used unfiltered ORB median (1.0pt). At G4+ filter:
  median risk=6.3pt, friction=9.1% of risk. Comparable to MNQ (7.0%), better than MES (15.7%).
- **Dollar gate PASSES.** $11.71 expected profit > $7.46 threshold.
- **True binding blocker: RR lock + LIVE_MIN.** MAX_SHARPE locks RR1.0 where full-sample
  ExpR=0.186 < LIVE_MIN=0.22. At RR1.5: ExpR=0.235 > LIVE_MIN. This is a design choice, not a bug.
- **noise_risk is secondary under current RR lock; becomes decision-relevant if RR policy changes.** Sigma miscalibrated 2.21x (should be 0.54, not 1.2).
  5-seed pilot: 0 E2 survivors at sigma=0.54 vs 1827 at sigma=1.2. Floor drops dramatically.
  `calibrate_null_sigma.py` built (uncommitted). 100-seed rerun deferred until RR policy resolved.
- **One honest positive island:** TOKYO_OPEN E2 ORB_G4/G4_CONT, N=175-177 (PRELIMINARY),
  ExpR=0.19-0.26, all wf_passed=True. Real edge under honest gates.
- **Verdict: NOT LIVE, SHADOW-TRACK ONLY.** MGC blocked by RR lock + LIVE_MIN interaction,
  not friction. Path to live exists IF RR lock policy relaxed AND noise floor recalibrated.
- **Next task: RR policy audit** — is MAX_SHARPE→RR1.0 the right lock policy when RR1.5
  has higher ExpR? Design question, not a bug fix.

#### 9. #82 Holdout Contamination Cleanup
- MNQ experimental_strategies cleared and re-discovered with `--holdout-date 2026-01-01`
- 3 apertures (O5/O15/O30) rebuilt, 31,104 strategies. 757 validated (was 389).
- Edge families: 228 (was 153). RR locks: 249 (was 157).
- **MNQ live: 7** (was 8, lost CME_PRECLOSE_E2_X_MGC_ATR70_S075 — honest holdout effect).
- Drift check #82: NOW PASSING. Down from 18 violations to 3 (all in frozen ML code).

#### 10. MGC Research Truth Audit
- See section 8 (corrected MGC truth block) for canonical MGC findings.
- Key additions from institutional audit: noise floor sigma overshoot (MGC 2.54x trimmed std)
  is a known method defect with unproven bias direction. Block bootstrap calibration needed.
- p_values propagated to validated_setups (772/772, verified matching).

#### 11. Spec Policy Audit
- **VERDICT: SPEC SET BIASED + GENUINE STRUCTURAL WEAKNESS.**
- Code is instrument-agnostic. Spec set covers CME_PRECLOSE/COMEX_SETTLE/CME_REOPEN.
- MES/MGC edges are on different sessions, all blocked by noise_risk or double-break regardless of specs.
- No spec expansion warranted — would produce 0 new live strategies.
- Latest-window-only fragility is real but causes 0 actual false negatives currently.

### Truth State (verified Mar 23 2026, post-#82 cleanup)
- **validated_setups:** 772 rows (MGC 6, MES 9, MNQ 757). All wf_passed=True, holdout-clean.
- **family_rr_locks:** 258 rows (MGC 2, MES 7, MNQ 249).
- **edge_families:** 237 rows (MGC 2, MES 7, MNQ 228).
- **regime_validated (rolling):** 8,795 rows. All event-based session names.
- **LIVE_PORTFOLIO:** 8 specs (2 CORE, 6 REGIME). Resolves MNQ 7, MES 1, MGC 0 (8 total).
- **noise_risk:** Fully populated. Zero NULLs. MNQ 50 clean (was 42).
- **Drift checks:** 74 pass, 3 violations (all ML #61, frozen). #82 RESOLVED.
- **ML:** Still frozen. V2 gate active. NOT READY (0 clean MGC/MES baselines).

#### 12. RR Policy Audit + JK Fallback (commit `6a0d853`)
- **Audit:** MAX_SHARPE has 73% RR1.0 bias (mechanical Sharpe advantage at lower RR).
  Sharpe monotonically decreases with RR system-wide. 44% of families suppress higher-ExpR alt.
  7 families where suppression crosses LIVE_MIN gate.
- **Fix:** JK-equal liveability tiebreaker in `_load_best_regime_variant`. When locked RR
  fails LIVE_MIN, tries JK-equal alternatives (vs locked RR's Sharpe, rho=0.7, p>0.05).
  Among gate-passers: FDR-significant first, then highest Sharpe.
- **Result:** MGC TOKYO_OPEN: RR1.0 (ExpR=0.186) -> RR1.5 (ExpR=0.235, p=0.894).
  RR blocker REMOVED. noise_risk=True is now binding. MGC still 0-live.
- **No impact:** MNQ 7 strategies, MES 1 strategy — unchanged. family_rr_locks untouched.
- **Code reviewed (2x):** FDR priority fix applied. All callers updated. No import cycle.

### Next Steps (for incoming session)
1. **Layer 8** — forward test / paper trade the 8-strategy portfolio (MNQ 7 + MES 1)
2. **MGC noise floor rerun** — 100-seed with sigma=0.54, now decision-relevant (RR resolved)
3. **Phase 3 fairness audit** — MGC 11 years vs MNQ 6: is "all years positive" the right test?
4. **TOKYO_OPEN spec** — consider adding to LIVE_PORTFOLIO if noise floor clears
5. **Live execution** — orphan containment gate is ready for paper/live sim testing

### Known issues
- ML #61: 3 violations in features.py (frozen, fix when ML unfrozen)
- DOUBLE_BREAK_THRESHOLD=0.67: HEURISTIC, proximity warning active
- MGC: 0 live — noise_risk is binding blocker (RR lock resolved via JK fallback)
- Spec set: MNQ-shaped by construction, not by architecture bug
- Live execution: orphan containment is session-scoped (process restart clears _blocked_strategies). Startup orphan check (query_open) covers restart case independently.
- Pre-existing (NOT blocking): `on_exit_sent` in position_tracker.py doesn't guard against duplicate PENDING_EXIT transition (unreachable in practice — called once before retry loop)
- Pre-existing (NOT blocking): `_emergency_flatten` doesn't cancel brackets before exit (broker order management handles this)
- Scratch accounting (VERIFIED, CLOSED): 12-15% of RR1.0 outcomes are scratches (pnl_r=NULL, excluded from ExpR/WR/N). Intentional design (strategy_discovery.py:428-430). TRUE session-end MTM VERIFIED from bars_1m: MNQ avg=+0.004R med=0.0R, MES avg=-0.007R med=0.0R, MGC avg=-0.041R med=-0.049R. Zero live strategy sign flips under 0R/-0.25R/-0.50R stress. Recheck if portfolio moves to RR2.0+ or MGC goes live.

---

## Prior Session
- **Tool:** Claude Code (ML Audit + Fix Planning Terminal)
- **Date:** 2026-03-21 (night)
- **Summary:** ML audit complete. Fix plan designed. Live portfolio found EMPTY (42 specs no match). Version gate deployed. Implementation paused for pipeline rebuild.

## Prior Session
- **Tool:** Claude Code
- **Date:** 2026-03-21 (earlier)
- **Summary:** Multi-RR portfolio built. ML audit found 4 FAILs. Bootstrap 5K code committed. Confluence design started. Session crashed mid-brainstorm.

---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Profit Extraction Roadmap — 2026-04-15 Master Plan

**Status:** MASTER PLANNING DOC. Updated when evidence or priorities change.
**Scope:** Every potential profit avenue for the canompx3 trading system, rated by evidence, cost, blast radius, and priority.
**Trigger:** User demanded "step back, systematically and thoroughly enumerate all potential areas of profit, no bias or lookahead or pigeonholing, logical trader brain."
**Governing rules:** `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, `.claude/rules/backtesting-methodology.md`, `docs/institutional/pre_registered_criteria.md`.

---

## Section 1 — Current state (honest baseline)

### Portfolio
- **1 account live:** `topstep_50k_mnq_auto` (TopStep 50K XFA, practice-account signal-mode currently).
- **6 MNQ lanes deployed** per `docs/runtime/lane_allocation.json` (2026-04-13 rebalance):

| Lane | Session | RR | Filter | Trailing ExpR | R/yr |
|---|---|---|---|---|---|
| L1 | EUROPE_FLOW | 1.5 | ORB_G5 | +0.185 | 44.3 |
| L2 | SINGAPORE_OPEN O30 | 1.5 | ATR_P50 | +0.241 | 44.0 |
| L3 | COMEX_SETTLE | 1.5 | OVNRNG_100 | +0.261 | 39.8 |
| L4 | NYSE_OPEN | 1.0 | ORB_G5 | +0.119 | 28.2 |
| L5 | TOKYO_OPEN | 1.5 | ORB_G5 | +0.089 | 21.6 |
| L6 | US_DATA_1000 O15 | 1.5 | VWAP_MID_ALIGNED | +0.210 | 22.1 |

Total ~200 R/yr at 1ct MNQ. TopStep cap = 1ct (Apr 7 2025 tail-day constraint).

### Validated universe
- **124 total validated strategies** in `validated_setups` (MNQ 101, MES 14, MGC 9).
- Only MNQ currently deploys live per TopStep scaling constraints.
- Research-only: MES/MGC lanes available if account profile activates them.

### Hard constraints (immutable today)
- TopStep 1ct MNQ cap (MLL tail-day protection)
- 2026-01-01 sacred holdout for new discovery
- F-1 orchestrator wiring needed before any live XFA (latent — live today = signal mode)
- Flat-risk-per-trade assumption in execution_engine / risk_manager

### Known unknowns
- DSR at honest N_eff for existing validated strategies
- How our lanes perform under correlation stress on multi-lane-fire days
- Monte Carlo MLL survival for any sizing variation

---

## Section 2 — Profit avenue enumeration

Seven ways the portfolio can make more money. Listed completely so nothing is pigeonholed away.

### Avenue A — Better existing deployed lanes (portfolio optimization)
Swap underperforming lanes for higher-trailing-ExpR validated alternatives.
- **Example:** L5 TOKYO_OPEN ORB_G5 trailing +0.089 — could be swapped for a higher-ExpR TOKYO_OPEN validated alternative.
- **Evidence:** already in `validated_setups`, trailing stats live.
- **Constraint:** correlation gate (`trading_app/lane_correlation.py`) and session uniqueness per allocator.

### Avenue B — Add NEW validated strategies (expand portfolio)
Run discovery → validator → pass all 12 criteria → register new validated_setup.
- **Example seed (strongest evidence this session):** MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_SKIP_GARCH_70 (overlay on L4 with inverse-garch skip).
- **Evidence bar:** all 12 pre_registered_criteria must pass (hypothesis file, BH-FDR, DSR, WF, Monte Carlo).
- **Blast radius:** one new filter in `trading_app/config.py`, one new strategy_id in validated_setups.

### Avenue C — Regime overlays / sizer (R3/R5 in mechanism_priors.md)
Apply directional or sizing tilts to existing lanes based on orthogonal regime signals (garch, rel_vol, etc.) WITHOUT changing the base filter.
- **Strongest evidence this session:** garch all-sessions scan shows 21/32 (inst, session) families have directional garch effect (BH-FDR on binomial sign at K=32).
- **Blast radius:** high — touches execution_engine, risk_manager, HWM.
- **Gating:** Topstep 1ct cap binds; cannot upsize. Only viable as **SKIP signals** (avoid trades) within current Topstep context.

### Avenue D — New entry models (E_RETEST, E3_RETRACE)
Currently all deployed = E2 (stop-market at break). E_RETEST (limit-on-retest after failed first break) is a different trade timing on the same edges.
- **Evidence:** spec exists at `docs/audit/hypotheses/phase-c-e-retest-entry-model.md`.
- **Blast radius:** NEW entry model in outcome_builder, new backtest path, new execution logic. 2-3 weeks infrastructure.
- **Upside:** potentially doubles trade count on existing sessions with uncorrelated timing.

### Avenue E — Non-ORB strategy classes (Phase E SC2.x)
Fade, mean-rev-within-range, cross-asset, vol-overlay. Fundamentally different trade class.
- **Evidence:** stub at `docs/audit/hypotheses/phase-e-non-orb-strategy-class.md`.
- **Blocker:** Dalton (Markets in Profile) + Murphy PDFs not acquired — literature gate not passed.
- **Potential:** uncorrelated returns = true diversification.

### Avenue F — HTF level signals (Path A)
Prev-week / prev-month high-low breaks + first-touch hypothesis on ORB cells.
- **Evidence:** stub at `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md`.
- **Blocker:** `prev_week_*`, `prev_month_*` features not built. 2-4h pipeline feature-engineering.
- **Upside:** big-structure signals, trader-intuitive, economically grounded (Fitschen + Murphy).

### Avenue G — Multi-firm scaling (horizontal)
Same lanes, more accounts. TopStep 5 XFA + Bulenox 11 + MFFU 5 = 21 concurrent accounts per MEMORY topstep_scaling_corrected_apr15.
- **Evidence:** prop firm research complete; canonical audit in `docs/audit/2026-04-15-topstep-scaling-reality-audit.md`.
- **Blocker:** 
  - TopStep XFA↔LFA exclusivity (destroys XFAs on LFA promotion)
  - Need Rithmic integration to unlock Bulenox
  - MFFU 5-sim + 5-DTF structure requires different execution wiring
- **Upside:** linear scaling of existing edge. No new research needed.

---

## Section 3 — Evidence matrix per avenue

| Avenue | Evidence strength | Data quality | Replication | Ready to pre-reg? |
|---|---|---|---|---|
| A — swap lanes | Strong (already validated) | Live trailing 12mo | N/A | N/A (execution task) |
| B — NYSE_OPEN SKIP_GARCH | **Strong** (p=6.9×10⁻⁸ BH-FDR K=32) | Canonical 6+ yr | Cross-instrument partial | **YES — tonight** |
| B — Phase-D TOKYO sizer | Strong sign-pattern (51/54) | 6+ yr | Cross-instrument full | Yes — after Topstep cap resolution |
| C — garch multi-session overlay | 21/32 families significant | Canonical | Sign-test aggregation | Shadow-log first |
| D — E_RETEST entry model | Spec only; no data yet | Needs infra | N/A | No — build first |
| E — non-ORB (SC2.1-4) | Stub only; literature-blocked | N/A | N/A | No — acquire literature |
| F — HTF levels | Stub only; features not built | N/A | N/A | No — build features first |
| G — multi-firm scaling | Research-complete | Firm docs | N/A | Post-Rithmic integration |

---

## Section 4 — Implementation cost per avenue

| Avenue | Effort | Files touched | Tests needed | Risk |
|---|---|---|---|---|
| A — lane swap | 30min (allocator rebalance) | `docs/runtime/lane_allocation.json` | None (allocator gate checks) | Low |
| B — NYSE_OPEN SKIP pre-reg + filter | 2-4h | `trading_app/config.py` (+1 filter), `docs/audit/hypotheses/<date>-nyse-open-skip.md`, validator run | Filter unit + sync drift | Low |
| C — shadow logger + dashboard | 1-2 weeks | New `shadow_journal.py`, schema update, dashboard tile | Schema drift, logger tests | Medium (no behavior change, only additive) |
| D — E_RETEST entry model | 2-3 weeks | `pipeline/outcome_builder.py`, `trading_app/execution_engine.py`, config registry, new drift checks | Outcome-builder regression + entry-model validator | High (new execution path) |
| E — non-ORB classes | 1-3 months | New strategy class framework, new entry engine, new validation path | Extensive | Very high (new product) |
| F — HTF levels | 2-3 weeks | `pipeline/build_daily_features.py` + new columns + pre-reg + scan | Feature drift + data-coverage | Medium (pipeline change) |
| G — multi-firm scaling | 1-3 months | New account profile types, broker integrations (Rithmic, TradingView, AMP), risk-manager multi-account | Extensive | High (broker integrations) |

---

## Section 5 — Blast radius per avenue

### Low blast (safe to execute)
- **A, B, C shadow-only:** additive or configuration-only. Validator gates protect against bad promotions.

### Medium blast
- **F (HTF features):** pipeline schema change. `pipeline/build_daily_features.py` plus drift check updates. Rollback via migration.
- **Lane swaps (A):** execution unchanged; only which validated lane the allocator picks. Self-healing via next rebalance.

### High blast
- **D (E_RETEST):** new outcome-builder path. All downstream relies on outcome columns. Extensive regression risk. Staged stage-gate approach mandatory.
- **C sizer deployment (post-shadow):** touches risk_manager, HWM, execution_engine, prop_profiles. Weeks of testing.
- **E (non-ORB):** new strategy class framework. Fundamental.
- **G (multi-firm):** broker-specific integration. Each firm = new product surface.

---

## Section 6 — Priority ranking (Expected Value × Probability / Cost × Risk)

Ranking = honest trader assessment. Numbers are rough orders-of-magnitude, not precise forecasts.

| Rank | Avenue | EV | Prob | Cost | Risk | Score | Do it? |
|---|---|---|---|---|---|---|---|
| 1 | **NYSE_OPEN SKIP_GARCH_70 pre-reg + filter** | +5-10 R/yr on L4 | 70% | 4h | Low | **Excellent** | **YES — this session** |
| 2 | Lane-swap audit (L5 TOKYO ExpR=+0.089 — can we do better?) | +5-15 R/yr | 50% | 1h | Low | Strong | Next session |
| 3 | Shadow-decision logger for garch multi-session | +0 direct, data for Stage 5 | 90% | 1-2w | Medium | Strong infrastructure | Stage B |
| 4 | HTF level features (Path A) | +10-30 R/yr potential | 30% | 3-5h build + scan | Medium | Good | After 1-3 land |
| 5 | Pre-reg TOKYO/SINGAPORE/EUROPE garch sizer family | 0 today (Topstep cap) | — | 1h | Low | Doc-only for now | File pre-reg, wait for cap unlock |
| 6 | E_RETEST entry model (Phase C) | +50-100% throughput if works | 20% | 2-3w | High | Strategic bet | Mid-priority |
| 7 | Non-ORB SC2.x (Phase E) | Uncorrelated +unknown | 15% | 1-3mo | Very high | Strategic bet | After literature |
| 8 | Multi-firm scaling prep | Linear multiplier | 60% | 1-3mo | High | Capex-heavy | After Rithmic integration |

### The take
- **This session:** ship the #1 action — NYSE_OPEN SKIP pre-reg + filter registration.
- **Next 30 days:** #2 lane-swap audit + file sizer-family pre-regs (#5).
- **Next 90 days:** build HTF features (#4) and begin shadow-logger infrastructure (#3).
- **Next 12 months:** E_RETEST entry model (#6), non-ORB (#7), multi-firm scaling (#8).

---

## Section 7 — 30-day action plan (concrete)

### Week 1 (now–next session)
1. ✅ **TONIGHT:** NYSE_OPEN SKIP_GARCH_70 pre-reg file filed (docs/audit/hypotheses/). Validator-ready.
2. **Next session:** Register `ORB_G5_SKIP_GARCH_70` filter in `trading_app/config.py`. Run discovery + validator. If passes 12 criteria → new validated_setup. If fails → document and kill.
3. **Next session:** Run lane-swap audit — for each of L1-L6, compare deployed strategy ExpR trailing 12mo vs available alternatives in validated_setups at same (session, instrument, apt). Propose swaps where trailing alternative ExpR > deployed × 1.25.
4. **Next session:** File sizer-family pre-regs for TOKYO_OPEN, SINGAPORE_OPEN, EUROPE_FLOW, LONDON_METALS garch-positive findings. Doc-only (no deployment). Pre-reg enables future Monte Carlo + pilot if Topstep cap lifts.

### Week 2-3
5. Build HTF level features in `pipeline/build_daily_features.py`: `prev_week_high/low/close`, `prev_month_high/low/close`. ~3-5h of work. Drift check for new columns.
6. Run Path A HTF-level discovery scan per stub `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md`.
7. Begin shadow-decision logger design document (not implementation yet).

### Week 4
8. Phase E non-ORB — acquire Dalton and Murphy PDFs (personal time / request). Extract literature to `docs/institutional/literature/`. Write first-pass SC2.1 direct-level-fade spec.

---

## Section 8 — 90-day plan

### Discovery/research
- Path A HTF-level scan complete → survivors T0-T8 → pre-reg shadow if any pass.
- All sizer pre-regs filed (garch TOKYO/SINGAPORE/EUROPE/LONDON — each a session-specific hypothesis).
- E_RETEST entry-model design doc finalized (before any code).
- Non-ORB SC2.1 literature-grounded spec complete.

### Infrastructure
- Shadow decision table + logger shipped (Stage A from earlier framework).
- Dashboard garch regime tile for pre-session operator discretion (Stage B).
- Rithmic integration research (for multi-firm scaling prep).

### Portfolio
- Any new validated strategies from Path A (expected 0-3) registered.
- NYSE_OPEN SKIP_GARCH variant deployed if validator-pass.
- Quarterly lane-swap audit baked into routine.

---

## Section 9 — 12-month plan

- E_RETEST entry model launched (if spec + shadow data show edge).
- Phase D Carver forecast combiner pilot on 1 MNQ lane (post-shadow, post-MLL Monte Carlo).
- Non-ORB SC2.1 or SC2.2 (whichever has stronger evidence) pilot.
- Multi-firm scaling: Bulenox or MFFU account opened as 2nd firm proof-of-concept.
- DSR re-audit all 124 validated_setups with empirically calibrated var_sr.
- Holdout extended forward → 9-12 months OOS for any new strategies.

---

## Section 10 — Honest risk register

### Methodological
- **Data-snooping risk.** We've run many scans this session. K_global=1579 in latest scan. Relying on family-level sign tests mitigates but doesn't eliminate.
- **Regime change.** 5.5-year IS window covers COVID-2020 vol, 2022-23 bear, 2024-25 grind. 2026+ regime may not match.
- **Topstep regime coupling.** All our edges are measured against 1ct sizing. Correlation stress on big-risk-day multi-lane fires is a blind spot.

### Execution
- **F-1 latent gate.** Live XFA requires orchestrator wiring of `RiskLimits.topstep_xfa_account_size` which is currently None. Must be done before any real-money XFA.
- **Broker integration fragility.** TopStep bot must be API-based; if native-platform blocks bots, the whole pipeline stops.
- **Signal latency.** Garch feature needs 252 prior closes to compute; new instruments need 1yr history before garch is available.

### Strategic
- **Opportunity cost of research vs implementation.** Every week on new discovery = week not scaling existing edge. Balance required.
- **Topstep-only ceiling.** Even with 5 XFA + LFA, ~$25K/yr realistic TopStep-alone. Multi-firm needed for meaningful scaling.
- **Psychological / operator risk.** All discretionary operator-tile proposals require operator discipline. Bot-only is safer but less flexible.

### Acceptable risk
- Lane swaps (well-understood)
- Pre-reg only (no deployment)
- Shadow logging (no behavior change)
- NYSE_OPEN SKIP as NEW filter (replaces nothing, validator gates)

### Unacceptable risk (won't do without explicit approval)
- Production sizing changes
- Schema changes without drift check coverage
- Multi-account deployment without Rithmic integration tested
- Any filter that's not registered in `ALL_FILTERS` and validated

---

## Appendix — What was explored this session (for memory)

- Path C: H2/rel_vol book closed. Composite subsumed. DSR ambiguous at honest K.
- H2 exploitation audit: garch appears to add sizer edge conditional on deployed filters (later revised).
- Cross-lane R5 replication: 0/36 survivors at K=36 — portfolio-wide R5 sizer REJECTED.
- Institutional trader-discipline battery: H2 passes K_lane=5 BH-FDR, variance ratio ~1.0, not leverage illusion.
- Cross-instrument replication: 8/8 COMEX_SETTLE cells directionally positive; binomial sign p=0.0039; LONDON_METALS does NOT transfer.
- All-sessions universality: 21/32 (inst, session) families BH-FDR significant on binomial sign. Strongest SKIP finding: MNQ_NYSE_OPEN inverse (p=6.9×10⁻⁸).

Supersedes: `docs/audit/results/2026-04-15-path-c-h2-closure.md` (outdated framing), `docs/audit/results/2026-04-15-path-c-self-audit-addendum.md` (superseded by final verdict).

---

**Status:** Living document. Update after each major session with new evidence, completed avenues, or re-prioritization.

**Next review:** After NYSE_OPEN SKIP pre-reg + validator run (next session).

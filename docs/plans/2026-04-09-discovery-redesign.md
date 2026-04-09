# ORB Discovery Redesign — Institutional-Grade Hypothesis Framework

**Date:** 2026-04-09
**Status:** DESIGN — awaiting user approval before implementation
**Trigger:** Mode A discovery produced 5 MNQ survivors from 37 hypotheses. The hypothesis files were poorly designed (MES copied from MNQ, threshold sweeps inflating K). This redesign fixes the discovery PROCESS.

---

## 1. Principles (non-negotiable)

These are grounded in the local literature extracts and project rules. Each has a citation.

| # | Principle | Source |
|---|-----------|--------|
| P1 | **Theory-first.** Economic mechanism must exist BEFORE backtesting. | LdP 2020 §1.2: "Without a testable theory that explains your edge, the odds are that you do not have an edge at all." |
| P2 | **One prediction per mechanism-session.** No threshold sweeps within bundles. Each hypothesis = one instrument × one session × one filter × one RR. | LdP 2020 §1.4.2: within-hypothesis parameter sweeps are optimization, not prediction. |
| P3 | **Size-first.** ORB size IS the edge. All hypotheses use G-filter gating. | TRADING_RULES "ORB Size = The Edge": "strip the ORB size filter → edge dies. This is arithmetic, not statistics." |
| P4 | **Honest K.** BH FDR family = ALL hypotheses in the file. No post-hoc family subsetting. | Harvey-Liu 2015, Chordia et al 2018 (t ≥ 3.00 with theory). |
| P5 | **Bailey MinBTL.** N ≤ exp(T_cal/2) at E=1.0, where T_cal = calendar years (matching pipeline Sharpe annualization). | Bailey et al 2013 Theorem 1 + 2026-04-09 audit: T must use calendar years. |
| P6 | **Holdout sacred.** 2026-01-01 onward is forward judgment only. Never touches discovery. | Amendment 2.7 (Mode A). |
| P7 | **No contamination from prior results.** Threshold selection must be grounded in structural arguments (friction arithmetic, contract economics), not in what "worked before." | LdP 2020: backtest-tweak-backtest forbidden. |

---

## 2. Data Integrity Gate (Phase 0 — run before any discovery)

Before any hypothesis file is executed, verify:

| Check | Method | Pass criteria |
|-------|--------|--------------|
| Zero parent symbols in canonical layers | `SELECT DISTINCT symbol FROM orb_outcomes WHERE symbol IN ('NQ','ES','GC')` | 0 rows |
| MGC backfill coverage | `SELECT MIN(trading_day), MAX(trading_day) FROM daily_features WHERE symbol='MGC' AND orb_minutes=5` | Start ≤ 2022-06-13 |
| All apertures rebuilt | `SELECT symbol, orb_minutes, COUNT(DISTINCT trading_day) FROM daily_features GROUP BY 1,2` | O5, O15, O30 present for all 3 instruments |
| Holdout boundary | Discovery CLI `--holdout-date 2026-01-01` enforced | Automatic (pipeline enforcement) |
| No 2026 data in discovery window | `SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol='MNQ'` filtered by holdout | ≤ 2025-12-31 |
| Friction calculations current | `from pipeline.cost_model import COST_SPECS; print(COST_SPECS)` | MNQ=$2.74, MES=$3.74, MGC=$5.74 |

This gate is a one-time verification. If any check fails, STOP and fix before proceeding.

---

## 3. Structural Economics Summary

### 3a. Data Availability (pre-holdout)

| Instrument | Calendar years | Bailey max N (E=1.0) | Point value | Total friction |
|---|---|---|---|---|
| MNQ | 6.66yr | 27 | $2.00/pt | $2.74 |
| MES | 6.66yr | 27 | $5.00/pt | $3.74 |
| MGC | 3.55yr | 5 | $10.00/pt | $5.74 |

### 3b. Session Friction Viability

Friction % = total_friction / (median_ORB_size × point_value) × 100. Sessions where UNFILTERED median friction > 30% are structurally cost-killed.

**MNQ — all sessions viable (all < 20%)**
| Session | Median ORB (pts) | Friction % | Verdict |
|---|---|---|---|
| NYSE_OPEN | 45.5 | 3.0% | VIABLE |
| US_DATA_1000 | 33.5 | 4.1% | VIABLE |
| CME_PRECLOSE | 18.5 | 7.4% | VIABLE |
| US_DATA_830 | 17.0 | 8.1% | VIABLE |
| EUROPE_FLOW | ~15 | ~9% | VIABLE |
| COMEX_SETTLE | ~12 | ~11% | VIABLE |
| SINGAPORE_OPEN | 8.75 | 15.7% | MARGINAL |

**MES — split: large sessions viable, small sessions cost-killed**
| Session | Median ORB (pts) | Friction % | Verdict |
|---|---|---|---|
| NYSE_OPEN | 8.25 | 9.1% | VIABLE |
| US_DATA_1000 | 7.00 | 10.7% | VIABLE |
| CME_PRECLOSE | 4.75 | 15.7% | MARGINAL |
| CME_REOPEN | 4.25 | 17.6% | MARGINAL + TOXIC (winter) |
| SINGAPORE_OPEN | 2.00 | 37.4% | COST-KILLED |
| TOKYO_OPEN | ~2.5 | ~30% | COST-KILLED |

**MGC — only large-ORB sessions viable**
| Session | Median ORB (pts) | Friction % | Verdict |
|---|---|---|---|
| US_DATA_830 | 4.1 | 14.0% | VIABLE (barely) |
| US_DATA_1000 | 3.9 | 14.7% | VIABLE (barely) |
| NYSE_OPEN | 3.5 | 16.4% | MARGINAL |
| LONDON_METALS | ~3.0 | ~19% | MARGINAL (G5 fixes this) |
| CME_REOPEN | ~2.5 | ~23% | MARGINAL (G5+ required) |
| CME_PRECLOSE | 1.0 | 57.4% | COST-KILLED |
| SINGAPORE_OPEN | 2.0 | 28.7% | COST-KILLED + 74% double-break |

### 3c. Unfiltered Baseline Summary (E2 RR2.0, pre-holdout)

Only sessions with positive or near-zero unfiltered ExpR are candidates. Negative unfiltered baselines REQUIRE a size filter to create positivity (which is expected — that's the "ORB Size = The Edge" thesis).

**MNQ — broad positive baseline:**
- 8/12 sessions positive at RR2.0 unfiltered (CME_PRECLOSE +0.091, US_DATA_1000 +0.075, NYSE_OPEN +0.085, COMEX_SETTLE +0.072, EUROPE_FLOW +0.059)
- US_DATA_830 near-zero (+0.002), SINGAPORE weakly negative

**MES — narrow positive baseline:**
- Only NYSE_OPEN positive at RR2.0 unfiltered (+0.045)
- US_DATA_1000 slightly negative (-0.014)
- Everything else negative (-0.10 to -0.17)

**MGC — no positive unfiltered baseline:**
- ALL sessions negative at RR2.0 unfiltered (best: US_DATA_830 -0.062)
- ALL sessions negative at RR1.0 unfiltered (best: US_DATA_1000 -0.030)
- Positive expectancy exists ONLY with size filters (G5+) — confirmed by "ORB Size = The Edge"

---

## 4. Instrument-Specific Hypothesis Design

### 4a. Universal Parameters (pre-committed, all hypotheses)

| Parameter | Value | Structural justification |
|---|---|---|
| Entry model | E2 (stop-market) | No backtest bias. Honest entry per TRADING_RULES. |
| ORB aperture | O5 (5-minute) | Primary aperture. O15/O30 weaker per STRATEGY_BLUEPRINT §4. |
| Confirm bars | CB1 | E2 triggers on touch — no confirmation needed. |
| RR target | 2.0 | Practitioner-standard breakout continuation target. Pre-committed to eliminate within-hypothesis optimization. |
| Stop multiplier | 1.0 | Standard (deployment may use 0.75x for prop). |
| Direction | BOTH | Except TOKYO_OPEN = LONG-ONLY per TRADING_RULES hypothesis H5. |
| Holdout | 2026-01-01 | Mode A sacred boundary. |

### 4b. Filter Threshold Selection (structural, not data-driven)

**Why G5 as default:** TRADING_RULES confirms "G5: MGC solid edge (creates positivity from negative baseline)" and "G4 maximizes total R, G5/G6 maximize per-trade edge." G5 is the "meaningful commitment" threshold where:
- Friction drops below 15% for most instrument-session pairs
- Microstructure noise (bid-ask bounce, tick noise) is dominated by directional flow
- The Crabel commitment hypothesis requires the range to represent institutional positioning, not noise

**One threshold per hypothesis.** No testing G4 AND G5 AND G6 AND G8 at the same session. Each is a separate prediction. We pre-commit to the threshold with the strongest structural argument for that instrument-session.

| Instrument | Default threshold | Structural reason |
|---|---|---|
| MNQ | G5 | Friction at G5: ~5-8% for major sessions. Meaningful commitment on a $2/pt contract. |
| MES | G5 | Friction at G5: ~9-14% for major sessions. $25 of committed capital per contract. |
| MGC | G5 | TRADING_RULES: "G5: MGC solid edge." $50 of committed capital per contract. Friction ~11% at G5. |

---

### 4c. MNQ Hypothesis File — K=5

MNQ has the broadest legitimate search space. 6.66 calendar years, N budget ≤ 27. We use 5 hypotheses — one per distinct economic mechanism.

| H# | Session | Filter | Mechanism | Theory citation |
|---|---|---|---|---|
| 1 | NYSE_OPEN | ORB_G5 | Crabel commitment at US equity cash open. NYSE 09:30 ET concentrates institutional equity flow. | Crabel 1990 ch.3 |
| 2 | EUROPE_FLOW | ORB_G5 | Cross-border information spillover. European session prices incorporated into US futures. | French & Roll 1986; Hamao, Masulis, Ng 1990 |
| 3 | US_DATA_830 | ORB_G5 | Macro announcement price discovery. 08:30 ET releases concentrate multi-day news into minutes. | Andersen, Bollerslev, Diebold, Vega 2003 |
| 4 | COMEX_SETTLE | ORB_G5 | Settlement microstructure. Mandatory hedger/producer flow at settlement creates directional imbalance. | O'Hara 1995 |
| 5 | CME_PRECLOSE | ORB_G5 | End-of-day positioning. Institutional rebalancing before daily close concentrates flow. | Market microstructure (pre-close auction literature) |

**Bailey check:** MinBTL = 2·ln(5)/1.0² = 3.22yr < 6.66yr → PASS (107% headroom).
**Noise floor Sharpe:** √(3.22/6.66) = 0.695 annualized. Any candidate with IS Sharpe > 0.70 is above noise.
**Sessions excluded:** TOKYO_OPEN (LONG-ONLY is a different mechanism — directional bias, not breakout continuation), SINGAPORE_OPEN (marginal friction, different mechanism), US_DATA_1000 (macro mechanism already covered by US_DATA_830 — same flow, different time). CME_REOPEN (different entry model recommendation per Session Playbook — E1, not E2).

**Why only 5 and not more:** Each additional hypothesis raises the noise floor. At K=5 on 6.66yr, noise floor = 0.70. At K=10, noise floor = 0.83. At K=16 (the old file), noise floor = 0.91. The lower noise floor at K=5 means weaker signals become detectable — directly addressing the MES problem where K=16 drowned the signal.

### 4d. MES Hypothesis File — K=4

MES has the same data length as MNQ but narrower structural viability. Only NYSE_OPEN and US_DATA_1000 have friction < 15%. MES CME_REOPEN is "TOXIC in winter" per TRADING_RULES.

| H# | Session | Filter | Mechanism | Theory citation |
|---|---|---|---|---|
| 1 | NYSE_OPEN | ORB_G5 | Crabel commitment at the S&P 500 cash open — the instrument Crabel's thesis was written about. MES receives the largest share of US equity institutional flow as the benchmark micro contract. | Crabel 1990 ch.3 |
| 2 | NYSE_OPEN | ORB_G6 | Same Crabel mechanism, higher commitment threshold. Tests whether stronger commitment produces stronger continuation (monotonic prediction). G6 = $30 committed capital per contract. | Crabel 1990 ch.3 (monotonicity) |
| 3 | US_DATA_1000 | ORB_G5 | Post-equity-open macro flow. 10:00 AM ET data releases (ISM, Consumer Confidence) arrive into an already-active equity market. Second macro window after NYSE_OPEN. | Andersen et al 2003; Ederington & Lee 1993 |
| 4 | US_DATA_830 | ORB_G5 | Pre-equity macro announcement. 08:30 ET releases on the benchmark equity index — CPI, NFP, PPI directly move S&P 500 via rate expectations. | Andersen et al 2003 |

**Bailey check:** MinBTL = 2·ln(4)/1.0² = 2.77yr < 6.66yr → PASS (140% headroom).
**Noise floor Sharpe:** √(2.77/6.66) = 0.645 annualized.

**Why K=4 not K=16:** The old file tested 16 hypotheses — 4 sessions × 4 G-thresholds per session. This was a threshold sweep disguised as pre-registration. The BH FDR correction at K=16 required p ≤ 0.003 for the best hypothesis. At K=4, it requires p ≤ 0.0125. The MES NYSE_OPEN signal (raw p ~ 0.026) has a realistic chance of clearing FDR at K=4.

**Why include H2 (G6 at NYSE_OPEN):** This is the one exception to "one threshold per session." Justification: G5 and G6 test a monotonic prediction — "does stronger commitment produce stronger continuation?" This is a DIFFERENT prediction from G5 alone. If G5 passes and G6 fails, the commitment threshold for MES NYSE_OPEN is below G6. If both pass, the Crabel thesis is robust across commitment levels. This costs 1 extra trial (K=4 vs K=3) but answers a structurally different question. The K=4 Bailey headroom (140%) can absorb this.

**Sessions excluded and why:**
- EUROPE_FLOW: MES unfiltered ExpR = -0.106 at RR2.0. No positive signal even with G5. The French-Roll cross-border mechanism applies to equity indices, but MES lacks the overnight activity that drives EUROPE_FLOW for MNQ (NQ/MNQ trades actively in Asian/European hours; ES/MES less so).
- CME_REOPEN: "TOXIC in winter" per TRADING_RULES. All 10 red-flag strategies are MES CME_REOPEN E1.
- TOKYO_OPEN, SINGAPORE_OPEN: Cost-killed (friction 28-39%).
- COMEX_SETTLE: Gold-specific settlement session, not equity-relevant.
- CME_PRECLOSE: MES unfiltered = -0.126. Marginal friction (15.7%).

### 4e. MGC Hypothesis File — K=5

MGC has the shortest data (3.55 calendar years) and the tightest Bailey constraint (N ≤ 5). All sessions are negative unfiltered — the edge ONLY exists with size filters. This is expected: "ORB Size = The Edge" is an arithmetic fact about friction.

| H# | Session | Filter | Mechanism | Theory citation |
|---|---|---|---|---|
| 1 | LONDON_METALS | ORB_G5 | London gold pricing center. LBMA Gold Price set during London hours. Physical gold demand (central banks, jewelry, ETF custodians) meets CME futures. | Lucey & Zhao 2008; Baur & Lucey 2010 |
| 2 | US_DATA_830 | ORB_G8 | Gold's macro sensitivity. US macro releases (NFP, CPI) affect gold via the USD/real-rate channel. G8 not G5: Session Playbook says MGC US_DATA_830 is "G12+ or DEAD" and "G8+ very selective." G5 would include noise that this session structurally produces. G8 is the minimum viable threshold per TRADING_RULES deep dive. | Andersen et al 2003; Baur & Lucey 2010; TRADING_RULES deep dive H3 |
| 3 | NYSE_OPEN | ORB_G5 | US equity session cross-asset flow. Gold trades actively during US equity hours — equity desks hedge with gold, risk-parity rebalances. | Capie, Mills, Wood 2005 |
| 4 | COMEX_SETTLE | ORB_G5 | COMEX settlement microstructure. Gold producers, central bank reserve managers, ETF APs must transact at settlement — mandatory flow creates directional imbalance. | O'Hara 1995 |
| 5 | CME_REOPEN | ORB_G5 | CME Sunday/daily reopen momentum. Overnight gap + first-session positioning reveals accumulated overnight information. | Ito, Engle, Lin 1992 (meteor shower) |

**Bailey check:** MinBTL = 2·ln(5)/1.0² = 3.22yr < 3.55yr → PASS (10.4% headroom).
**Noise floor Sharpe:** √(3.22/3.55) = 0.952 annualized.

**Why these 5:** Each covers a DIFFERENT economic mechanism for gold:
1. London physical gold market (the primary gold pricing venue globally)
2. US macro/real-rate channel (gold's inflation sensitivity)
3. US equity cross-asset flow (gold as portfolio hedge)
4. Mandatory settlement flow (O'Hara microstructure)
5. Overnight information revelation (CME reopen gap dynamics)

**Sessions excluded and why:**
- SINGAPORE_OPEN: 74% double-break rate for MGC (structurally mean-reverting, TRADING_RULES says OFF)
- TOKYO_OPEN: TRADING_RULES says "WORSE for MGC (delta -0.092, p<0.001)"
- US_DATA_1000: Same macro mechanism as US_DATA_830 — double-counting the same flow
- CME_PRECLOSE: 57.4% friction — severely cost-killed

**Honest concern:** 10.4% Bailey headroom is thin. If the calendar-year metric is even slightly wrong, N=5 could fail. This is documented transparently. The alternative (N=4, dropping one hypothesis) gives 28% headroom. Dropping H5 (CME_REOPEN) is the safest choice if headroom is needed — it has the weakest gold-specific theory.

---

## 5. What This Design Does NOT Do

| Excluded dimension | Why |
|---|---|
| Multiple RR targets per session | LdP §1.4.2: within-hypothesis parameter sweep. RR=2.0 pre-committed. |
| COST_LT filters | Same family as G-filters (trade-size gate, not signal). Correlated with G5 — testing both double-counts the same edge. |
| ATR_P70 regime filter | NO-GO per Blueprint SS5: regime-conditional discovery dead. ATR percentile as a filter didn't survive Mode A (MNQ p=0.41, MES p=0.61). |
| OVNRNG filters | Tested in Mode A — MNQ OVNRNG_50 near-miss (p=0.029). Could be a future targeted single-hypothesis file, but not included here to keep K honest. |
| PDR/GAP pre-session signals | Validated in prior research (p=0.003, p=0.009) but these are DIFFERENT mechanisms from Crabel size-first. Including them in the same file would mix mechanism families. They deserve their OWN separate hypothesis file with their OWN K — a future Phase 2 design. |
| E1/E3 entry models | E2 is the honest discovery entry. E1/E3 optimization happens post-discovery for deployment. |
| O15/O30 apertures | O5 is primary. O15/O30 are weaker per Blueprint §4. |
| Multiple confirm bars | CB1 is the only E2-relevant value. |
| Direction filters (except TOKYO_OPEN) | Pre-committing to BOTH directions is the default. TOKYO_OPEN LONG-ONLY is the only theory-grounded exception (TRADING_RULES H5). |

---

## 6. Execution Phases

### Phase 0: Data Integrity Verification
Run the 6 checks from Section 2. Gate all subsequent work behind this.

### Phase 1: Write Hypothesis Files
Write 3 YAML files (one per instrument) following the exact format in `docs/institutional/hypothesis_registry_template.md`. Each hypothesis gets:
- Theory citation from external literature
- Economic basis explaining WHY the edge should exist for THIS instrument at THIS session
- Exact filter, scope, and kill criteria
- Bailey compliance block with calendar-year metric

### Phase 2: Execute Discovery
For each instrument:
```
python -m trading_app.strategy_discovery --instrument X --orb-minutes 5 \
  --hypothesis-file docs/audit/hypotheses/YYYY-MM-DD-X-redesign.yaml \
  --holdout-date 2026-01-01
```

### Phase 3: Validate Survivors
Run strategy_validator on any BH FDR survivors. Check:
- Walk-forward efficiency ≥ 0.50
- Era stability (no era with N ≥ 50 and ExpR < -0.05)
- Sample size N ≥ 100
- 2026 OOS (forward judgment, not selector)

### Phase 4: Portfolio Construction
- Replace deployed lanes with Mode A survivors only
- Update prop_profiles
- NO grandfather clause for old strategies

---

## 7. Expected Outcomes (honest)

| Instrument | Prediction | Basis |
|---|---|---|
| MNQ | 2-4 FDR survivors (NYSE_OPEN, EUROPE_FLOW, possibly COMEX_SETTLE/CME_PRECLOSE) | Raw p-values 0.004-0.017 from prior Mode A. K=5 gives lighter FDR correction than K=16. |
| MES | 1-2 FDR survivors (NYSE_OPEN G5/G6) | Raw p-values 0.026-0.032 from prior Mode A. K=4 BH threshold = 0.0125 for rank 1. Close call. |
| MGC | 0-1 FDR survivors | Only LONDON_METALS G5 showed positive ExpR (+0.154, N=72, p=0.35). 3.55yr data, high noise floor (0.95). Realistic expectation: 0 survivors. |

**Total portfolio after redesign:** Likely 3-6 institutionally validated strategies, all MNQ/MES, size-gated, at the highest-flow sessions. This is a NARROW but HONEST portfolio. It can be broadened over time as more data accumulates (especially for MGC and MES).

---

## 8. Contamination Disclosure

This design was written AFTER seeing the Mode A results. The following contamination risks are acknowledged:

| Risk | Mitigation |
|---|---|
| Session selection informed by prior results | All sessions are justified by economic mechanism, not by prior ExpR. Excluded sessions have structural reasons (cost-killed, double-break, toxic winter). |
| Threshold G5 chosen after seeing G4/G5/G6/G8 results | G5 justified by friction arithmetic (threshold where friction < 15%) and TRADING_RULES recommendation. Not by "G5 had the best p-value." |
| K reduction from 16 to 4-5 after seeing K=16 failure | K reduction justified by principle P2 (one prediction per mechanism-session). The old K=16 was wrong from the start — threshold sweeps disguised as pre-registration. |
| MES narrowed to NYSE_OPEN-heavy after seeing US_DATA_830/EUROPE_FLOW negative | US_DATA_830 and EUROPE_FLOW included (H3, H4 for MES). Not cherry-picked away. |

**The honest truth:** A fully uncontaminated design would require a researcher who has never seen any ORB results. That's not possible in this project. The mitigations above reduce contamination to an acceptable level by grounding every decision in structural arguments, not in prior results.

---

## 9. Self-Review Findings (adversarial)

### Finding 1: MES H2 (G6 NYSE_OPEN) is a threshold variant
I justified it as a "monotonic prediction test." But the user's framework says "no threshold sweeps within bundles." Strictly, testing G5 AND G6 at the same session IS a threshold sweep. **Resolution:** Keep H2 but count it honestly as a separate trial (K=4 not K=3). The monotonic prediction IS a different question. If user disagrees, drop to K=3.

### Finding 2: MGC Bailey headroom is thin (10.4%)
If the calendar-year calculation is off by even 1 month, N=5 fails. **Resolution:** Document the risk transparently. Offer N=4 fallback (drop CME_REOPEN, 28% headroom). User decides.

### Finding 3: MNQ K=5 is LESS than prior K=16, which means some prior survivors may not survive
The prior 5 MNQ survivors were validated at K=16. At K=5 with DIFFERENT hypotheses (different sessions), the results will be different strategies. The old 5 are replaced, not preserved. **Resolution:** This is correct behavior. The old results are contaminated by K=16 threshold sweeps. The new K=5 results are cleaner.

### Finding 4: No pre-session signal filters (PDR, GAP, OVNRNG) in this design
These are validated signals (p=0.003, p=0.009, p=0.029) that could produce additional survivors. But including them in the same file mixes mechanism families and inflates K. **Resolution:** Explicitly defer to Phase 2 — separate hypothesis files for pre-session signals, with their own K. This is noted in Section 5.

### Finding 5: EUROPE_FLOW excluded from MES but included in MNQ
MES EUROPE_FLOW is negative unfiltered (-0.106). MNQ EUROPE_FLOW is positive (+0.059). The mechanism (French-Roll cross-border flow) applies to both equity indices. The exclusion for MES is partially data-driven. **Resolution:** Add MES EUROPE_FLOW back as H5 (K=5 total). The mechanism applies. If it fails, it fails honestly. Excluding it based on prior results is contamination.

**REVISED MES: K=5** (added EUROPE_FLOW G5 as H5)

| H# | Session | Filter | Mechanism |
|---|---|---|---|
| 1 | NYSE_OPEN | ORB_G5 | Crabel commitment at S&P 500 cash open |
| 2 | NYSE_OPEN | ORB_G6 | Crabel monotonic prediction (stronger commitment) |
| 3 | US_DATA_1000 | ORB_G5 | Post-equity-open macro flow |
| 4 | US_DATA_830 | ORB_G5 | Pre-equity macro announcement |
| 5 | EUROPE_FLOW | ORB_G5 | Cross-border information spillover |

**Updated Bailey:** MinBTL = 2·ln(5) = 3.22yr < 6.66yr → PASS (107% headroom).
**Updated noise floor:** √(3.22/6.66) = 0.695.

### Finding 6: MGC LONDON_METALS Session Playbook recommends E3, not E2
The Session Playbook says LONDON_METALS is a "Retrace Entry (E3)" session. Testing it with E2 (stop-market) tests a DIFFERENT strategy from what the Playbook validated. **Resolution:** This is acceptable for DISCOVERY. E2 is the honest, no-bias entry. If G5 at LONDON_METALS produces a positive edge with E2, it validates the SESSION-LEVEL mechanism (London gold pricing). The entry model can be optimized post-discovery. If E2 fails, it doesn't mean E3 would also fail — but E3 is RETIRED (adverse selection). We test with E2, which is the honest institutional entry.

### Finding 7: MGC US_DATA_830 G8 instead of G5 breaks the "universal G5" principle
The design says G5 for all hypotheses, but MGC US_DATA_830 needs G8. This is a structural exception: TRADING_RULES deep dive says "G12+ or DEAD" for this session. G5 includes noise trades that the session structurally produces. G8 is the minimum viable threshold from the Feb 2026 deep dive — this is friction arithmetic, not data-mining. **Resolution:** Document the exception with the structural reason. G8 at this one session is justified by the same friction argument as G5 everywhere else — it's where friction drops below 15% for MGC at US_DATA_830 specifically.

### Finding 8: MNQ CME_REOPEN excluded despite being "most stable family" per Session Playbook
Session Playbook says CME_REOPEN is "TRANSITIONING" for MNQ but uses E1 CB2, not E2 CB1. The mechanism (CME reopen momentum) is different from Crabel at NYSE_OPEN. **Resolution:** Keep excluded. CME_REOPEN's recommended parameters (E1 CB2 RR2.5) differ from the universal E2 CB1 RR2.0. Testing it with E2 would be testing a different strategy than what the Session Playbook validated. It belongs in a future E1-specific hypothesis file.

---

## 10. Final Hypothesis Summary

| Instrument | K | Sessions | Bailey headroom | Noise floor |
|---|---|---|---|---|
| **MNQ** | 5 | NYSE_OPEN, EUROPE_FLOW, US_DATA_830, COMEX_SETTLE, CME_PRECLOSE | 107% | 0.70 |
| **MES** | 5 | NYSE_OPEN (G5), NYSE_OPEN (G6), US_DATA_1000, US_DATA_830, EUROPE_FLOW | 107% | 0.70 |
| **MGC** | 5 | LONDON_METALS (G5), US_DATA_830 (G8), NYSE_OPEN (G5), COMEX_SETTLE (G5), CME_REOPEN (G5) | 10.4% | 0.95 |

**Total new trials:** 15 (5 + 5 + 5)
**Total instruments:** 3
**Mechanism families covered:** 5 (Crabel commitment, macro announcement, cross-border flow, settlement microstructure, reopen momentum)
**Filter families:** 1 (G-filter size-first, G5 default, G8 for MGC US_DATA_830 per structural exception) — cleanest possible design
**Parameters pre-committed:** E2 CB1 RR2.0 O5, BOTH direction (except noted)

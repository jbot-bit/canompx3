---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# ML V3 Research Design — First-Principles Redesign

**Date:** 2026-03-28
**Status:** SPIKE 1A COMPLETE — data results below
**Prerequisite reading:** This document is the design. Implementation requires explicit user approval.

---

## 0. SPIKE 1A RESULTS (executed 2026-03-28)

Data: 1,247,328 rows. 3 instruments (MNQ/MGC/MES) × 12 sessions × 6 RR targets. E2 CB1.

### Partial Correlations with mae_r (controlling for orb_size_norm)

| Feature | raw_r | partial_r | p-value | N |
|---------|-------|-----------|---------|---|
| atr_20_pct | 0.0594 | **0.0577** | 5e-147 | 199,679 |
| rel_vol | -0.0440 | -0.0383 | 5e-68 | 206,687 |
| break_delay_min | -0.0246 | -0.0182 | 1e-16 | 207,888 |
| prev_day_range_norm | 0.0218 | 0.0273 | 1e-35 | 207,888 |
| atr_vel_ratio | 0.0201 | 0.0247 | 3e-29 | 206,631 |

**Kill threshold (|partial_r| > 0.05): 1/8 features pass for mae_r.** atr_20_pct is the only independent predictor of adverse excursion.

### Partial Correlations with mfe_r (controlling for orb_size_norm)

| Feature | partial_r | p-value |
|---------|-----------|---------|
| **rel_vol** | **0.0545** | 1e-135 |
| **atr_vel_ratio** | **0.0520** | 1e-123 |
| atr_20_pct | 0.0464 | 1e-95 |

**2/8 pass for mfe_r.** rel_vol and atr_vel_ratio independently predict favorable excursion.

### Quintile Scans (RR2.0 pooled)

| Feature → Target | Q1→Q5 Spread | Spearman ρ | p | WR Spread |
|------------------|-------------|-----------|---|-----------|
| rel_vol → pnl_r | **+0.2027** | **1.000** | **0.000** | **+5.8%** |
| atr_20_pct → pnl_r | +0.1215 | 0.900 | 0.037 | +1.5% |
| rel_vol → mfe_r | +0.2122 | 1.000 | 0.000 | +4.2% |
| atr_20_pct → mae_r | +0.0683 | 1.000 | 0.000 | +2.2% |
| break_delay → pnl_r | -0.0275 | -0.300 | 0.624 | -2.4% |

**rel_vol is the dominant signal.** Perfect monotonicity. break_delay is NOT monotonic — WR spread only 2.4%, not the 6%+ claimed from per-session V1 analysis.

### Cross-Instrument Stability (rel_vol → pnl_r)

| Inst | RR1.0 Spread | RR2.0 Spread | RR3.0 Spread | WR Spread (RR2.0) |
|------|-------------|-------------|-------------|-------------------|
| MNQ | +0.18 | +0.17 | +0.16 | +4.7% |
| MGC | +0.22 | +0.22 | +0.21 | +6.6% |
| MES | +0.22 | +0.24 | +0.22 | +7.0% |

**CONSISTENT across all 3 instruments, all RR levels.** This is not an artifact.

### Arithmetic vs Signal Test (orb_size held constant at Q3)

| rel_vol Quintile | pnl_r | WR |
|-----------------|-------|-----|
| Q1 (lowest) | -0.1430 | 34.3% |
| Q3 | -0.0249 | 37.9% |
| Q5 (highest) | +0.0340 | 40.9% |

**WR spread = 6.6% at constant ORB size. SIGNAL, not arithmetic.**

### T0 Tautology Check

| Feature pair | Pearson r |
|-------------|-----------|
| rel_vol ↔ orb_size_norm | 0.203 |
| rel_vol ↔ atr_20_pct | -0.020 |
| atr_20_pct ↔ orb_size_norm | -0.052 |

**No tautology.** All inter-feature correlations < 0.30. rel_vol is genuinely independent.

### Bootstrap (1000 perms, skip bottom 20% rel_vol)

| Combo | Real Delta | p-value | Verdict |
|-------|-----------|---------|---------|
| MNQ RR1.0 | +2.7R | 0.001 | PASS |
| MNQ RR2.0 | +226.6R | 0.001 | PASS |
| MGC RR1.0 | +199.0R | 0.001 | PASS |
| MGC RR2.0 | +273.7R | 0.001 | PASS |
| MES RR1.0 | +219.1R | 0.001 | PASS |
| MES RR2.0 | +310.0R | 0.001 | PASS |

**All 6 combos pass at resolution floor.** Needs 5K perms for precise p-values.

### OOS Simulation (temporal 80/20 split, skip bottom 20% rel_vol)

| Inst | RR | N_test | Base R | After Filter | Delta |
|------|----|--------|--------|-------------|-------|
| MNQ | 1.0 | 14,966 | +1325 | +1328 | +3 |
| MNQ | 2.0 | 13,942 | +838 | +1065 | **+227** |
| MGC | 1.0 | 12,091 | +8 | +207 | **+199** |
| MGC | 2.0 | 11,140 | -163 | +111 | **+274** |
| MES | 1.0 | 9,422 | -107 | +112 | **+219** |
| MES | 2.0 | 8,749 | -405 | -96 | **+310** |

**Simple filter improves all 6 combos. Largest effects on negative-baseline instruments (MGC/MES).**

### ML (5-feature RF) vs Simple Filter

| Inst | RR | Simple dR | ML dR | ML - Simple |
|------|-----|-----------|-------|-------------|
| MNQ | 1.0 | -7 | -139 | **-132** |
| MNQ | 2.0 | +217 | -50 | **-267** |
| MGC | 1.0 | +202 | +232 | +30 |
| MGC | 2.0 | +273 | +354 | +82 |
| MES | 1.0 | +207 | +258 | +51 |
| MES | 2.0 | +295 | +611 | **+316** |

**Mixed.** ML HURTS MNQ (the strongest instrument). Helps MGC/MES. Simple filter is more robust.

### RF Regression on pnl_r (prediction quintiles)

RR1.0 pooled, 80/20 temporal split:

| Pred Quintile | N | Predicted Mean | Actual Mean | Actual WR |
|--------------|-----|---------------|-------------|-----------|
| Q1 (worst) | 7,040 | -0.175 | -0.096 | 53.7% |
| Q3 | 7,039 | -0.046 | +0.046 | 56.8% |
| Q5 (best) | 7,040 | +0.065 | **+0.145** | **60.3%** |

**Spearman ρ = 0.235 (p < 1e-300).** Model SORTS correctly even though R² ≈ 0. The ranking works; the point estimates don't.

RR2.0 pooled:

| Pred Quintile | Actual Mean | Actual WR |
|--------------|-------------|-----------|
| Q1 | -0.156 | 33.1% |
| Q5 | +0.122 | 40.0% |

**Spread: 0.278R, WR spread 6.9%.** Sorting works at RR2.0 too.

### VERDICT

1. **rel_vol is a REAL, NOVEL signal** — WR varies 6.6% at fixed ORB size, monotonic across all instruments and RR levels, bootstrap p < 0.001.
2. **break_delay is NOT the signal it was claimed to be** — WR spread only 2.4% pooled, not monotonic. Per-session V1 result was likely noise or confounded.
3. **Simple rel_vol filter beats ML for MNQ** — the dominant instrument. ML overfits threshold optimization.
4. **RF regression SORTS correctly** (Spearman 0.24) but explains near-zero variance (R² < 0.001). The sorting ability is what matters for trade selection.
5. **MAE/MFE prediction failed** — RF test R² negative. Features predict excursion DIRECTION (high vol = bigger moves both ways) but not magnitude beyond what orb_size already captures.
6. **The honest path forward: simple rel_vol filter as a new validated signal, NOT ML.** It's a one-line filter that adds +200-300R per instrument on OOS data. ML adds complexity that hurts the strongest instrument.

### NEXT STEPS

1. Run 5K bootstrap on rel_vol filter for precise p-values (1000 perms hit resolution floor)
2. Test rel_vol as a filter in the existing discovery grid (add to ALL_FILTERS)
3. Test rel_vol × G-filter interaction (are they additive or redundant?)
4. If rel_vol survives 5K bootstrap at BH FDR: implement as production filter
5. Microstructure features (Spike 2) still worth testing — genuinely new info dimension

---

## 1. Literature Review

### 1.1 Lopez de Prado — *ML for Asset Managers* (2020)

**Local PDF status:** `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` contains Section 1 ONLY (Cambridge preview, 45 physical pages). Sections 5-8 (meta-labeling details, feature importance, CPCV) are NOT in the local file. Claims below that reference Sections 5-8 are from training memory — **NOT verified against local PDF**.

**What IS grounded from local PDF:**

- **Classification over regression** (p.21-22): "Predicting the sign of an outcome is often more important than predicting its size, and a reason for favoring classifiers over regression methods in finance." Furthermore: "the sign and size of an outcome depend on different features, so jointly forecasting the sign and size of an outcome with a unique set of features can lead to subpar results." **→ Meta-labeling separates sign (primary model) from size (meta-label).**

- **High-tech horoscopes** (p.19): "Without this theory-ML interplay, investors are placing their trust on high-tech horoscopes." ML without testable economic theory = pure curve-fitting. **→ Every V3 framing must have a structural mechanism, not just pattern discovery.**

- **Overfitting control** (p.8): Three methods: (1) cross-validating forecasts, (2) limiting tree depth, (3) adding more trees. No specific depth/leaf recommendations for finance. **→ Our RF params (max_depth=6, min_samples_leaf=100) are defensible but arbitrary.**

- **MDA feature importance** (p.5): MDA (mean decrease accuracy) described as the honest importance method — shuffle a feature, measure accuracy decay. MDI (mean decrease impurity) is mentioned but not compared in this extract. **→ From AFML training memory: MDI is biased toward high-cardinality features; MDA with purged CV is honest. Cannot verify from local file.**

**From training memory (UNVERIFIED against local PDF):**
- Meta-labeling requires the primary model to have directional skill (positive edge). Meta-labeling on negative baselines = threshold artifact (our V1/V2 confirmed this empirically).
- Triple barrier method: vertical (time), horizontal-upper (profit), horizontal-lower (stop). Differs from fixed RR by letting time exit interact with P&L barriers.
- Sample uniqueness weighting: observations that overlap in time should be downweighted. Not implemented in our system.
- CPCV: C(N,k) combinatorial splits with purge + embargo. Our implementation (10 groups, k=2, 45 splits) matches the AFML recommendation.

### 1.2 Aronson — *Evidence-Based Technical Analysis* (2005)

**Local PDF: FULL TEXT available** (544 pages). All citations verified.

- **Curse of dimensionality** (p.481/465): "If 100 observations meet data density requirements at the two-dimensional level, 1,000 would be required at three dimensions and 10,000 at four dimensions." **→ 10x per dimension. With 5 features, we need 100,000+ samples for adequate density. Our pooled active set is 208K — just sufficient for 5 features. Per-session (3K-8K) is clearly insufficient.**

- **Sample size is the most important factor** (p.315/299): With 10 months of data, best-of-1024 rules has bias ~84%. With 1000 months, bias <12%. **→ Our 10 years (120 months) is moderate. With K>100 configs, data-mining bias is significant.**

- **When data mining FAILS** (p.325/309): "when the number of observations is small, data mining does not work. One would do just as well picking a rule by guess." **→ Per-session ML with N=3K-8K and 5 features is marginal. Pooling is not optional — it's necessary.**

- **Complex rules outperform** (p.466/450): 82% of significant rules were complex (nonlinear), despite being only 8% of the total. **→ ML SHOULD be better than rules, but ONLY with sufficient data.**

- **No significance test exists for complex nonlinear rules** (p.472/456): Must rely on train/test/validate three-segment approach. **→ Our CPCV + bootstrap is the best available methodology.**

- **6,402 rules tested, 0 survived WRC** (p.463/447): On S&P 500 with proper correction. **→ Honest methodology kills most things. This is expected, not a failure.**

### 1.3 Chan — *Algorithmic Trading* (2013)

**Local PDF: FULL TEXT available.**

- **Neural nets = guaranteed overfit** (p.41): "With at least 100 parameters, we can certainly fit the model to any time series we want and obtain a fantastic Sharpe ratio. Needless to say, it will have little or no predictive power going forward." **→ Our RF has ~200 trees × 6 depth × sqrt(5) features ≈ modest effective parameters. But threshold optimization adds degrees of freedom.**

- **Regime = human judgment** (p.207): Chan treats regime detection as discretionary, not ML. "Their role is to make judicious high-level judgment calls." **→ Framing D (daily regime classifier) has no Chan support.**

- **Kelly is fragile** (p.190): "the consequence of using an overestimated mean or an underestimated variance is dire." **→ Framing I (Kelly sizing via ML) is dangerous — estimation error in ML predictions compounds Kelly's existing fragility.**

- **Walk-forward = 50% haircut** (p.25): "Most traders would be happy to find that live trading generates a Sharpe ratio better than half of its backtest value." **→ Our WFE threshold of 0.50 is exactly Chan's rule of thumb.**

### 1.4 Carver — *Systematic Trading* (2015)

**Local PDF: FULL TEXT available.**

- **Equal blend beats fitted selection** (p.75): Choosing best rule each year gives SR=0.07. Random selection: SR=0.20. Equal blend of all: SR=0.33. **→ Feature selection and hyperparameter fitting may HURT. Simpler is better.**

- **40 years to confirm SR=0.3** (p.77): At realistic SR ≈ 0.3, you need ~40 years of data to be statistically sure. **→ With 10 years, we cannot reliably distinguish SR=0.3 from noise.**

- **Pool data across instruments** (p.82): "Only if there is a statistically significant difference in performance between various rules across instruments should you fit them individually. In practice this is rarely necessary." **→ STRONG support for pooled training. Per-session models are the opposite of this advice.**

- **Position sizing on predicted return, not win probability** (p.128): "forecasts are proportional to expected risk adjusted returns." **→ Our binary win/loss target is misaligned with Carver's framework. Continuous return prediction aligns better.**

- **ML never mentioned. Implicitly hostile** (p.276): "Don't try anything too clever; it is probably unnecessary and it's more likely to go wrong." **→ Carver would not use ML at all. He'd use simple rules with equal weighting.**

### 1.5 Bailey & Lopez de Prado — *Deflated Sharpe Ratio* (2014)

**Key findings:**
- **N=46 trials at SR=2.5: DSR=0.95** (p.10). With N=88+: DSR drops below significance. **→ Our K=108 (V2 configs) is well past the danger zone.**
- **Optimal stopping rule** (p.10-11): Try 37% of configs randomly, then stop at first that beats all prior. **→ We exhaustively searched all 108 — maximum data-mining penalty.**
- **5 years of data allows max ~45 independent configs** (Pseudo-Math p.8). **→ With 10 years, ~200 configs is the ceiling. Our K=108 is within budget but tight.**

### 1.6 Chordia, Goyal & Saretto — *Two Million Trading Strategies* (2018)

**Key findings:**
- **t ≥ 3.79 threshold** (p.6) under FDP-StepM at 5%/5%. **→ Our BH FDR threshold of ~2.5-3.0 is lenient by this standard.**
- **75-99% of classical discoveries are false; 98% under FDP-StepM** (p.40). **→ Validates our aggressive kill rate. Most things SHOULD fail.**
- **More strategies ≠ more bias for adaptive methods** (p.31-32): BH and FDP-StepM are adaptive — having more strategies doesn't impose statistical bias. **→ Pooling strategies for BH FDR is safe and may improve power.**

### 1.7 Man AHL — *Overfitting* (2015)

- **"Culture of failure"** (p.7, Campbell Harvey): Reward negative research results. If researchers are punished for null findings, they data-mine until something looks significant. **→ Our ML being DEAD is a legitimate and valuable finding, not a failure.**

---

## 2. V2 Post-Mortem

### A. DATA VOLUME

Per-session at E2 RR1.0 CB1: MNQ ranges from 2,738 (CME_REOPEN) to 7,837 (BRISBANE_1025). With 60% train split → ~1,643 to 4,702 training samples. At ~55% WR → ~903 to 2,586 positives. With 5 features, EPV = 181 to 517. **EPV is adequate at per-session level for RR1.0.**

**But at RR2.0:** WR drops to ~30% → positives = ~493 to 1,411. EPV = 99 to 282. Still above 10, but Aronson's curse-of-dimensionality demands 10^5 for 5 dimensions. Per-session N of 3K-8K is **orders of magnitude too small** by Aronson's standard.

**De Prado (AFML, from training memory):** No explicit minimum N stated for RF, but CPCV requires enough data in each of C(N,k) splits. With 10 groups and k=2, each test set is 20% of data. At N=3K, test set = 600. Marginal.

**Verdict:** Per-session N is adequate for EPV but insufficient for high-dimensional density. Pooling is required.

### B. TARGET VARIABLE

The binary target `(pnl_r > 0)` is misaligned with the system's edge mechanism. Project research conclusively showed:
- WR is flat at 58-60% across most filters (ARITHMETIC_ONLY verdict, Mar 24 2026)
- Edge comes from **payoff asymmetry** — bigger ORBs have lower friction, so wins pay more relative to costs
- WR is NOT the predictable dimension; payoff is

**De Prado's classification recommendation** (p.21-22) assumes sign and size are both informative. In our system, sign (win/loss) is nearly random (noise-floor WR ~58%), while size (pnl_r magnitude) varies meaningfully with ORB size and session. Binary classification is trying to predict the LESS predictable dimension.

**BUT:** De Prado also says sign and size depend on different features. Maybe sign features exist that our filters don't capture. The V2 result (0/12 BH survivors) suggests they don't — or that 5 features are insufficient to find them.

**Verdict:** Binary target is WRONG for this system. Regression on pnl_r or MAE/MFE prediction aligns with the actual edge mechanism.

### C. FEATURE OVERLAP

- `orb_size_norm` ≈ G-filter pass/fail: `corr ≈ 0.85+` by construction (G-filters threshold on orb_size)
- `atr_20_pct` ≈ ATR70_VOL filter: `corr ≈ 0.90+` by construction (ATR70 thresholds at 70th percentile)
- Of the 5 V2 features, 2 are near-duplicates of existing filters. Truly novel: `gap_open_points_norm`, `orb_pre_velocity_norm`, `prior_sessions_broken`. That's 3 features with genuinely new information.

**Verdict:** V2 had 3 novel features, not 5. The RF was partially learning the existing filter stack, which it can't beat (deterministic rules always outperform probabilistic learning of the same signal).

### D. PRE-BREAK CONSTRAINT

The Blueprint says ML must be pre-break. But `quant-audit-protocol.md` lists `break_delay_min` as "trade-time-knowable." The ML config blacklists it with the note: "VALID theory but unknown pre-break."

**This is an architecture choice, not a data constraint.** Two legitimate architectures exist:

1. **Pre-break (current):** Decide before placing the stop-market order. Cost of skip = zero. Features limited to what's known at ORB close.
2. **At-break:** Decide after the break bar prints but before entry confirmation completes. Cost of skip = possible fill already in (E2 stop-market fires on break). Features include break_delay, break_bar_volume, rel_vol.

V2 threw away its three highest-theory features (break_delay, break_bar_volume, break_bar_continues) to maintain pre-break architecture. The quintile feature scan (Mar 20) found `break_delay_min` has genuine WR monotonicity — one of the few SIGNAL (not ARITHMETIC_ONLY) features ever found in this project.

**For E2 (stop-market entry):** The break fires the stop. You're already getting filled. Pre-break filtering means deciding whether to PLACE the stop. At-break filtering means deciding whether to HOLD the position. Different decisions, different costs.

**Verdict:** The pre-break constraint sacrificed the best available signal. V3 should explore at-break architecture, especially for exit management (Framing C).

### E. POOLING

Per-session was chosen because cross-session features leak session identity (79.7% accuracy). V2 comments note this explicitly.

**What pooling gains:** N goes from 3K-8K per session to 60K-90K pooled (active instruments). Aronson's 10^5 threshold becomes reachable.

**What pooling loses:**
- Session-specific patterns (COMEX has different dynamics than NYSE_OPEN)
- Risk of model learning "session = good/bad" rather than feature-outcome relationships
- Interpretability — can't inspect per-session importance

**Carver's answer** (p.82): Pool by default. Fit individually only if there's a statistically significant difference. We've never tested whether session-specific ML outperforms pooled ML on the same features.

**Verdict:** Pooling should be the DEFAULT. Per-session is a complexity choice that must justify itself with superior performance.

### F. FEATURES THE MODEL NEVER SAW

Available in DB but never given to ML:
- `mae_r`, `mfe_r` — post-trade excursions (targets for Framing C, not input features)
- `prev_day_high`, `prev_day_low`, `prev_day_close` — prior day levels (absolute prices, need normalization)
- `overnight_range`, `overnight_high/low` — blacklisted as session-dependent lookahead for Asian sessions, but CLEAN for US/London sessions
- `session_asia_high/low`, `session_london_high/low`, `session_ny_high/low` — session-window extremes (lookahead for sessions within window, clean for later ones)
- `took_pdh/pdl_before_1000`, `overnight_took_pdh/pdl` — binary sweep signals (blacklisted as session-dependent)
- `garch_forecast_vol`, `garch_atr_ratio` — removed as "statistical artifact" but never formally tested via T1 WR monotonicity
- `compression_z`, `compression_tier` — ORB compression (tested, NO-GO as continuous predictor; AVOID gate works as binary)
- `rsi_14_at_CME_REOPEN` — removed as "guilty until proven"
- All `bars_1m` microstructure (volume profiles, price velocity during ORB window, VWAP slope) — never computed
- `break_delay_min`, `break_bar_volume`, `rel_vol` — blacklisted by pre-break constraint but have genuine theory

**Key insight:** The best untapped feature source is bars_1m microstructure (16.8M rows). The ORB window itself contains volume and price information that has never been extracted as ML features. Example: "Was volume front-loaded in the first 2 minutes of the ORB?" or "Did price accelerate toward the end of the ORB window?" These are pre-break, trade-time-knowable, and potentially novel.

---

## 3. Framing Evaluation Matrix

### Framing A: Pooled Binary Meta-Label (Fix V2's Data Volume Problem)

**Description:** Same binary target `(pnl_r > 0)`, but pool all sessions + instruments. Session and instrument as categorical features.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | MODERATE | De Prado endorses classification. Carver: "pool by default." |
| Data availability | READY | No new features needed. 208K rows pooled. |
| N adequacy | SUFFICIENT | 208K pooled, ~115K positives at RR1.0 |
| Novelty vs filters | LOW | Same features, just more data. Still 3 novel features. |
| Overfit risk | LOW | Large N, few features |
| Honest testability | CPCV + bootstrap + BH FDR on pooled model |

**Biggest risk:** WR is flat at ~58% across all conditions. If there's no WR signal in the features, more data won't help — you'll just measure noise more precisely.

**Pre-registered hypothesis:** "Pooled binary RF at RR1.0 with session+instrument as categoricals achieves bootstrap p < 0.05 on total R delta and BH FDR survival at K=3 (one model per RR)."

**If WR monotonicity (T1) fails on ANY feature in the pooled model → framing is DEAD.** Binary classification requires WR variation.

### Framing B: Regression on pnl_r

**Description:** Continuous target = pnl_r. Predict E[pnl_r], take trades where E[pnl_r] > cost threshold.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | WEAK-MODERATE | De Prado favors classification for sign prediction. BUT our edge is in SIZE not sign. Chan: don't use regression for heavy tails. Carver: size forecasts for continuous signals. |
| Data availability | READY | pnl_r exists for all rows |
| N adequacy | SUFFICIENT | Same 208K pooled |
| Novelty vs filters | MEDIUM | Can capture payoff-varying features that binary can't |
| Overfit risk | MEDIUM | Heavy tails in pnl_r (std ~0.86-1.66x) increase noise |
| Honest testability | Bootstrap on total R delta using predicted E[pnl_r] threshold |

**Biggest risk:** pnl_r has heavy tails (kurtosis >> 3). A few extreme outcomes dominate regression fit. Winsorization or quantile regression may be needed.

**Pre-registered hypothesis:** "Gradient-boosted regression (LightGBM) on winsorized pnl_r achieves positive R delta vs baseline on 20% test set, confirmed by 5K bootstrap at p < 0.05."

### Framing C: MAE/MFE Prediction for Dynamic Exits

**Description:** Don't predict take/skip. Predict HOW the trade develops. Model 1: E[mae_r] → dynamic stop tightening. Model 2: E[mfe_r] → dynamic target adjustment.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | MODERATE | De Prado triple barrier has time dimension. Chan: stop/target management is core. Our own S0.75 research showed tight stops add value (T6 PASSES all 9 combos at p≤0.002). |
| Data availability | READY | mae_r and mfe_r exist for all outcomes. MNQ E2 RR2.0: median MAE=0.79R, median MFE=0.88R. Rich distribution. |
| N adequacy | SUFFICIENT | 208K pooled, mae/mfe are continuous targets |
| Novelty vs filters | HIGH | Existing filters select WHICH trades. This optimizes HOW trades are managed. Completely different value add. |
| Overfit risk | MEDIUM | Predicting continuous excursion — same heavy-tail concern as B |
| Honest testability | Simulate P&L with dynamic stops/targets vs fixed. Bootstrap on net R delta. |

**Why this is the most promising framing:**
1. Works on ALL trades, not just filtered ones. Even "bad" sessions benefit from better exit management.
2. S0.75 (tight stop) already validated at p≤0.002 across all 9 instrument×RR combos. ML can optimize WHEN to tighten.
3. At-break features (break_delay, rel_vol) are the strongest theoretical predictors and are legitimate inputs for exit management.
4. No binary classification → no WR-flatness problem. MAE is a continuous, well-distributed target.
5. Doesn't compete with filter stack — complements it. A trade that passes G8 + ATR70 can STILL benefit from dynamic stops.

**Biggest risk:** Implementation complexity. Need to simulate intraday P&L paths with dynamic levels, not just outcome lookup.

**Pre-registered hypothesis:** "RF regression on mae_r using pre-break + at-break features, applied as a dynamic S multiplier (S = f(predicted_mae_r)), produces total R improvement > 5% vs fixed S0.75, confirmed at bootstrap p < 0.05."

### Framing D: Daily Regime Classifier

**Description:** Binary: "Is today a good day for session X?" — aggregate day-level prediction.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | WEAK | Chan treats regime as human judgment. No academic support for daily binary regime from features. |
| Data availability | READY | Aggregate pnl_r per (trading_day, session) |
| N adequacy | INSUFFICIENT | ~2,900 trading days for MNQ. Per session: ~2,900 × 1 obs each. With 5 features, need 10^5 (Aronson). |
| Novelty vs filters | LOW-MEDIUM | ATR70_VOL already captures vol regime. Calendar overlays = NO-GO. |
| Overfit risk | HIGH | Very low N, many possible day-level features |
| Honest testability | Bootstrap, but N is too small for reliable p-values |

**Biggest risk:** N ≈ 2,900 days is far too small. Aronson: "when the number of observations is small, data mining does not work."

**Verdict: SKIP.** Insufficient N. The simple ATR percentile filter already captures the regime signal that matters.

### Framing E: Filter Stack as Features (Ensemble)

**Description:** Binary inputs from existing filter pass/fail, ML learns optimal combination.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | MODERATE | Carver's forecast combination. But his answer is: EQUAL BLEND beats fitted. |
| Data availability | READY | Filter pass/fail computable from existing data |
| N adequacy | SUFFICIENT | Same 208K pooled |
| Novelty vs filters | LOW | ML learns filter interactions, but our grid search already tests filter combinations |
| Overfit risk | MEDIUM | Many binary features → many possible interactions |
| Honest testability | Bootstrap vs best individual filter |

**Biggest risk:** The filter grid search already tested all pairwise combinations. ML would need to find 3-way+ interactions — these are typically noise at our N.

**Carver's warning** (p.75): Fitted selection (SR=0.07) < random selection (SR=0.20) < equal blend (SR=0.33). ML-fitted filter combination likely WORSE than equal-weight blend.

**Verdict: SKIP.** Carver's evidence is damning. Equal blend of filters already explored. ML adds complexity without clear advantage.

### Framing F: Microstructure from bars_1m

**Description:** Compute ORB-window features from 1-minute OHLCV: volume acceleration, price velocity, volume concentration, VWAP slope.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | MODERATE | Microstructure is a genuine information dimension. Volume profile analysis has academic support. But no specific ORB-window microstructure literature. |
| Data availability | NEEDS_ENGINEERING | bars_1m has 16.8M rows but no pre-computed microstructure features. Would need a feature engineering pipeline to compute per-ORB-window volume profiles. |
| N adequacy | SUFFICIENT | Same 208K outcomes, just richer features per outcome |
| Novelty vs filters | HIGH | Genuinely new information. No existing filter captures intra-ORB volume dynamics. |
| Overfit risk | HIGH | Many possible microstructure features → feature selection bias. Aronson warns about this. |
| Honest testability | Univariate T1 scan on each microstructure feature first. Then RF with survivors only. |

**Biggest risk:** Feature engineering creates degrees of freedom. Each computation choice (window size, normalization, binning) is a parameter that contributes to multiple testing.

**Pre-registered hypothesis:** "At least 2 of 5 pre-specified microstructure features (volume acceleration slope during ORB window, volume concentration in first/last 2 minutes, price velocity slope, VWAP-close divergence, bar range expansion ratio) show WR monotonicity at p < 0.05 after BH FDR at K=5."

### Framing G: Multi-Class or Ordinal Target

**Description:** Target = {big_win (>1.5R), small_win, scratch, small_loss, big_loss}.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | WEAK | No specific support for ordinal classification in trading literature. De Prado: separate sign from size. |
| Data availability | READY | Derivable from pnl_r |
| N adequacy | MARGINAL | Class imbalance at extreme categories (big_win at RR2.0 ≈ 10%) |
| Novelty vs filters | MEDIUM | Preserves payoff structure better than binary |
| Overfit risk | HIGH | Multiple thresholds to define classes → hidden parameter search |
| Honest testability | Complex — multi-class metrics are less clean than binary/continuous |

**Verdict: SKIP.** Regression (Framing B) captures the same information more cleanly without arbitrary class boundaries.

### Framing H: Survival Analysis (Time-to-Event)

**Description:** Predict time to target OR time to stop using Cox/AFT models.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | WEAK | No trading-specific survival analysis literature in resources/. De Prado triple barrier has time dimension but doesn't frame it as survival. |
| Data availability | NEEDS_ENGINEERING | exit_ts exists but need full path from entry to exit, not just endpoints |
| N adequacy | SUFFICIENT | Same N, more complex model |
| Novelty vs filters | HIGH | Time dynamics are completely unexploited |
| Overfit risk | HIGH | Specialized models, many assumptions (hazard ratios, proportional hazards) |
| Honest testability | Complex — specialized validation for survival models |

**Verdict: DEFER.** Interesting but high complexity, low literature support. If Framing C (MAE/MFE) works, survival analysis adds marginal value.

### Framing I: Kelly Sizing via ML

**Description:** ML predicts E[return] and Var[return], Kelly formula sizes position.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | WEAK | Chan (p.190): "overestimated mean or underestimated variance is dire." Kelly is fragile. ML estimation error compounds this. |
| Data availability | READY | Mean and variance computable from predictions |
| N adequacy | SUFFICIENT | |
| Novelty vs filters | MEDIUM | |
| Overfit risk | HIGH | Two predictions (mean + variance) from same features, each with estimation error |
| Honest testability | Simulate Kelly sizing vs fixed sizing. Bootstrap on risk-adjusted return. |

**Verdict: SKIP.** Chan warns explicitly. ML estimation error + Kelly fragility = ruin risk.

### Framing J: De Prado-Aligned Triple Barrier Regression (Literature-Derived)

**Description:** Instead of fixed RR targets, use triple barrier labels: (time barrier, profit barrier, stop barrier). Target = the barrier that was hit first (as a categorical) or the P&L at barrier exit (as continuous). This directly implements de Prado's labeling recommendation.

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Literature support | STRONG | De Prado Ch 5 (from training memory — Section 5 not in local PDF). The triple barrier is his RECOMMENDED labeling method. |
| Data availability | NEEDS_ENGINEERING | We have fixed-RR outcomes, not triple-barrier outcomes. Need to compute time-barrier exits from bars_1m. |
| N adequacy | SUFFICIENT | Same N with richer labels |
| Novelty vs filters | HIGH | Time dimension is unexploited. A trade that "almost hit target but timed out" is currently counted as a loss. |
| Overfit risk | MEDIUM | Well-defined labels reduce researcher degrees of freedom |
| Honest testability | Compare triple-barrier-labeled model vs fixed-RR model on same features. Bootstrap on total R delta. |

**Note:** Our `ts_outcome` and `ts_pnl_r` columns in orb_outcomes already capture time-stop conditional outcomes. This is a partial triple barrier — we have the data for time exits at fixed horizons. What we don't have is adaptive barriers.

**Biggest risk:** This is a target engineering change, not a feature change. If the FEATURES are the bottleneck (not enough signal), better labels won't help.

---

## 4. Ranking + Top Candidates

### Ranking by Probability of Surviving Honest Testing

| Rank | Framing | Probability | Reasoning |
|------|---------|-------------|-----------|
| 1 | **C: MAE/MFE Dynamic Exits** | 35-45% | Genuinely new dimension (exit management vs entry selection). S0.75 already validated. At-break features available. Doesn't compete with filter stack. |
| 2 | **B: Regression on pnl_r** | 15-25% | Targets the right dimension (payoff, not WR). But heavy tails and noise in pnl_r distribution may overwhelm signal. |
| 3 | **F: Microstructure from bars_1m** | 15-20% | Genuinely new information. But high feature engineering risk. Univariate scan first to validate. |
| 4 | **A: Pooled Binary Meta-Label** | 5-10% | More data doesn't help if WR is flat. V2 already showed the ceiling. |
| 5 | **J: Triple Barrier Regression** | 10-15% | Strong literature support, but target engineering alone unlikely to overcome feature bottleneck. |

### Top 3 for Research Spikes

**SPIKE 1: Framing C — MAE/MFE Dynamic Exits (HIGHEST PRIORITY)**

Why: It's the only framing that doesn't try to predict take/skip — it tries to improve HOW trades are managed. This sidesteps the fundamental problem (WR is flat, binary prediction has no signal) by targeting a different outcome (excursion magnitude).

Single biggest risk: **The at-break features (break_delay, rel_vol) may not predict MAE/MFE any better than ORB size alone.** If `corr(orb_size, mae_r) ≈ corr(model_prediction, mae_r)`, the model adds nothing over the existing size filter.

Kill test: Run univariate correlation between each candidate feature and mae_r. If no feature achieves partial correlation > 0.05 after controlling for orb_size → framing is dead.

**SPIKE 2: Framing F — Microstructure Features (MEDIUM PRIORITY)**

Why: Genuinely new information dimension. No existing filter captures intra-ORB volume dynamics. If there's signal here, it's signal that ONLY ML can capture.

Single biggest risk: **Feature engineering creates hidden degrees of freedom.** Every computation choice (how to measure "volume acceleration," what window to use) is a parameter. With 5 candidate features and 3 computation variants each, K = 15 effective trials.

Kill test: Pre-register exactly 5 features with exact computation formulas BEFORE looking at data. Run T1 WR monotonicity. If 0/5 show spread > 3% → framing is dead.

**SPIKE 3: Framing B — Regression on pnl_r (LOWER PRIORITY)**

Why: Targets the right dimension (payoff asymmetry). If there ARE features that predict trade SIZE (not just direction), regression captures them while binary classification throws that information away.

Single biggest risk: **pnl_r has heavy tails (std ~0.86-1.66x mean).** A few extreme outliers dominate the regression. Winsorization at ±3σ may help, but may also clip the actual edge (which IS in the tails).

Kill test: Compare R² of regression model vs R² of "just predict mean(pnl_r) for this session×instrument" naive baseline. If improvement < 1% R² → framing is dead.

---

## 5. Pre-Registered Hypotheses (Falsifiable)

### H1: MAE/MFE Prediction (Framing C)
"If at-break features predict adverse excursion, then an RF regression model using (orb_size_norm, atr_20_pct, gap_open_points_norm, break_delay_min, rel_vol) to predict mae_r will achieve partial correlation > 0.05 controlling for orb_size_norm alone, on 20% holdout, confirmed at bootstrap p < 0.05 with 5K permutations."

**If partial correlation ≤ 0.05:** The features don't predict excursion beyond what ORB size already tells us. Framing C is DEAD.

**If pass:** Implement as dynamic stop multiplier S = f(predicted_mae_r). Simulate on 2025 data. Evaluate total R improvement vs fixed S0.75.

### H2: Microstructure Signal (Framing F)
"If intra-ORB volume dynamics predict outcome quality, then at least 2 of 5 pre-specified microstructure features (see §6.2) will show WR monotonicity with quintile spread > 3% and bootstrap p < 0.05, after BH FDR at K=5."

**If 0-1 features show spread > 3%:** Microstructure doesn't contain tradeable signal at 1-minute resolution. Framing F is DEAD at current data resolution.

**If pass:** Add surviving features to Framing C's feature set. Retrain MAE/MFE model with microstructure features.

### H3: Regression on pnl_r (Framing B)
"If features predict payoff magnitude, then LightGBM regression on winsorized pnl_r achieves test R² > 0.01 (explaining >1% of variance) and positive total R delta on 20% holdout, confirmed at bootstrap p < 0.05."

**If R² ≤ 0.01:** The features don't explain payoff variance. Framing B is DEAD.

---

## 6. V3 Spec Drafts

### 6.1 SPIKE 1: MAE/MFE Dynamic Exit Model

**Feature list:**
| Feature | Source | Computation | Novelty |
|---------|--------|-------------|---------|
| orb_size_norm | daily_features | orb_size / atr_20 | Control variable |
| atr_20_pct | daily_features | Rolling 252-day percentile | Existing |
| gap_open_points_norm | daily_features | gap_open_points / atr_20 | Existing |
| break_delay_min | daily_features | Minutes from ORB close to first break | **AT-BREAK (restored)** |
| rel_vol | daily_features | break_bar_volume / 20-session median | **AT-BREAK (restored)** |
| orb_pre_velocity_norm | daily_features | Pre-session momentum / atr_20 | Existing |
| prior_sessions_broken | computed | Count of prior session ORB breaks | Existing |

**Target:** mae_r (continuous, from orb_outcomes). Secondary: mfe_r.

**Model:** RandomForest Regressor, same hyperparameters as V2 (max_depth=6, min_samples_leaf=100, n_estimators=200). RF handles nonlinearity without heavy-tail sensitivity of linear regression.

**Training protocol:**
- **Pooled:** All 3 active instruments × all 12 sessions. Session + instrument as additional features.
- Split: 60% train / 20% validation / 20% test (temporal)
- CPCV within train split (10 groups, k=2)
- Expected N: ~60K-80K rows with non-null mae_r

**Validation:**
- CPCV R² on train split
- Test-set R² vs naive baseline (predict mean)
- Partial correlation controlling for orb_size_norm
- 5K bootstrap on total R delta when applied as dynamic stop

**Kill criteria:**
- Partial correlation ≤ 0.05 → DEAD
- Bootstrap p > 0.05 on R delta → NOT SIGNIFICANT
- Dynamic-stop simulation R improvement < 5% → NOT WORTH COMPLEXITY

**Reusable code:** CPCV (trading_app/ml/cpcv.py), feature extraction (features.py backbone), bootstrap (scripts/tools/ml_bootstrap_test.py). New: regression target extraction, dynamic-stop simulation.

**Estimated compute:** Training ~2 min, bootstrap ~8 hrs (5K perms), simulation ~30 min.

### 6.2 SPIKE 2: Microstructure Features

**Pre-specified feature list (5 features, exact computation):**

1. **Volume acceleration slope:** Linear regression slope of volume across ORB-window 1-minute bars, normalized by mean volume. Positive = volume increasing through window.
2. **Volume concentration ratio:** Volume in first 2 bars / volume in last 2 bars of ORB window. >1 = front-loaded, <1 = back-loaded.
3. **Price velocity slope:** Linear regression slope of close prices across ORB-window bars, normalized by ATR/window_length.
4. **VWAP-close divergence:** (VWAP − close) at ORB end, normalized by ORB size. Positive = price above VWAP (buying pressure).
5. **Bar range expansion ratio:** Max(bar range) / min(bar range) within ORB window. Measures intra-ORB volatility expansion.

**Data source:** bars_1m joined to daily_features on (trading_day, symbol). Filter to bars within ORB window timestamps.

**Testing protocol:** Univariate T1 scan (quintile binning) on each feature × mae_r. BH FDR at K=5.

**Dependencies:** New feature engineering pipeline step. Must run after build_daily_features, before ML training.

**Kill criteria:** 0-1 features show WR quintile spread > 3% → DEAD.

### 6.3 SPIKE 3: pnl_r Regression (if C or F survive)

**Features:** Same as Spike 1 + any Spike 2 survivors.

**Target:** pnl_r winsorized at ±3σ per (instrument, rr_target).

**Model:** LightGBM regressor. Advantages over RF for regression: better handling of continuous targets, built-in L1/L2 regularization, faster training.

**Kill criteria:** R² < 0.01 on test set → DEAD.

---

## 7. Open Questions

1. **Is there ANY feature that predicts WR (not just payoff)?** V2's binary classification failure may be because WR genuinely has no predictable component at ORB timescales. Framing A would answer this definitively with maximum power (208K pooled samples). If pooled binary at 208K still shows no signal, WR prediction is permanently closed.

2. **Does break_delay_min predict mae_r?** The quintile scan (Mar 20) showed WR monotonicity for break_delay. But does early break also predict LOWER adverse excursion? If yes, it's a double signal (WR + MAE). If no, break_delay is a WR signal only, which binary classification should have captured (but didn't at per-session N).

3. **Should we use LightGBM instead of RF?** For regression targets, GBM typically outperforms RF. But GBM has more hyperparameters → more overfit risk. The conservative choice is RF (fewer knobs). The pragmatic choice is GBM with aggressive regularization (max_depth=4, min_child_samples=200, learning_rate=0.01, early_stopping).

4. **Can we use E1 data to increase N?** E1 (market order) and E2 (stop-market) have different fill mechanics but similar outcomes at RR1.0 (WR differs by <2%). Pooling E1+E2 doubles N. But E1 is "not recommended" and has slight adverse selection. Need to verify E1/E2 outcome distributions are statistically indistinguishable before pooling.

5. **What about the 2026 sacred holdout?** V3 ML results on 2016-2025 data are in-sample by design. The 2026 holdout could serve as the FINAL test — but only 3 months of data exist (Jan-Mar 2026). At ~20 trades/session/month, that's ~60-240 trades per session. Marginal for bootstrap. Better to use expanding walk-forward on 2016-2025, with 2026 as directional confirmation only.

6. **Is the V2 "ML DEAD" verdict final?** V2 tested binary classification with 5 features on per-session data. It did NOT test: regression targets, at-break features, microstructure features, pooled training, or exit management. The framing was exhaustively tested; the PROBLEM SPACE is not. ML V3 is a different question, not a retry of V2.

---

## 8. Recommended Execution Order

```
SPIKE 1A: Univariate scan (1 day)
├── Compute partial correlation of each feature with mae_r, controlling for orb_size
├── If no feature achieves partial corr > 0.05 → Framing C is DEAD
└── If PASS → proceed to 1B

SPIKE 1B: MAE/MFE regression model (2 days)
├── Train pooled RF regressor on mae_r
├── CPCV + test-set evaluation
├── Dynamic-stop simulation on 2025 replay
├── 5K bootstrap on R delta
└── If bootstrap p > 0.05 → DEAD. If PASS → candidate for paper trade.

SPIKE 2: Microstructure features (3 days)
├── Build feature extraction pipeline from bars_1m
├── Compute 5 pre-specified features
├── T1 quintile scan on each
├── BH FDR at K=5
└── If <2 survive → DEAD. If PASS → add to Spike 1 model and retrain.

SPIKE 3: pnl_r regression (2 days, ONLY if 1 or 2 survive)
├── Train pooled LightGBM on winsorized pnl_r
├── Compare R² vs naive baseline
├── Bootstrap on R delta
└── If R² < 0.01 → DEAD.

SPIKE 4: Pooled binary meta-label (1 day, OPTIONAL)
├── Run pooled binary RF at 208K samples
├── This is the DEFINITIVE test of whether WR is predictable
├── If 0 signal → WR prediction is PERMANENTLY CLOSED for this system
└── If signal exists → investigate why per-session missed it
```

**Total estimated time:** 7-9 days of compute + analysis, spread over 2-3 weeks with verification gates.

**Decision tree:**
- If Spike 1 fails AND Spike 2 fails → ML provides no value for this system. Close permanently.
- If Spike 1 passes → Deploy dynamic stops. ML adds value through exit management, not entry selection.
- If Spike 2 passes → Rich feature discovery. Feed into both exit (Spike 1) and entry (Spike 3) models.
- If Spike 3 passes → pnl_r regression is the portfolio's ML component. Replace binary classification entirely.

---

## 9. The Honest Assessment

**What the literature says about our situation:**

- **Aronson** (p.325): Data mining works when N is large. Our pooled N=208K is sufficient for 5-7 features. Per-session N=3K-8K is NOT.
- **Carver** (p.75): Equal blend beats fitted selection. ML adds complexity that usually HURTS. Only deploy if the improvement is large and robust.
- **De Prado** (p.19): Without theory-ML interplay = horoscopes. Every framing needs a mechanism.
- **Chan** (p.41): 100 parameters = guaranteed overfit. Keep models simple.
- **Chordia** (p.40): 98% of discoveries are false. Our kill rate should be 95%+.

**What our project's history says:**

Every exciting ML number was an artifact:
- V1: +1,345R delta → threshold artifact (p=0.35)
- V1 replay: +12.20R → pre-fix methodology, 3 FAILs open
- V2: 0/12 BH survivors after fixing all FAILs

**What's genuinely different about V3:**

1. **Different question:** V2 asked "should I take this trade?" V3C asks "how should I manage this trade's exit?" — a fundamentally different ML application.
2. **Different target:** V2 predicted win/loss (flat at 58%). V3C predicts MAE/MFE (continuous, well-distributed).
3. **Different features:** V2 excluded at-break features. V3C restores them for exit management (where they're legitimate).
4. **Pooled training:** V2 used per-session (N=3K-8K). V3 pools (N=60K-208K).

These are structural changes, not parameter tweaks. The V2 "ML DEAD" verdict applies to binary per-session meta-labeling with pre-break features. It does NOT apply to pooled regression on continuous targets with at-break features.

**But:** The probability of ANY ML framing surviving honest testing is still < 50%. The base rate for trading ML surviving proper validation is ~2% (Chordia). Our structural advantages (10-year dataset, pre-computed outcomes, established test methodology) improve that, but not by an order of magnitude.

**The honest expectation:** 1-2 spikes will die. If Spike 1C (MAE/MFE) survives, it's worth deploying. If everything dies, ML is permanently closed for this system, and the raw baseline portfolio (which already works) is the final answer.

That's also a perfectly good outcome.

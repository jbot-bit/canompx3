# Strategy Research Blueprint

**This doc defines HOW strategy research is MEANT to be done.** The process, the gates, the order of operations, the kill criteria. Follow this regardless of what the current numbers say.

Sections §1-§3, §6, §10-§11 are **METHODOLOGY** — the permanent process. They don't change when data changes.
Sections §4-§5, §7-§9 are **CURRENT STATE** — volatile snapshots. Query canonical sources, don't cite from memory.

**Authority:** This doc ROUTES to governing docs. It does not override them.
- Trading logic → `TRADING_RULES.md`
- Research methodology → `RESEARCH_RULES.md`
- Code structure → `CLAUDE.md`
- Feature specs → `docs/specs/`

**Freshness:** Last structural audit: 2026-04-13 (6 filter discovery null results added to NO-GO). Re-audit after: any NO-GO declaration, new instrument/session/entry model, pipeline rebuild, ML finding, or validated_setups change.

**Core principle:** Our system is not proven right — it's proven not-yet-wrong. Every finding is provisional until forward-tested. The methodology exists to catch our own mistakes before they cost capital.

---

## 1. Quick Reference — "What Am I Doing?"

| If you're... | Go to section | Also read | Think about |
|-------------|--------------|-----------|-------------|
| Researching a new idea | §3 Research Test Sequence | RESEARCH_RULES.md | Is this in the NO-GO registry (§5)? Does a mechanism exist? What variable space will I search? |
| Training/evaluating ML | §6 ML Sub-Pipeline | `trading_app/ml/config.py` | Have I checked univariate signal first? Am I testing all RR/aperture combinations? Bootstrap is mandatory. |
| Building/changing a portfolio | §4 Variable Space | TRADING_RULES.md Session Playbook | What are the deployment constraints (DD limits, position limits)? Correlation between strategies? Stop multiplier (0.75x for prop)? |
| Setting up paper trading | §7 Paper Trading Checklist | Pre-registration doc | Kill criteria defined BEFORE starting? Cost model verified? 2026 holdout is SACRED? |
| Running a pipeline rebuild | §8 Pipeline Order | `docs/ARCHITECTURE.md` | Dependency order matters. init_db first if adding sessions. FK constraints block deletes. |
| Checking if something is dead | §5 NO-GO Registry | TRADING_RULES.md "What Doesn't Work" | What would REOPEN this path? Has the variable space been fully explored? |
| Declaring something dead | §5 + §10 | §3 Gate 2 (CRITICAL RULE) | Tested ≥3 values per dimension? Bootstrap run? Fresh audit done? |

---

## 2. The One Thing That Matters

**ORB size relative to friction IS the primary edge gate.** 16-year regime audit (Apr 2026) proved: the edge is cost-gated (ARITHMETIC_ONLY — trades where cost/risk < ~10% are profitable, above ~15% they're not). G-filters approximate this gate at current prices. Gross R (before friction) is positive across ALL eras (2010-2025); early-era negative net R is entirely explained by friction drag from small absolute ORBs. ~~Break delay was previously claimed as a conviction signal~~ **but institutional test (2026-03-30, 7.2M trades) killed it: per-session d<0.2, O5/O30 direction flip, Simpson's paradox in pooled data. See NO-GO registry §5.** Source: TRADING_RULES.md "ORB Size = The Edge" + `research/break_delay_institutional_test.py`.

**Current reality (from gold.db, verified 2026-03-24, 10-year backfill):**
- **MNQ E2:** Positive unfiltered at 5yr (2021-2025). At 16yr: CME_PRECLOSE and NYSE_OPEN are structural (positive with G5 in ALL eras including 2010-15). COMEX_SETTLE marginal (p=0.060). EUROPE_FLOW dead at 16yr unfiltered (G5 early N=89, inconclusive). US_DATA_1000 partially regime-dependent (G5 early -0.072R N=920, not purely friction — may reflect pre-2016 US data release dynamics). Most early-era kills are friction drag, not regime change (Apr 2 audit).
- **MGC E2 unfiltered:** Negative. Positive ONLY under combined gate (friction <10% + timeout <=10m): +0.148R.
- **MES E2 unfiltered:** Negative. Positive under combined gate: +0.116R.
- **E1 unfiltered:** Negative everywhere including MNQ.
- **Validated setups:** Query `validated_setups` for current count — do not cite from memory. Counts change after every rebuild.

**All tables in §4 below are MNQ E2 unless stated otherwise.** For MGC/MES session-specific performance, see TRADING_RULES.md Session Playbook.

→ `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`  # ['MES', 'MGC', 'MNQ']
→ `from trading_app.config import ENTRY_MODELS`  # ['E1', 'E2', 'E3']

---

## 3. Research Test Sequence

When evaluating ANY new idea, follow these gates IN ORDER. Each gate has a kill criterion.

### Gate 1: Mechanism Check
**Question:** Why should this work? What structural market reason exists?
**Reference:** RESEARCH_RULES.md "The Mechanism Test"
**Pass:** Plausible structural mechanism (friction, liquidity, institutional flow)
**Kill:** "The numbers show it works" is not a mechanism. → STOP or add extra scrutiny.
**Warning:** Mechanism check does NOT prove causation. It screens obvious artifacts.

### Gate 2: Baseline Viability
**Question:** Is there positive ExpR at any point in the variable space?
**Test:** Query `orb_outcomes` grouped by relevant variables.
**Kill:** Negative ExpR at ALL tested combinations → DEAD for this approach.
**CRITICAL RULE:** Before declaring a variable dimension dead, test at LEAST 3 values across that dimension.
- RR: test 1.0, 1.5, 2.0 minimum
- Aperture: test O5, O15, O30
- Entry model: test E1, E2
- Sessions: test ALL sessions, not just 2-3

### Gate 3: Statistical Significance
**Question:** Does the positive baseline survive multiple testing?
**Test:** BH FDR at honest test count (not cherry-picked count)
**Reference:** RESEARCH_RULES.md "Significance Testing"
**Thresholds:** p < 0.05 minimum to note. p < 0.01 to recommend. p < 0.005 for discovery claims.
**Kill:** 0 survivors after FDR → noise, not edge.
**Always report:** N trades, time span, number of variations tested, exact p-values.

### Gate 4: Out-of-Sample Validation
**Question:** Does it hold on unseen data?
**Test:** Walk-forward, holdout split, or pre-registered forward test.
**Reference:** RESEARCH_RULES.md "In-Sample vs Out-of-Sample"
**Kill:** WFE < 50% (Sharpe decay > 50% from IS to OOS) → likely overfit.
**2026 holdout is SACRED:** 3 pre-registered MNQ strategies only. See `docs/pre-registrations/`.

### Gate 5: Adversarial Audit
**Question:** What could make this result fake?
**Tests (pick applicable):**
- Bootstrap permutation (mandatory for ML, recommended for novel findings)
- Sensitivity analysis: change param ±20%, does result survive?
- Fresh agent audit: new context, no prior bias
**Kill:** Bootstrap p > 0.05, or ±20% param change kills the finding → artifact.

### Gate 6: Replay Validation
**Question:** Does the ExecutionEngine match the statistical expectations?
**Test:** `paper_trader.py` replay on historical data, compare to orb_outcomes stats.
**Kill:** Material discrepancy (>10% trade count difference, opposite sign on PnL) → execution bug.

### Gate 7: Paper Trade with Kill Criteria
**Question:** Does it work forward?
**Pre-register:** Kill criteria BEFORE starting. Lock in a doc with git commit.
**Reference:** `docs/pre-registrations/` for existing kill criteria.
**Kill criteria template:**
- After N trades per session: ExpR < threshold → STOP that session
- After N trades: slippage > threshold → STOP (cost model wrong)
- After N total: portfolio ExpR < threshold → STOP everything

---

## 4. Variable Space

Every tunable parameter, its values, and canonical source. QUERY these, never cite from memory.

### Instruments
| Instrument | Status | Unfiltered E2 O5 RR1.0 | Source |
|-----------|--------|------------------------|--------|
| MNQ | **ACTIVE** | +0.085R (POSITIVE) | `ACTIVE_ORB_INSTRUMENTS` |
| MGC | ACTIVE (filter-dependent) | −0.157R unfiltered, positive with G5+ on select sessions | `ACTIVE_ORB_INSTRUMENTS` |
| MES | ACTIVE (filter-dependent) | −0.061R unfiltered, positive with G4+ on select sessions | `ACTIVE_ORB_INSTRUMENTS` |
| M2K | DEAD | 0/18 families survive noise | Removed 2026-03-18. **Trap:** `orb_active=True` in ASSET_CONFIGS but excluded by `DEAD_ORB_INSTRUMENTS`. Always use `ACTIVE_ORB_INSTRUMENTS`, never `orb_active` directly. |
| MCL, SIL, M6E, MBT | DEAD | 0 validated | `ASSET_CONFIGS` (present, not active) |

→ `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`

### Entry Models
| Model | Status | Mechanism | Notes |
|-------|--------|-----------|-------|
| E2 | **PRIMARY** | Stop-market at ORB + slippage | Honest, includes fakeouts |
| E1 | Active (conservative) | Market on next bar, ~1.18x risk | No backtest bias |
| E3 | RETIRED | Limit retrace at ORB edge | Adverse selection. Per-combo only. In `SKIP_ENTRY_MODELS` — skipped at runtime. |
| E0 | **PURGED** | 3 optimistic biases | Never use. |

→ `from trading_app.config import SKIP_ENTRY_MODELS`  # frozenset({"E3"}) — excluded from outcome_builder and grid search

→ `from trading_app.config import ENTRY_MODELS`

### RR Targets (MNQ E2 O5 NO_FILTER, verified 2026-03-21)
| RR | ExpR | WR | Status |
|----|------|-----|--------|
| 1.0 | +0.085 | 54.8% | Strongest unfiltered baseline |
| 1.5 | +0.083 | 42.5% | Strong |
| 2.0 | +0.077 | 34.4% | ML signal at O30 (AUC 0.658, bootstrap p=0.005) |
| 2.5 | +0.053 | 28.3% | Weakening |
| 3.0 | +0.033 | 23.9% | Marginal |
| 4.0 | −0.010 | 17.9% | DEAD |

**MGC/MES:** All RR targets negative without size filters. With G5+ filter, MGC CME_REOPEN becomes positive (see TRADING_RULES.md Session Playbook). Size filter is mandatory for MGC/MES.

### Confirm Bars
| Value | Notes |
|-------|-------|
| CB1 | Default for E2 (no confirmation needed — stop-market triggers on touch) |
| CB2 | Optimal for E1 on G5+ days per TRADING_RULES.md |
| CB3-CB5 | Identical to CB1-CB2 for E3 (same limit order) |

### Direction
| Value | When |
|-------|------|
| BOTH | Default. Most sessions. |
| LONG-ONLY | TOKYO_OPEN (shorts negative across all instruments — TRADING_RULES.md) |
| DIR_LONG / DIR_SHORT | Available as discovery grid filters via `config.ALL_FILTERS` |

### Stop Multiplier
| Value | When | Source |
|-------|------|--------|
| 1.0 | Standard (self-funded, backtesting) | Default |
| 0.75 | Prop firm accounts (survival over income) | `manual-trading-playbook.md` |

Prop 0.75x reduces risk per trade by 25% — stop is 75% of ORB range from entry. Affects position sizing, drawdown, and ExpR. Use `--stop-multiplier 0.75` on paper_trader CLI.

### Instrument × Session Mapping
Not all sessions are enabled for all instruments. Canonical source:
→ `from pipeline.asset_configs import ASSET_CONFIGS`  # `ASSET_CONFIGS[instrument]["enabled_sessions"]`

Key exclusions:
- **MGC SINGAPORE_OPEN: OFF** (74% double-break rate — structurally mean-reverting)
- See `config.get_excluded_sessions(instrument)` for runtime exclusions

### Deployment Constraints (Prop Firms)
| Constraint | Value | Impact |
|-----------|-------|--------|
| Apex max trailing DD | $2,500-$6,000 (depends on account) | Limits position count and risk per trade |
| Tradeify max DD | $2,000-$3,000 | Same |
| TopStep max DD | $2,000-$3,000 | Same |
| Max concurrent | Typically 1-3 positions | `max_concurrent_positions` param |
| Automation allowed | Apex: NO. Tradeify/TopStep: YES. | Signal-only for Apex, auto for others |

→ See `docs/plans/manual-trading-playbook.md` for full prop deployment plan

### Portfolio Correlation
Session correlations affect portfolio diversification. Key pairs:
- **TOKYO_OPEN vs LONDON_METALS: −0.39** (negative — genuine diversification)
- **MNQ vs MES: +0.83** (same asset class — DON'T stack)
- **MGC vs MNQ/MES: +0.04** (near-zero — independent)

→ `portfolio.correlation_matrix()` for computed values. `RiskManager` enforces `max_correlation` at entry.

### Apertures (MNQ E2 RR1.0 unfiltered)
| Aperture | ExpR | Notes |
|----------|------|-------|
| O5 | +0.085 | Primary. Highest per-trade edge. |
| O15 | +0.075 | Weaker. TOKYO_OPEN is exception (O15 > O5). |
| O30 | +0.065 | Weakest baseline. But ML discriminates here at RR2.0. |

### Sessions (MNQ E2 O5 RR1.0 NO_FILTER, verified 2026-03-21)
| Session | ExpR | N | Notes |
|---------|------|---|-------|
| CME_PRECLOSE | +0.202 | 1,254 | Strongest. Pre-registered for 2026. |
| US_DATA_1000 | +0.140 | 1,311 | Strong |
| COMEX_SETTLE | +0.121 | 1,266 | Pre-registered for 2026 |
| NYSE_OPEN | +0.117 | 1,311 | Pre-registered for 2026 |
| EUROPE_FLOW | +0.097 | 1,313 | |
| NYSE_CLOSE | +0.073 | 1,102 | Low WR (30.8%). Excluded from raw baseline. |
| TOKYO_OPEN | +0.064 | 1,314 | LONG-ONLY per TRADING_RULES.md |
| CME_REOPEN | +0.059 | 1,169 | Friday skip recommended |
| US_DATA_830 | +0.055 | 1,310 | |
| LONDON_METALS | +0.053 | 1,313 | |
| SINGAPORE_OPEN | +0.029 | 1,314 | MGC excluded (74% double-break) |
| BRISBANE_1025 | +0.011 | 1,314 | Weakest |

→ `from pipeline.dst import SESSION_CATALOG`

### Filters
| Filter | What it does | Source |
|--------|-------------|--------|
| NO_FILTER | Take all trades (pass-through) | `config.ALL_FILTERS["NO_FILTER"]` |
| ORB_G4 through ORB_G8 | ORB size >= N points | `config.ALL_FILTERS` |
| ATR70_VOL | ATR percentile >= 70 | `config.ALL_FILTERS["ATR70_VOL"]` |
| DIR_LONG / DIR_SHORT | Direction filter | `config.ALL_FILTERS` |
| VOL_RV12_N20 | Relative volume >= 1.2× median | `config.ALL_FILTERS` |
| X_MES_ATR60 / X_MES_ATR70 | Cross-asset MES ATR filter | `config.ALL_FILTERS` |
| X_MGC_ATR60 / X_MGC_ATR70 | Cross-asset MGC ATR filter | `config.ALL_FILTERS` |
| Composites (ORB_G4_NODBL etc) | Size + no-double-break | `config.ALL_FILTERS` |

**CRITICAL:** `filter_type` must EXACTLY match a key in `ALL_FILTERS`. Unknown strings cause silent trade drops (fail-closed). Use `"NO_FILTER"`, never `"NONE"` or `"BASE"`.

→ `from trading_app.config import ALL_FILTERS`

### Cost Models (verified 2026-03-21)
| Instrument | Point Value | Total Friction | Tick Size |
|-----------|-------------|---------------|-----------|
| MNQ | $2.00 | $2.74 | 0.25 |
| MGC | $10.00 | $5.74 | 0.10 |
| MES | $5.00 | $3.74 | 0.25 |

→ `from pipeline.cost_model import COST_SPECS`

---

## 5. NO-GO Registry

Everything confirmed dead. Do NOT re-test without a fundamentally new approach.

| Path | Verdict | Evidence | What Would Reopen |
|------|---------|----------|-------------------|
| MGC/MES unfiltered ORB | DEAD | All negative. 0 survivors. | New instrument economics (lower friction) |
| E0 entry model | PURGED | 3 compounding biases | Nothing — structurally flawed |
| E3 entry model | RETIRED | Adverse selection. 19/20 both negative. | At-break architecture (not pre-break) |
| RR4.0 unfiltered baseline | DEAD | 0/12 sessions positive (T5), IS negative (T2), null p>0.87 (T6), 4/11 years positive (T7). Audit 2026-03-30. Filter-carried variants (G8, ATR70) exist on SINGAPORE_OPEN but ride a negative baseline — filters do all the work. | New session with structural low-WR tolerance |
| ML on PORTFOLIO-LEVEL negative baselines | DEAD | Threshold artifact, bootstrap p=0.35 | Nothing — mathematical trap |
| ML on PER-SESSION negative baselines | DEAD | V1 was p=0.005 on NYSE_OPEN but EPV=2.4. V2 fixed all 3 FAILs → 0/12 BH survivors | Fundamentally new features or 2x more data |
| ML V2 meta-labeling (5 features) | DEAD | 10/12 neg baseline, 2 bootstrapped, 0 BH survivors at K=12. NYSE_CLOSE p=0.039 raw, p=0.473 adjusted | 2x data growth or new feature discovery |
| **ML V3 RR-stratified pooled meta-label (6 features)** | **DEAD — PERMANENT** | **Sprint 2026-04-11.** 29 active MNQ+MES validated strategies, 29,488 pooled trades, positive baseline (WR 51.82%), CPCV-45, Youden-J-calibrated thresholds, 200-perm permutation test. All 3 RR-stratified trials: CPCV AUC ~0.50 (pure noise), Null A p > 0.43, BH FDR adj p > 0.73. **Holdout: ML actively destroyed −$18K across RR=1.0/1.5 strata** (baseline +$16.8K → ML −$1.5K), per-strategy local losses on 9 of 20 tested strategies. Kill criteria C1 (BH FDR), C3/C4 (paired bootstrap lower 95% CI), C8 (dollar lift), C9 (per-strategy local loss) all triggered. `orb_volume_norm` was TOP MDA feature (0.20-0.25) but the model's splits on it did not discriminate — **structural finding: filters are not interchangeable with multivariate ML features** because the extreme-value signal that makes volume-class features produce 48 BH FDR survivors as univariate filters is smoothed away in a multivariate RF. V1/V2/V3 all DEAD across 2 months of institutional audit. ML subsystem deleted Stage 4 (`trading_app/ml/`, tests, 8 drift checks, ExecutionEngine hooks). **Postmortem:** `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md`. **Stage 3 commit:** `c80869d9`. **Stage 0 verification:** `docs/audit/ml_v3/2026-04-11-stage-0-verification.md`. | Fundamentally new feature class with theory for why the signal would NOT smooth in a tree ensemble (tick microstructure from L2, order-book imbalance, at-break architecture instead of pre-break). Pre-registration in `docs/audit/hypotheses/` required BEFORE any rebuild. |
| Calendar blanket skip | DEAD | Mixed results (some days BETTER) | Per-combo only, never blanket |
| Non-ORB strategies | DEAD | 6 archetypes, 540 tests, 0 survivors | Fundamentally different market model |
| MCL, SIL, M6E, MBT, M2K ORB | DEAD | 0 validated per instrument | New data source or contract change |
| Quintile conviction | DEAD | Vol filter artifact on best decade | — |
| RSI, MACD, Bollinger, MA cross | GUILTY | RESEARCH_RULES.md: guilty until proven | Sensitivity + OOS + mechanism required |
| Pyramiding | DEAD | Correlated intraday = tail risk | — |
| Breakeven trail stops | DEAD | −0.12 to −0.17 Sharpe | — |
| O15/O30 ORB aperture (as portfolio) | ARITHMETIC_ONLY | Friction artifact — O5 gross R > O15 by 0.031R/trade on 24K matched pairs. Same family as G-filters. Mar 2026 | New mechanism hypothesis (not friction/size related) |
| Break speed / break delay filter | DEAD | 7.2M trades, BH K=96. 0 survivors at deployed apertures (O15). Simpson's paradox in pooled data. O5/O30 direction FLIP kills mechanism claim. Per-session d<0.2. `research/break_delay_institutional_test.py`. Mar 2026 | New mechanism that explains aperture direction flip |
| Late entry timing filter | DEAD | Same test battery. EARLY vs LATE collinear with break speed (not independent). 0 survivors outside CME_PRECLOSE (FRAGILE). Mar 2026 | — |
| New-on-new confluence stacking (Phase 3b pairs) | DEAD | Phase 4 OOS lift: 0/8 DEPLOY. IS lift collapsed 72-100% OOS. CI includes zero for all. Carver 1.5x: 0/8 pass. Tested: 8 pairs of new confluence features AND-combined (rel_vol+orb_size_norm, rel_vol+orb_volume, etc.). NOT tested: existing deployed filter + new confluence, veto-style stacking, cross-session. ATR70_VOL (deployed, working) is a composite — stacking is NOT universally dead. `phase4_oos_lift.py`. Mar 2026 | OOS evidence for the specific untested categories above |
| Regime-conditional discovery (rolling window) | INVESTIGATED — NO-GO | 16yr→10yr kills 16.2% of strategies. Kill mechanism is friction drag (not regime). Gross R positive all eras. CME_PRECLOSE G5: +0.084R in 2010-15. Filter grid already handles cost gating. Rolling window would admit ~50% FP (Carver). Chi2 p=6e-256 confirms kill rate varies by filter. Apr 2026 | Evidence that microstructure (not cost) changed — currently only US_DATA_1000 shows this (G5 early -0.072R) |
| Vol-regime adaptive parameter switching | DEAD | `research_vol_regime_switching.py`: H0 not rejected. Best static params indistinguishable from adaptive. Mar 2026 | Fundamentally new regime detection method |
| NR4/NR7 (Narrow Range) filter | DEAD | Crabel pattern. NR4 fires 50%, NR7 fires 33% on 23h futures (vs ~14% equities). Lift direction flips across instruments/sessions: NR7 MES CME_REOPEN +0.24R (t=6.3) but MGC EUROPE_FLOW -0.15R (t=-5.3). Not a systematic edge. E2/CB1, RR2.0, pre-holdout. `filter_discovery_all_group_a.py`. Apr 2026 | Crabel NR is for equities with 6.5h days. 23h CME micros dilute the contraction signal to noise. Would need a volatility-adjusted NR definition specific to futures, not the raw N-day lookback. |
| IBS (Internal Bar Strength) filter | DEAD | Kinlay/Quantified Strategies mean-reversion signal tested on ORB momentum strategy. No monotonic quartile effect. No IBS-direction alignment signal survived t >= 1.5. Scattered single-quartile hits (e.g., MNQ CME_PRECLOSE Q1 t=3.1) are noise from 135+ comparisons. E2/CB1, RR2.0. `filter_discovery_all_group_a.py`. Apr 2026 | New entry model that exploits mean-reversion (not ORB breakouts) |
| Gap direction x break alignment filter | DEAD | CME micro futures have near-zero overnight gaps (median: MNQ 0.50pts, MES 0.25pts, MGC 0.10pts on instruments at 5K-17K). Lift direction flips: MNQ COMEX_SETTLE aligned +0.15R (t=5.7) but MNQ US_DATA_1000 opposing +0.10R (t=-3.5). MGC CME_REOPEN had one strong result (t=6.6) but single-instrument/single-session. `filter_discovery_a4_a5_fix.py`. Apr 2026 | Different market with real overnight gaps (equities, not 23h futures) |
| ORB compression x overnight range interaction filter | DEAD (redundant) | Crabel double-contraction thesis. Massive t-stats (t=-17 at TOKYO_OPEN) but Compressed+Quiet lift is entirely driven by `overnight_range_pct`, not the compression tier. 2x2 matrix: ALL tiers show same quiet-vs-active split. Compression adds no incremental signal over existing `OvernightRangeFilter`. `filter_discovery_a4_a5_fix.py`. Apr 2026 | Evidence that compression tier adds signal BEYOND overnight_range_pct alone (independent test) |
| Crabel Stretch filter | DEAD (impractical) | 10-day SMA of min(open-low, high-open). MES LONDON_METALS: +0.60R lift (t=4.2, Bonf) but N=102 exceeded (0.7% pass rate). MNQ: no significant lift (exceeded is WORSE, t=-2.1). Too selective to deploy. `filter_discovery_all_group_a.py`. Apr 2026 | Stretch threshold calibration for CME micros (current definition from equities produces <1% pass rate) |
| EMA20 trend alignment filter | DEAD (look-ahead artifact) | Uncorrected: t=42 (Bonf) using `daily_close` in EMA — tautological (today's close predicts today's direction). Corrected with `prev_day_close` (no look-ahead): 0/24 instrument-session combos reach t >= 2.0. Max |t| = 1.71. Trend alignment using yesterday's close has ZERO predictive power for next-day ORB break outcomes. E2/CB1, RR2.0, pre-holdout. Apr 2026 | Evidence that multi-day trend (not just yesterday's close) predicts intraday breakout direction — would need 50/100/200-day EMA tested on prev_day_close with theoretical justification |

Full NO-GO table with details: TRADING_RULES.md "What Doesn't Work"

---

## 6. ML Sub-Pipeline — PERMANENTLY DEAD (deleted 2026-04-11)

**`trading_app/ml/` was deleted in the ML V3 sprint Stage 4 on 2026-04-11.** V1, V2, and V3 all reached the same verdict across two months of institutional audit: pooled meta-labeling over validated ORB strategies does not provide net lift and can actively destroy value.

### Timeline

| Version | Date | Methodology | Verdict |
|---|---|---|---|
| V1 | Mar 2026 | Per-aperture meta-label, 23 features, 60/20/20 split | DEAD — threshold artifact on negative baselines, bootstrap p=0.35 |
| V2 | Mar 27 2026 | Per-session 5-feature RF, CPCV, 5K bootstrap, BH FDR K=12 | DEAD — 0/12 BH survivors, EPV=2.4 trap |
| V3 | Apr 11 2026 | RR-stratified pooled meta-label, 6 features (5 V2 core + orb_volume_norm), pre-registered hypothesis, CPCV-45, 200-perm null | DEAD — 0/3 trials survive, -$18K holdout dollar lift, per-strategy local losses |

### V3 structural finding

The V3 sprint tested the most charitable configuration: pooled across 29 positive-baseline MNQ+MES strategies (51.82% pooled WR), theory-grounded features, pre-registered kill criteria, Mode A sacred holdout. CPCV AUC came out at 0.50 (pure noise) on all 3 RR-stratified trials despite `orb_volume_norm` being the top MDA feature (0.20-0.25 importance). The model uses the feature; the splits do not discriminate.

**The structural conclusion:** filters and ML features are not interchangeable. The extreme-value signal that made `orb_volume_norm`/`rel_vol`-class features produce 48 BH FDR survivors as univariate filters in the March 2026 confluence program is smoothed away when they enter a multivariate random forest. This is not a feature-engineering problem to iterate — it is a property of the signal shape in this data.

### Reopen criteria (for a future V4, if ever)

A V4 ML sprint would need to satisfy ALL of the following before code is written:
- Fundamentally new feature class with theory for why the signal would NOT smooth under tree ensembles (tick microstructure from L2, order-book imbalance, at-break architecture rather than pre-break)
- Pre-registered hypothesis file in `docs/audit/hypotheses/` per Criterion 1 BEFORE any data is touched
- New dataset (Databento L2 tick pilot, or a different instrument class) — more trades on the current 6 features will not change the verdict
- Explicit critique of the V3 postmortem showing why the structural finding does not apply

### Canonical record

- **Postmortem:** `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md`
- **Stage 0 pre-flight verification:** `docs/audit/ml_v3/2026-04-11-stage-0-verification.md`
- **Stage 2 pre-registered hypothesis (v2):** `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence.yaml` (commit `a165ff80`)
- **Stage 3 execution commit:** `c80869d9`
- **Memory topic file:** `ml_v3_dead.md`

**Do not re-implement any ML path without going through the reopen criteria above. This is a hard NO-GO registered in § 5.**

---

## 7. Paper Trading Checklist

Before deploying paper trading:

- [ ] Pre-registration doc exists in `docs/pre-registrations/` (git-committed BEFORE testing)
- [ ] Kill criteria defined (per-session ExpR threshold, slippage threshold, portfolio threshold)
- [ ] Portfolio built from canonical source (not hardcoded)
- [ ] `filter_type` matches portfolio intent AND exists in `ALL_FILTERS` (unknown keys = silent trade drops)
- [ ] Replay validation matches expectations (trade count, WR, PnL within 10%)
- [ ] Cost model verified: `from pipeline.cost_model import get_cost_spec`
- [ ] 2026 holdout is SACRED — no "quick checks" on 2026 data

### Raw Baseline Paper Trading
```bash
python -m trading_app.paper_trader --instrument MNQ --raw-baseline --rr-target 1.0 \
  --start 2025-01-01 --end 2025-12-31 --quiet
```

### ML-Filtered Paper Trading — REMOVED
ML subsystem deleted 2026-04-11 (V1/V2/V3 DEAD). The `--use-ml` flag has no effect. See § 6.

### Live Signal-Only
```bash
PYTHONPATH=. python scripts/run_live_session.py --instrument MNQ --signal-only --raw-baseline
```

---

## 8. Pipeline Order

```
Databento .dbn.zst
  → ingest_dbn.py       → bars_1m (14.9M rows)
  → build_bars_5m.py    → bars_5m (3.1M rows)
  → build_daily_features.py → daily_features (34K rows)
  → outcome_builder.py  → orb_outcomes (8.4M rows)
  → strategy_discovery.py → experimental_strategies
  → strategy_validator.py → validated_setups (11 rows, all MNQ)
  → build_edge_families.py → edge_families (0 rows — needs rebuild)
  # ML meta_label step removed 2026-04-11 (V1/V2/V3 DEAD — see § 5, § 6)
  → portfolio builder    → Portfolio object
  → paper_trader.py      → trade journal
```

Row counts verified 2026-03-21. Commands: `docs/ARCHITECTURE.md`.

**Rebuild dependencies:** Outcomes depend on daily_features. Discovery depends on outcomes. Validation depends on discovery. Edge families depend on validation. ML depends on outcomes + daily_features. Pipeline is ONE-WAY (pipeline/ → trading_app/, never reversed).

**When adding a session:** init_db.py → rebuild daily_features → rebuild outcomes → rebuild discovery → rebuild validation → rebuild edge families. See hard_lessons.md #15.

---

## 9. Active Research Threads

**⚠ VOLATILE — update every session. Query for current state.**

| Thread | Stage | Next Step | Blocking? |
|--------|-------|-----------|-----------|
| MNQ RR1.0 raw baseline | Gate 7 (paper trade) | Deploy signal-only. Kill criteria in pre-reg doc. Code ready (`--raw-baseline`). | No — other terminal running |
| MNQ RR2.0 O30 ML | **DEAD** (V2 Phase 1: 0/12 BH survivors at K=12. All 3 FAILs fixed, methodology clean. Phase 2 cancelled.) | None — path closed Mar 2026 | No |
| Confluence univariate scan | Gate 3 (tested) | 48 BH FDR survivors, 25/25 WF DEPLOYABLE. New-on-new stacking DEAD (0/8 OOS). Deployed+new & veto UNTESTED. | No |
| 2026 holdout test | Gate 5 (monitoring) | Apr 2026 forward data collecting. Pre-registered strategies deployed. | Time-gated |
| Simple regime filter (ATR>50pct) | Gate 2 (untested) | Run quartile comparison vs ML. Lower complexity alternative. | Deferred |
| Edge families rebuild | Infrastructure | Run build_edge_families.py (0 rows currently) | Needed for fitness tracking |

**⚠ This table goes stale fast. When starting a session, query the actual state rather than trusting these rows.**

---

## 10. What We Might Be Wrong About

Epistemic humility. These are assumptions baked into the system that COULD be wrong.

| Assumption | Why we believe it | What would prove us wrong | How to test |
|-----------|------------------|--------------------------|-------------|
| ORB size is THE edge | Feb 2026 stress test, friction mechanism | Size filter stops working (gold returns to $1800, ORBs shrink) | Monitor avg ORB size vs filter gate. If G5+ qualifies < 5 days/month → edge dying. |
| MNQ E2 baselines are real | BH FDR at N=55, yearly consistency | 2026 forward test fails (pre-registered, binding) | April 2026: N≥100 per session |
| ~~ML at RR2.0 O30 has genuine skill~~ | **FALSIFIED Mar 2026.** V2 Phase 1: 0/12 BH survivors at K=12 after fixing all 3 FAILs. Phase 2 cancelled. Raw baselines are the portfolio. | N/A — assumption dead | Do not re-test without fundamentally new feature source. |
| Cost model is accurate ($2.74 MNQ) | Industry standard + 1-tick slippage | Real slippage > 1 tick systematically | Paper trade kill criterion: avg slippage > 3 ticks → STOP |
| E2 stop-market is unbiased | Includes fakeouts, uses slippage | E2 still optimistic vs real fills (spread widens at session opens) | Compare paper trade fills to backtest fills |
| 60/20/20 time split is appropriate | Standard ML practice | Market regime shifted at split boundary (val period was hot) | Walk-forward validation with multiple split points |
| Sessions are stationary | Event-based (DST-clean) | CME changes trading hours, new session opens | Monitor SESSION_CATALOG against exchange calendars |

**Rule:** When planning research, check this table. If your plan depends on one of these assumptions, note it explicitly. If the assumption breaks, your plan breaks.

---

## 11. Failure Patterns — What Catches Us

| Pattern | Example | Prevention |
|---------|---------|------------|
| **Incomplete variable search** | ML tested at RR1.0 only → missed RR2.0 signal | Gate 2: test ≥3 values per dimension |
| **Missing adversarial gate** | ML +2,366R → p=0.35 artifact | Gate 5: bootstrap mandatory for ML |
| **Stale information** | VWAP "57%" → actually 99% | Always query canonical sources |
| **Blanket rules** | SINGAPORE_OPEN off → only MGC should be off | Per instrument × session, never blanket |
| **No plan-first** | 30+ sessions jumping to code | Hard lesson #14: plan is first output |
| **Metadata trusted** | Strategy_fitness table doesn't exist | Hard lesson #7: verify schema before SQL |

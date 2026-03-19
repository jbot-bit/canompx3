# Adversarial Audit Framework — Full System Bias Review

**Date:** 2026-03-19
**Status:** DESIGN (pending MNQ null test result)
**Trigger:** Zero-context adversarial review found system at 15-25% live profitability estimate.
**Prerequisite:** MNQ 100-seed null test (running overnight, sigma=5.0 calibrated)

---

## Current System State (Pre-Audit)

| Instrument | Validated | Edge Families | Status | Notes |
|------------|-----------|---------------|--------|-------|
| MGC | 0 | 0 | Active | Regime-dependent on elevated ATR. Zero survivors. |
| MES | 0 | 0 | Active | Zero survivors. |
| MNQ | 11 | 0 (not built) | Active | **Pending null test.** If MNQ E2 ceiling > 0.37R → all dead. |
| M2K | 95 | 46 (28 PURGED) | **DEAD** | Null test killed it (0/18 families survive noise floor). |

**Critical:** Noise floors TEMPORARILY ZEROED in `config.py:97-100` during per-instrument calibration.
MGC floors known (E1=0.22, E2=0.32). MNQ/MES floors pending.

**Total experimental strategies tested:** ~117k across all instruments (including dead).

---

## Audit Objectives

Answer five questions with raw data and statistical tests:

1. **Does ORB breakout have a real edge, or is the apparent edge an artifact of gates and filters?**
2. **Is session-specificity genuine, or is it grid-search overfitting?**
3. **Which gates are statistically justified vs compensatory masking?**
4. **What is the honest IS vs OOS performance decay?**
5. **What is the realistic probability of live profitability?**

---

## Module 1: Raw Performance Extraction (Unfiltered Baseline)

### Purpose
Establish the naked ORB breakout performance before any gates, filters, or optimizations.

### Method
```
For each active instrument (MGC, MNQ, MES):
  For each session (10 sessions per TRADING_RULES.md):
    For each entry model (E1, E2):
      Query orb_outcomes with NO filter applied:
        - Pull ALL outcomes regardless of ORB size, confirm bars, direction
        - Compute: win_rate, mean_R, median_R, std_R, Sharpe, N, time_span
        - Compute per-RR target (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
```

### Key Questions
- What is the **baseline ExpR** with zero filtering? (Expected: negative, per TRADING_RULES.md)
- How negative? Is it -0.05R (marginal) or -0.50R (structurally broken)?
- Does the baseline vary by instrument? By session? By entry model?

### Statistical Tests
- One-sample t-test: H0: mean_R = 0 (is the baseline significantly different from zero?)
- Kruskal-Wallis: H0: distributions equal across sessions (is session variation real?)
- For each instrument pair: Mann-Whitney U on per-trade R distributions

### Output
Table: instrument × session × entry_model → raw ExpR, Sharpe, N, p-value (H0: ExpR=0)

---

## Module 2: Gate-by-Gate Marginal Impact Analysis

### Purpose
Measure what each gate adds or removes, in isolation and cumulatively.

### Gates to Test (in pipeline order)

| Gate | What It Does | Hypothesis |
|------|-------------|-----------|
| G4 (ORB >= 4pt) | Removes small ORBs | Primary edge source per TRADING_RULES.md |
| G5 (ORB >= 5pt) | Tighter size filter | Marginal improvement over G4? |
| G6 (ORB >= 6pt) | Most selective | Best per-trade but reduces N |
| CB optimization | 1-5 confirm bars | Refinement or overfitting? |
| RR optimization | 1.0-4.0 targets | Refinement or overfitting? |
| Direction filter | Long/short only | Session-dependent or noise? |
| ATR velocity | Skip contracting ATR | MGC-only confirmed signal |
| VOL filter | RV12_N20 volume | MNQ-specific regime gate |
| Cross-asset ATR | MES/MGC ATR → MNQ | Cross-market signal |

### Method
```
For each gate:
  1. Compute ExpR WITH gate applied (eligible days only)
  2. Compute ExpR WITHOUT gate (all days)
  3. Delta = ExpR_with - ExpR_without
  4. Test: paired t-test on per-trade R (matched days)
  5. Report: delta, p-value, N_removed, % of days removed

Cumulative analysis:
  Apply gates in order: G4 → CB → RR → direction → regime filters
  At each step: ExpR, N, Sharpe, marginal delta

Reverse cumulative:
  Start with all gates, remove one at a time
  Which single gate removal destroys the most edge?
```

### Critical Test: ORB Size as THE Edge
TRADING_RULES.md claims "ORB size IS the edge." Verify:
- Compute ExpR at G0 (no size filter) → expect negative
- Compute ExpR at G4, G5, G6, G8 → expect monotonically improving
- Is the improvement statistically significant at each step? (Fisher exact test on win rates)
- Does the size-edge pattern hold across ALL instruments and sessions?
- If yes → size filter is justified. If instrument-dependent → flag as potential overfitting.

### Output
Table: gate × instrument × session → ExpR_with, ExpR_without, delta, p_value, N_with, N_without

---

## Module 3: Unified vs Segmented Parameter Test

### Purpose
Determine whether per-session parameter optimization is real signal or grid-search noise.

### Method A: Fixed-Parameter Portfolio
```
Choose ONE parameter set (the most common surviving combo):
  - E2, CB1, RR2.0, G4
Apply to ALL instrument × session combos
Compute: portfolio ExpR, Sharpe, N
Compare to: current optimized per-session parameters
```

### Method B: Leave-One-Session-Out Cross-Validation
```
For each session S:
  1. Optimize parameters on ALL sessions EXCEPT S (training)
  2. Apply optimized params to session S (test)
  3. Compare test ExpR to:
     a. S-specific optimized ExpR (current system)
     b. Fixed-parameter ExpR (Method A)
  4. If S-specific >> fixed AND cross-validated: real session effect
  5. If S-specific >> fixed BUT NOT cross-validated: overfitting
```

### Method C: Permutation Test
```
Shuffle session labels 1000 times (break session-parameter association)
For each shuffle:
  Run discovery grid with shuffled labels
  Record best ExpR per "session"
Compare shuffled distribution to actual best-per-session ExpR
If actual >> shuffled (p < 0.01): session-specificity is real
If actual ≈ shuffled: apparent specificity is noise
```

### Statistical Tests
- Jobson-Korkie: H0: Sharpe_unified = Sharpe_segmented
- Bootstrap confidence intervals on the Sharpe difference
- Permutation test p-value

### Output
- Fixed-param portfolio vs optimized portfolio: ExpR, Sharpe, MaxDD
- Cross-validation results per session
- Permutation test p-value for session-specificity

---

## Module 4: RED-Flag Verification

### 4A: Walk-Forward Query — orb_minutes Join Check

**Risk:** If WF validation queries `daily_features` without `orb_minutes` filter, sample inflated 3x (O5/O15/O30 rows per day).

**Method:**
1. Read `trading_app/walkforward.py` — find the exact SQL query for OOS window
2. Check whether `orb_minutes` is in the WHERE clause
3. If missing: count actual rows returned vs expected (1 per trading day)
4. If inflated: ALL walk-forward results are invalid. Full revalidation needed.

**Verification:** Run a single WF window for one strategy, count rows vs trading days.

### 4B: Singleton Family Trap

**Risk:** N=1 strategies with 100 trades + Sharpe 1.0 → ROBUST. Weak threshold.

**Method:**
1. Query edge_families WHERE member_count = 1 AND robustness_status IN ('ROBUST', 'SINGLETON')
2. For each: pull full_sample_sharpe, full_sample_exp_r, oos_exp_r (if available)
3. Compare singleton strategies to family-head strategies (N >= 5)
4. Test: are singletons systematically worse OOS? (t-test on WFE)

**Evidence needed:** Distribution of WFE for singletons vs robust families.

### 4C: Session-Specific Filter Origin (Git Blame)

**Risk:** MGC SINGAPORE_OPEN/LONDON_METALS filters in `config.py:get_filters_for_grid()` — were these discovered on training data, then hardcoded?

**Method:**
1. `git log -p --follow` on config.py lines 365-440
2. When were these filters added?
3. What was the stated rationale at commit time?
4. Was a temporal hold-out used? Or full-dataset analysis → hardcode?

**If discovered on full dataset:** Look-ahead bias. Filters should be walk-forward validated.

### 4D: WF Trade-Count Override Impact

**Risk:** MGC WF windows lowered to 10 trades (now 30 per config.py:132).

**Method:**
1. Re-run WF for MGC strategies at 15 trades/window (default) vs 30 (current)
2. How many strategies pass at 15 that fail at 30? Vice versa?
3. Is the 30-trade window justified by regime diversity, or does it just lower the bar?

### 4E: Noise Floor Calibration Gap

**Risk:** Noise floors ZEROED in production. Any strategies validated during this window are unfiltered.

**Method:**
1. Check git log: when were floors zeroed?
2. Were any strategies validated/promoted AFTER zeroing?
3. If yes: those strategies bypassed the noise gate. Flag for revalidation.

**Critical:** The 11 MNQ survivors — were they validated before or after zeroing?

### Output
Per RED-flag: CONFIRMED (bias exists) / CLEARED (no bias) / INCONCLUSIVE (more data needed)

---

## Module 5: IS vs OOS Sharpe Decay (Bailey 2014)

### Purpose
Measure the shrinkage from in-sample discovery to out-of-sample validation.

### Method
```
For each validated strategy (MNQ 11 + M2K 95 historical):
  1. Pull discovery-time ExpR and Sharpe (from experimental_strategies)
  2. Pull WF aggregate OOS ExpR and Sharpe (from validation run)
  3. Compute: WFE = OOS_Sharpe / IS_Sharpe
  4. Compute: ExpR_decay = 1 - (OOS_ExpR / IS_ExpR)
```

### Expected Results (Bailey 2014 benchmark)
- WFE > 0.50: Strategy likely real (OOS retains >= 50% of IS)
- WFE 0.25-0.50: Gray zone — possible overfit
- WFE < 0.25: Likely overfit

### Statistical Tests
- Distribution of WFE across all validated strategies
- One-sample t-test: H0: mean(WFE) >= 0.50 (is the portfolio honestly validated?)
- Correlation: WFE vs N (do higher-sample strategies have better OOS retention?)

### Output
- Histogram of WFE across all strategies
- Mean, median, std of WFE
- % strategies with WFE < 0.25 (likely overfit)
- % strategies with WFE > 0.50 (likely real)

---

## Module 6: Null Test Integration

### Purpose
Gate all findings through per-instrument noise floors.

### Method
```
After MNQ null test completes:
  1. Extract MNQ E1 and E2 noise ceilings (mean + 2*std of per-seed maxima)
  2. Apply to all 11 MNQ survivors: ExpR > MNQ_ceiling?
  3. Count survivors after honest calibration

After MES null test (if run):
  4. Same process for MES

Final portfolio:
  5. Only strategies above per-instrument noise floor are "real"
  6. Report: how many survive? What is portfolio ExpR/Sharpe?
```

### Decision Matrix

| MNQ E2 Ceiling | Survivors (of 11) | Implication |
|----------------|-------------------|-------------|
| <= 0.32 | 11 (all) | MNQ edge may be real. Proceed to paper trading. |
| 0.32-0.37 | Some | Marginal. Need more data. |
| > 0.37 | 0 | **All MNQ strategies are noise.** System has ZERO edge. |

### If Zero Survivors
This is not a failure of the audit — it's the audit working correctly. If zero strategies survive honest calibration across all instruments, the honest conclusion is:

> "ORB breakout as implemented in this system does not have a statistically detectable edge above the noise floor of random entry. The apparent edge was an artifact of grid search over ~117k combinations with insufficient multiple testing correction at the per-instrument level."

This finding would be consistent with:
- The academic literature (ORB edge is "real but fragile, regime-dependent")
- The 15-25% live profitability estimate from the zero-context adversarial review
- E0 purge (33/33 artifact) removing the strongest apparent signal

---

## Execution Plan

### Phase 1: Pre-Audit Checks (30 min)
- [ ] Verify MNQ null test completed and extract results
- [ ] Run RED-flag 4A (WF query orb_minutes check) — blocking if inflated
- [ ] Run RED-flag 4E (noise floor zeroing timeline) — blocking if strategies bypassed gate

### Phase 2: Raw Extraction (1 hour)
- [ ] Module 1: Unfiltered baseline for all instrument × session × entry model
- [ ] Module 6: Apply null test floors, gate all subsequent analysis

### Phase 3: Gate Analysis (1 hour)
- [ ] Module 2: Gate-by-gate marginal impact
- [ ] Module 2 critical test: ORB size as THE edge (cross-instrument verification)

### Phase 4: Specificity Test (1-2 hours)
- [ ] Module 3A: Fixed-parameter portfolio
- [ ] Module 3B: Leave-one-session-out cross-validation
- [ ] Module 3C: Permutation test (if compute allows — 1000 shuffles × grid = heavy)

### Phase 5: Decay & Verification (30 min)
- [ ] Module 5: IS vs OOS Sharpe decay for all validated strategies
- [ ] Module 4B-D: Remaining RED-flag checks

### Phase 6: Synthesis (30 min)
- [ ] Compile all findings into audit report
- [ ] Honest assessment of strategy viability
- [ ] Recommendations: continue, pivot, or abandon

---

## Dependencies & Blockers

| Blocker | Impact | Resolution |
|---------|--------|-----------|
| MNQ null test not complete | Cannot run Module 6, cannot gate Module 1-5 findings | Wait for overnight completion |
| RED-flag 4A confirmed (WF inflation) | ALL validation results invalid | Must fix query + full revalidation before any other module |
| RED-flag 4E confirmed (strategies validated with zeroed floors) | MNQ 11 survivors may be ungated | Revalidate after floors restored |
| MES null test not run | Cannot set MES-specific floor | Currently MES has 0 validated → low impact |

---

## What This Audit Will NOT Do

- Propose new strategies or parameter sets (this is an audit, not discovery)
- Cherry-pick time periods or instruments to make results look better
- Smooth over inconvenient findings with caveats
- Recommend live trading unless the raw numbers unambiguously support it
- Write any production code (design mode only)

---

## Success Criteria

The audit is successful if it produces:
1. A clear YES/NO on ORB edge viability, supported by p-values
2. A ranked list of every gate's marginal contribution (justified vs compensatory)
3. An honest WFE distribution showing IS→OOS decay
4. RED-flag dispositions (CONFIRMED/CLEARED/INCONCLUSIVE)
5. A go/no-go recommendation for paper trading

The audit is NOT successful if it:
- Confirms the hypothesis without testing it
- Hides the null case (zero edge)
- Reports only favorable findings
- Produces recommendations without supporting statistics

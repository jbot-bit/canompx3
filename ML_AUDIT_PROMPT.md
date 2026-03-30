# ML & Confluence System — Honest Audit Prompt

Use this prompt to interrogate the ORB trading system's ML and confluence implementation. Copy-paste to a fresh session or use as a structured self-audit checklist.

---

## THE PROMPT

You are auditing a futures ORB breakout trading system's ML and confluence layer. The system declared "ML is DEAD" after V2 meta-labeling produced 0/12 BH FDR survivors. Before accepting that verdict or reopening ML, answer every question below honestly, grounded in data — not assumptions. Show your work.

### PART 1: Was the Right Question Asked?

1. **Framing check:** The ML system uses meta-labeling (RandomForest binary classifier predicting win/loss per trade). Is this the right framing for ORB confluences? Confluences in discretionary trading don't predict win/loss — they shift the *distribution* of outcomes (better R:R, tighter MAE, higher MFE). Did we test for distributional shifts, or only binary classification accuracy?

2. **Base rate problem:** ORB breaks already have 55-70% win rate at RR=1.0. A classifier needs to improve on that. But confluences might matter more at RR=1.5+ or RR=2.0+ where the base rate is lower and the signal-to-noise ratio is different. Were higher RR targets tested with the same rigor, or were they washed out by low sample sizes at higher RR?

3. **Feature interaction vs. feature importance:** Random Forest importance measures marginal contribution. But confluences work as *conjunctions* — PDH proximity matters WHEN the gap is in the same direction AND ATR is expanding. Were any interaction terms tested? Were any conditional subsets examined (e.g., "prior_sessions_broken > 2 AND gap_up AND atr_vel_ratio > 1.05")?

4. **Per-session vs. pooled:** The V2 test ran per-session models (correct — sessions have different microstructure). But did we check whether confluences that are noise globally have signal in specific sessions? For example: does PDH proximity matter for NYSE_OPEN but not TOKYO_OPEN? A global "PDH = noise" verdict may hide session-specific signal.

### PART 2: Were the Right Features Tested?

5. **Feature completeness audit.** List every market structure concept that experienced ORB/futures traders use for confluence. For each one, state: (a) Is it computed in `daily_features`? (b) Was it tested in ML? (c) Was it tested as a standalone univariate filter on positive baselines? (d) If not tested, why not — principled exclusion or oversight?

   Minimum checklist:
   - Prior day high / prior day low (as support/resistance)
   - VWAP (developing, session, prior day)
   - Initial balance (first 30/60 min range)
   - Overnight high / overnight low
   - Prior session ORB levels (as magnets)
   - Volume profile / POC / value area
   - Gap fill tendency (does gap direction predict ORB direction?)
   - Break velocity / momentum at break
   - Time-of-day within session (early vs late breaks)
   - Multi-timeframe trend alignment (is 1h/4h trend with or against the break?)
   - Round numbers / psychological levels
   - Cumulative tick / delta (order flow)
   - Relative volume at ORB formation (not just break bar)
   - Compression/expansion sequence (consecutive compressed days → expansion)

6. **Feature engineering quality.** For each feature that IS computed:
   - Is the computation correct? (Trace the code path from `bars_1m` → `daily_features` → ML feature)
   - Is normalization appropriate? (ATR-normalized? Percentile-ranked? Raw points are non-stationary)
   - Is the lookback window defensible? (Why 20 for ATR? Why 5 bars for pre-velocity? Why 252 for percentile?)
   - Does the feature have look-ahead contamination? (Check the blacklist — is it complete?)
   - Is the feature computed at the right TIME POINT? (Pre-ORB? Pre-break? Post-break? Each has different valid-use windows)

7. **Feature timing audit.** The system draws a hard line: ML features must be known BEFORE break (pre-break). But some of the most informative features are known AT break (break bar volume, break velocity, break delay). These were removed from ML. Was an "at-break" model variant tested separately? The trading decision for stop-market entries (E2) happens pre-break, but for limit entries the at-break information is available. Is the timing constraint correct for all entry models?

### PART 3: Was the Statistical Methodology Sound?

8. **CPCV implementation.** The system uses Combinatorial Purged Cross-Validation (de Prado). Verify:
   - Are purge and embargo periods sufficient? (1 day each — is that enough given ORB sessions can have correlated outcomes across consecutive days?)
   - Is the time-ordering correct? (Bars assigned to trading days correctly? DST handled?)
   - Does CPCV handle the class imbalance? (55-70% wins = mild imbalance, but at RR=2.0 it could be 30-40% wins)

9. **Bootstrap test power.** 5000 permutations were used. For the 2 sessions that passed baseline:
   - What effect size was the bootstrap powered to detect? (If the true AUC improvement is 0.02, is 5000 perms enough?)
   - Were permutation labels shuffled correctly (within time blocks, not globally)?
   - Was the test statistic appropriate (AUC? ΔExpR? Something else?)

10. **BH FDR correction.** K=12 (session family). But:
    - Is K=12 the right family? The pre-registration tested 108 configs. Even with per-session selection reducing to 12, the selection step itself introduces multiplicity.
    - Were the 10 sessions excluded by baseline gate counted in K? (They should be — excluding them deflates K and inflates significance)
    - What would results look like at K=108 (all configs)?

11. **Negative baseline gate.** 10/12 sessions had negative train ExpR and were skipped. But:
    - Negative baseline ≠ "ML can't help." ML could theoretically flip a negative baseline positive by skipping the worst trades. Was this tested?
    - The gate is conservative (good for avoiding false positives) but could hide real ML signal. What's the cost of the gate?

### PART 4: Is the Feature Pipeline Correct End-to-End?

12. **Join integrity.** The triple-join requirement (trading_day + symbol + orb_minutes) is documented. Verify:
    - Does the ML training pipeline apply it correctly? (Check `meta_label.py` query)
    - Does the feature extraction pipeline apply it correctly? (Check `features.py` joins)
    - Are there any code paths that join on only 2 of the 3 keys?

13. **Data leakage scan.** Beyond the documented blacklist:
    - Does any feature use information from AFTER the ORB window of the session being traded? (Not just after the break — after the ORB)
    - Cross-session features (`prior_sessions_broken`): if session X starts at 10:00 and session Y starts at 10:30, does the feature for Y correctly exclude X's break if X broke after 10:30?
    - VWAP computation: "from day start to session start" — what's "day start"? Is it 09:00 Brisbane? If so, for CME_REOPEN (also ~09:00), the VWAP window is near-zero.

14. **Survivorship / availability bias.**
    - Are features available for ALL trading days in the sample, or do early dates have NULLs (e.g., ATR_20 needs 20 prior days)?
    - How are NULLs handled? (Dropped? Imputed? If imputed, with what — and does the imputation introduce bias?)
    - Does the training set include the first N days where features are being "warmed up"?

### PART 5: What Would "Done Right" Look Like?

15. **Univariate filter audit.** Before any ML, each candidate confluence should be tested as a simple binary filter on positive-baseline sessions:
    - Split the population: filter ON vs filter OFF
    - Compare: ExpR, win rate, Sharpe, N (both sides)
    - Statistical test: two-sample t-test or permutation test on per-trade PnL
    - Correction: BH FDR across all filters tested
    - This was NOT done systematically. The system jumped from "features in RF" to "ML is dead." The intermediate step — "which features work as simple filters?" — is missing.

16. **Distributional analysis.** For the top 3-5 confluence features:
    - Plot the PnL distribution (histogram/KDE) for trades WITH vs WITHOUT the confluence
    - Compare MAE distributions (do confluences reduce adverse excursion?)
    - Compare MFE distributions (do confluences increase favorable excursion?)
    - This tests the hypothesis "confluences shift the distribution" which meta-labeling doesn't capture.

17. **Regime-conditional testing.** Some confluences may only matter in specific regimes:
    - High-ATR days: does VWAP proximity matter more when volatility is elevated?
    - Gap days: does gap direction interact with break direction?
    - Compressed spring days: does prior_sessions_broken matter more after compression?
    - These are conditional hypotheses that RF importance averaging washes out.

18. **Implementation order.** If rebuilding the confluence system from scratch:
    - What's the correct order of operations? (Feature engineering → univariate testing → conditional testing → interaction testing → model-based integration)
    - What gates should exist between each step?
    - What sample size is needed at each step?
    - What's the minimum data requirement (years × instruments × sessions) before ML is even worth attempting?

### PART 6: Honest Assessment

19. **Kill vs. pause.** Is "ML is DEAD" the right verdict, or should it be "ML V2 meta-labeling with 5 features is dead; univariate confluence testing is NOT done"? There's a difference between "the approach failed" and "the features don't have signal."

20. **What WOULD change the verdict?** The NO-GO says "reopen condition: 2x data or new features." Is that sufficient? Or does the methodology need to change (e.g., distributional testing instead of classification, regime-conditional instead of pooled)?

21. **Opportunity cost.** Is the time spent on ML better spent on: more sessions, more instruments, better execution modeling, walk-forward validation of existing filters, or live paper trading? What's the highest-value next step for the system — honestly?

---

## HOW TO USE THIS PROMPT

1. Read `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, and `docs/STRATEGY_BLUEPRINT.md` first
2. For each question, query the actual data (MCP tools or direct SQL) — do NOT answer from memory
3. Show code paths, row counts, and statistical results for every claim
4. If a question reveals a gap, flag it explicitly: "GAP: [description]"
5. If a question reveals a contradiction, flag it: "CONTRADICTION: [description]"
6. At the end, produce a ranked list of gaps and recommended next steps
7. Do NOT recommend reopening ML until univariate filter testing is complete

---

## EXPECTED OUTPUT

A structured audit report with:
- **CONFIRMED:** Things that were done correctly
- **GAPS:** Things that were never tested or tested incorrectly
- **CONTRADICTIONS:** Places where code, docs, and data disagree
- **RECOMMENDATIONS:** Ordered list of what to do next, with estimated effort and expected value

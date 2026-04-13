# IBS Filter Validation — Claude Code Prompt

Drop into a fresh Claude Code session pointed at the repo.
Run AFTER or IN PARALLEL with the NR7 prompt (they are independent filters).

---

## Prompt

```
# Task: Validate IBS (Internal Bar Strength) as a Session-Specific Filter

## What we already know (exploratory analysis 2026-04-13)

IBS = (prev_day_close - prev_day_low) / (prev_day_high - prev_day_low)
Measures where the prior day's close sits within its range. 0 = closed at low, 1 = closed at high.
Source: Jonathan Kinlay, Quantified Strategies (78% win rate on SPY mean-reversion).

### Key finding: IBS is SESSION-SPECIFIC and DIRECTION-SPECIFIC

IBS does NOT work as a universal filter. It works differently per session:

**HIGH_IBS (> 0.75) helps CME_PRECLOSE:**
- MNQ CME_PRECLOSE E2 RR1.0 CB1 5m: HIGH_IBS+G5 avg_r=0.1704 t=4.65 N=588
  vs OTHER+G5 avg_r=0.0678 t=2.36 N=1012
- This is the STRONGEST single-filter t-stat found in the entire dataset
- Year-by-year: 2019 -6.99R, 2020 +28.85R, 2021 +13.47R, 2022 +31.92R, 
  2023 +2.35R, 2024 +3.26R, 2025 +16.92R
- Thesis: Prior day closed near high = momentum continuation into 
  end-of-day session. Institutional rebalancing flows carry through.

**LOW_IBS (< 0.25) helps COMEX_SETTLE:**
- MNQ COMEX_SETTLE E2 RR1.0 CB1 5m: LOW_IBS+G5 avg_r=0.1614 t=3.23 N=327
  vs OTHER+G5 avg_r=0.0531 t=1.92 N=1078
- MNQ COMEX_SETTLE E2 RR1.5 CB1 5m: LOW_IBS+G5 avg_r=0.1815 t=2.98 N=364
- Year-by-year: 2019 -2.04R, 2020 +6.66R, 2021 -5.57R (BAD), 
  2022 +18.91R, 2023 +6.79R, 2024 +11.05R, 2025 +19.14R
- Thesis: Prior day closed near low = mean-reversion bounce at 
  COMEX settle. Sellers repricing after prior-day weakness.
- NOTE: 2021 was negative. Era stability criterion requires ≥ -0.05 per era.
  2020-2022 era needs checking: combined it may pass, but 2021 alone fails.

**IBS does NOT help:**
- NYSE_OPEN: No meaningful differentiation
- EUROPE_FLOW: No signal (NR7 owns this session)
- TOKYO_OPEN: LOW_IBS shows moderate signal (t=2.12) but below threshold

### Correlation with other filters:
- IBS vs NR7: r = -0.01 (perfectly uncorrelated)
- IBS vs overnight_range_pct: not computed yet (do in validation)
- IBS vs ORB_G5: not computed yet (do in validation)
- IBS is computed from different columns (prev_day_close position) than 
  any existing filter (size, volume, cost, ATR, overnight range)

## Task: Formal Validation

### Phase 1: Pre-Registration

1. Read docs/institutional/hypothesis_registry_template.md
2. Create docs/audit/hypotheses/2026-04-13-ibs-session-filter.yaml with:
   - Hypothesis 1: HIGH_IBS (>0.75) improves MNQ CME_PRECLOSE ORB outcomes
   - Hypothesis 2: LOW_IBS (<0.25) improves MNQ COMEX_SETTLE ORB outcomes
   - Trials: 4 total: {CME_PRECLOSE, COMEX_SETTLE} × {RR1.0, RR1.5}
   - Entry model: E2, confirm_bars: 1, orb_minutes: 5, ORB_G5 base filter
   - IBS buckets: pre-specified at 0.25 and 0.75 (standard quartile boundaries)
   - Kill criteria: t < 2.0 after Bonferroni (4 comparisons → t ≥ 2.24)
   - Theory: 
     * HIGH_IBS → momentum continuation (Kinlay, mean-reversion literature inverted)
     * LOW_IBS → mean-reversion bounce (Kinlay, Quantified Strategies)
3. Do NOT test NYSE_OPEN, EUROPE_FLOW, or TOKYO_OPEN with IBS (no signal)

### Phase 2: Statistical Testing

Write a disposable Python script. Do NOT modify production code.

For each of the 4 trials:

1. Compute IBS from daily_features columns:
   `IBS = (prev_day_close - prev_day_low) / (prev_day_high - prev_day_low)`
   Exclude rows where prev_day_high == prev_day_low (zero-range days)

2. Join: daily_features → orb_outcomes on (trading_day, symbol, orb_minutes)
   Both tables filtered to orb_minutes = 5

3. Apply base filter: orb_{session}_size >= 5

4. Split by IBS bucket (HIGH > 0.75 for CME_PRECLOSE, LOW < 0.25 for COMEX)

5. For each bucket compute:
   - N, win_rate, mean_pnl_r, median_pnl_r, stddev_pnl_r
   - One-sample t-statistic: mean / (std / sqrt(N))
   - Two-sample Welch's t-test between IBS-filtered and non-IBS groups
   - 95% CI on mean_pnl_r
   - Cohen's d effect size

6. WHERE trading_day < '2026-01-01' (holdout fence)

7. Era stability: 2019, 2020-2022, 2023, 2024-2025
   Each era with ≥50 trades must show ExpR ≥ -0.05
   ⚠️ COMEX LOW_IBS 2021 was -5.57R on N=46 trades — check if the 
   2020-2022 era combined (N≈172) passes the threshold

8. Bonferroni: 4 comparisons → adjusted α = 0.0125, required t ≥ 2.24

### Phase 3: DSR Computation

For trials that pass Phase 2:

1. Compute DSR with n_trials = 4 (pre-registered, not bulk discovery)
2. With K=4 and observed Sharpe from the IBS-filtered data, 
   DSR should be dramatically higher than your existing strategies' DSR
3. Target: DSR > 0.95

### Phase 4: Correlation Verification

1. Compute pairwise Pearson correlation between IBS signal and ALL 
   existing filter signals on the same trading days:
   - IBS vs NR7 (expect ~-0.01, already checked)
   - IBS vs orb_size (ORB_G5 proxy)
   - IBS vs overnight_range_pct (OVNRNG proxy)
   - IBS vs COST ratio (if computable)
   - IBS vs ATR percentile
2. All correlations must be |r| < 0.3 to confirm independence

### Phase 5: Comparison to Existing Family Heads

For CME_PRECLOSE HIGH_IBS:
- Compare to MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 (current head, 
  $1003/yr, 101 trades/yr)
- Does HIGH_IBS replace X_MES_ATR60, or stack on top?
- Test: HIGH_IBS AND X_MES_ATR60 combined — does double filtering improve 
  or just cut N?

For COMEX_SETTLE LOW_IBS:
- Compare to MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 (current head, 
  $892/yr, 87 trades/yr)
- From exploratory: LOW_IBS + OVNRNG_100 showed N=346, avg_r=0.1667, t=3.44
  That's better ExpR than either alone. Check if it's a genuine stack.

### Phase 6: Implementation Proposal (design only)

IF any trials survive:

1. Propose adding `ibs_prior_day` column to daily_features 
   (continuous 0-1, computed in build_daily_features.py)
2. Propose filter types: IBS_HIGH (>0.75), IBS_LOW (<0.25)
3. Session-specificity: IBS_HIGH for CME_PRECLOSE only, 
   IBS_LOW for COMEX_SETTLE only — encode this constraint in the filter
4. Present as design proposal per CLAUDE.md gate

## DB access
- `from pipeline.paths import GOLD_DB_PATH`
- `con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)`
- Canonical tables only: daily_features, orb_outcomes
- NEVER write to the database

## Mandatory reads before starting
1. TRADING_RULES.md
2. RESEARCH_RULES.md  
3. docs/institutional/pre_registered_criteria.md
4. docs/institutional/hypothesis_registry_template.md
5. trading_app/config.py (existing filter patterns)

## What NOT to do
- Do NOT test IBS on NYSE_OPEN, EUROPE_FLOW, TOKYO_OPEN (no signal)
- Do NOT test MES with IBS (insufficient data / no signal found)
- Do NOT modify any production files
- Do NOT use n_trials > 4 for DSR computation
- Do NOT change the 0.25/0.75 IBS bucket boundaries after seeing results
- Do NOT skip the 2021 era problem on COMEX (investigate and document)
```

---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# NR7 Filter Validation — Claude Code Prompt

Drop into a fresh Claude Code session pointed at the repo.

---

## Prompt

```
# Task: Validate NR7 as a Stacking Filter for Existing ORB Strategies

## What we already know (from exploratory analysis on 2026-04-13)

NR7 (Narrow Range 7 — Crabel) was tested across all MNQ sessions with E2 
entry, RR1.0 and RR1.5, on pre-2026 data. Results:

### NR7 CREATES edge (baseline was noise or weak):
- EUROPE_FLOW RR1.0 G5: NR7 avg_r=0.0869 t=3.81 N=1573 vs NO avg_r=0.0139 t=0.88
- EUROPE_FLOW RR1.5 G5: NR7 avg_r=0.0830 t=2.87 N=1573 vs NO avg_r=0.0229 t=1.16
- COMEX_SETTLE RR1.0 G5: NR7 avg_r=0.0795 t=3.32 N=1486 vs NO avg_r=0.0475 t=2.96
- TOKYO_OPEN RR1.5 G5: NR7 avg_r=0.0916 t=3.20 N=1586 vs NO avg_r=0.0480 t=2.45

### NR7 HURTS (do not apply):
- NYSE_OPEN: NR7 drops avg_r, kills RR1.5
- CME_PRECLOSE: Small drag
- MES CME_PRECLOSE: Kills edge entirely

### Key properties:
- Correlation with ORB_G5: -0.05 (uncorrelated)
- Correlation with overnight_range_pct: -0.03 (uncorrelated)
- EUROPE_FLOW NR7+G5 year-by-year: 2019 bad (-20.89R), 2020-2025 all positive
- Dollar estimate: NR7+G5 EUROPE_FLOW = ~$1005/yr vs $633/yr G5-only 
  (60% more dollars from half the trades)

### Structural thesis:
NR7 = prior day range was tightest in 7 days = volatility contraction.
Mid-session ORBs (EUROPE_FLOW, COMEX_SETTLE) benefit because institutional 
flow on the compressed day creates a directional release through these sessions.
NYSE_OPEN doesn't benefit because the open itself IS the volatility release.

## What this task needs to do

This is a FORMAL VALIDATION of the NR7 filter, following the project's 
pre-registered criteria (docs/institutional/pre_registered_criteria.md).

### Phase 1: Pre-Registration (BEFORE any new analysis)

1. Read docs/institutional/hypothesis_registry_template.md
2. Create docs/audit/hypotheses/2026-04-13-nr7-stacking-filter.yaml with:
   - Hypothesis: NR7 improves ExpR on EUROPE_FLOW, COMEX_SETTLE, TOKYO_OPEN 
     ORB breakouts when stacked with ORB_G5
   - Exactly 6 trials: {EUROPE_FLOW, COMEX_SETTLE, TOKYO_OPEN} × {RR1.0, RR1.5}
   - Entry model: E2, confirm_bars: 1, orb_minutes: 5
   - Kill criteria: t < 2.0 after Bonferroni (6 comparisons → t ≥ 2.64)
   - Theory citation: Crabel volatility contraction-expansion thesis
3. DO NOT test NYSE_OPEN or CME_PRECLOSE with NR7 (known negative from 
   exploratory phase — would inflate trial count for no reason)

### Phase 2: Rigorous Statistical Testing

Write a disposable Python script. Do NOT modify production code.

For each of the 6 pre-registered trials:

1. Compute NR7 from daily_features.prev_day_range:
   ```
   NR7 = true when prev_day_range == min(prev_day_range) over 
   trailing 7 rows partitioned by symbol, and window is full (7 rows)
   ```

2. Join daily_features → orb_outcomes on (trading_day, symbol, orb_minutes)

3. Apply ORB_G5 filter: orb_{session}_size >= 5

4. Split into NR7=true vs NR7=false buckets

5. For each bucket compute:
   - N, win_rate, mean_pnl_r, median_pnl_r, stddev_pnl_r
   - t-statistic: mean / (std / sqrt(N))
   - 95% CI on mean_pnl_r
   - Welch's t-test between NR7 and non-NR7 groups (two-sample)

6. WHERE trading_day < '2026-01-01' (holdout fence)

7. Era stability check per criterion #9:
   Eras: 2019, 2020-2022, 2023, 2024-2025
   Each era with ≥50 trades must show ExpR ≥ -0.05

8. Bonferroni correction: 6 comparisons → adjusted α = 0.05/6 = 0.00833
   Required t ≥ 2.64 for significance

### Phase 3: DSR Computation

For the trials that pass Phase 2:

1. Compute Deflated Sharpe Ratio using:
   - n_trials = 6 (this is a targeted pre-registered test, NOT the bulk 
     discovery of 35K trials — that's the whole point)
   - SR observed from the NR7-filtered outcomes
   - Skewness and kurtosis from the pnl_r distribution
   - SR0 = max Sharpe among the 6 trials under the null

2. Target: DSR > 0.95 per criterion #5
   With K=6, the DSR hurdle is DRAMATICALLY lower than K=35616.
   This is why pre-registration matters — it resets the trial counter.

### Phase 4: Comparison to Existing Family Heads

For the trials that pass Phases 2-3:

1. Query validated_setups for the matching session (e.g., 
   MNQ_EUROPE_FLOW_E2_RR1.0_CB1_* family heads)
2. Compare: NR7+G5 vs existing filter (COST_LT12, ORB_G5, OVNRNG_100)
3. Does NR7 REPLACE an existing filter, or STACK on top?
4. Compute the 5000-permutation bootstrap lift test (if available in 
   trading_app) to check if NR7 adds marginal edge over the baseline

### Phase 5: Implementation Proposal (design only, no code changes)

IF any trials survive all phases:

1. Propose adding NR7 as a new daily_features column: 
   `is_nr7` (boolean, computed in build_daily_features.py)
2. Propose adding NR7 filter type to trading_app/config.py StrategyFilter
3. Propose composite filters: NR7_G5 (NR7 AND ORB_G5)
4. Specify which sessions it applies to (EUROPE_FLOW, COMEX_SETTLE, 
   TOKYO_OPEN only — never NYSE_OPEN or CME_PRECLOSE)
5. Present as a design proposal per CLAUDE.md design gate

## DB access
- `from pipeline.paths import GOLD_DB_PATH`
- `con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)`
- Canonical tables only: daily_features, orb_outcomes
- NEVER write to the database

## Mandatory reads before starting
1. TRADING_RULES.md
2. RESEARCH_RULES.md  
3. docs/institutional/pre_registered_criteria.md (the 12 criteria)
4. docs/institutional/hypothesis_registry_template.md
5. trading_app/config.py (existing filter patterns)
6. pipeline/build_daily_features.py (where NR7 column would go)

## What NOT to do
- Do NOT test NYSE_OPEN or CME_PRECLOSE with NR7 (known negative)
- Do NOT test MES with NR7 (known negative on CME_PRECLOSE)
- Do NOT modify any production files
- Do NOT use n_trials=35616 for DSR — this is a pre-registered K=6 test
- Do NOT expand scope beyond the 6 pre-registered trials
- Do NOT skip the era stability check
- Do NOT claim "done" without showing all statistical output
```

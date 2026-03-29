# APERTURE RESEARCH MASTER PROMPT
## O15/O30 Edge Discovery — From Pipeline-Verified Data

**Created:** 2026-03-29
**Prerequisite:** PIPELINE_VERIFICATION_MASTER.md — 26/26 PASS (2026-03-29). Pipeline math is clean. This research trusts the data.
**Authority:** RESEARCH_RULES.md (methodology), STRATEGY_BLUEPRINT.md §3 (gate sequence), TRADING_RULES.md (trading logic).

---

## §0. CONTEXT — WHAT WE KNOW AND DON'T KNOW

### Known (verified)
- **O5 baseline:** MNQ E2 RR1.0 unfiltered = +0.085R. MGC/MES need size filters (G5+/G4+).
- **O15 baseline:** MNQ E2 RR1.0 unfiltered = +0.075R (weaker per-trade than O5).
- **O30 baseline:** MNQ E2 RR1.0 unfiltered = +0.065R (weakest).
- **TOKYO_OPEN O15 premium:** +0.208R over O5 (validated, revalidated Mar 2026). Only session where O15 > O5.
- **Pipeline:** 573 O15 + 256 O30 strategies in validated_setups. 3 O15 strategies in LIVE_PORTFOLIO. 0 O30 in production.
- **Paper trader fix:** Aperture mismatch bug fixed (V2). O15/O30 now load correct `orb_minutes` in execution engine.
- **Outcome replay:** 100 O15 + 100 O30 outcomes verified bar-by-bar (§4 of verification master). Math is correct.
- **ML at O30:** DEAD. 0/12 BH FDR survivors at K=12. Do not revisit.

### Unknown (this research answers)
1. Which O15/O30 strategies survive forward replay with the fixed paper trader?
2. Is the O15 edge real beyond TOKYO_OPEN, or is it one-session artifact?
3. Does O30 have ANY tradeable edge, or is it dead like ML?
4. Do O15/O30 edges survive walk-forward (WFE > 50%)?
5. Are O15/O30 strategies additive to the O5 portfolio (low correlation) or redundant?
6. Is the wider ORB a genuine mechanism (more information = better breakout signal) or just noise reduction by fewer trades?

---

## §1. OPERATING RULES — NON-NEGOTIABLE

1. **CANONICAL LAYERS ONLY.** All queries hit `bars_1m`, `daily_features`, `orb_outcomes`. Never use `validated_setups` or `edge_families` for discovery truth.

2. **2026 HOLDOUT IS SACRED.** Discovery uses data through 2025-12-31 ONLY. 2026 data is forward-test monitoring. Do not optimize on it. Do not peek. The moment you look at 2026 results to decide whether a strategy is "good," you've contaminated the holdout.

3. **BH FDR ON EVERY CLAIM.** Every "this works" claim must survive Benjamini-Hochberg correction at the correct K. Report both global K (all strategies tested) and instrument/family K (for promotion decisions). Never swap K post-hoc.

4. **MECHANISM FIRST.** Before declaring any O15/O30 edge real, articulate WHY wider apertures would produce a structural advantage. "The numbers show it" is not a mechanism. If you can't explain the mechanism, the edge is likely noise that hasn't been corrected for yet.

5. **SENSITIVITY OR IT DIDN'T HAPPEN.** Every parameterized finding must survive ±20% perturbation. If O15 works at 15 minutes but not at 13 or 17, it's curve-fit.

6. **NO COMPARISONS WITHOUT MATCHING.** When comparing O5 vs O15 vs O30, match on SAME trading days, SAME session, SAME direction, SAME instrument. Unmatched comparisons confound calendar effects with aperture effects.

7. **FAIL-CLOSED ON AMBIGUITY.** If a result is borderline (p ≈ 0.05, WFE ≈ 0.50, sensitivity fails 1 of 9 cells), the answer is NOT PROVEN. Don't round up.

8. **REPORT NEGATIVE RESULTS.** If O30 is dead, say so. If O15 only works on TOKYO_OPEN and nowhere else, say so. Negative results save future sessions from re-testing dead ends. Add confirmed dead ends to the NO-GO registry in STRATEGY_BLUEPRINT.md §5.

---

## §2. GATE SEQUENCE — FOLLOW THIS ORDER

This follows STRATEGY_BLUEPRINT.md §3, adapted for aperture-specific research.

### Gate 0: Data State Verification
Before touching anything, confirm the data is fresh and the paper trader fix is deployed.

```python
"""
CHECKS:
1. orb_outcomes has O15 and O30 rows for all active instruments
2. Paper trader fix commit is in the current branch (V2 orb_minutes fix)
3. daily_features has O15/O30 rows with non-NULL orb_size_pts
4. No stale outcomes (rebuilt AFTER paper trader fix date)
"""
```

```sql
-- Outcome freshness by aperture
SELECT symbol, orb_minutes, COUNT(*) AS n_outcomes,
       MIN(trading_day) AS first_day, MAX(trading_day) AS last_day
FROM orb_outcomes
WHERE symbol IN ('MGC', 'MNQ', 'MES')
GROUP BY 1, 2
ORDER BY 1, 2;

-- Verify O15/O30 are populated
SELECT orb_minutes, COUNT(DISTINCT symbol) AS instruments,
       COUNT(DISTINCT orb_label) AS sessions
FROM daily_features
WHERE orb_minutes IN (15, 30)
GROUP BY 1;
```

**HALT if:** O15/O30 rows are missing, stale, or pre-fix.

### Gate 1: Mechanism Test — WHY Would Wider Apertures Work?

**Do this BEFORE looking at any results.** Write down the structural hypothesis FIRST.

Candidate mechanisms (evaluate each):

| Mechanism | Hypothesis | Testable Prediction | Test |
|-----------|-----------|---------------------|------|
| **Information content** | Wider ORB captures more price discovery → more reliable level | O15/O30 breakouts should have higher win rate than O5 | Compare WR by aperture, matched days |
| **Noise reduction** | Wider ORB filters out opening noise → fewer false breaks | O15/O30 should have fewer scratches and stop-outs | Compare scratch rate and time-to-stop |
| **Volatility absorption** | Wider ORB = bigger range = higher absolute risk → filters out low-vol days naturally | O15/O30 should have larger avg ORB size → acts like implicit G4+ filter | Compare orb_size_pts distribution by aperture |
| **Time compression** | Less time remaining after wider ORB → time-stop more likely → worse outcomes | O30 should have more time-stops than O15, O15 more than O5 | Compare exit_type distribution |
| **Opportunity cost** | Waiting 30 min to define ORB = missing the real move | O30 breakouts that DO work should show less available range post-break | Compare MFE (max favorable excursion) by aperture |

```python
"""
For each mechanism, write the query, run it, and record:
- Does the prediction hold? (Yes/No/Partial)
- Effect size (how much, not just statistically significant)
- Is the mechanism STRUCTURAL (would hold in any market) or EMPIRICAL (only this dataset)?

A mechanism that is purely empirical → treat the edge as provisional.
A mechanism that is structural → higher confidence in forward persistence.
"""
```

**PASS CRITERION:** At least ONE mechanism must be structural and confirmed by data. If all mechanisms are empirical only → O15/O30 edge is SUSPECT and requires extra scrutiny in later gates.

### Gate 2: Unfiltered Baseline — All Instruments × All Sessions × All Apertures

The foundation. No filters. Just raw edge by aperture.

```sql
-- Unfiltered baseline: ExpR by instrument × session × aperture × entry_model
-- Discovery window ONLY (through 2025-12-31)
SELECT
    o.symbol,
    o.orb_label,
    o.orb_minutes,
    o.entry_model,
    o.direction,
    COUNT(*) AS n_trades,
    ROUND(AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END), 4) AS win_rate,
    ROUND(AVG(pnl_r), 4) AS expr,
    ROUND(STDDEV(pnl_r), 4) AS std_r,
    ROUND(AVG(pnl_r) / NULLIF(STDDEV(pnl_r), 0), 4) AS sharpe_per_trade
FROM orb_outcomes o
WHERE o.trading_day < '2026-01-01'
  AND o.symbol IN ('MGC', 'MNQ', 'MES')
  AND o.rr_target = 1.0          -- baseline RR
  AND o.confirm_bars = 1         -- baseline CB
  AND o.pnl_r IS NOT NULL        -- exclude scratches
GROUP BY 1, 2, 3, 4, 5
HAVING COUNT(*) >= 30            -- minimum viable sample
ORDER BY 1, 2, 3, 4, 5;
```

**Build the full heatmap:** instrument × session × aperture. Color-code:
- GREEN: ExpR > +0.05R AND N ≥ 100
- YELLOW: ExpR > 0 AND N ≥ 30
- RED: ExpR ≤ 0 OR N < 30

**Key questions this answers:**
- Which sessions show O15 > O5? (Already know: TOKYO_OPEN. Any others?)
- Which sessions show O30 > O5? (Expect: few or none)
- Is the O15 premium instrument-specific (MNQ only) or cross-instrument?
- Is O30 dead across the board or alive in specific niches?

### Gate 3: Matched-Day Comparison — Aperture Effect Isolation

**CRITICAL:** Unmatched comparison confounds aperture with calendar. A day that has an O5 break may not have an O15 break (wider ORB = fewer breaks). To isolate the aperture effect, compare ONLY on days where ALL three apertures have an outcome.

```sql
-- Find days with outcomes at ALL three apertures for same session
WITH triple_days AS (
    SELECT symbol, trading_day, orb_label, entry_model, direction
    FROM orb_outcomes
    WHERE trading_day < '2026-01-01'
      AND pnl_r IS NOT NULL
      AND rr_target = 1.0 AND confirm_bars = 1
    GROUP BY 1, 2, 3, 4, 5
    HAVING COUNT(DISTINCT orb_minutes) = 3  -- all three apertures fired
)
SELECT
    o.symbol, o.orb_label, o.orb_minutes,
    COUNT(*) AS n_matched,
    ROUND(AVG(o.pnl_r), 4) AS matched_expr,
    ROUND(AVG(CASE WHEN o.outcome = 'win' THEN 1.0 ELSE 0.0 END), 4) AS matched_wr
FROM orb_outcomes o
JOIN triple_days t ON o.symbol = t.symbol
    AND o.trading_day = t.trading_day
    AND o.orb_label = t.orb_label
    AND o.entry_model = t.entry_model
    AND o.direction = t.direction
WHERE o.rr_target = 1.0 AND o.confirm_bars = 1
  AND o.pnl_r IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
```

**Then compute the paired difference:**
```sql
-- Paired test: O15 - O5 per matched day
WITH pairs AS (
    SELECT
        o5.symbol, o5.trading_day, o5.orb_label,
        o5.pnl_r AS pnl_r_o5,
        o15.pnl_r AS pnl_r_o15,
        o15.pnl_r - o5.pnl_r AS delta_r
    FROM orb_outcomes o5
    JOIN orb_outcomes o15 ON o5.symbol = o15.symbol
        AND o5.trading_day = o15.trading_day
        AND o5.orb_label = o15.orb_label
        AND o5.entry_model = o15.entry_model
        AND o5.direction = o15.direction
        AND o5.rr_target = o15.rr_target
        AND o5.confirm_bars = o15.confirm_bars
    WHERE o5.orb_minutes = 5 AND o15.orb_minutes = 15
      AND o5.pnl_r IS NOT NULL AND o15.pnl_r IS NOT NULL
      AND o5.trading_day < '2026-01-01'
      AND o5.rr_target = 1.0 AND o5.confirm_bars = 1
)
SELECT
    symbol, orb_label,
    COUNT(*) AS n_pairs,
    ROUND(AVG(delta_r), 4) AS avg_delta,
    ROUND(STDDEV(delta_r), 4) AS std_delta,
    -- Paired t-test: t = avg_delta / (std_delta / sqrt(n))
    ROUND(AVG(delta_r) / NULLIF(STDDEV(delta_r) / SQRT(COUNT(*)), 0), 3) AS t_stat,
    -- Direction: positive = O15 better, negative = O5 better
    CASE WHEN AVG(delta_r) > 0 THEN 'O15_BETTER' ELSE 'O5_BETTER' END AS winner
FROM pairs
GROUP BY 1, 2
HAVING COUNT(*) >= 30
ORDER BY 1, 2;
```

**PASS CRITERION:** The paired test is the honest comparison. If O15 is not significantly better (p < 0.05 two-tailed) on matched days → the unfiltered "O15 premium" is a SELECTION ARTIFACT (different days, not better performance).

### Gate 4: Filter Interaction — Does Filtering Change the Aperture Story?

The O5 portfolio uses specific filters (G5, G8, ATR70_VOL, etc.). Do these filters help O15/O30 the same way?

```python
"""
For each active filter in ALL_FILTERS:
1. Apply filter to O5, O15, O30 separately
2. Compare filtered ExpR across apertures
3. Check: does the filter HELP O15/O30 more than O5? Same? Less?

Key hypothesis: Wider ORB may act as a NATURAL size filter (bigger ORB = bigger range).
If so, explicit size filters (G4, G5, G8) may be REDUNDANT for O15/O30.
Test this explicitly:
- O5 + G5 vs O15 unfiltered → are they equivalent?
- O15 + G5 vs O15 unfiltered → does G5 still add value on top of wider aperture?
"""
```

```sql
-- Filter interaction: does G5 help O15 as much as it helps O5?
-- Step 1: O5 unfiltered vs O5+G5
-- Step 2: O15 unfiltered vs O15+G5
-- Step 3: Compare the deltas

-- (Requires joining daily_features for filter columns)
-- Write the specific queries per filter when executing
```

**Key output:** Filter × Aperture interaction matrix. For each filter:
- Does it improve O15? By how much?
- Does it improve O30? By how much?
- Is the improvement larger or smaller than for O5?
- Is the filter REDUNDANT for wider apertures?

### Gate 5: RR Target × Aperture Grid

O5 portfolio mostly uses RR1.0. Is the optimal RR different for wider apertures?

```sql
-- Full RR × Aperture grid (unfiltered, discovery window only)
SELECT
    o.symbol, o.orb_label, o.orb_minutes, o.rr_target,
    COUNT(*) AS n,
    ROUND(AVG(o.pnl_r), 4) AS expr,
    ROUND(AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END), 4) AS wr
FROM orb_outcomes o
WHERE o.trading_day < '2026-01-01'
  AND o.symbol IN ('MGC', 'MNQ', 'MES')
  AND o.entry_model = 'E2'
  AND o.confirm_bars = 1
  AND o.pnl_r IS NOT NULL
GROUP BY 1, 2, 3, 4
HAVING COUNT(*) >= 30
ORDER BY 1, 2, 3, 4;
```

**Hypothesis:** Wider ORB = bigger stop distance = RR2.0+ may be more reachable (wider range gives more room to run). If true → O15/O30 optimal RR may be higher than O5.

**Test:** Is ExpR maximized at a different RR for O15/O30 vs O5? Is the difference significant or flat (Sharpe surface analysis — if CV < 10% across RR values, it's a flat surface and RR doesn't matter)?

### Gate 6: Walk-Forward Validation

**THIS IS THE GATE THAT HAS NEVER BEEN RUN FOR O15/O30.**

```python
"""
For every O15/O30 strategy that passed Gates 2-5 (positive ExpR, BH significant):

1. Anchored expanding walk-forward:
   - Train: start → 2021-12-31, test: 2022
   - Train: start → 2022-12-31, test: 2023
   - Train: start → 2023-12-31, test: 2024
   - Train: start → 2024-12-31, test: 2025
   (MGC uses WF_START_OVERRIDE = 2022-01-01 — skip pre-2022 low-ATR regime)

2. Compute per-window OOS metrics:
   - OOS ExpR, WR, Sharpe for each test year

3. Compute WFE = OOS_avg_Sharpe / IS_Sharpe
   - Must be > 0.50 (hard gate, no waivers)

4. Year-over-year stability:
   - ALL test years must be positive (or regime waiver for seasonal strategies)
   - If 1 year is negative → FAIL (hard gate per TRADING_RULES.md)

5. Report WFE distribution for O15 vs O5:
   - Are O15 strategies MORE or LESS stable than O5?
   - Are O30 strategies MORE or LESS stable than O5?
"""
```

**PASS CRITERION:** WFE > 0.50 AND all OOS years positive AND BH FDR significant at correct K.

### Gate 7: Portfolio Correlation — Additive or Redundant?

**Purpose:** Even if O15/O30 strategies are independently valid, they're worthless if they correlate 0.95 with existing O5 portfolio.

```python
"""
1. Take the current LIVE_PORTFOLIO (O5 strategies)
2. Compute daily portfolio P&L for the O5 book
3. For each candidate O15/O30 strategy:
   a. Compute its daily P&L time series
   b. Correlate with O5 portfolio P&L
   c. Compute marginal Sharpe contribution (portfolio Sharpe WITH vs WITHOUT)

Key metric: Does adding this O15/O30 strategy IMPROVE the portfolio Sharpe?
If correlation > 0.7 with existing book → REDUNDANT (skip)
If correlation < 0.3 and ExpR > 0 → ADDITIVE (promote)
"""
```

```sql
-- Daily P&L correlation scaffold
-- Step 1: Build O5 portfolio daily P&L
WITH o5_daily AS (
    SELECT trading_day, SUM(pnl_r) AS portfolio_pnl_r
    FROM orb_outcomes
    WHERE orb_minutes = 5
      AND (symbol, orb_label, entry_model, rr_target, confirm_bars, direction)
          IN (/* current LIVE_PORTFOLIO params */)
      AND trading_day < '2026-01-01'
    GROUP BY trading_day
),
-- Step 2: Build candidate O15 daily P&L
o15_daily AS (
    SELECT trading_day, pnl_r AS candidate_pnl_r
    FROM orb_outcomes
    WHERE /* candidate O15 strategy params */
      AND trading_day < '2026-01-01'
)
-- Step 3: Correlate on matched days
SELECT CORR(o5.portfolio_pnl_r, o15.candidate_pnl_r) AS correlation
FROM o5_daily o5
JOIN o15_daily o15 ON o5.trading_day = o15.trading_day;
```

### Gate 8: Sensitivity Analysis — ±20% Perturbation

**MANDATORY per RESEARCH_RULES.md.** Every parameterized finding must survive perturbation.

```python
"""
For each promoted O15/O30 strategy:

1. Aperture sensitivity: Test at 13m and 17m (for O15), 25m and 35m (for O30)
   - If edge vanishes at ±2 minutes → PARAMETER FRAGILE → do not promote
   - Implementation: requires raw bar replay (can't query orb_outcomes at non-standard apertures)

2. RR sensitivity: ±0.5 RR around optimal
   - If ExpR flips sign within ±0.5 RR → FRAGILE

3. Filter threshold sensitivity: ±20% on filter thresholds
   - If ExpR flips sign → FRAGILE

4. Confirm bars sensitivity: CB ±1
   - If ExpR flips sign → FRAGILE

Build a 3×3 sensitivity grid for each candidate (aperture × RR × filter):
- All 9 cells must be positive ExpR
- Central cell must not be >2× any neighbor (peak, not plateau)
"""
```

**PASS CRITERION:** 9/9 cells positive. Central cell within 2× of neighbors. If not → NOT ROBUST.

### Gate 9: 2026 Forward Test — MONITORING ONLY

**This gate does NOT make promotion decisions.** It only checks for catastrophic failure.

```sql
-- 2026 forward performance (monitoring, not optimization)
SELECT
    o.symbol, o.orb_label, o.orb_minutes, o.entry_model,
    o.rr_target, o.confirm_bars, o.direction,
    COUNT(*) AS n_2026,
    ROUND(AVG(o.pnl_r), 4) AS expr_2026,
    ROUND(SUM(o.pnl_r), 2) AS total_r_2026
FROM orb_outcomes o
WHERE o.trading_day >= '2026-01-01'
  AND o.pnl_r IS NOT NULL
  -- Only strategies that passed Gates 1-8
GROUP BY 1, 2, 3, 4, 5, 6, 7
ORDER BY 1, 2, 3;
```

**RED FLAGS (halt and investigate, do not promote):**
- Total R < -5R in 2026 (catastrophic drawdown)
- Win rate < 35% (structural break from IS)
- 3+ consecutive losing months (regime shift)

**GREEN FLAGS (proceed to paper trading):**
- 2026 performance within 1σ of IS expectation
- No structural break in monthly WR or ExpR

---

## §3. DELIVERABLES — WHAT THIS RESEARCH PRODUCES

At the end of all gates, produce:

### 3.1 Aperture Verdict Table
```
| Aperture | Instrument | Status     | Evidence Summary                        |
|----------|-----------|------------|-----------------------------------------|
| O15      | MNQ       | PROMOTE/DEAD/MONITOR | Mechanism X, WFE Y, corr Z with O5 |
| O15      | MGC       | ...        | ...                                     |
| O15      | MES       | ...        | ...                                     |
| O30      | MNQ       | ...        | ...                                     |
| O30      | MGC       | ...        | ...                                     |
| O30      | MES       | ...        | ...                                     |
```

### 3.2 Promotion Candidates (if any)
For each strategy promoted past Gate 8:
- Full parameter set (instrument, session, aperture, entry model, RR, CB, filter, direction)
- N trades, ExpR, WR, Sharpe, max DD
- WFE
- Correlation with existing O5 portfolio
- Sensitivity grid (9 cells)
- Mechanism (structural or empirical)

### 3.3 NO-GO Registry Updates
Any aperture × session × instrument combo confirmed dead → add to STRATEGY_BLUEPRINT.md §5 NO-GO registry with:
- Date tested
- N tested
- Reason for NO-GO
- "Do not re-test until: [condition]"

### 3.4 TRADING_RULES.md Updates
Any confirmed findings (positive or negative) → update TRADING_RULES.md aperture section with:
- `@research-source` annotation
- `@revalidated-for` current entry models
- Data state timestamp

---

## §4. SESSION PLAN — EXECUTION ORDER

### Session 1: Foundation (Gates 0-1)
- Data state verification
- Mechanism test (BEFORE looking at results)
- Deliverable: Mechanism hypothesis matrix, data freshness confirmed

### Session 2: Baseline Discovery (Gates 2-3)
- Unfiltered baseline heatmap (all instrument × session × aperture)
- Matched-day paired comparison (the honest test)
- Deliverable: Baseline heatmap, paired t-test results, preliminary verdict on O15 vs O5

### Session 3: Filter & RR Interaction (Gates 4-5)
- Filter × aperture interaction matrix
- RR × aperture grid
- Deliverable: Optimal filter/RR for O15/O30 (if edge exists), redundancy analysis

### Session 4: Walk-Forward (Gate 6) — THE BIG ONE
- Full walk-forward on all surviving O15/O30 candidates
- WFE computation
- Year-over-year stability
- Deliverable: WFE distribution, kill list for strategies that fail OOS

### Session 5: Portfolio & Sensitivity (Gates 7-8)
- Correlation analysis with O5 book
- Marginal Sharpe contribution
- ±20% sensitivity on all parameters
- Deliverable: Final candidate list with full evidence

### Session 6: Forward Test & Verdicts (Gate 9 + §3)
- 2026 monitoring check (flags only, no optimization)
- Build final verdict table
- Update NO-GO registry and TRADING_RULES.md
- Deliverable: Aperture Verdict Table, promotion candidates, NO-GO updates

---

## §5. ANTI-PATTERNS — THINGS THAT WILL RUIN THIS RESEARCH

```
❌ Looking at 2026 results before completing Gates 1-8 — contaminates holdout
❌ Declaring O15 "works" because TOKYO_OPEN works — one session is not a finding
❌ Comparing O5 vs O15 on different days — unmatched comparison is confounded
❌ Promoting O30 strategies with N < 100 — REGIME class only, never standalone
❌ Using validated_setups as truth — derived layer, possibly stale
❌ Skipping mechanism test — "the numbers show it" is not a mechanism
❌ Testing sensitivity at ±1% instead of ±20% — too narrow to find fragility
❌ Promoting a strategy with WFE = 0.49 because "it's close" — hard gate, no waivers
❌ Adding O15/O30 to portfolio without checking correlation — redundant strategies increase DD without increasing Sharpe
❌ Declaring O30 dead after testing 1 session — test ALL sessions before NO-GO (Blueprint §3 Gate 2: ≥3 values per dimension)
❌ Importing outcome_builder or strategy_discovery to "verify" — independent replication means INDEPENDENT code
❌ Inlining research stats in code comments — they go stale, use @research-source annotations
```

---

## §6. EXPECTED OUTCOMES — HONEST PRIORS

Based on what we already know, here's what I expect this research to find. Record these BEFORE running so we can compare against actual results (pre-registration).

| Prediction | Confidence | Basis |
|-----------|-----------|-------|
| O15 TOKYO_OPEN will pass all gates (already validated) | 90% | Existing validation + mechanism (Tokyo open auction provides genuine price discovery) |
| O15 will show edge in ≤3 other sessions beyond TOKYO_OPEN | 50% | Weaker baseline, no confirmed mechanism for other sessions |
| O30 will be dead across all sessions | 70% | Weakest baseline, time compression mechanism works against it |
| O15/O30 optimal RR will be higher than O5 (RR1.5-2.0 vs RR1.0) | 60% | Wider ORB = more room to run, but also wider stop |
| O15/O30 correlation with O5 will be > 0.5 on matched days | 75% | Same instrument, same session, same direction — hard to decorrelate |
| G5/G8 filters will be less impactful on O15/O30 than O5 | 65% | Wider ORB acts as implicit size filter |

**If actual results differ significantly from priors → investigate WHY before promoting.**

---

## §7. SUCCESS CRITERIA

This research is complete when:
1. All 6 instrument × aperture combos have a verdict (PROMOTE/DEAD/MONITOR)
2. Every verdict has full evidence chain (mechanism, BH FDR, WFE, sensitivity, correlation)
3. NO-GO registry updated for dead combos
4. TRADING_RULES.md updated with confirmed findings
5. Any promoted strategies have complete parameter sets ready for paper trading
6. Pre-registered predictions in §6 compared against actuals (calibration check)
7. All results archived in `docs/audits/` or `research/output/`

---

## §8. FINAL VERDICT — 2026-03-29

### O15/O30 Aperture Edge: **NO-GO (ARITHMETIC_ONLY)**

**Date:** 2026-03-29
**Method:** Gate 0-3 of aperture research protocol. Gate 3b (friction artifact test) was decisive.
**Data:** 24,056 matched-day paired trades (MNQ E2 CB1 RR1.0, pre-2026 holdout).

**Evidence:**
- Gross R (before friction): O5 = +0.158, O15 = +0.127. **O5 wins by 0.031R/trade.**
- Net R (after friction): O5 = -0.019, O15 = +0.007. O15 "wins" only because friction hurts O5 more.
- Win rate DECREASES with wider aperture: MNQ 58.9% → 56.4% → 55.4%
- Scratch rate DOUBLES: 7.6% → 13.9% → 15.8%
- Time to resolution 4x longer: 27min → 63min → 102min
- Direction concordance 87% on matched days

**Mechanism:** Wider ORB = larger risk denominator = friction is smaller fraction of risk.
This is identical to the cost/risk% friction gate (ARITHMETIC_ONLY, 2026-03-24).
The wider aperture does NOT predict better breakouts. It reduces friction impact per R-multiple by increasing the dollar risk.

**Correct framing:** Natural minimum-viable-trade-size screen. Same family as G-filters.
**Do NOT call:** "aperture edge", "wider ORB advantage", "information content premium"
**Do NOT re-test until:** A new mechanism hypothesis emerges that was NOT tested here.

### Disposition of 208 O15/O30 Validated Strategies
These strategies are not adding alpha. They express the same edge as O5 strategies with size filters, but with:
- Fewer trades (less opportunity)
- Higher capital at risk per trade
- Longer time to resolution
- Worse gross R per trade

**Recommendation:** Do not promote O15/O30 strategies to LIVE_PORTFOLIO. Existing O5 + G5/G8 filters achieve superior results.

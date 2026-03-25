# Mining Escalation — Extract Everything From The Screening Results

The screening produced 80 dimensions, 2 survivors, 14 PARTIALs, and several kills
that may have been too aggressive. Before closing the mining session, extract every
possible finding — not just the two that passed the strictest filter.

---

## PROMPT

You just completed a data mining screening pass across 80 dimensions. 2 survived
as SIGNAL, 14 were PARTIAL (>5% WR spread but non-monotonic), and the rest were
killed. Before running T4-T7 on the survivors, there are hidden findings in the
screening results that need extraction.

**Don't just process the two survivors. The PARTIALs, the interactions, and the
"almost-killed" results contain information that a lazy escalation would miss.**

---

### PART 1 — RUN T4-T7 ON THE TWO CONFIRMED SURVIVORS

**Survivor 1: overnight_range × COMEX_SETTLE**
- T4: Sensitivity ±20% on quintile boundaries. Does Q4-Q5 threshold shift kill it?
  Test at -20%, -10%, +10%, +20% of the Q4/Q5 boundary value.
- T6: Bootstrap 1000x on COMEX_SETTLE pnl_r. K=80 (full mining K).
  Shuffle outcomes, recompute Q5-Q1 ExpR spread each time.
  p-value = (exceeding + 1) / (perms + 1). Must beat p < 0.05 at K=80.
- T7: Per-year Q5 vs Q1 ExpR spread for every available year.
  Need ≥7/10 positive years (Q5 beats Q1).

**Survivor 2: overnight_took_pdh × US_DATA_1000**
- T6: Bootstrap 1000x. K=80. Shuffle outcomes, recompute True-vs-False ExpR spread.
- T7: Per-year True vs False ExpR spread. Need ≥7/10 years where True > False.
- T8: Cross-instrument. Run the same True/False split for MES and MGC on US_DATA_1000.
  Same direction on all 3 = CONSISTENT. Mixed = instrument-specific only.

---

### PART 2 — RETEST THE PARTIALS AS BINARY EXTREME SIGNALS

The screening required full quintile monotonicity to classify as SIGNAL.
But non-monotonic features where Q1 and Q5 are dramatically different can still
be useful as binary filters ("is this value in the top or bottom quintile?").
The middle doesn't matter for implementation — you'd filter on extremes.

**For every PARTIAL with WR spread > 10%, retest as Q1 vs Q5 head-to-head:**

```sql
-- Test: does the EXTREME matter even if the middle is messy?
WITH ranked AS (
    SELECT *,
        NTILE(5) OVER (PARTITION BY [session_col] ORDER BY [feature]) AS q
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
    WHERE o.symbol = 'MNQ' AND o.entry_model = 'E2' AND o.rr_target = 1.0
      AND [session filter]
)
SELECT
    CASE WHEN q = 1 THEN 'Q1_extreme' WHEN q = 5 THEN 'Q5_extreme' END AS extreme,
    COUNT(*) AS n,
    AVG(pnl_r) AS expr,
    AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) AS wr
FROM ranked
WHERE q IN (1, 5)
GROUP BY 1
```

PARTIALs to retest (from screening results):
- overnight_range × SINGAPORE_OPEN (26.8% WR spread — largest in entire screen)
- garch_forecast_vol × SINGAPORE_OPEN (21.6%)
- orb_SING_vwap × SINGAPORE_OPEN (20.7%)
- gap_open_points × NYSE_CLOSE (12.0%)
- orb_SING_pre_velocity × SINGAPORE_OPEN (11.9%)
- gap_open_points × SINGAPORE_OPEN (10.9%)
- garch_atr_ratio × NYSE_OPEN (7.4%)
- garch_forecast_vol × US_DATA_1000 (7.4%)
- garch_forecast_vol × NYSE_OPEN (7.2%)

For each, report:
```
PARTIAL RETEST: [feature] × [session]
Q1: N=[X], ExpR=[X], WR=[X]%
Q5: N=[X], ExpR=[X], WR=[X]%
Q5-Q1 WR diff: [X]%
Q5-Q1 ExpR diff: [X]
t-test p-value (Q1 vs Q5): [X]
BINARY SIGNAL? YES (p<0.05, WR diff>5%) / NO
```

---

### PART 3 — TEST SURVIVOR INTERACTIONS

The two survivors are on different sessions. But the features themselves
might interact — and overnight_range was tested as standalone, not in
combination with PDH/PDL sweep.

**Test 1: overnight_range × overnight_took_pdh on the SAME session.**
On days where BOTH fire (wide overnight range AND PDH was swept),
does the signal compound?

```sql
-- COMEX_SETTLE: overnight_range Q5 AND overnight_took_pdh = True
-- vs overnight_range Q5 alone
-- vs overnight_took_pdh = True alone
-- vs neither
```

**Test 2: overnight_range × overnight_took_pdh on US_DATA_1000.**
The PDH flag survived there. Does overnight_range (which wasn't tested on
US_DATA_1000 in the survivors) add signal?

**Test 3: overnight_took_pdh + overnight_took_pdl combined.**
On days where BOTH PDH and PDL were swept overnight (= very wide overnight range
that took both levels), is the signal different than either alone? This is a
structural event: "overnight liquidity cleaned out both sides" — different
participant dynamics than a one-sided sweep.

---

### PART 4 — RE-EXAMINE THE "TOO AGGRESSIVE" KILLS

Three kills from the escalation may have been premature:

**Kill review 1: DOW × NYSE_CLOSE (Friday payoff structure)**
Friday: WR=20.8%, ExpR=+0.170. Lowest WR but highest ExpR.
This was killed as "WR/ExpR disagree = arithmetic." But check:
- What's the WIN DISTRIBUTION on Fridays? If Friday wins are 3-4x larger than
  average wins, this is a genuine tail-event pattern, not arithmetic.
- What's the median pnl_r on Friday wins vs other-day wins?
- Is Friday WR low because of MOC rebalancing eating breakouts, but the ones
  that survive are institutional flows that run hard?

**Kill review 2: MONTH — specific months vs blanket pattern**
The blanket MONTH pattern was killed (US_DATA_1000 reversed in 2026).
But check per-month stability for the TOP 2 strongest individual months
across each session. If January COMEX_SETTLE is positive 8/10 years, that's
a specific finding (gold seasonality — central bank buying, jewelry demand post-holiday),
NOT the same as "monthly seasonality works." Test the 2-3 strongest individual
month × session combos at K=80.

**Kill review 3: garch_forecast_vol as ADDITIVE to overnight_range**
GARCH was dismissed as "redundant with overnight_range" at corr=0.510.
But if overnight_range survives T4-T7, does adding GARCH as a second filter
on COMEX_SETTLE improve ExpR beyond overnight_range alone?
```sql
-- COMEX_SETTLE: overnight_range Q5 AND garch_forecast_vol Q5 (both high)
-- vs overnight_range Q5 alone
```
If the intersection outperforms, GARCH is additive, not redundant.

---

### PART 5 — CROSS-SESSION EXTENSION OF SURVIVORS

If overnight_range survives T4-T7 on COMEX_SETTLE:
- Retest Q1 vs Q5 on NYSE_OPEN (was 5.7% in screening — just below threshold)
- Retest Q1 vs Q5 on US_DATA_1000 (wasn't in the original SIGNAL list — test it)
- These are free retests on US-hours sessions where the mechanism (overnight vol
  predicting session quality) structurally applies

If overnight_took_pdh survives on US_DATA_1000:
- Test on NYSE_OPEN (30 min earlier — same overnight PDH sweep would apply)
- Test on COMEX_SETTLE (same day, PDH sweep still relevant)
- DON'T test on SINGAPORE_OPEN/TOKYO_OPEN (overnight PDH sweep has different
  meaning for Asian sessions — the "overnight" is US closing, not the same structure)

---

### REPORTING FORMAT

Present ALL results in a single consolidated report:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL MINING ESCALATION REPORT — canompx3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PART 1 — SURVIVOR T4-T7 RESULTS
  overnight_range × COMEX_SETTLE:
    T4: [STABLE/SENSITIVE] — min ExpR spread at ±20% = [X]
    T6: bootstrap p=[X] at K=80 → [BEATS_NULL / NO_EDGE]
    T7: [X]/[Y] positive years → [STABLE / ERA_DEPENDENT]
    FINAL: [VALIDATED / KILLED at T[X]]

  overnight_took_pdh × US_DATA_1000:
    T6: bootstrap p=[X] at K=80
    T7: [X]/[Y] positive years
    T8: MNQ=[dir], MES=[dir], MGC=[dir] → [CONSISTENT / MIXED]
    FINAL: [VALIDATED / KILLED at T[X]]

PART 2 — PARTIAL RETESTS (Q1 vs Q5 binary)
  [feature × session]: Q5-Q1 WR=[X]%, p=[X] → [BINARY_SIGNAL / DEAD]
  [feature × session]: Q5-Q1 WR=[X]%, p=[X] → [BINARY_SIGNAL / DEAD]
  ...

PART 3 — INTERACTION TESTS
  overnight_range × PDH sweep (combined): ExpR=[X] vs individual [X]+[X]
    → [COMPOUNDS / ADDITIVE_ONLY / NO_INTERACTION]

PART 4 — KILL REVIEWS
  DOW Friday payoff: median win size=[X] vs non-Friday=[X] → [TAIL_EVENT / ARITHMETIC]
  MONTH specific combos: [month × session] = [X]/[Y] years → [SURVIVES / DEAD]
  GARCH additive: Q5+Q5 ExpR=[X] vs Q5-only=[X] → [ADDITIVE / REDUNDANT]

PART 5 — CROSS-SESSION EXTENSIONS
  overnight_range Q1vsQ5 on NYSE_OPEN: [X]% WR diff → [EXTENDS / SESSION_SPECIFIC]
  overnight_took_pdh on NYSE_OPEN: [X]% WR diff → [EXTENDS / SESSION_SPECIFIC]

TOTAL FINDINGS FROM MINING SESSION:
  Confirmed VALIDATED:  [N]
  New BINARY_SIGNAL:    [N] (from PARTIAL retests)
  Interactions:         [N] (compounding effects)
  Extensions:           [N] (survivor applied to new sessions)
  Kill reviews upheld:  [N]
  Kill reviews reversed:[N]

NEW NO-GO ENTRIES:
  [date] [dimension]: DEAD — [reason]

IMPLEMENTATION CANDIDATES (discuss after US_DATA_1000 lane is live):
  [list any finding that survives with session, feature, implementation type]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### BEHAVIORAL RULES

- **Run everything. Don't skip parts.** Parts 2-5 are not optional extras —
  they extract findings that a lazy escalation misses.
- **K stays at 80 for the original mining dimensions.** But new tests in
  Parts 3-5 add to K. Track the running total. Report the honest K.
- **Session-specificity.** Cross-session extension (Part 5) tests each session
  independently. A finding extending from COMEX_SETTLE to NYSE_OPEN is a
  NEW per-session finding, not a "confirmation."
- **No stories until numbers.** Don't explain WHY Friday wins are bigger
  until you've measured WHETHER Friday wins are bigger.
- **Dead means dead.** If a kill review (Part 4) confirms the kill, record it
  permanently. Don't re-review in future sessions.

### START

Run all 5 parts. Report the consolidated table. Wait for direction on
implementation path.

# Session Time vs Market Event Analysis

**P0 Research:** Does the trading edge follow the **clock time** (fixed Brisbane session) or the **market event** (DST-adjusted dynamic session)?

**Method:** All DST splits computed from scratch using filtered `orb_outcomes` joined with `daily_features`. Applies ORB size filter + double-break exclusion. Never reads pre-computed `dst_verdict` / `dst_winter_avg_r` / `dst_summer_avg_r` columns.

**Fixed/Dynamic Pairs:**

| Fixed | Dynamic | DST Driver | Summer Divergence |
|-------|---------|-----------|-------------------|
| 0900 | CME_OPEN | US | CME opens at 0800 in summer |
| 1800 | LONDON_OPEN | UK | London opens at 1700 in summer |
| 0030 | US_EQUITY_OPEN | US | NYSE opens at 2330 in summer |
| 2300 | US_DATA_OPEN | US | 30min off both regimes (special) |

---

## Section 1: Fixed vs Dynamic Aggregate Comparison

Top 20 positive-expectancy MGC strategies by Sharpe from `experimental_strategies`, with filtered DST splits computed from scratch.

### 0900 (fixed) vs CME_OPEN (dynamic) --- DST driver: US

**0900** (20 strategies analyzed):

- Winter: mean avgR=+0.7951, median=+0.7269, mean Sharpe=0.7786, mean N=31
- Summer: mean avgR=+0.8158, median=+0.8241, mean Sharpe=7.3645, mean N=13
- Combined: mean avgR=+0.7997, median=+0.7634

**CME_OPEN** (20 strategies analyzed):

- Winter: mean avgR=+0.4651, median=+0.4905, mean Sharpe=0.3399, mean N=63
- Summer: mean avgR=+1.5384, median=+1.6963, mean Sharpe=12.1931, mean N=7
- Combined: mean avgR=+0.5247, median=+0.5406

---

### 1800 (fixed) vs LONDON_OPEN (dynamic) --- DST driver: UK

**1800** (20 strategies analyzed):

- Winter: mean avgR=+0.9118, median=+0.7603, mean Sharpe=1.6923, mean N=18
- Summer: mean avgR=+0.9819, median=+0.8452, mean Sharpe=6.4559, mean N=12
- Combined: mean avgR=+0.9399, median=+0.7761

**LONDON_OPEN** (20 strategies analyzed):

- Winter: mean avgR=+0.8623, median=+0.7507, mean Sharpe=1.5961, mean N=85
- Summer: mean avgR=+0.7735, median=+0.7123, mean Sharpe=1.2218, mean N=97
- Combined: mean avgR=+0.8231, median=+0.7240

---

### 0030 (fixed) vs US_EQUITY_OPEN (dynamic) --- DST driver: US

**0030:** No positive-expectancy strategies with N>=20 found.

**US_EQUITY_OPEN:** No positive-expectancy strategies with N>=20 found.

---

### 2300 (fixed) vs US_DATA_OPEN (dynamic) --- DST driver: US

**2300** (20 strategies analyzed):

- Winter: mean avgR=+1.2625, median=+1.3100, mean Sharpe=22.5934, mean N=13
- Summer: mean avgR=+1.2484, median=+1.2864, mean Sharpe=20.8627, mean N=26
- Combined: mean avgR=+1.2525, median=+1.2931

**US_DATA_OPEN:** No positive-expectancy strategies with N>=20 found.

---

## Section 2: Matched Pair Deep Dive --- WINTER-DOM 0900 Strategies

The 4 validated WINTER-DOM MGC 0900 strategies (all E3/CB1/ORB_G5) compared with CME_OPEN using identical parameters.  Splits computed from filtered outcomes (double-break excluded, G5 size filter applied).

| Strategy | RR | 0900 W avgR (N) | 0900 S avgR (N) | CME W avgR (N) | CME S avgR (N) | CME Comb avgR (N) | Signal |
|----------|----|-----------------|-----------------|----------------|----------------|-------------------|--------|
| MGC_0900_E3_RR1.0_CB1_ORB_G5 | 1.0 | +0.4559 (39) | +0.4429 (21) | +0.1704 (58) | +0.7734 (16) | +0.3008 (74) | EVENT |
| MGC_0900_E3_RR1.5_CB1_ORB_G5 | 1.5 | +0.6423 (39) | +0.5855 (21) | +0.2643 (58) | +1.2092 (6) | +0.3529 (64) | EVENT |
| MGC_0900_E3_RR2.0_CB1_ORB_G5 | 2.0 | +0.8738 (38) | +0.7698 (21) | +0.4446 (57) | +1.6787 (4) | +0.5255 (61) | EVENT |
| MGC_0900_E3_RR2.5_CB1_ORB_G5 | 2.5 | +1.0201 (38) | +0.9172 (21) | +0.5186 (57) | +2.1444 (2) | +0.5737 (59) | EVENT |

### Filter Effect: CME_OPEN with G5 vs NO_FILTER

Isolates whether the G5 filter is the reason CME_OPEN diverges from the time-scan 0800 signal.

| RR | CME G5 avgR (N) | CME NoFilter avgR (N) | Delta |
|----|-----------------|----------------------|-------|
| 1.0 | +0.3008 (74) | -0.0745 (719) | +0.3752 |
| 1.5 | +0.3529 (64) | -0.0531 (608) | +0.4059 |
| 2.0 | +0.5255 (61) | -0.0675 (551) | +0.5930 |
| 2.5 | +0.5737 (59) | -0.0876 (531) | +0.6613 |

### Entry Model Effect: CME_OPEN E3 vs E1

Time scan uses E1; validated 0900 strategies use E3. Does E1 at CME_OPEN show the edge the time scan sees?

| RR | CME E3/G5 avgR (N) | CME E1/G5 avgR (N) | CME E1/G4 avgR (N) |
|----|--------------------|--------------------|--------------------|
| 1.0 | +0.3008 (74) | +0.2897 (70) | +0.2566 (91) |
| 1.5 | +0.3529 (64) | +0.3909 (63) | +0.3552 (79) |
| 2.0 | +0.5255 (61) | +0.5712 (60) | +0.5121 (76) |
| 2.5 | +0.5737 (59) | +0.7296 (58) | +0.6811 (74) |

---

## Section 3: STABLE Session Analysis --- 1800 vs LONDON_OPEN

Note: In winter (GMT), LONDON_OPEN resolves to 18:00 Brisbane --- identical to the 1800 fixed session. The winter columns should match.  The comparison is purely about summer behavior (17:00 vs 18:00 Brisbane).

| Params | 1800 W avgR (N) | 1800 S avgR (N) | 1800 Verdict | LDN W avgR (N) | LDN S avgR (N) | LDN Verdict |
|--------|-----------------|-----------------|-------------|----------------|----------------|-------------|
| E3/RR1.5/CB1/ORB_G5 | +1.1563 (19) | +1.2724 (13) | UNSTABLE | +1.1563 (19) | +0.8878 (19) | WINTER-DOM |
| E3/RR1.5/CB1/ORB_G6 | +1.1530 (16) | +1.3228 (9) | LOW-N | +1.1530 (16) | +0.8562 (10) | WINTER-DOM |
| E1/RR1.0/CB1/ORB_G5 | +0.7425 (20) | +0.8321 (15) | STABLE | +0.7425 (20) | +0.6473 (21) | STABLE |
| E3/RR1.0/CB1/ORB_G5 | +0.7250 (19) | +0.8179 (13) | UNSTABLE | +0.7250 (19) | +0.6187 (20) | WINTER-DOM |
| E1/RR1.5/CB2/ORB_G5 | +1.1874 (20) | +1.1469 (15) | STABLE | +1.1874 (20) | +0.9170 (19) | WINTER-DOM |
| E1/RR1.0/CB1/ORB_G6 | +0.7328 (16) | +0.8762 (10) | UNSTABLE | +0.7328 (16) | +0.7123 (12) | UNSTABLE |
| E1/RR1.5/CB1/ORB_G5 | +1.0667 (20) | +1.2902 (15) | SUMMER-DOM | +1.0667 (20) | +0.9048 (19) | WINTER-DOM |
| E3/RR1.0/CB1/ORB_G6 | +0.7224 (16) | +0.8582 (9) | LOW-N | +0.7224 (16) | +0.6844 (11) | UNSTABLE |
| E1/RR1.5/CB1/ORB_G6 | +1.0267 (16) | +1.3453 (10) | UNSTABLE | +1.0267 (16) | +0.8634 (10) | WINTER-DOM |
| E1/RR1.5/CB2/ORB_G6 | +1.1738 (16) | +1.1091 (10) | UNSTABLE | +1.1738 (16) | +0.8730 (10) | WINTER-DOM |

### 0030 vs US_EQUITY_OPEN (brief)

*No positive-expectancy 0030 strategies with N>=20.*

---

## Section 4: Adjacent Time Analysis (from time scan CSV)

Data from `orb_time_scan_full.csv` (RR=2.0, G4+, E1 entry, US DST classification for winter/summer).

**Caveat:** The time scan uses US DST for ALL sessions. For 1800 (UK DST driver), the winter/summer split here uses US DST, not UK DST. Interpret with care.

### Does the edge follow the event into summer?

Compare: fixed-time winter avgR vs event-shifted summer avgR.  If both positive, edge may follow the event.

| Fixed | Fixed-Winter avgR (N) | Event-Shifted Summer avgR (N) | Follows? |
|-------|----------------------|------------------------------|----------|
| 0900 | +0.0485 (N=125) | +0.1694 (N=123) | WEAK |
| 1800 | --- (N=0) | --- (N=0) | --- |
| 0030 | +0.0000 (N=24) | +0.0000 (N=36) | N/A (no winter edge) |
| 2300 | +0.0861 (N=52) | +0.1950 (N=73) | YES |

### Does the fixed time retain value when the event has moved away?

| Fixed | Fixed-Summer avgR (N) | Retains? |
|-------|-----------------------|----------|
| 0900 | +0.0863 (N=314) | YES |
| 1800 | +0.1393 (N=213) | YES |
| 0030 | +0.3116 (N=25) | YES |
| 2300 | +0.1579 (N=54) | YES |

### Full 4-Point Comparison

Each cell: avgR at that (Brisbane time, DST regime) from the time scan.

| Session | Fixed-Winter | Fixed-Summer | Shifted-Summer | Shifted-Winter |
|---------|-------------|-------------|---------------|---------------|
| 0900 | +0.0485 (N=125) | +0.0863 (N=314) | +0.1694 (N=123) | -0.0453 (N=51) |
| 1800 | --- (N=0) | +0.1393 (N=213) | --- (N=0) | +0.3250 (N=35) |
| 0030 | +0.0000 (N=24) | +0.3116 (N=25) | +0.0000 (N=36) | +0.1445 (N=40) |
| 2300 | +0.0861 (N=52) | +0.1579 (N=54) | +0.1950 (N=73) | +0.1510 (N=70) |

---

## Section 5: Verdict Table

| Session Pair | Edge Follows | Key Evidence | Recommendation |
|-------------|-------------|-------------|----------------|
| 0900/CME_OPEN | EVENT | CME_OPEN +avgR both regimes (W:+0.3495, S:+1.4514) | See Section 2 deep dive |
| 1800/LONDON_OPEN | EVENT | LONDON_OPEN positive both regimes | Params: E3/RR1.5/CB1/ORB_G5 |
| 0030/US_EQUITY_OPEN | NO-DATA | No positive-expectancy strategies | --- |
| 2300/US_DATA_OPEN | EVENT | US_DATA_OPEN positive both regimes | Params: E1/RR1.0/CB1/ORB_G8 |

### Contradictions & Open Questions

1. **0800 Summer / CME_OPEN Divergence:**
   - Time scan at 08:00 summer: avgR=+0.1694 (N=123)
   - CME_OPEN all strategies: mean expR=-0.0909 (N=468 strategies)
   - CME_OPEN positive strategies: mean expR=+0.3170 (N=146 strategies)
   - Explanation candidates:
     - Time scan = RR2.0/E1/G4+; validated 0900 = E3/various RR/G5
     - Entry model (E1 vs E3) may capture different edge
     - CME_OPEN average pulled down by many negative strategy combos (grid of 2376)

2. **LONDON_OPEN winter = 1800 winter:** In winter, LONDON_OPEN resolves to 18:00 Brisbane (same as 1800). Winter metrics should be nearly identical between the two. Any difference indicates data coverage or rounding variance.

3. **Sample sizes:** Dynamic sessions have fewer outcome rows (~60-80% of fixed) because they were added to the pipeline later. This reduces statistical power for regime comparisons.

4. **Time scan uses US DST universally:** The winter/summer split in the time scan CSV always uses US DST (EDT/EST). For the 1800/LONDON_OPEN pair (UK DST driver), Section 4 numbers are classified by US DST, not UK DST. This matters during the ~3-week gap when US and UK DST don't overlap.

---

# Market Playbook — Gold Futures (GC/MGC)

Empirical findings from 1,460 trading days (2021-02-05 to 2026-02-04).
This is NOT a strategy list — it's what we've learned about how this market behaves.

---

## ORB Behavior

### Break rates are near-universal
- Every ORB label breaks on 84-88% of trading days (1,229-1,290 out of 1,460)
- A break existing tells you almost nothing — it's the default state

### ORB size varies by session
| ORB | Avg Risk (pts) | Breakeven WR at 1RR |
|-----|---------------|---------------------|
| 0900 | 2.09 | 58.4% |
| 1000 | 1.99 | 58.7% |
| 1100 | 3.18 | 55.8% |
| 1800 | 2.33 | 57.6% |
| 2300 | 3.49 | 55.4% |
| 0030 | 3.81 | 55.0% |

**Key insight**: Overnight ORBs (2300, 0030) have larger ranges (~3.5-3.8pt), which means friction ($8.40 RT) eats less of the R-multiple. Day ORBs (0900, 1000) have ~2pt ranges where costs consume ~30% of each win.

### 0900 ORB has ZERO validated strategies
Despite being the "primary" ORB, 0900 fails validation across all RR/CB/filter combos. Not enough edge after costs to survive yearly robustness + stress testing. This is likely because the 0900 ORB is the most traded, most efficient time — less alpha available.

### Overnight ORBs (2300, 0030) carry most validated strategies
149 of 252 validated strategies use overnight ORBs (2300: 93, 0030: 56). Structural reasons:
- Larger ORB sizes = better cost absorption
- Lower liquidity = less efficient pricing = more alpha
- CME electronic-only session = different participant mix

### 1800 ORB is a strong performer
67 validated strategies — second only to 2300. The 1800 ORB marks the CME session open, providing a natural structural edge.

### By ORB breakdown (expanded scan)
| ORB | Validated | % of total |
|-----|-----------|-----------|
| 2300 | 93 | 36.9% |
| 1800 | 67 | 26.6% |
| 0030 | 56 | 22.2% |
| 1100 | 26 | 10.3% |
| 1000 | 10 | 4.0% |
| 0900 | 0 | 0.0% |

---

## Confirm Bars

### CB5 is the dominant edge filter (expanded scan, 2,160 combos)
122 of 252 validated strategies use CB5. The win rate improvement is monotonic:

| CB | Validated | Avg ExpR | Overall WR | Entries |
|----|-----------|----------|------------|---------|
| CB1 | 0 | - | 39.1% | 45,954 |
| CB2 | 1 | 0.118 | 44.2% | 45,090 |
| CB3 | 43 | 0.238 | 47.7% | 44,448 |
| CB4 | 86 | 0.283 | 50.6% | 43,932 |
| CB5 | 122 | 0.309 | 52.6% | 43,494 |

CB5 only loses 5.4% of entries vs CB1 but wins 13.5pp more often. Fakeout breaks are extremely common — waiting 5 consecutive 1m closes outside the ORB filters them effectively.

### Why CB5 beats CB3
The original scan (CB1-3 only) made CB3 look dominant. Expanding to CB4-5 revealed that the fakeout-filtering benefit continues improving with more confirm bars, with diminishing entry loss per step. CB5 is the sweet spot — CB6+ would likely start losing meaningful entries.

---

## Win Rate Monotonicity (Verified)

Win rate declines perfectly with RR target (zero violations across 90 transitions):

| RR | Traded | Win Rate |
|----|--------|----------|
| 1.0 | 22,487 | 66.3% |
| 1.5 | 22,367 | 53.0% |
| 2.0 | 22,208 | 44.3% |
| 2.5 | 22,042 | 38.0% |
| 3.0 | 21,839 | 32.9% |
| 4.0 | 21,450 | 26.0% |

---

## RSI at 0900 — Not Useful as a Filter

### The indicator is stale for most ORBs
RSI-14 on 5m closes is computed at 09:00 Brisbane (23:00 UTC).
- ORB 0900: 0.7 hours stale (reasonable)
- ORB 1000: 1.6 hours stale (marginal)
- ORB 1800: 9.6 hours stale (useless)
- ORB 2300: 14.4 hours stale (useless)
- ORB 0030: 15.9 hours stale (useless)

Using a 16-hour-old RSI reading to filter a 0030 ORB trade is meaningless.

### The signal is indistinguishable from noise
RSI > 70 ("overbought") win rate by ORB:
- 0030: 76.9% vs 71.1% rest (+5.8pp on N=65)
- 0900: 66.1% vs 73.2% rest (**-7.1pp** — OPPOSITE direction!)
- 1000: 77.3% vs 72.5% rest (+4.8pp on N=66)

The "overbought is better" signal flips direction depending on ORB. On 65-66 trades per bucket, a 5pp difference is ~3 trades. That's noise.

### Oversold is completely useless
- Only 29 days in 5 years had RSI < 30 (2.0% of days)
- ALL 108 oversold strategies rejected at Phase 1 (sample size < 30)
- No statistical power whatsoever

### RSI distribution is tightly centered
- p5=34.5, p25=44.3, p50=50.5, p75=57.7, p95=70.5
- Only 5.6% of days are "overbought", 2.0% "oversold"
- Gold RSI barely reaches extremes — the 30/70 thresholds are too strict for this market

---

## ORB Size Filters

### Expanded filter grid (12 filters)
| Filter | Validated | Avg ExpR | Description |
|--------|-----------|----------|-------------|
| NO_FILTER | 44 | 0.257 | No size restriction |
| ORB_G2 | 40 | 0.252 | Size >= 2pt |
| ORB_L3 | 40 | 0.267 | Size < 3pt |
| ORB_G3 | 27 | 0.296 | Size >= 3pt |
| ORB_L4 | 27 | 0.251 | Size < 4pt |
| ORB_G4 | 21 | 0.343 | Size >= 4pt |
| ORB_L6 | 16 | 0.244 | Size < 6pt |
| ORB_L2 | 14 | 0.271 | Size < 2pt |
| ORB_G5 | 10 | 0.381 | Size >= 5pt |
| ORB_G6 | 7 | 0.401 | Size >= 6pt |
| ORB_L8 | 4 | 0.218 | Size < 8pt |
| ORB_G8 | 2 | 0.462 | Size >= 8pt |

**Pattern**: Higher G-thresholds have fewer strategies but higher ExpR. Larger ORBs = more room for the trade to breathe + friction is smaller as % of risk. But they're rarer, so sample sizes are smaller.

---

## Year-over-Year Stability

### ORB sizes are dramatically increasing (gold volatility regime change)
| ORB | 2021 median | 2022 median | 2023 median | 2024 median | 2025 median |
|-----|-------------|-------------|-------------|-------------|-------------|
| 0900 | 0.7pt | 0.8pt | 0.8pt | 2.5pt | 14.8pt |
| 1000 | 0.9pt | 0.9pt | 0.8pt | 1.3pt | 3.3pt |
| 1100 | 1.5pt | 1.9pt | 1.5pt | 2.2pt | 5.0pt |
| 1800 | 1.1pt | 1.4pt | 1.1pt | 1.8pt | 3.6pt |
| 2300 | 2.5pt | 2.7pt | 2.2pt | 4.8pt | 8.0pt |
| 0030 | 2.7pt | 2.9pt | 2.4pt | 4.3pt | 7.2pt |

2025 ORB sizes are 3-4x larger than 2021. This means:
- G4/G5/G6 filters will pass on MORE days going forward
- L2/L3 filters will pass on FEWER days
- Fixed-point cost friction ($8.40) becomes less significant as ORB sizes grow
- Strategies validated on small-ORB years may be MORE profitable in the current regime

### Break rates stable across years
84-88% break rate every year, every ORB. No structural shift.

### Win rates positive every year for top strategies
The most consistent strategies maintain positive expectancy in every individual year (2021-2025), not just in aggregate. No "one good year carrying the average."

### Most consistent strategy
MGC_1800_RR2.0_CB4_ORB_L3 — lowest year-to-year standard deviation (StdR=0.0227), positive every year.

---

## Cost Model Reality

### MGC friction is significant on small ORBs
- Total RT friction: $8.40 (commission $2.40 + spread $2.00 + slippage $4.00)
- On a 2pt ORB: friction = 0.84pt = 42% of risk
- On a 4pt ORB: friction = 0.84pt = 21% of risk
- On a 10pt ORB: friction = 0.84pt = 8.4% of risk

**The ORB_G4 filter (size > 4pt) exists because small ORBs get eaten by costs.**

### 1RR is the most cost-sensitive target
At 1RR, your win gives you less than 1R (friction reduces the win).
For a 2pt ORB: win_r = 0.70, so you're risking 1R to make 0.70R.
You need 59% win rate just to break even.
At RR1.5-2.0 the math is kinder per trade but you need fewer, bigger wins.

---

## Data Quality Notes

### GC (not MGC) provides the bar data
- GC has ~96% 1m bar coverage (median 1,377 bars/day of 1,440 possible)
- MGC only had ~78% coverage — not enough for accurate ORBs
- Prices are identical (same underlying, same exchange)

### Missing bars = missing volume, not missing price
Databento only emits a bar when there's a trade. Gaps mean no trades, not missing data. The OHLCV of the next bar after a gap already captures the full price range.

### Late-session data thins out
- 2,887 "ghost scratches" — entries with zero bars after entry
- 97% are on ORB 0030 and 2300 (overnight ORBs near end of CME session)
- GC volume dies after ~14:00 UTC. Entries after that may fire on the last trade of the day.
- These are classified as "scratch" (technically correct — no target/stop hit observed)
- Impact: worst case ~0.9pp on aggregate win rate

### ~56-bar days are weekends
10 days have <500 bars. All are Sundays/holidays with thin electronic-only trading. ORBs on these days exist but are unreliable.

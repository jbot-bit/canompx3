# Market Playbook — Gold Futures (GC/MGC)

Empirical findings from 1,460 trading days (2021-02-05 to 2026-02-04).
Database now contains 2,922 trading days (2016-02-01 to 2026-02-04) but
outcomes/strategies below are computed on the 2021-2026 slice.
~3M outcomes computed across 7 instruments with entry models (E0/E1/E3).
618 strategies validated active (post-cost, FDR-corrected, yearly robust).

**IMPORTANT**: These findings supersede the earlier 252-strategy scan which had
two bugs: entry at ORB level (unreachable price) and missing friction on wins.
The current scan uses honest entry prices and correct cost deductions.

---

## Entry Models (Phase 5b Fix)

Three entry models replace the original broken entry logic:

| Model | Entry Price | When | Fill Rate |
|-------|-----------|------|-----------|
| **E1** (Market-On-Next-Bar) | Next bar OPEN after confirm | 1 bar after confirmation | ~100% |
| **E2** (Market-On-Confirm-Close) | Confirm bar CLOSE | At confirmation | 100% |
| **E3** (Limit-At-ORB) | ORB level (orb_high/orb_low) | First retrace after confirm | ~96-97% on G5+ days |

**E3** fill rate is much higher on G5+ days (~97%) vs overall (~70-80%). On large-ORB
days, price retraces are common because the initial breakout is more volatile.

### Entry model performance depends on ORB time
| ORB | Best EM | Why |
|-----|---------|-----|
| CME_REOPEN | **E1** (+38.2R) | Fast momentum entry; extra confirmation wastes edge |
| TOKYO_OPEN | **E1** (+18.6R) | Same pattern as CME_REOPEN |
| LONDON_METALS | **E3** (+22.6R) | GLOBEX open spikes through then retraces; E3 catches the pullback |
| US_DATA_830 | **E3** (+12.3R) | Overnight session has mean-reversion character |

**Key insight**: E1 and E3 exploit opposite mechanisms. E1 = ride the momentum.
E3 = get a better price on the pullback. The right model depends on the session character.

---

## The Central Finding: ORB Size Is the Edge

The single most important variable is **ORB size**. Larger ORBs produce better outcomes because:
1. More room for directional follow-through relative to noise
2. Fixed friction ($8.40 RT) is a smaller percentage of risk
3. Large ORBs signal genuine institutional participation

### Win rate by CME_REOPEN ORB size (E1 RR2.5 CB2):
| ORB Size | N | Win Rate | Avg R |
|----------|---|----------|-------|
| < 2pt | 948 | 27.2% | -0.47 |
| 2-4pt | 144 | 25.7% | -0.30 |
| 4-6pt | 49 | 38.8% | +0.19 |
| 6-10pt | 42 | 50.0% | +0.59 |
| 10-20pt | 20 | 36.8% | +0.23 |

**Breakeven is around 4pt ORB size.** Below 4pt, the house wins.

### ORB size filter sensitivity (CME_REOPEN E1 RR2.5 CB2):
| Filter | N | WR | TotalR | AvgR | Notes |
|--------|---|-----|--------|------|-------|
| G4+ | 125 | 40.0% | +38.2 | +0.31 | Best total R |
| G5+ | 102 | 41.2% | +37.0 | +0.36 | Good balance |
| G6+ | 76 | 40.8% | +28.8 | +0.38 | Highest per-trade edge |
| G8+ | 47 | 36.2% | +12.4 | +0.26 | Declining (overfitting) |
| G10+ | 34 | 32.4% | +5.0 | +0.15 | Too restrictive |

**G4+ maximizes total R; G5+/G6+ maximize per-trade edge.**

### Percentage-based filter as alternative
A 0.15% of price filter approximates G5 point-based (TotalR: +37.2 vs +37.0)
but auto-adapts to gold price level. A 5pt ORB at $4800 (0.10%) is different
from 5pt at $1800 (0.28%). Consider implementing for regime stability.

### Validated strategies by filter
| Filter | Validated | Avg ExpR | Description |
|--------|-----------|----------|-------------|
| ORB_G6 | 97 | 0.30 | Size >= 6pt (most validated) |
| ORB_G5 | 90 | 0.28 | Size >= 5pt |
| ORB_G4 | 73 | 0.22 | Size >= 4pt |
| ORB_G3 | 34 | 0.17 | Size >= 3pt |
| ORB_G8 | 18 | 0.21 | Size >= 8pt |
| NO_FILTER | 0 | - | ALL negative ExpR |
| ORB_L* | 0 | - | ALL negative ExpR |

**NO_FILTER and L-filter strategies are ALL negative.** Without filtering for large ORBs, there is no edge after costs.

---

## Confirm Bar Analysis

### CB behavior depends on ORB time AND entry model

On **G5+ days** (the tradeable universe), CB2 is optimal for momentum entries:

**CME_REOPEN E1 RR2.5 G5+ (momentum entry):**
| CB | N | WR | TotalR | AvgR |
|----|---|-----|--------|------|
| CB1 | 102 | 38.2% | +28.9 | +0.30 |
| **CB2** | **102** | **41.2%** | **+37.0** | **+0.38** |
| CB3 | 99 | 35.4% | +21.1 | +0.23 |
| CB4 | 99 | 33.3% | +15.0 | +0.16 |
| CB5 | 98 | 29.6% | +4.3 | +0.05 |

**LONDON_METALS E3 RR2.0 G5+ (retrace entry):**
| CB | N | WR | TotalR | AvgR |
|----|---|-----|--------|------|
| CB1 | 81 | 43.2% | +13.8 | +0.17 |
| CB2 | 80 | 45.0% | +17.6 | +0.22 |
| CB3 | 78 | 46.2% | +19.6 | +0.25 |
| CB4 | 77 | 46.8% | +20.6 | +0.27 |
| **CB5** | **75** | **48.0%** | **+22.6** | **+0.30** |

**Why the reversal**: On G5+ days, CME_REOPEN breakouts are genuine (large institutional
moves). Extra confirmation bars waste the edge — CB2 is enough to confirm direction,
then E1 captures momentum. But for LONDON_METALS E3, more confirm bars = more time for the
retrace to develop = better fill quality at the ORB level.

---

## RR Target Optimization

**CME_REOPEN E1 CB2 G5+:**
| RR | N | WR | TotalR | AvgR |
|----|---|-----|--------|------|
| 1.0 | 102 | 56.9% | +7.5 | +0.08 |
| 1.5 | 102 | 50.0% | +18.6 | +0.19 |
| 2.0 | 102 | 45.1% | +28.4 | +0.29 |
| **2.5** | **102** | **41.2%** | **+37.0** | **+0.38** |
| 3.0 | 102 | 32.4% | +27.1 | +0.29 |
| 4.0 | 102 | 24.5% | +21.7 | +0.23 |

**RR2.5 is the sweet spot** for CME_REOPEN: enough room for the trend, not so far it rarely hits.

**LONDON_METALS E3 CB5 G5+:**
| RR | N | WR | TotalR | AvgR |
|----|---|-----|--------|------|
| 1.0 | 75 | 64.0% | +11.2 | +0.15 |
| 1.5 | 75 | 54.7% | +17.5 | +0.23 |
| **2.0** | **75** | **48.0%** | **+22.6** | **+0.30** |
| 2.5 | 75 | 40.0% | +20.2 | +0.27 |

**RR2.0 is optimal for LONDON_METALS E3**: lower target works because E3 gets a better entry price
(closer to the ORB level), so the risk-to-reward math favors a slightly closer target.

---

## Direction Bias

**CME_REOPEN/TOKYO_OPEN: Long breakouts are stronger than short**

| Session | Direction | N | WR | AvgR | TotalR |
|---------|-----------|---|-----|------|--------|
| CME_REOPEN | LONG | 56 | 44.6% | +0.50 | +26.8 |
| CME_REOPEN | SHORT | 46 | 37.0% | +0.23 | +10.2 |
| TOKYO_OPEN | LONG | 48 | 41.7% | +0.41 | +18.6 |
| TOKYO_OPEN | SHORT | 40 | 30.0% | -0.04 | -1.8 |

**TOKYO_OPEN SHORT is actually negative** — long-only filter adds significant edge at TOKYO_OPEN.
CME_REOPEN short is still positive but weaker. Both-directions CME_REOPEN gives more total R (+37 vs +27).

**LONDON_METALS shows no strong directional bias** with E3 entry (long +10.7R, short +4.4R at RR2.5).

---

## Correlation Between ORBs

Different ORBs show **negative PnL correlation** (even better than independence):

| Pair | PnL Correlation | Interpretation |
|------|----------------|---------------|
| CME_REOPEN vs TOKYO_OPEN | -0.04 | Near zero |
| CME_REOPEN vs LONDON_METALS | -0.12 | Mild hedging |
| TOKYO_OPEN vs LONDON_METALS | **-0.39** | **Strong hedging** |
| CME_REOPEN vs US_DATA_830 | +0.11 | Mild positive |
| TOKYO_OPEN vs US_DATA_830 | +0.29 | Some overlap |
| LONDON_METALS vs US_DATA_830 | -0.09 | Near zero |

**TOKYO_OPEN and LONDON_METALS are strongly negatively correlated**: when one loses, the other tends
to win. This makes them an excellent pair for portfolio construction.

Note: Trade DATE overlap between ORBs is high (91-100%, same G5+ days), but break
directions are ~50% random between ORBs, creating genuine PnL independence.

---

## Optimized Portfolio (4 Legs)

Each leg exploits a **different mechanism** at a different session:

### Leg 1: CME_REOPEN E1 CB2 RR2.5 G4+ (both directions)
- **Mechanism**: Momentum breakout at CME reopen
- N=125, WR=40.0%, TotalR=+38.2, AvgR=+0.31
- Avg risk: 12.9pt ($129/contract)
- Positive every year 2022-2026

### Leg 2: TOKYO_OPEN E1 CB2 RR2.5 G5+ LONG-ONLY
- **Mechanism**: Gold intraday long bias on large-move days
- N=48, WR=41.7%, TotalR=+18.6, AvgR=+0.39
- Avg risk: 12.5pt ($125/contract)
- Only trades 2025+ in meaningful volume

### Leg 3: LONDON_METALS E3 CB5 RR2.0 G5+ (both directions)
- **Mechanism**: London metals open spike-and-retrace; limit entry at ORB level
- N=75, WR=48.0%, TotalR=+22.6, AvgR=+0.30
- Avg risk: 8.9pt ($89/contract)
- Highest win rate of any leg

### Leg 4 (optional): US_DATA_830 E3 CB4 RR1.5 G8+
- **Mechanism**: Overnight session retrace entry on very large ORB days
- N=50, WR=52.0%, TotalR=+12.3, AvgR=+0.25
- Requires G8+ filter (stricter)
- Adds +12.3R and reduces max DD by 1.2R

### Combined performance

**3-leg portfolio:**
| Year | N | R | Notes |
|------|---|---|-------|
| 2021 | 9 | -4.0 | Low vol, few trades |
| 2022 | 9 | +3.5 | |
| 2023 | 6 | +6.4 | |
| 2024 | 16 | +0.8 | |
| 2025 | 157 | +56.5 | Regime shift kicks in |
| 2026 | 51 | +16.3 | YTD (Jan-Feb) |
| **TOTAL** | **248** | **+79.4** | |

- Max Drawdown: 9.6R
- Max Consecutive Losses: 7
- Avg: 19 trades/month, +6.5R/month (2025-2026)

**4-leg portfolio (with US_DATA_830):**
| Year | N | R |
|------|---|---|
| 2021 | 10 | -5.0 |
| 2022 | 10 | +4.8 |
| 2023 | 6 | +6.4 |
| 2024 | 24 | +1.7 |
| 2025 | 188 | +62.6 |
| 2026 | 60 | +21.3 |
| **TOTAL** | **298** | **+91.7** |

- Max Drawdown: 8.3R
- Max Consecutive Losses: 7

### Dollar economics (1 MGC contract per signal)
| Metric | 3-leg | 4-leg |
|--------|-------|-------|
| Avg risk/trade | $116 | ~$110 |
| 2025 return | $6,570 | $7,260 |
| Max DD | $1,115 | $963 |
| Capital needed | ~$2,300 | ~$2,200 |
| 2025 ROI | ~283% | ~330% |

*Capital = margin ($1,210) + max DD buffer*

---

## Worst-Case Scenarios

| Metric | Value |
|--------|-------|
| Worst month | 2025-06: -2.7R ($314) |
| Max consecutive losses | 7 |
| Max drawdown | 9.6R (3-leg) / 8.3R (4-leg) |
| Longest flat period | 334 days (Dec 2023 - Nov 2024) |
| Max win streak | 7 |

---

## The Regime Shift (Critical Context)

Gold prices and ORB sizes have changed dramatically:

| Year | Avg Gold Price | CME_REOPEN Median ORB | G5+ Days (CME_REOPEN) |
|------|---------------|-----------------|-----------------|
| 2021 | $1,794 | 0.7pt | 7 (3%) |
| 2022 | $1,807 | 0.9pt | 9 (3%) |
| 2023 | $1,955 | 0.9pt | 2 (1%) |
| 2024 | $2,404 | 1.0pt | 7 (3%) |
| 2025 | $3,463 | 2.5pt | 70 (27%) |
| 2026 | $4,768 | 14.8pt | 21 (88%) |

**2021-2024**: G5+ days were rare (2-9/year). Strategies had tiny samples.
**2025-2026**: G5+ days are common (70+/year). The strategies generate real trade volume.

### Multi-ORB day frequency (2025)
| G5+ ORBs triggered | Days | % |
|--------------------|------|---|
| 0 | 139 | 44% |
| 1 | 69 | 22% |
| 2 | 44 | 14% |
| 3 | 31 | 10% |
| 4 | 33 | 10% |

108/282 days (38%) had 2+ G5+ ORBs triggered simultaneously.
Position sizing must account for correlated exposure on multi-ORB days.

### Regime risk
- If gold returns to $1800, G5+ days drop to ~5/year (untradeable frequency)
- The edge requires elevated volatility/price levels
- A 5pt ORB at $4800 gold = 0.10% — increasingly normal

---

## Day of Week (Low Confidence)

| Day | N | WR | AvgR | Confidence |
|-----|---|-----|------|-----------|
| Monday | 44 | 34.1% | +0.21 | Medium (large N) |
| Tuesday | 17 | 52.9% | +0.71 | Low (small N) |
| Wednesday | 15 | 33.3% | +0.06 | Low |
| Thursday | 12 | 41.7% | +0.34 | Low |
| Friday | 14 | 57.1% | +0.83 | Low |

**Monday weakness is the only pattern with reasonable sample.** Tuesday/Friday
appear strong but N is too small to be reliable. Not recommended as a filter.

---

## Volatility Persistence

Previous day's ORB size mildly predicts today's edge:

| Previous ORB | N | WR | AvgR |
|-------------|---|-----|------|
| Small (<3pt) | 36 | 41.7% | +0.39 |
| Medium (3-6pt) | 28 | 32.1% | +0.11 |
| Large (6-10pt) | 14 | 50.0% | +0.59 |
| Huge (10+pt) | 24 | 45.8% | +0.52 |

**Consecutive G5+ days** are stronger: when yesterday was also G5+, WR=45.1%,
AvgR=+0.47 vs WR=37.3%, AvgR=+0.25 for non-streak days (N=51 each).

---

## RSI as Directional Confirmation

RSI at CME_REOPEN predicts outcome **when combined with direction**:

| RSI + Direction | N | WR | AvgR | TotalR |
|----------------|---|-----|------|--------|
| RSI 60+ LONG | 24 | 54.2% | +0.81 | +19.5 |
| RSI 40-60 LONG | 21 | 47.6% | +0.52 | +10.9 |
| RSI 40-60 SHORT | 21 | 42.9% | +0.36 | +7.6 |
| RSI 60+ SHORT | 14 | 28.6% | -0.00 | -0.0 |
| RSI <40 SHORT | 11 | 36.4% | +0.25 | +2.7 |
| RSI <40 LONG | 11 | 18.2% | -0.34 | -3.7 |

**Key pattern**: RSI confirms direction. Long in bullish RSI (60+) = +0.81 AvgR.
Long in bearish RSI (<40) = -0.34 AvgR. Short in bullish RSI = breakeven.

**Not yet actionable**: Small samples (N=11-24). Monitor but don't filter yet.
If confirmed with more data, RSI 60+ LONG-only could be a very strong signal.

---

## MAE/MFE Analysis

### Trade excursion patterns (CME_REOPEN E1 CB2 RR2.5 G5+)
| Outcome | N | Avg MAE | Med MAE | Avg MFE | Med MFE |
|---------|---|---------|---------|---------|---------|
| Win | 42 | 0.27R | 0.22R | 2.45R | 2.39R |
| Loss | 56 | 1.04R | 1.00R | 0.62R | 0.41R |
| Scratch | 4 | 0.54R | 0.54R | 0.80R | 0.74R |

**Winners barely dip** (MAE 0.22R median) before going to target.
**45% of losses went 0.5R favorable before stopping out** (25 of 56).
**27% of losses went 1.0R favorable** (15 of 56) — trailing stop candidates.

### Trailing stop opportunity (theoretical)
Moving stop to breakeven at +1.0R would rescue losses that went favorable:

| Leg | Current R | Losses Rescued | R Saved | With Trailing |
|-----|-----------|---------------|---------|---------------|
| CME_REOPEN E1 CB2 RR2.5 G4+ | +38.2 | 18 | +18.0 | +56.2 |
| TOKYO_OPEN E1 CB2 RR2.5 G5+ LONG | +18.6 | 4 | +4.0 | +22.6 |
| LONDON_METALS E3 CB5 RR2.0 G5+ | +22.6 | 9 | +9.0 | +31.6 |
| US_DATA_830 E3 CB4 RR1.5 G8+ | +12.3 | 1 | +1.0 | +13.3 |
| **TOTAL** | **+91.7** | **32** | **+32.0** | **+123.7** |

*Upper bound — requires bar-by-bar simulation to verify ordering.
Some wins that dipped back below entry after +1.0R would also become breakeven.*

### Loss speed patterns
| Duration | N | Avg MFE |
|----------|---|---------|
| <30min (fast stop) | 19 | 0.13R |
| 30-60min | 12 | 0.41R |
| 1-2hr | 21 | 0.67R |
| 2hr+ (slow bleed) | 19 | 1.09R |

Fast losses (< 30min) never went favorable — genuine immediate reversals.
Slow losses (2hr+) went 1.09R favorable on average — prime trailing stop targets.

---

## Multi-Trade Day Analysis

When the portfolio fires multiple signals on the same day:

| Trades/Day | Days | Avg Day R | Total R |
|-----------|------|-----------|---------|
| 1 trade | 64 | -0.02 | -1.5 |
| 2 trades | 44 | +0.49 | +21.3 |
| 3 trades | 20 | +2.46 | +49.2 |
| 4 trades | 9 | +1.65 | +14.9 |

**3-4 trade days are the money makers.** When multiple ORBs trigger G5+,
it signals a high-volatility day where breakouts tend to follow through.

Worst single day: -2.0R (capped by portfolio diversification).
Best single day: +5.8R.

---

## Gold Price Level Effect

| Price Level | N | WR | AvgR | Notes |
|------------|---|-----|------|-------|
| <$2000 | 10 | 20.0% | -0.18 | Low vol, small ORBs |
| $2000-3000 | 12 | 41.7% | +0.39 | |
| $3000-4000 | 25 | 60.0% | +0.90 | Sweet spot |
| $4000+ | 55 | 36.4% | +0.21 | Still positive |

**$3000-4000 is the sweet spot** (WR=60%, AvgR=+0.90). At $4000+, the edge narrows
but remains positive. This is because 5pt at $4800 is only 0.10% of price — the
point-based filter becomes less selective. Consider switching to percentage-based.

---

## Cost Model

### MGC friction per round-trip
- Commission: $2.40/RT
- Spread: $2.00/RT (1 tick x $10)
- Slippage: $4.00/RT
- **Total: $8.40/RT = 0.84pt**

### Impact by ORB size
| ORB Size | Friction % of Risk | Win at 1RR | Need WR |
|----------|-------------------|------------|---------|
| 2pt | 42% | 0.58R | 63.3% |
| 4pt | 21% | 0.79R | 55.9% |
| 6pt | 14% | 0.86R | 53.7% |
| 10pt | 8.4% | 0.92R | 52.2% |

---

## Data Quality Notes

### GC (not MGC) provides the bar data
- GC has ~96% 1m bar coverage (median 1,377 bars/day of 1,440 possible)
- MGC only had ~78% coverage — not enough for accurate ORBs
- Prices are identical (same underlying, same exchange)

### 689,310 outcomes computed
- 6 ORBs x 6 RRs x 5 CBs x 3 entry models x ~1,460 days
- Entry models: E1 (next bar open), E2 (confirm close), E3 (limit at ORB retrace)
- All outcomes include realistic entry prices from actual bar data

### Validation criteria
- Exclude 2021 (structurally different low-vol regime, 3% G5+ rate)
- 80% of included years must be positive
- Post-cost ExpR > 0
- Survive 1.5x cost stress test
- Minimum 50 trades

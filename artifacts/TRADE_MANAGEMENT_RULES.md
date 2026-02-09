# Trade Management Rules — 1800 E3 Session

Verified 2026-02-09 against gold.db (1,286 trades, 2021-2026, G4+ filter).

All claims independently re-verified from raw data. Sample sizes noted for each.

---

## Glossary — What All the Codes Mean

### Sessions (ORB Labels)
| Code | Plain English | Time (Brisbane) | Time (UTC) |
|------|--------------|-----------------|------------|
| 0900 | Asia open | 9:00 AM | 23:00 prev day |
| 1000 | Asia mid-morning | 10:00 AM | 00:00 |
| 1100 | Asia late morning | 11:00 AM | 01:00 |
| 1800 | Evening session | 6:00 PM | 08:00 |
| 2300 | Late night session | 11:00 PM | 13:00 |
| 0030 | After midnight | 12:30 AM | 14:30 |

### Entry Models
| Code | Plain English | How It Works |
|------|--------------|-------------|
| E1 | Market entry | Price breaks ORB + confirm bars close outside. You buy/sell at the NEXT bar's open price. Fastest entry. |
| E2 | Confirm close entry | Same as E1 but you enter at the confirm bar's closing price (not next bar open). Rarely used. |
| E3 | Limit retrace entry | Price breaks ORB + confirm bars close outside. You place a LIMIT ORDER back at the ORB edge and wait for price to come back to you. Better entry price, but sometimes price never comes back and you miss the trade. |

### Confirm Bars (CB)
| Code | Plain English |
|------|--------------|
| CB1 | 1 consecutive 1-minute bar must close outside the ORB |
| CB2 | 2 consecutive 1-minute bars must close outside the ORB |
| CB3 | 3 consecutive (and so on) |
| CB4 | 4 consecutive |
| CB5 | 5 consecutive |

**Important:** If ANY bar closes back inside the ORB, the count resets to zero. It's not a timer — it's consecutive closes outside.

**Important:** For E3 entries, CB1-CB5 all produce the exact same entry price (see Rule 1). They only differ for E1/E2 entries.

### Risk-Reward (RR)
| Code | Plain English | Example (10-pt ORB, long) |
|------|--------------|--------------------------|
| RR1.5 | Target = 1.5x the risk | Risk = 10 pts, Target = 15 pts above entry |
| RR2.0 | Target = 2x the risk | Risk = 10 pts, Target = 20 pts above entry |
| RR2.5 | Target = 2.5x the risk | Risk = 10 pts, Target = 25 pts above entry |
| RR3.0 | Target = 3x the risk | Risk = 10 pts, Target = 30 pts above entry |

**Risk = ORB size** (distance from ORB high to ORB low). Stop loss is always at the other side of the ORB.

### ORB Size Filters (G = "Greater than")
| Code | Plain English | What It Does |
|------|--------------|-------------|
| NO_FILTER | Take every trade | No minimum ORB size. Historically LOSES money. |
| L3 | Small ORBs only | Only trade if ORB is LESS than 3 points. Loses money. |
| G4 (or G4+) | Medium+ ORBs | Only trade if ORB is 4 points or bigger. This is where edge starts. |
| G5 | Bigger ORBs | Only trade if ORB is 5 points or bigger. Stronger filter. |
| G6 | Large ORBs | Only trade if ORB is 6+ points. Fewer trades but higher quality. |
| G8 | Very large ORBs | Only trade if ORB is 8+ points. Very selective, very few trades. |
| G10 | Huge ORBs | 10+ points. Rare but strong. |

**Key fact:** ORB size IS the edge. Below 4 points, the house wins. Above 4, you start winning. Above 6, strong edge.

### Other Terms
| Term | Plain English |
|------|--------------|
| ORB | Opening Range Breakout. The high and low of the first 5 minutes of a session. |
| ORB edge | The high (if long) or low (if short) of the ORB. Where E3 limit orders sit. |
| R or R-multiple | Your profit/loss measured in units of risk. +1R = you won 1x your risk. -1R = full stop loss hit. |
| ExpR | Expected R per trade (average across all trades). Positive = profitable system. |
| WR | Win rate (percentage of trades that hit target). |
| Sharpe | Risk-adjusted return. Higher = smoother equity curve. Above 0.15 = decent. |
| MaxDD | Maximum drawdown in R-multiples. Worst peak-to-trough losing streak. |
| MFE | Maximum Favourable Excursion. How far price went in your favour before the trade ended. |
| MAE | Maximum Adverse Excursion. How far price went against you before the trade ended. |
| PDH / PDL | Previous Day High / Previous Day Low. Key support/resistance levels. |
| HOD / LOD | High of Day / Low of Day. Today's extremes so far. |
| MGC | Micro Gold futures (the contract you trade). $10 per point, $8.40 round-trip cost. |
| GC | Full-size Gold futures. Same price as MGC, just 10x the contract size. We use GC price data for accuracy but trade MGC for smaller position size. |

### Reading a Strategy ID

Example: `MGC_1800_E3_RR2.0_CB2_ORB_G4`

```
MGC     = instrument (Micro Gold)
1800    = session (6 PM Brisbane)
E3      = entry model (limit retrace)
RR2.0   = risk-reward 2:1
CB2     = 2 confirm bars
ORB_G4  = filter: ORB size must be 4+ points
```

---

## Rule 1: CB Overlap — You Only Have 2 Strategies, Not 10

**Claim:** CB1 through CB5 produce identical E3 entry prices and nearly identical outcomes.

| Metric | RR 1.5 | RR 2.0 |
|--------|--------|--------|
| CB pairs checked | 1,258 | 1,258 |
| Same entry price | 100.0% | 100.0% |
| Same outcome | 93.8% | 96.3% |

**Why:** E3 entry = limit order at ORB edge. Once any CB confirms, the limit sits at the same price regardless of how many bars confirmed. More confirm bars just delays the order placement — but the fill price is identical.

**Implication:** Never run multiple CB variants on the same session. Pick one CB (CB2 is fine) and treat CB1-CB5 as one strategy. Your real strategy count for 1800 E3 is **2** (RR1.5 G5+ and RR2.0 G4+), not 10.

---

## Rule 2: 30-Minute Position Check

**Claim:** If losing at 30 minutes after fill, win probability drops significantly.

| Position at 30 min | N | Win Rate |
|--------------------|---|----------|
| In profit | 505 | **70%** |
| In loss | 239 | **24%** |

**Scope:** 1800 E3, RR 1.5 + 2.0, G4+ filter, trades still open at 30 min (N=744).

**Action:** At 30 minutes post-fill, check if price is above (long) or below (short) your entry:
- Green = 70% win rate. Hold.
- Red = 24% win rate. **Strongly consider manual exit.** You save ~76% of those eventual losses.

**Correction note:** An earlier in-session estimate quoted "10% win rate if losing at 30 min." That was from an all-sessions query. The 1800-specific verified number is **24%** — still a strong kill signal but less extreme.

---

## Rule 3: Retrace Dwell Time — The Best Signal

**Claim:** After a trade goes +0.3R or more, if price returns to entry zone (within 0.15R of entry), the number of consecutive minutes it stays there predicts outcome.

### RR 2.0 (N=362 trades that retraced after going green)

| Consecutive min at entry | N | Win Rate | Action |
|--------------------------|---|----------|--------|
| 1-2 min (quick touch) | 30 | **100%** | HOLD — always recovers |
| 3-5 min | 59 | 39% | Watch closely |
| 6-10 min | 47 | 55% | Coin flip |
| 11-15 min | 63 | 44% | Edge fading |
| 16-20 min | 32 | **16%** | EXIT |
| 21-30 min | 59 | 31% | Dead trade |
| 31+ min | 72 | 38% | Dead trade |

### RR 1.5 (N=313 trades that retraced)

| Consecutive min at entry | N | Win Rate |
|--------------------------|---|----------|
| 1-2 min | 37 | **100%** |
| 10+ min | 176 | 33% |
| 16-20 min | 22 | **0%** |

### Total time near entry (RR 2.0)

| Total min near entry | N | Win Rate |
|---------------------|---|----------|
| 1-3 min | 32 | **94%** |
| 4-8 min | 71 | 46% |
| 9-15 min | 46 | **28%** |
| 16+ min | 213 | 37% |

**Action:**
1. Quick touch and bounce (1-2 min) = normal, don't panic, 100% win rate.
2. Sitting at entry 3-8 min = watch, no action yet.
3. **10+ consecutive minutes at entry zone = EXIT.** Win rate drops to 33-36%.
4. **16+ consecutive minutes = definitely exit.** Win rate is 0-16%.

**Definition of "entry zone":** Price within 0.15R of entry price (about 2.4 pts on a 16-pt ORB).

---

## Rule 4: No Time-Based Cutoff

**Claim:** There is no wall-clock time after which it's better to close a trade.

| Time still open | N | Win Rate | Avg R |
|----------------|---|----------|-------|
| 2 hours | 242 | 55% | +0.39R |
| 3 hours | 128 | 55% | +0.38R |
| 4 hours | 105 | 56% | +0.43R |
| 5 hours | 34 | 47% | +0.21R |

**Implication:** Do NOT use a fixed time cutoff. Trades that take 4 hours to resolve still have positive expected value (+0.43R). The 1800 session is slow money — let stop and target work.

The only valid early exit signals are Rule 2 (30-min check) and Rule 3 (retrace dwell time).

---

## Summary: Decision Flowchart

```
TRADE FILLS (E3 retrace at ORB edge)
  |
  +-- 30 min check: Am I in profit?
  |     YES -> HOLD (70% WR)
  |     NO  -> EXIT (only 24% WR)
  |
  +-- Price goes green (+0.3R), then retraces to entry:
  |     Quick touch (1-2 min) -> HOLD (100% WR)
  |     Sits 3-8 min -> WATCH
  |     Sits 10+ min consecutive -> EXIT (33% WR)
  |     Sits 16+ min consecutive -> DEFINITELY EXIT (0-16% WR)
  |
  +-- No retrace, trade grinding toward target:
        -> Let it run. No time cutoff improves EV.
```

---

## Methodology

- Database: gold.db, 1800 ORB, E3 entry model, G4+ ORB size filter
- Period: 2021-2026 (all available orb_outcomes data)
- orb_minutes: 5 (standard 5-min ORB)
- Entry zone definition: within 0.15R of entry price
- "Went green" threshold: +0.3R from entry
- 1-minute bars used for dwell time counting (bars_1m table)
- All timestamps UTC, converted from Brisbane trading day boundaries
- pnl_r is post-cost (MGC friction $8.40 RT deducted before R conversion)

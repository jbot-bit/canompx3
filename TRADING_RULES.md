# Trading Rules — Gold Futures (GC/MGC)

Single source of truth for live trading. All claims verified against gold.db.
For research deep-dives and data tables, see `docs/RESEARCH_ARCHIVE.md`.

---

## Glossary

### Sessions (ORB Labels — 11 total)

**Fixed sessions (7):**
| Code | Plain English | Time (Brisbane) | Time (UTC) |
|------|--------------|-----------------|------------|
| 0900 | Asia open | 9:00 AM | 23:00 prev day |
| 1000 | Asia mid-morning | 10:00 AM | 00:00 |
| 1100 | Asia late morning | 11:00 AM | 01:00 |
| 1130 | Asia late morning 2 | 11:30 AM | 01:30 |
| 1800 | Evening session | 6:00 PM | 08:00 |
| 2300 | Late night session | 11:00 PM | 13:00 |
| 0030 | After midnight | 12:30 AM | 14:30 |

**Dynamic sessions (4, DST-aware — times shift with daylight saving):**
| Code | Plain English | Reference TZ |
|------|--------------|-------------|
| CME_OPEN | CME daily open | 5:00 PM CT |
| US_EQUITY_OPEN | US equity open | 9:30 AM ET |
| US_DATA_OPEN | US economic data | 8:30 AM ET |
| LONDON_OPEN | London open | 8:00 AM UK |

**Aliases:** TOKYO_OPEN → 1000, HK_SG_OPEN → 1130
**Module:** `pipeline/dst.py` (SESSION_CATALOG + resolvers)
**Per-asset enabled sessions:** `pipeline/asset_configs.py`

### Entry Models
| Code | How It Works |
|------|-------------|
| E1 | Price breaks ORB + confirm bars close outside. Enter at NEXT bar's open. Fastest. |
| E3 | Price breaks ORB + confirm bars close outside. LIMIT ORDER at ORB edge, wait for retrace. Better price, ~3% miss rate on G5+ days. |

*E2 removed — identical to E1 on 1m bars.*

### Confirm Bars (CB)
CB1-CB5 = 1-5 consecutive 1m bars must close outside ORB. If ANY bar closes back inside, count resets. For E3, CB1-CB5 produce identical entry prices (same limit order).

### Risk-Reward (RR)
Target = RR × risk. Risk = ORB size (high - low). Stop = other side of ORB.
Example: 10pt ORB, RR2.0 = 20pt target, 10pt stop.

### ORB Size Filters
| Code | Meaning | Edge |
|------|---------|------|
| NO_FILTER | Take every trade | LOSES money. No edge. |
| L3/L4/L6/L8 | ORB < N points | LOSES money on 5m ORB. |
| G4+ | ORB >= 4 points | Breakeven starts here. |
| G5+ | ORB >= 5 points | Solid edge. |
| G6+ | ORB >= 6 points | Strongest per-trade edge, fewer trades. |
| G8+ | ORB >= 8 points | Very selective. 2300 only. |

### Other Terms
| Term | Meaning |
|------|---------|
| ORB | Opening Range Breakout. High/low of first 5 minutes of a session. |
| R-multiple | Profit/loss in units of risk. +1R = won 1× risk. -1R = full stop hit. |
| ExpR | Expected R per trade (average). Positive = profitable. |
| Sharpe | Risk-adjusted return. Above 0.15 = decent. |
| MaxDD | Maximum drawdown in R. Worst peak-to-trough. |
| MGC | Micro Gold futures. $10/point, $8.40 RT friction. |
| MNQ | Micro Nasdaq futures. $2/point, $2.74 RT friction. |
| MES | Micro S&P 500 futures. $1.25/point, $2.10 RT friction. |
| MCL | Micro Crude Oil futures. NO EDGE (0 validated). |
| Edge family | Group of strategies with identical trade days. 1 head per family. |

### Reading a Strategy ID
`MGC_1800_E3_RR2.0_CB2_ORB_G4` = Micro Gold, 6PM session, limit retrace entry, 2:1 target, 2 confirm bars, ORB >= 4pt filter.

---

## Edge Summary

### What Works
1. **ORB size filter is THE edge.** Below 4pt, house wins. Above 4pt, you start winning. Above 6pt, strong edge.
2. **E1 momentum for 0900/1000.** Fast entry captures breakout energy.
3. **E3 retrace for 1800/2300.** GLOBEX open spikes then pulls back; limit entry catches the pullback.
4. **Kill losers early** at 0900 (15 min) and 1000 (30 min). Proven Sharpe improvement.
5. **Negative correlation between sessions** (1000 vs 1800: -0.39). Portfolio diversification is real.
6. **1000 LONG-ONLY.** Short breakouts at 1000 are negative. Filter to longs.

### What Doesn't Work (Confirmed NO-GOs)
| Idea | Result |
|------|--------|
| No filter / L-filters | ALL negative ExpR. 0 validated without G4+. |
| Breakeven trail stops | -0.12 to -0.17 Sharpe. Gold too volatile. |
| VWAP direction overlay | Worse than baseline (-0.085 ExpR). |
| RSI reversion (intraday) | N=4216, ExpR=-0.278. Money incinerator. |
| Session fade (Asia→London) | N=777, ExpR=-0.162. Consistently negative. |
| Gap fade | Gold gaps avg 0.80pt — too small vs ORB sizes. |
| Double-break fade/reversal | Failed OOS / regime-specific, too few trades. |
| ADX overlay (all regimes) | Low-vol ExpR negative. Regime amplifier, not edge generator. |
| Systematic retrace dwell | Kills winners at 1000. Discretionary only for 1800 E3. |
| Pyramiding | Structural NO-GO. Correlated intraday positions = tail risk. |
| Inside Day / Ease Day filters | All WORSE vs G4 baseline. |
| V-Box CLEAR filter (E1) | Filters nothing (98% trades pass). |
| Volume confirmation | Volume at break does NOT predict follow-through. |
| Session cascade | Prior session range does NOT predict next session quality. |
| Multi-day trend alignment | No improvement on established sessions. |
| Alt strategies (Gap Fade, VWAP Pullback, Value Area, Concretum) | ALL NO-GO. ORB breakout IS the edge in gold. |
| Non-ORB structural (Time-of-day, Range Expansion, Opening Drive) | ALL NO-GO. No risk structure, friction kills. |

> **Key insight**: Gold trends intraday more than it mean-reverts. The only confirmed edges are momentum breakouts with size filters. Full research details: `docs/RESEARCH_ARCHIVE.md`.

---

## Session Playbook

### 0900 Asia Open — Momentum Breakout

| Parameter | Value |
|-----------|-------|
| Entry model | **E1** (market on next bar) |
| Filter | **G4+** minimum (G5/G6 for higher per-trade edge) |
| RR target | **2.5** (sweet spot for 0900) |
| Confirm bars | **CB2** (optimal on G5+ days) |
| Direction | Both (long stronger: +0.50R vs +0.23R short) |
| Early exit | **Kill losers at 15 minutes.** Only 24% of 15-min losers eventually win. +26% Sharpe, -38% MaxDD. |
| Do NOT | Use 30-min kill (too late, 41% still win). Use breakeven trail. |

**Full-period stats (E1 CB2 RR2.5 G4+):** N=125, WR=40%, TotalR=+38.2, AvgR=+0.31
**Rolling eval:** TRANSITIONING (score 0.42-0.54, passes 9-11/19 windows). High ExpR but inconsistent.

### 1000 Asia Mid-Morning — Momentum (LONG ONLY) + IB-Conditional Exit

| Parameter | Value |
|-----------|-------|
| Entry model | **E1** (market on next bar) |
| Filter | **G3+** (CORE tier, smallest stable filter for 1000) |
| RR target | **2.5** (fixed target while IB pending) |
| Confirm bars | **CB2** |
| Direction | **LONG ONLY.** 1000 short is negative (WR=30%, AvgR=-0.04). |
| Early exit | **Kill losers at 30 minutes.** Only 12% of 30-min losers eventually win. 3.7× Sharpe improvement. |
| Exit mode | **IB-conditional** (see below) |
| Do NOT | Kill at 10 min (too early, actively harmful). |

**IB-Conditional Exit (1000 only):**
1. On entry, trade starts in `ib_pending` mode with fixed target active.
2. IB = 120 minutes from 0900 (23:00 UTC). After IB forms, detect first break of IB high or low.
3. **IB aligned** (breaks same direction as ORB trade): cancel fixed target, hold for 7 hours. Stop still active.
4. **IB opposed** (breaks opposite direction): exit at market immediately.
5. **IB not yet broken**: keep fixed target active (limbo defense).
6. Opposed kill is **regime insurance** (stop usually fires before IB breaks), not active alpha.

Config: `SESSION_EXIT_MODE["1000"] = "ib_conditional"`, `IB_DURATION_MINUTES = 120`, `HOLD_HOURS = 7`.

**Rolling eval:** STABLE (score 0.68-0.83, passes 13-16/19 windows). The MOST stable family.
**Nested 15m ORB:** 1000 is the ONLY session where 15m ORB beats 5m (+0.208R premium, 90% of pairs improve).

### 1800 Evening — Retrace Entry (GLOBEX Open)

| Parameter | Value |
|-----------|-------|
| Entry model | **E3** (limit at ORB edge) |
| Filter | **G4+** (G6 for tightest drawdown) |
| RR target | **2.0** (E3 gets better entry, closer target optimal) |
| Confirm bars | **CB2** (CB1-CB5 identical for E3) |
| Direction | Both |
| Early exit | **No systematic rule helps.** |
| Trade management | See 1800 E3 Management Rules below. |
| Do NOT | Use time-based cutoff (trades at 4h still +0.43R EV). Use breakeven trail. |

**Full-period stats (E3 CB5 RR2.0 G6+):** N=50, WR=52%, ExpR=+0.43, Sharpe=0.31, MaxDD=3.3R
**Rolling eval:** AUTO-DEGRADED (81% double-break rate). Only works in trending regimes.

### 2300 Late Night — Retrace Entry

| Parameter | Value |
|-----------|-------|
| Entry model | **E3** (limit at ORB edge) |
| Filter | **G8+** (very selective, requires large ORB) |
| RR target | **1.5** |
| Confirm bars | **CB4** |
| Direction | Both |
| Early exit | **No early exit helps.** |

**Full-period stats (E3 CB4 RR1.5 G8+):** N=50, WR=52%, TotalR=+12.3, AvgR=+0.25
**Rolling eval:** AUTO-DEGRADED (70% double-break rate).

### 1100 Session — PERMANENTLY OFF (2026-02-13)

No tradeable edge. 74% double-break rate (structurally mean-reverting). Hard exclusion in `execution_engine.py`, `portfolio.py`, `strategy_fitness.py`. Revisit only with fade/reversal model. Full research: `docs/RESEARCH_ARCHIVE.md`.

---

## 1800 E3 Management Rules

*Source: `artifacts/TRADE_MANAGEMENT_RULES.md` (verified against 1,286 trades)*

### 30-Minute Position Check
At 30 min post-fill, check mark-to-market:
- **In profit**: 70% WR. HOLD.
- **In loss**: 24% WR. **EXIT.** Saves ~76% of eventual losses.

### Retrace Dwell Time (after trade goes +0.3R then returns to entry zone)
| Time at entry zone | Win Rate | Action |
|--------------------|----------|--------|
| 1-2 min (quick touch) | **100%** | HOLD — always recovers |
| 3-8 min | 39-55% | WATCH |
| 10+ min consecutive | **33%** | **EXIT** |
| 16+ min consecutive | **0-16%** | **DEFINITELY EXIT** |

Entry zone = within 0.15R of entry price (~2.4pt on 16pt ORB).

### No Time Cutoff
Trades still open at 4 hours: WR=56%, AvgR=+0.43R. Let stop/target work.

---

## Filters

### ORB Size (THE Edge)
Win rate by 0900 ORB size (E1 RR2.5 CB2):

| ORB Size | N | WR | AvgR |
|----------|---|-----|------|
| < 2pt | 948 | 27% | -0.47 |
| 2-4pt | 144 | 26% | -0.30 |
| 4-6pt | 49 | 39% | +0.19 |
| 6-10pt | 42 | 50% | +0.59 |
| 10-20pt | 20 | 37% | +0.23 |

**G4+ maximizes total R. G5+/G6+ maximize per-trade edge.** No validated strategy exists without G4+.

### Volume Filter (Relative Volume)
`rel_vol = break_bar_volume / median(same UTC minute, 20 prior days)`
Fail-closed: missing data = ineligible. 0900 rejects 34.7% of days. Config: `VOL_RV12_N20`.

### Direction Filter
1000: LONG ONLY. Short breakouts negative (WR=30%, AvgR=-0.04).
0900: both positive, long stronger (+0.50 vs +0.23 AvgR).

### Filter Deduplication Rules
Many filter variants produce the SAME trade set.

- G2/G3 pass 99%+ of days on most sessions — cosmetic label, not real filtering
- Only G4+ filters meaningfully filter (>5% filter rate)
- Always group by `(session, EM, RR, CB)` and count unique trades

### Annualized Sharpe (ShANN)
- Per-trade Sharpe is MEANINGLESS without trade frequency context
- ShANN = per_trade_sharpe × sqrt(trades_per_year)
- Minimum bar: ShANN >= 0.5 with 150+ trades/year
- Strong: ShANN >= 0.8 | Institutional: ShANN >= 1.0

---

## Live Portfolio Configuration

*Source: `trading_app/live_config.py`*

### Tier 1: CORE (always on)
| Family | Session | EM | Filter | Exit Mode | Gate |
|--------|---------|-----|--------|-----------|------|
| 0900_E1_ORB_G5 | 0900 | E1 | G5+ | Fixed target | None |
| 1000_E1_ORB_G5 | 1000 | E1 | G5+ | IB-conditional | None |

### Tier 2: HOT (rolling-eval gated)
| Family | Session | EM | Filter | Gate |
|--------|---------|-----|--------|------|
| 0900_E1_ORB_G4 | 0900 | E1 | G4+ | Rolling stability >= 0.6 |
| 0900_E3_ORB_G4 | 0900 | E3 | G4+ | Rolling stability >= 0.6 |
| 1000_E1_ORB_G4 | 1000 | E1 | G4+ | Rolling stability >= 0.6 |

### Tier 3: REGIME (fitness-gated)
| Family | Session | EM | Filter | Gate |
|--------|---------|-----|--------|------|
| 0900_E1_ORB_G6 | 0900 | E1 | G6+ | high_vol (must be FIT) |
| 1800_E3_ORB_G6 | 1800 | E3 | G6+ | high_vol (must be FIT) |

### Portfolio Parameters
- Max concurrent positions: 3
- Max daily loss: 5.0R
- Position sizing: 2% of equity per trade
- Account equity: $25,000 default

### Correlation
| Pair | Correlation | Interpretation |
|------|------------|---------------|
| 0900 vs 1000 | -0.04 | Near zero |
| 0900 vs 1800 | -0.12 | Mild hedging |
| 1000 vs 1800 | **-0.39** | Strong hedging |

---

## Reporting Rules

### Family-Level Reporting (MANDATORY)

**Never cite the performance of a single parameter variant.** Always report the family average.

A "family" = one unique combination of `(session, entry_model, filter_level)`. All RR/CB variants within a family share 85-100% of the same trade days.

**Correct**: "The 0900 E1 G5 family averages +0.34R ExpR across RR/CB variants."
**Wrong**: "MGC_0900_E1_RR3.0_CB2_ORB_G6 has +0.39R ExpR."

**Dedup rule**: Group by `(session, EM, filter_level)` first. Report unique family count, not parameter variant count.

---

## Regime Warnings

### The 2025 Shift
| Year | Median 0900 ORB | G5+ Days (0900) |
|------|-----------------|-----------------|
| 2021-2024 | 0.7-1.0pt | 2-9/year (1-3%) |
| 2025 | 2.5pt | 70/year (27%) |
| 2026 (Jan) | 14.8pt | 21/month (88%) |

- Validated strategies collapse to **2 truly STABLE families** under rolling evaluation
- **If gold returns to $1800 levels, G5+ days drop to ~5/year (untradeable frequency)**

### Rolling Evaluation Summary
| Family | Score | Windows Passed | Status |
|--------|-------|----------------|--------|
| 1000_E1_G2 | 0.68-0.83 | 13-16/19 | STABLE |
| 0900_G3+ families | 0.42-0.54 | 9-11/19 | TRANSITIONING |
| 1800/2300/1100/0030 | <0.30 | AUTO-DEGRADED | Double-break >67% |

### Double-Break Frequency
| Session | Rate | Implication |
|---------|------|------------|
| 0900 | 57% | Survives |
| 1000 | 57% | Survives |
| 1100 | 74% | Mean-reverting (OFF) |
| 1800 | 81% | Breakout fails most of the time |
| 2300 | 70% | Unreliable for breakout |
| 0030 | 76% | Unreliable for breakout |

### ATR Regime Decision (2026-02-13)
- **NO hard ATR pre-trade gate.** Threshold-sensitive, family-inconsistent, inflated by prior lookahead.
- **KEEP ATR for position sizing:** Turtle-style vol normalization in `portfolio.py`.
- **KEEP rolling fitness:** `strategy_fitness.py` + `live_config.py` regime tier = the regime gate.
- Full ATR research: `docs/RESEARCH_ARCHIVE.md`.

---

## Cost Model

| Component | Amount |
|-----------|--------|
| Commission | $2.40/RT |
| Spread | $2.00/RT (1 tick × $10) |
| Slippage | $4.00/RT |
| **Total friction** | **$8.40/RT = 0.84pt** |

Friction as % of risk: 42% on 2pt ORB, 21% on 4pt, 14% on 6pt, 8.4% on 10pt.
This is why small ORBs lose — friction eats the edge.

---

## Confirmed Edges

| Finding | Evidence | Status |
|---------|----------|--------|
| ORB size G4+ breakout | Walk-forward positive, validated strategies | DEPLOYED |
| 0900 kill losers at 15 min | +26% Sharpe, -38% MaxDD | DEPLOYED |
| 1000 kill losers at 30 min | 3.7× Sharpe, -35% MaxDD | DEPLOYED |
| 1000 LONG-ONLY filter | Short WR=30%, AvgR=-0.04 | DEPLOYED |
| 1800 E3 30-min check + retrace dwell | In-loss 24% WR; dwell >10min = 33% WR | DEPLOYED (discretionary) |
| Nested 15m ORB for 1000 | +0.208R premium, 90% pairs improve | VALIDATED |
| IB 120m direction alignment | Opposed=3% WR. Cross-validated 0900+1000. | DEPLOYED |
| 1100 permanent exclusion | 74% double-break, all signals failed | DEPLOYED |

## Pending / Inconclusive
| Idea | Status | Notes |
|------|--------|-------|
| 30m nested ORB | NOT BUILT | 15m done, 30m not yet populated |
| RSI directional confirmation | Low confidence | RSI 60+ LONG = +0.81R but N=24. Monitor. |
| Percentage-based ORB filter | Not tested | 0.15% of price ≈ G5. Auto-adapts to price level. |
| E3 V-Box Inversion | Promising | CHOP +0.503 ExpR (N=127). Future E3-specific retest. |
| Low-vol counterbalance | NOT RESEARCHED | Mean-reversion partner for High Vol Breakout system. |

---

## Reference Documents

| Document | Content |
|----------|---------|
| `docs/RESEARCH_ARCHIVE.md` | All research deep-dives, data tables, scripts |
| `MARKET_PLAYBOOK.md` | Full backtest results, yearly breakdowns |
| `artifacts/TRADE_MANAGEMENT_RULES.md` | 1800 E3 management rules with full data |
| `artifacts/EARLY_EXIT_RULES.md` | Early exit research per-session |
| `trading_app/live_config.py` | Declarative live portfolio (code) |
| `trading_app/config.py` | Filter definitions, entry models (code) |

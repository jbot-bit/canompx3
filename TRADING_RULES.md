# Trading Rules — Gold Futures (GC/MGC)

Single source of truth for live trading. All claims verified against gold.db.
For deep-dive data, see linked reference documents.

---

## Glossary

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
| Code | How It Works |
|------|-------------|
| E1 | Price breaks ORB + confirm bars close outside. Enter at NEXT bar's open. Fastest. |
| E2 | Same as E1 but enter at confirm bar's closing price. Rarely used. |
| E3 | Price breaks ORB + confirm bars close outside. LIMIT ORDER at ORB edge, wait for retrace. Better price, ~3% miss rate on G5+ days. |

### Confirm Bars (CB)
CB1-CB5 = 1-5 consecutive 1-minute bars must close outside the ORB. If ANY bar closes back inside, count resets. For E3, CB1-CB5 produce identical entry prices (same limit order). Pick one CB per session.

### Risk-Reward (RR)
Target = RR x risk. Risk = ORB size (high - low). Stop = other side of ORB.
Example: 10pt ORB, RR2.0 = 20pt target, 10pt stop.

### ORB Size Filters
| Code | Meaning | Edge |
|------|---------|------|
| NO_FILTER | Take every trade | LOSES money. No edge. |
| L3/L4/L6/L8 | ORB LESS than N points | LOSES money on 5m ORB. |
| G4+ | ORB >= 4 points | Breakeven starts here. |
| G5+ | ORB >= 5 points | Solid edge. |
| G6+ | ORB >= 6 points | Strongest per-trade edge, fewer trades. |
| G8+ | ORB >= 8 points | Very selective. 2300 only. |

### Other Terms
| Term | Meaning |
|------|---------|
| ORB | Opening Range Breakout. High/low of first 5 minutes of a session. |
| R-multiple | Profit/loss in units of risk. +1R = won 1x risk. -1R = full stop hit. |
| ExpR | Expected R per trade (average). Positive = profitable. |
| Sharpe | Risk-adjusted return. Above 0.15 = decent. |
| MaxDD | Maximum drawdown in R. Worst peak-to-trough. |
| MGC | Micro Gold futures. $10/point, $8.40 RT friction. |

### Reading a Strategy ID
`MGC_1800_E3_RR2.0_CB2_ORB_G4` = Micro Gold, 6PM session, limit retrace entry, 2:1 target, 2 confirm bars, ORB >= 4pt filter.

---

## Edge Summary

### What Works
1. **ORB size filter is THE edge.** Below 4pt, the house wins. Above 4pt, you start winning. Above 6pt, strong edge.
2. **E1 momentum for 0900/1000.** Fast entry captures breakout energy.
3. **E3 retrace for 1800/2300.** GLOBEX open spikes then pulls back; limit entry catches the pullback.
4. **Kill losers early** at 0900 (15 min) and 1000 (30 min). Proven Sharpe improvement.
5. **Negative correlation between sessions** (1000 vs 1800: -0.39). Portfolio diversification is real.
6. **1000 LONG-ONLY.** Short breakouts at 1000 are actually negative. Filter to longs.

### What Doesn't Work (Confirmed NO-GOs)
| Idea | Result | Detail |
|------|--------|--------|
| No filter / L-filters | ALL negative ExpR | 0/312 validated without G4+ |
| Breakeven trail stops | DESTRUCTIVE everywhere | -0.12 to -0.17 Sharpe. Gold too volatile. 45-62% of would-be winners stopped out. |
| VWAP direction overlay | Worse than baseline | -0.085 ExpR vs size-only |
| RSI reversion (intraday) | Total money incinerator | N=4216, ExpR=-0.278. Gold does NOT mean-revert on simple indicators. |
| Session fade (Asia->London) | Consistently negative | N=777, ExpR=-0.162. Asia H/L not reliable S/R. |
| Gap fade | Insufficient signal | Gold gaps avg 0.80pt absolute -- too small vs ORB sizes. |
| Double-break fade (v1) | Failed OOS | 1800 OOS = -26.7R |
| Double-break reversal (v2) | Regime-specific, too few trades | Full-period driven by 5 Jan-2026 trades. Walk-forward killed it. |
| ADX overlay (all regimes) | NO-GO | Low-vol ExpR negative. Regime amplifier, not edge generator. |
| Systematic retrace dwell | Neutral/harmful pooled | Kills winners at 1000 E3 (67-75% original WR). Discretionary only for 1800 E3. |

> **Key insight**: Gold trends intraday more than it mean-reverts. The only confirmed edges are momentum breakouts with size filters.

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

### 1000 Asia Mid-Morning — Momentum (LONG ONLY)

| Parameter | Value |
|-----------|-------|
| Entry model | **E1** (market on next bar) |
| Filter | **G3+** (CORE tier, smallest stable filter for 1000) |
| RR target | **2.5** |
| Confirm bars | **CB2** |
| Direction | **LONG ONLY.** 1000 short is negative (WR=30%, AvgR=-0.04). |
| Early exit | **Kill losers at 30 minutes.** Only 12% of 30-min losers eventually win. 3.7x Sharpe improvement. |
| Do NOT | Kill at 10 min (too early, actively harmful). |

**Rolling eval:** STABLE (score 0.68-0.83, passes 13-16/19 windows). The MOST stable family.
**Nested 15m ORB:** 1000 is the ONLY session where 15m ORB beats 5m (+0.208R premium, 90% of pairs improve).

### 1800 Evening — Retrace Entry (GLOBEX Open)

| Parameter | Value |
|-----------|-------|
| Entry model | **E3** (limit at ORB edge) |
| Filter | **G4+** (G6 for tightest drawdown) |
| RR target | **2.0** (E3 gets better entry, so closer target is optimal) |
| Confirm bars | **CB2** (CB1-CB5 identical for E3, pick one) |
| Direction | Both (no strong directional bias with E3) |
| Early exit | **No systematic rule helps.** |
| Trade management | See 1800 E3 Management Rules below. |
| Do NOT | Use time-based cutoff (trades at 4h still +0.43R EV). Use breakeven trail. |

**Full-period stats (E3 CB5 RR2.0 G6+):** N=50, WR=52%, ExpR=+0.43, Sharpe=0.31, MaxDD=3.3R
**Rolling eval:** AUTO-DEGRADED (81% double-break rate). Only works in trending regimes.
**Regime warning:** Best Sharpe in full period, but structurally unreliable under rolling eval.

### 2300 Late Night — Retrace Entry

| Parameter | Value |
|-----------|-------|
| Entry model | **E3** (limit at ORB edge) |
| Filter | **G8+** (very selective, requires large ORB) |
| RR target | **1.5** |
| Confirm bars | **CB4** |
| Direction | Both |
| Early exit | **No early exit helps.** Baseline already marginal. |

**Full-period stats (E3 CB4 RR1.5 G8+):** N=50, WR=52%, TotalR=+12.3, AvgR=+0.25
**Rolling eval:** AUTO-DEGRADED (70% double-break rate).

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
| 1-2 min (quick touch) | **100%** | HOLD -- always recovers |
| 3-8 min | 39-55% | WATCH |
| 10+ min consecutive | **33%** | **EXIT** |
| 16+ min consecutive | **0-16%** | **DEFINITELY EXIT** |

Entry zone = within 0.15R of entry price (~2.4pt on 16pt ORB).

### No Time Cutoff
Trades still open at 4 hours: WR=56%, AvgR=+0.43R. Let stop/target work. The ONLY valid exits are the 30-min check and retrace dwell.

---

## Filters

### ORB Size (THE Edge)
The single most important variable. Win rate by 0900 ORB size (E1 RR2.5 CB2):

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
Abstain when volume is abnormally low. Fail-closed: missing data = ineligible.
0900: rejects 34.7% of days. Encoded in `config.py` as `VOL_RV12_N20`.

### Direction Filter
1000 session: LONG ONLY. Short breakouts negative (WR=30%, AvgR=-0.04).
0900: both directions positive, but long stronger (+0.50 vs +0.23 AvgR).

### Confirmed NO-GO Filters
- **VWAP direction**: Worse than baseline (-0.085 ExpR)
- **ADX overlay** (all regimes): Low-vol regime ExpR negative
- **ADX+VWAP combined**: Best in high-vol (+0.431 ExpR) but NO-GO overall (low-vol negative)
- **All overlays are regime amplifiers, not edge generators**

### Filter Deduplication Rules
Many filter variants produce the SAME trade set. Always report both counts.

**Key findings:**
- MNQ: 57 "strategies" = 16 unique trades (group by session, EM, RR, CB)
- G2/G3 pass 99%+ of days on most sessions -- they don't change the trade, just the label
- G8 genuinely filters 1100 session (29% cut), barely filters 1800 (4% cut)
- Some strategies are negative unfiltered -- filter IS the edge there
- Only G4+ filters meaningfully filter (>5% filter rate)

**Annualized Sharpe (ShANN):**
- Per-trade Sharpe is MEANINGLESS without trade frequency context
- ShANN = per_trade_sharpe * sqrt(trades_per_year)
- Minimum bar: ShANN >= 0.5 with 150+ trades/year
- Strong: ShANN >= 0.8 | Institutional: ShANN >= 1.0

---

## Live Portfolio Configuration

*Source: `trading_app/live_config.py`*

### Tier 1: CORE (always on)
| Family | Session | EM | Filter | Gate |
|--------|---------|-----|--------|------|
| 1000_E1_ORB_G3 | 1000 | E1 | G3+ | None (always trade) |

Rolling eval: STABLE (score 0.68-0.83). Positive in 13-16 of 19 windows.

### Tier 2: REGIME (fitness-gated)
| Family | Session | EM | Filter | Gate |
|--------|---------|-----|--------|------|
| 0900_E1_ORB_G4 | 0900 | E1 | G4+ | high_vol (must be FIT) |

Rolling eval: TRANSITIONING (score 0.42-0.54). Excellent when vol is high, negative when low.

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

## Research Registry

### Confirmed Edges
| Finding | Evidence | Status |
|---------|----------|--------|
| ORB size G4+ breakout | 313 validated strategies, walk-forward positive | DEPLOYED |
| 0900 kill losers at 15 min | +26% Sharpe, -38% MaxDD | PROVEN |
| 1000 kill losers at 30 min | 3.7x Sharpe, -35% MaxDD | PROVEN |
| 1000 LONG-ONLY filter | Short WR=30%, AvgR=-0.04 | PROVEN |
| 1800 E3 30-min position check | In-loss 24% WR vs in-profit 70% | PROVEN (discretionary) |
| 1800 E3 retrace dwell >10 min exit | 33% WR, 16+ min = 0-16% | PROVEN (discretionary) |
| Nested 15m ORB for 1000 | +0.208R premium, 90% pairs improve | VALIDATED |
| 1000 session most stable (rolling) | Score 0.68-0.83, 13-16/19 windows | VALIDATED |
| CB1-CB5 identical for E3 | 100% same entry price, 93-96% same outcome | PROVEN |
| IB 120m direction alignment | Opposed=3% WR (structural). Cross-validated 0900+1000. | VALIDATED |

### Confirmed NO-GOs
| Idea | Evidence | Do NOT revisit |
|------|----------|----------------|
| NO_FILTER / L-filters on 5m | 0/312 validated | Ever |
| Breakeven trail | -0.12 to -0.17 Sharpe all sessions | Ever |
| VWAP direction overlay | -0.085 ExpR vs baseline | Ever |
| RSI intraday reversion | N=4216, ExpR=-0.278 | Ever |
| Session fade (Asia->London) | N=777, ExpR=-0.162 | Ever |
| Gap fade | Avg gap 0.80pt, insufficient signal | Ever |
| Systematic retrace dwell | Kills winners at 1000; neutral/harmful pooled | As systematic rule |
| ADX all-regime overlay | Low-vol ExpR negative | Unless regime-gated |
| Inside Day filter | All WORSE vs G4 baseline on 0900/1000. N=17-27 on compound. | Ever |
| Ease Day filter (close vs typical) | All WORSE standalone. EASE+G4 1000 BEAT on tiny N=66-73, noise. | Ever |
| V-Box CLEAR filter (E1 momentum) | 0900 E1 G4: CLEAR +0.019 Sharpe, 98% trades CLEAR. Filters nothing. | As E1 filter |

### Pending / Inconclusive
| Idea | Status | Notes |
|------|--------|-------|
| Prior Day H/L fade | TIMED OUT | Likely NO-GO based on all reversal failures |
| ADX as regime-only overlay | High-vol: +0.431 ExpR | Only valid if you accept regime switching |
| 30m nested ORB | NOT BUILT | 15m done, 30m not yet populated |
| RSI directional confirmation | Low confidence | RSI 60+ LONG = +0.81R but N=24. Monitor. |
| Percentage-based ORB filter | Not tested | 0.15% of price approximates G5. Auto-adapts to price level. |
| E3 V-Box Inversion (retest into HVN) | 0900 E3 G4: CHOP +0.503 ExpR (Sharpe 0.357) vs CLEAR -0.094. N=127 CHOP. | Future E3-specific retest strategy. Retrace INTO high-volume node = high fill probability + high WR. |

### EOD Exit Tournament (2026-02-12)

Tested: replace fixed RR target with time-based exit (stop still active). Mark-to-market at cutoff.

**Method**: For each trade, scan bars from entry. Stop takes priority. If not stopped by cutoff, exit at bar close. PnL via canonical `to_r_multiple()`. All RR targets pooled (1.5/2.0/2.5), G4+, E1+E3.

**Session results (Sharpe delta vs fixed RR baseline):**

| Session | EM | fixed_rr | 1h | 2h | 4h | 7h | Winner |
|---------|-----|----------|------|------|------|------|--------|
| 0900 | E1 | **0.112** | -0.035 | +0.001 | +0.002 | +0.036 | Fixed RR |
| 0900 | E3 | **0.121** | +0.004 | +0.068 | +0.074 | +0.044 | Fixed RR |
| 1000 | E1 | +0.029 | +0.047 | +0.013 | +0.013 | **+0.078** | 7h runner |
| 1000 | E3 | +0.036 | -0.054 | +0.012 | +0.008 | +0.038 | ~Same |
| 1800 | E1 | -0.043 | -0.035 | -0.011 | -0.058 | -0.069 | All negative |
| 1800 | E3 | -0.013 | -0.040 | -0.058 | -0.119 | -0.101 | All negative |

**1000 E1 7h Runner deep-dive (CB2 RR2.0 G4+, N=138):**

| Metric | Fixed RR | 7h Runner |
|--------|----------|-----------|
| N | 138 | 138 |
| WR | 38.4% | 29.7% |
| ExpR | +0.031 | +0.200 |
| Sharpe | 0.024 | 0.093 |
| MaxDD | -13.5R | -18.4R |
| TotalR | +4.3 | +27.6 |
| Exit dist | - | 68% stopped, 31% time exit |

**CRITICAL: Regime-dependent.** Yearly Sharpe delta (7h minus fixed):

| Year | N | Fixed Sharpe | 7h Sharpe | Delta |
|------|---|-------------|-----------|-------|
| 2020 | 14 | +0.372 | -0.055 | -0.428 |
| 2024 | 5 | +1.067 | +0.042 | -1.025 |
| 2025 | 93 | -0.035 | +0.057 | +0.092 |
| 2026 | 21 | -0.067 | +0.260 | +0.327 |

16/17 trades with >3R are from 2025-2026 (gold $3200-5200, trending hard).

**Verdict:**
- **0900**: Fixed RR wins. Do not change exit logic.
- **1000 7h runner**: Real edge in trending regime (2025+). LOSES in choppy regime (2020, 2024). **REGIME OVERLAY ONLY** -- not a universal replacement.
- **1800**: All exit types negative. Confirmed dead for E1.
- **1h/2h/4h**: No consistent improvement anywhere. Dead.

### IB Direction Alignment (2026-02-12)

**Initial Balance (IB)** = high/low of first N minutes after session open.
After IB forms, the first break of IB high or IB low determines IB direction.
If IB direction matches ORB break direction = **aligned**. Otherwise = **opposed**.

**Setup**: E1 CB2 RR2.0 G4+, IB=120m (fixed, no per-session optimization).

**Core finding**: Opposed trades have ~3% WR on 7h hold. This is mechanical --
if the wider market structure (IB) breaks against your ORB trade, you lose.

**Cross-validation** (0900 was BLIND test -- same 120m IB, zero re-optimization):

| Session | Group | N | WR | ExpR | Sharpe | MaxDD | TotalR |
|---------|-------|---|-----|------|--------|-------|--------|
| 1000 | All fixed RR | 138 | 38.4% | +0.031 | 0.024 | -13.6 | +4.3 |
| 1000 | Aligned 7h | 83 | 47.0% | +0.896 | 0.361 | -6.4 | +74.4 |
| 1000 | Opposed 7h | 54 | 3.7% | -0.860 | -1.158 | -45.4 | -46.4 |
| 0900 | All fixed RR | 148 | 45.3% | +0.229 | 0.169 | -7.5 | +33.8 |
| 0900 | Aligned 7h | 84 | 60.7% | +0.858 | 0.435 | -5.7 | +72.1 |
| 0900 | Opposed 7h | 63 | 3.2% | -0.876 | -1.198 | -46.7 | -55.2 |

**Blended strategy** (aligned=7h hold, opposed=fixed RR):

| Session | Fixed Sharpe | Blended Sharpe | Delta |
|---------|-------------|----------------|-------|
| 1000 | 0.024 | 0.174 | +0.150 |
| 0900 | 0.169 | 0.144 | -0.025 |

**Key findings**:
1. **Direction alignment is structural**: works on both sessions, ~3% opposed WR on both
2. **As a FILTER** (only trade aligned): strong on both (Sharpe 0.36+ aligned-only)
3. **As a HOLD/EXIT switch**: helps at 1000 (weak baseline), neutral at 0900 (strong baseline)
4. **Kill switch (opposite IB side)**: redundant with standard stop -- IB range contains stop level
5. **Signal wait**: median 138m after entry (2+ hours to know alignment)

**Yearly stability (1000 blended)**:

| Year | N | Aligned | Opposed | Fixed | Blended | Best |
|------|---|---------|---------|-------|---------|------|
| 2020 | 14 | 9 | 5 | 0.372 | 0.188 | fixed |
| 2024 | 5 | 3 | 2 | 1.067 | 0.611 | fixed |
| 2025 | 93 | 50 | 42 | -0.035 | 0.129 | blended |
| 2026 | 21 | 17 | 4 | -0.067 | 0.302 | blended |

**Regime dependence**: Blended wins in trending markets (2025-2026), fixed wins in choppy (2020).
Do not try to "fix" choppy years -- accept this requires a trending regime.

**Slippage**: Already in cost model ($8.40 RT friction per trade). Verified: `to_r_multiple()` deducts friction.

**Scripts**: `scripts/analyze_ib_alignment_v2.py` (canonical), `scripts/audit_ib_single_break.py` (audit trail).

---

## Regime Warnings

### The 2025 Shift
Gold volatility has changed dramatically:

| Year | Median 0900 ORB | G5+ Days (0900) |
|------|-----------------|-----------------|
| 2021-2024 | 0.7-1.0pt | 2-9/year (1-3%) |
| 2025 | 2.5pt | 70/year (27%) |
| 2026 (Jan) | 14.8pt | 21/month (88%) |

- 313 validated strategies **collapse to 2 truly STABLE families** under rolling evaluation
- G4+ filters show edge in 2025H1/2026 but NOT 2025H2
- G6 filter at 0900: 1-8 eligible days/year in 2016-2024, then 50 in 2025
- **If gold returns to $1800 levels, G5+ days drop to ~5/year (untradeable frequency)**

### Rolling Evaluation Summary
| Family | Score | Windows Passed | Status |
|--------|-------|----------------|--------|
| 1000_E2_G2 | 0.83 | 16/19 | STABLE |
| 1000_E1_G2 | 0.68-0.78 | 13-15/19 | STABLE |
| 0900_G3+ families | 0.42-0.54 | 9-11/19 | TRANSITIONING |
| 1800/2300/1100/0030 | <0.30 | AUTO-DEGRADED | Double-break >67% |

### Double-Break Frequency
| Session | Rate | Implication |
|---------|------|------------|
| 0900 | 57% | Survives double-break filter |
| 1000 | 57% | Survives |
| 1100 | 74% | Structurally mean-reverting |
| 1800 | 81% | Breakout fails most of the time |
| 2300 | 70% | Unreliable for breakout |
| 0030 | 76% | Unreliable for breakout |

---

## Cost Model

| Component | Amount |
|-----------|--------|
| Commission | $2.40/RT |
| Spread | $2.00/RT (1 tick x $10) |
| Slippage | $4.00/RT |
| **Total friction** | **$8.40/RT = 0.84pt** |

Friction as % of risk: 42% on 2pt ORB, 21% on 4pt, 14% on 6pt, 8.4% on 10pt.
This is why small ORBs lose -- friction eats the edge.

---

## Reference Documents (Deep-Dive)

| Document | Content |
|----------|---------|
| `MARKET_PLAYBOOK.md` | Full backtest results, tables, yearly breakdowns |
| `artifacts/TRADE_MANAGEMENT_RULES.md` | 1800 E3 management rules with full data tables |
| `artifacts/EARLY_EXIT_RULES.md` | Early exit research with per-session breakdowns |
| `trading_app/live_config.py` | Declarative live portfolio (code) |
| `trading_app/config.py` | Filter definitions, entry models (code) |

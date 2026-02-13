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
| 1100 early exit (30m time stop) | ExpR -0.051 -> -0.028 (still negative). MaxDD 25% tighter. Kills 21 winners per 129 cuts. | Unless edge found |

### 1100 Session Status: SHELVED (2026-02-13)

**Status**: No tradeable edge in current regime. Shelved, not permanent NO-GO.

**Research conducted**:
1. **Zero-lookahead signals**: Tested 0900 alignment, gap direction, prior day trend, ATR regime, 0900 ORB size. **None produced actionable signal.** Best split (G8+ aligned vs opposed) has N=42/56 -- too thin, yearly inconsistent.
2. **Timed early exit (15/20/30/45/60m)**: 30m is best threshold. Cuts 37% of trades, saves 0.078R/cut. ExpR improves from -0.051 to -0.028 -- **still negative**. MaxDD improves 25%. Not enough to flip session positive.
3. **Midday reversal to Asia open**: Only 12-16% of trades cross back through Asia open by 12:30. No systematic reversal pattern.
4. **MTM trajectory**: 73% of trades underwater at 11:15 (CB4 chase cost). Winners recover by 11:30, losers never do. But killing losers doesn't generate enough savings vs winners killed.

**Root cause**: 1100 has 74% double-break rate (structurally mean-reverting). Breakout strategy fights session structure. The problem isn't fat-tail losers -- it's insufficient winners.

**Revisit conditions**: New regime with lower double-break rate, or new entry model (fade/reversal) that exploits mean-reversion.

**Hypotheses tested and failed**:
- 0900 break direction alignment: No signal (yearly inconsistent, G8+ N too thin)
- Gap direction alignment: Mild (+0.036 vs -0.049 ExpR at RR2.0), Sharpe 0.027 = noise
- Prior day trend alignment: Reversed (opposed does better)
- ATR regime split: Tiny (+0.019 vs -0.034), not actionable
- 0900 ORB size as trend proxy: Zero signal
- 0900 IB (120m) direction alignment: DOES NOT TRANSFER to 1100. Opposed WR 27.9% (vs 3% on 0900/1000). Reason: IB range ends at 11:00 = 1100 ORB start, so IB and ORB measure same price action (71% same direction). Not independent signals.
- Timed early exit 30m: MaxDD -25% but ExpR still negative (-0.028). Kills 21 winners per 129 cuts.
- Midday reversal to Asia open: Only 12-16% cross back. No systematic reversal.

**Scripts**: `scripts/analyze_1100_zero_lookahead.py`, `scripts/analyze_1100_midday_reversal.py`, `scripts/backtest_1100_early_exit.py`, `scripts/analyze_1100_ib_alignment.py`.

### Timed Early Exit Research (2026-02-13)

**Rule**: At N minutes after fill, if bar close vs entry is negative, exit at bar close.

| Session | Threshold | Sharpe Delta | MaxDD Delta | Status |
|---------|-----------|-------------|-------------|--------|
| 0900 | 15 min | **+26%** | **-38%** | IMPLEMENTED |
| 1000 | 30 min | **+270%** (3.7x) | **-35%** | IMPLEMENTED |
| 1100 | 30 min | +34% (still negative) | -25% | NOT IMPLEMENTED |
| 1800 | any | No benefit | - | NOT IMPLEMENTED |
| 2300 | any | No benefit | - | NOT IMPLEMENTED |
| 0030 | any | No benefit | - | NOT IMPLEMENTED |

Config: `trading_app/config.py` -> `EARLY_EXIT_MINUTES` (0900=15, 1000=30, others=None).
Scripts: `scripts/backtest_1100_early_exit.py`, `scripts/analyze_1100_zero_lookahead.py`, `scripts/analyze_1100_midday_reversal.py`.

### Profit Factor Screen (2026-02-12)

**Method**: Recomputed metrics from `orb_outcomes` for 2024-02-12 to 2026-02-04 (2 years).
Screened for PF 1.5-2.0, ShANN 0.8-1.5, WR <= 75%, N >= 30.

**Result**: 26 "strategies" matched -- but they are **4 trade families, not 26 independent edges.**

**Family-level results (honest reporting -- averaged across all RR/CB variants that passed):**

| Family | Unique Days | Avg N | Avg WR | Avg PF | Avg ExpR | Avg ShANN | Regime |
|--------|------------|-------|--------|--------|----------|-----------|--------|
| 0900 E1 G5 | ~88 | 87 | 39% | 1.56 | +0.34R | 1.32 | High vol only |
| 0900 E1 G6 | ~64 | 64 | 39% | 1.59 | +0.36R | 1.19 | High vol only |
| 0900 E1 G8 | ~41 | 41 | 42% | 1.64 | +0.38R | 1.03 | High vol only |
| 1800 E1+E3 G6 | ~47 | 47 | 51% | 1.63 | +0.32R | 1.12 | High vol only |

**Critical findings:**
1. **98% of trades are from 2025-2026.** The 2-year window IS the hot streak. G6 at 0900 produced 1-8 eligible days/year in 2016-2024, then 50 in 2025.
2. **Tighter filters look better but are less trustworthy.** G8 (N=41) has best per-trade PF but widest confidence interval. G5 (N=88) is the most honest signal.
3. **All 4 families are one trade idea:** ORB breakout on high-vol expansion days. The filter level, RR, and CB are parameter knobs, not independent edges.
4. **Minimum Track Record Length**: Sharpe 1.0 needs ~3 years to be statistically significant. With 2 years of hot data, the confidence interval is too wide to bet the house.

**Honest log:**
- Strategies relying on G6+ filters are dormant in low volatility (2016-2024).
- Effectiveness is 100% correlated to Volatility Expansion (2025-2026).
- Unique Trade Count is low (~4 families), indicating one market behavior (High Vol Breakout), not multiple edges.
- Do NOT optimize 2021/2024 to be profitable. That is iterated out-of-sample overfitting.

### Pending / Inconclusive
| Idea | Status | Notes |
|------|--------|-------|
| Prior Day H/L fade | TIMED OUT | Likely NO-GO based on all reversal failures |
| ADX as regime-only overlay | High-vol: +0.431 ExpR | Only valid if you accept regime switching |
| 30m nested ORB | NOT BUILT | 15m done, 30m not yet populated |
| RSI directional confirmation | Low confidence | RSI 60+ LONG = +0.81R but N=24. Monitor. |
| Percentage-based ORB filter | Not tested | 0.15% of price approximates G5. Auto-adapts to price level. |
| E3 V-Box Inversion (retest into HVN) | 0900 E3 G4: CHOP +0.503 ExpR (Sharpe 0.357) vs CLEAR -0.094. N=127 CHOP. | Future E3-specific retest strategy. Retrace INTO high-volume node = high fill probability + high WR. |
| ATR regime on/off switch | **NO-GO** as pre-trade filter. Fails robustness (threshold-sensitive, family-inconsistent, inflated by prior lookahead). ATR kept for **position sizing only** (Turtle-style vol normalization). See Regime Switch section. |
| Low-vol counterbalance strategy | NOT RESEARCHED | Mean-reversion partner for the High Vol Breakout system. |

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

## Reporting Rules

### Family-Level Reporting (MANDATORY)

**Never cite the performance of a single parameter variant.** Always report the family average.

A "family" = one unique combination of `(session, entry_model, filter_level)`. All RR/CB variants within a family share 85-100% of the same trade days. They are parameter knobs on one trade idea, not independent strategies.

**Correct**: "The 0900 E1 G5 family averages +0.34R ExpR across RR/CB variants."
**Wrong**: "MGC_0900_E1_RR3.0_CB2_ORB_G6 has +0.39R ExpR."

If the family average is robust, the edge is real. If only one specific RR/CB combo works, it's noise.

**Dedup rule**: When counting strategies, group by `(session, EM, filter_level)` first. Report unique family count, not parameter variant count. 334 validated strategies = ~12 families. 26 PF-screen matches = 4 families.

### Regime On/Off Switch (NOT YET BUILT)

The G5/G6 breakout families bleed in low-vol regimes and print in high-vol regimes. They need an automatic on/off switch.

**Concept**: Pre-trade check using rolling ATR or rolling median ORB size:
- `IF rolling_ATR(20) < threshold THEN NO TRADE` for G5+ families
- This prevents the 2021/2024 bleed without overfitting entry logic
- The G4+ ORB size filter is a per-DAY vol check. ATR adds a per-REGIME vol check (are we in an expansion period at all?).

**What we already have (implicit regime switches):**
- ORB size filter (G4/G5/G6): per-day volatility gate. Works, but dormant for months in quiet regimes.
- Strategy fitness (`strategy_fitness.py`): 3-layer structural + rolling + decay check. Already gates the live portfolio.
- `live_config.py` regime tier: 0900 is gated by `high_vol` fitness status.

**What's missing:**
- ~~No ATR column in `daily_features`.~~ DONE (2026-02-12). `atr_20` column added and backfilled.
- No rolling median ORB size. Can compute from `orb_XXXX_size`.
- No explicit "regime OFF" state that prevents all G5+ trading for weeks/months.
- No low-vol counterbalance strategy in the portfolio (mean-reversion partner).

**ATR(20) by year — the regime map:**

| Year | ATR(20) | Med 0900 ORB | G4+ Days | G6+ Days | G4% | G6% |
|------|---------|-------------|----------|----------|-----|-----|
| 2016 | 17.4 | 0.60 | 7 | 1 | 2.6% | 0.4% |
| 2017 | 12.0 | 0.60 | 2 | 1 | 0.7% | 0.3% |
| 2018 | 11.5 | 0.40 | 1 | 1 | 0.3% | 0.3% |
| 2019 | 14.8 | 0.50 | 2 | 0 | 0.7% | 0.0% |
| 2020 | 30.6 | 1.20 | 25 | 9 | 8.6% | 3.1% |
| 2021 | 22.6 | 0.70 | 13 | 8 | 4.5% | 2.7% |
| 2022 | 24.3 | 0.90 | 11 | 7 | 3.8% | 2.4% |
| 2023 | 22.7 | 0.90 | 6 | 1 | 2.1% | 0.3% |
| 2024 | 30.9 | 1.00 | 11 | 5 | 3.8% | 1.7% |
| **2025** | **57.1** | **2.50** | **83** | **50** | **28.4%** | **17.1%** |
| **2026** | **110.3** | **14.80** | **24** | **20** | **100%** | **83.3%** |

**Structural breakpoints visible:**
- ATR < 20: Dormant. 0-2 G4+ days/year. ORB breakout strategies are OFF.
- ATR 20-30: Marginal. 6-25 G4+ days/year. Enough for G4 but not G6.
- ATR > 30: Active. G4+ days appear regularly. This is where the edge lives.
- ATR > 50: Expansion. G6+ days become frequent. PF screen strategies activate.

**ATR Regime Gate Backtest Results (2026-02-13, lag-fixed ATR):**

Tested ATR(20) thresholds [0, 20, 25, 30, 35, 40] across 8 families, 10 years data.
Fixed variant set (no survivorship bias). Family-level averages across RR/CB variants.
ATR(20) is now lag-fixed (excludes current day's True Range -- no lookahead).

| Finding | Detail |
|---------|--------|
| **0900_G4: ATR >= 25 marginal** | +2.87R at 25, but -1.05R at 30. Threshold-sensitive = fragile. |
| **0900_G5: ATR gating HURTS** | Negative at 25 (-0.29R), 30 (-1.68R). Only helps at 35 (+2.06R). |
| **0900_G6: ATR gating HURTS** | Negative at most thresholds. +1.41R only at 35. |
| **1000_G3: ATR >= 30 helps** | +4.55R. But ShANN only goes from 0.009 to 0.073 -- still untradeable. |
| **1000_G4: ATR gating HURTS** | Monotonically worse: -2.25R at 30, -7.36R at 40. |
| **1800 families: mixed** | E3 G4 stays negative at all thresholds. G6 improves but N drops to 44-50. |
| **ATR threshold frequency** | >= 20: 55.9%, >= 25: 35.6%, >= 30: 22.1%, >= 40: 12.3% of days |

**Honest caveats:**
- Previous results (2026-02-12) used ATR with lookahead (included current day's TR). Lag-fixed results are weaker.
- Selecting per-family thresholds from 6 values IS curve fitting. No single threshold works across families.
- 48 cells tested (6 thresholds x 8 families). Improvements are threshold-sensitive and non-robust.
- ATR >= 30 excludes 78% of trading days -- massive opportunity cost on positive-ExpR strategies.

**Decision (2026-02-13): NO hard ATR pre-trade gate.**
- DISCARDED: The ATR >= 25/30 gate fails robustness. Results are threshold-sensitive, family-inconsistent, and were inflated by lookahead in prior testing.
- KEEP ATR for POSITION SIZING: Turtle-style vol normalization (`compute_vol_scalar` + `compute_position_size_vol_scaled` in portfolio.py). High ATR = smaller size, low ATR = larger size. Normalizes risk without deleting trades.
- KEEP Rolling Fitness: `strategy_fitness.py` + `live_config.py` regime tier remains the regime gate. Multi-layer assessment (structural + rolling + decay) is more principled than a single ATR cutoff.

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

---

## Alternative Strategies Research (Feb 2026) -- ALL NO-GO

Tested 4 non-ORB strategies for diversification. Walk-forward OOS on 18 months (Aug 2024 - Feb 2026). Scripts in `research/`, artifacts in `artifacts/`.

### Gap Fade (Mean Reversion)
- **NO-GO.** Gold doesn't gap enough. Only 5-17 qualifying trades at best threshold (0.4x ATR). Walk-forward produced zero qualifying windows. Dead on arrival.

### VWAP Pullback (Trend Continuation)
- **NO-GO.** Abundant signals (642 OOS trades) but no edge. ExpR=-0.012R, ShANN=-0.33. Negative in both low-vol and high-vol regimes. VWAP pullbacks in gold are noise.

### Value Area Breakout (Volume Profile)
- **NO-GO after audit.** Initial OOS looked promising (N=365, ExpR=+0.085, ShANN=1.41) but:
  - First 5 months (Aug-Dec 2024): **-8.2R**. Edge only appeared in high-vol 2025. Regime-dependent, not structural.
  - Only 50% of walk-forward windows positive (coin flip).
  - Half the total profit came from 2 outlier months (Sep 2025: +16.2R alone).
  - After $8.40 RT friction: net edge = $10.50/trade on $222 risk. Profit/MaxDD ratio < 1.
  - Value Area Reversion mode was separately negative (N=82, ExpR=-0.024R).

### Concretum Bands (Dynamic Volatility Breakout)
- **NO-GO after audit.** Initial OOS looked strong (N=338, ExpR=+0.084, ShANN=1.14, regime-stable) but:
  - **It's ORB in a hat.** BM=0.5 sigma triggers on 89% of trading days -- same days ORB trades. Not a diversifier.
  - Only 61% of windows positive. Half the profit from 2 outlier months.
  - Parameter instability: switched from BM0.5 to BM2.0 in late 2025.
  - After friction: net edge = $7.80/trade. MaxDD=$2,934 vs total profit=$2,640. You lose more in drawdown than you make.

### Why They All Failed
1. **ORB breakout IS the edge in gold.** Alternative entry methods (VWAP, volume profile, volatility bands) don't find new edge -- they either replicate ORB or find nothing.
2. **Gap Fade requires gapping markets.** Gold is too continuous (23h/day trading) for meaningful gaps.
3. **ShANN can lie.** High trade frequency inflates annualized Sharpe even when per-trade edge is thin. Always check: (a) leave-one-out stability, (b) friction-adjusted dollars, (c) % positive windows.
4. **Regime-dependent results are not tradeable** unless you have a reliable regime detector. "Works in 2025" is not a strategy.

### Key Takeaway
Stop looking for alternative entries. The diversification path is **more instruments** (MNQ, other futures) or **overlay filters** (regime detection on existing ORB), not new strategy types on the same instrument.

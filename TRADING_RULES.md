# Trading Rules — Gold Futures (GC/MGC)

Single source of truth for live trading. All claims verified against gold.db.
For research deep-dives and data tables, see `docs/RESEARCH_ARCHIVE.md`.

---

## Glossary

### Sessions (ORB Labels — 13 total)

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

**Dynamic sessions (6, DST-aware — times shift with daylight saving):**
| Code | Plain English | Reference TZ |
|------|--------------|-------------|
| CME_OPEN | CME daily open | 5:00 PM CT |
| US_EQUITY_OPEN | US equity open | 9:30 AM ET |
| US_DATA_OPEN | US economic data | 8:30 AM ET |
| LONDON_OPEN | London open | 8:00 AM UK |
| US_POST_EQUITY | US post-equity-open | 10:00 AM ET |
| CME_CLOSE | CME equity pre-close | 2:45 PM CT |

**Aliases:** TOKYO_OPEN → 1000, HK_SG_OPEN → 1130
**Module:** `pipeline/dst.py` (SESSION_CATALOG + resolvers)
**Per-asset enabled sessions:** `pipeline/asset_configs.py`

**DST Audit (Feb 2026):** Fixed sessions 0900/1800/0030 drift ±1 hour vs actual market opens during DST (US ~Mar-Nov, UK ~Mar-Oct). Dynamic sessions track the real event. Initial audit showed fixed slightly outperformed dynamic on matched summer days (+0.13R MGC, +0.14R MES), and winter edges are stronger (MES +0.35R, MGC +0.18R, MNQ +0.09R). **CAUTION: do NOT rule out dynamic sessions or event-time ORBs.** In winter, fixed = event time (identical), so the comparison is summer-only. Both fixed AND event times may have independent edge — and there may be undiscovered profitable times at other clock positions. A full 24-hour ORB time scan (`research/research_orb_time_scan.py`) is pending to answer this definitively. Full results: `docs/RESEARCH_ARCHIVE.md`.

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
1. **ORB size IS the edge — not sessions, not parameters.** Cross-instrument stress test (Feb 2026) proved this: strip the size filter and ALL edges die. CB, RR, entry model are refinements on top of the one thing that matters: was the ORB big enough? See "ORB Size = The Edge" section below.
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
| Cross-instrument 1000 LONG portfolio (MGC+MNQ+MES) | NO-GO for diversification. MNQ/MES correlation = +0.83 (same trade). MGC/equity = +0.40-0.44 (moderate, not free). Adding MNQ+MES to 1000 LONG worsens Sharpe. Pick ONE equity micro, don't stack. Need truly uncorrelated asset for portfolio diversification. |

> **Key insight**: Gold trends intraday more than it mean-reverts. The only confirmed edges are momentum breakouts with size filters. Full research details: `docs/RESEARCH_ARCHIVE.md`.

---

## DST CONTAMINATION WARNING (Feb 2026)

**Stats below for sessions 0900, 1800, 0030, 2300 are BLENDED winter+summer averages.** Three sessions (0900/1800/0030) align with their market event in winter but miss by 1 hour in summer. 2300 is a special case — it NEVER aligns with the US data release but DST flips which side it sits on.

| Session | Winter (std time) | Summer (DST) | Shift |
|---------|------------------|--------------|-------|
| 0900 | = CME open exactly | CME opened 1hr ago (at 0800) | US DST |
| 1800 | = London open exactly | London opened 1hr ago (at 1700) | UK DST |
| 0030 | = US equity open exactly | Equity opened 1hr ago (at 2330) | US DST |
| 2300 | 30min BEFORE US data (data at 23:30) | 30min AFTER US data (data at 22:30) | US DST |

**2300 detail:** 2300 Bris = 13:00 UTC always. US data 8:30 ET = 13:30 UTC winter / 12:30 UTC summer. Winter = pre-positioning (less volume). Summer = post-data reaction (+76-90% volume). Edge is WINTER-DOM on MGC G8+ — the quieter pre-data window produces better breakouts.

**Sessions 1000, 1100, 1130 are CLEAN** — Asia has no DST. Dynamic sessions are also clean (resolvers adjust).

**Remediation status (Feb 2026 — DONE):**
- ✅ Winter/summer split baked into `strategy_validator.py` (DST columns on both strategy tables)
- ✅ 1272 strategies re-validated: 275 STABLE, 155 WINTER-DOM, 130 SUMMER-DOM, 10 SUMMER-ONLY. No validated strategies broken.
- ✅ Volume analysis confirms event-driven edges (`research/output/volume_dst_findings.md`)
- ✅ 24-hour ORB time scan with winter/summer split (`research/research_orb_time_scan.py`)
- ✅ Red flags: all 10 are MES 0900 E1 experimental (never in production). MGC 1800 confirmed STABLE.
- Rule: ALL future research at 0900/1800/0030/2300 MUST split by DST regime and report both halves.

---

## Session Playbook

### 0900 Asia Open — Momentum Breakout
**⚠️ DST-AFFECTED: Winter = CME open exactly. Summer = 1hr after CME open (63% volume drop).**
**DST revalidation (Feb 2026 — DONE):** MGC 0900 validated strategies are mostly STABLE or WINTER-DOMINANT. **MES 0900 is TOXIC in winter** — all 10 red-flag strategies (edge dies in one regime) are MES 0900 E1. Winter = when 0900 aligns with CME open exactly. The event alignment hurts MES breakouts. Volume analysis confirms 0900's edge is the CME open event (63% volume drop in summer when CME opens at 0800 instead).

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
**⚠️ DST-AFFECTED: Winter = London open exactly. Summer = 1hr after London open. MGC volume 31% higher in winter.**
**DST revalidation (Feb 2026 — DONE):** MGC 1800 validated strategies confirmed **STABLE** using UK DST split. E1 RR2.0 CB1 G5: +1.59W(15) / +1.54S(13). E3 RR2.0 CB1 G5: +1.57W(15) / +1.48S(11). Edge is genuine year-round. London is the primary gold market — MGC shows the strongest London-open sensitivity.

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
**⚠️ DST-AFFECTED: Winter = 30min pre-US-data (quieter, fewer trades). Summer = 30min post-US-data (higher volume). DST revalidation (Feb 2026): MGC 2300 validated strategies are WINTER-DOM. 4 validated strategies (all G8+). Edge works in the pre-data positioning window.**

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

### ORB Size = The Edge (CONFIRMED Feb 2026)

**This is the single most important finding in the project.**

Cross-instrument stress test (`scripts/tools/stress_test.py`) tested 11 top edges across MES, MGC, and MNQ. Results:
- **ALL 6 MES/MGC edges: DID NOT SURVIVE.** Failed size filter, YoY stability, and parameter sensitivity.
- **ALL 5 MNQ 0900 edges: SURVIVED or MOSTLY SURVIVED.** Positive all 3 years, bidirectional, ALL real parameter neighbors profitable.

The pattern across ALL instruments: strip the ORB size filter → edge dies. Small ORBs lose, large ORBs win. CB, RR, and entry model are second-order refinements. The size filter IS the strategy.

**Mechanism:** Friction as % of risk. A 2pt ORB has 42% friction (MGC). A 6pt ORB has 14%. Small ORBs don't have enough room for price to move beyond friction + noise. Large ORBs give the breakout space to work. This is arithmetic, not statistics — it persists as long as cost structure exists.

**Implication:** Stop chasing parameter combos. The G5+/G6+ gates are not "nice to have" — they ARE the strategy. Future research should focus on optimal size thresholds per session per instrument, not on CB/RR/EM tuning.

Win rate by 0900 ORB size (E1 RR2.5 CB2, MGC):

| ORB Size | N | WR | AvgR |
|----------|---|-----|------|
| < 2pt | 948 | 27% | -0.47 |
| 2-4pt | 144 | 26% | -0.30 |
| 4-6pt | 49 | 39% | +0.19 |
| 6-10pt | 42 | 50% | +0.59 |
| 10-20pt | 20 | 37% | +0.23 |

**G4+ maximizes total R. G5+/G6+ maximize per-trade edge.** No validated strategy exists without G4+.

**Deep Dive Results (Feb 2026)** — `scripts/tools/orb_size_deep_dive.py`:

Instrument-specific optimal gates (maximizing total R):
| Instrument | Session | Optimal Gate | Notes |
|-----------|---------|-------------|-------|
| MNQ | 0900 | G4+ | Deep dive suggested G3+ but hypothesis test (H1) disproved: 3-4pt band adds only 9 trades. |
| MNQ | 1000 | G4+ | Same — G3 band is 3 trades, all losses. |
| MES | 0900 | G3+ (NO cap) | Hypothesis test (H2) showed 12pt+ is BEST zone on 0900 (N=7, WR=85.7%). Do NOT cap. |
| MES | 1000 | G4+ with L12 cap | Hypothesis test (H2) confirmed: 12pt+ is TOXIC (N=9, avgR=-0.68). Band G4-L12 improves both avgR and totR. |
| MGC | 0900 | G5+ | Confirmed. Highest total R at G5. |
| MGC | 1000 | G5+ | Confirmed. |
| MGC | 1800 | G8+ | Only very large ORBs work in evening. |
| MGC | 2300 | G12+ or DEAD | Kill for MGC — consider removing from enabled sessions. |

Friction hypothesis (partially disproven):
- Theory: higher friction instruments need bigger ORBs. WRONG in absolute terms.
- MGC has LOWEST friction (0.84pt) but needs BIGGEST gate (G5+).
- MNQ has higher friction (1.37pt) but works from G4+.
- Reason: "points" mean different things. 5pt on MGC = 0.17% of price. 5pt on MNQ = 0.024%.
- TESTED & REJECTED: Percentage-based filter (H3) and ORB/ATR ratio (H4) do not beat instrument-specific fixed gates. See `docs/RESEARCH_ARCHIVE.md`.

Direction bias — hypothesis test confirmed for 1000 session (H5):
- **1000 session: LONG-ONLY confirmed across all 3 instruments.** MGC shorts are negative (-0.09 avgR). MNQ shorts near zero (+0.03). LONG-ONLY doubles avgR.
- 0900 session: Both directions work. Difference too small to justify halving N. Keep bidirectional.
- 1800 session: Dead regardless of direction. No filter rescues it.

MES 1000 upper cap (hypothesis test H2 confirmed):
- 12pt+ ORBs on S&P 1000 session are TOXIC (N=9, avgR=-0.68, WR=11.1%). Mean-reversion trap.
- Band filter G4-L12 beats uncapped G4+ on BOTH avgR (+0.36 vs +0.28) AND totR (+40.3 vs +34.2).
- Band filter G5-L12 is best per-trade: avgR=+0.52, N=67.
- **Exception:** MES 0900 12pt+ is the BEST zone (avgR=+1.47). Do NOT apply cap to 0900.

**Hypothesis test results:** `scripts/tools/hypothesis_test.py` — full results in `docs/RESEARCH_ARCHIVE.md`.
| Hypothesis | Verdict | Action |
|-----------|---------|--------|
| H1: G3 for MNQ | NO-GO | Band 3-4pt adds 9 trades (noise). Stick with G4+. |
| H2: MES 12pt cap | PASS (1000 only) | Add G4-L12, G5-L12 to MES 1000 grid. Do NOT cap 0900. |
| H3: % of price filter | NO-GO | Not better than tuned fixed gates. |
| H4: ORB/ATR ratio | HARD NO-GO | Scale mismatch — ORBs are 5-15% of daily ATR. |
| H5: Direction filter | PASS (1000) | Add LONG-ONLY for 1000 across MGC, MNQ, MES. |

### Volume Filter (Relative Volume)
`rel_vol = break_bar_volume / median(same UTC minute, 20 prior days)`
Fail-closed: missing data = ineligible. 0900 rejects 34.7% of days. Config: `VOL_RV12_N20`.

### Direction Filter (Hypothesis Test H5 — confirmed Feb 2026)
**1000 session: LONG ONLY across all instruments.**

| Instrument | LONG avgR | LONG N | SHORT avgR | SHORT N | BOTH avgR |
|-----------|----------|--------|-----------|---------|----------|
| MGC | +0.66 | 38 | -0.09 | 27 | +0.35 |
| MNQ | +0.26 | 178 | +0.03 | 203 | +0.14 |
| MES | +0.19 | 81 | +0.06 | 106 | +0.12 |

MGC 1000 shorts are actively negative. MNQ 1000 shorts are near-zero noise. LONG-ONLY doubles avgR for all three.

**0900 session: KEEP BIDIRECTIONAL.** Both directions positive. Difference too small (MGC: +0.87 LONG vs +0.63 SHORT) to justify halving trade count.

**1800 session: Dead regardless of direction.** MES 1800 is -0.23 LONG, -0.15 SHORT. No direction filter rescues a dead session.

### Filter Deduplication Rules
Many filter variants produce the SAME trade set.

- G2/G3 pass 99%+ of days on MGC — cosmetic label, not real filtering for gold
- G3 on MNQ was hypothesized as real but **disproved (H1)**: 3-4pt band adds only 9 trades on 0900, all other sessions negative. G3 is cosmetic on MNQ too (different reason: instrument lives at 12pt+).
- For MGC: only G4+ filters meaningfully filter (>5% filter rate)
- For MNQ/MES: G4+ is the minimum meaningful gate
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

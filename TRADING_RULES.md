# Trading Rules — Multi-Instrument Futures (MGC/MNQ/MES/M2K)

Single source of truth for live trading. All claims verified against gold.db.
For research deep-dives and data tables, see `docs/RESEARCH_ARCHIVE.md`.

---

## Glossary

### Sessions (ORB Labels — 10 total, all event-based)

All sessions are now dynamic/event-based. Times are resolved per-day from `pipeline/dst.py` SESSION_CATALOG, eliminating DST contamination.

| Code | Event | Reference TZ | Legacy Name |
|------|-------|-------------|-------------|
| CME_REOPEN | CME daily reopen | 5:00 PM CT | was 0900, CME_OPEN |
| TOKYO_OPEN | Tokyo open | 10:00 AM AEST | was 1000 |
| SINGAPORE_OPEN | Singapore/HK open | 11:00 AM AEST | was 1100, 1130 |
| LONDON_METALS | London metals open | 8:00 AM UK | was 1800, LONDON_OPEN |
| US_DATA_830 | US economic data release | 8:30 AM ET | was 2300, US_DATA_OPEN |
| NYSE_OPEN | NYSE equity open | 9:30 AM ET | was 0030, US_EQUITY_OPEN |
| US_DATA_1000 | US 10 AM data/post-open | 10:00 AM ET | was US_POST_EQUITY |
| CME_PRECLOSE | CME equity pre-close | 2:45 PM CT | was CME_CLOSE |
| COMEX_SETTLE | COMEX gold settlement | 1:30 PM ET | NEW |
| NYSE_CLOSE | NYSE equity close | 4:00 PM ET | NEW |

**Module:** `pipeline/dst.py` (SESSION_CATALOG + resolvers)
**Per-asset enabled sessions:** `pipeline/asset_configs.py`

**DST Resolution (Feb 2026):** All sessions are now event-based with per-day time resolution. The old fixed-clock sessions (0900/1800/0030/2300) that drifted ±1 hour during DST have been replaced. Winter edges are stronger across instruments (MES +0.35R, MGC +0.18R, MNQ +0.09R). Full results: `docs/RESEARCH_ARCHIVE.md`.

### Entry Models
| Code | How It Works | Risk per Trade | Known Backtest Biases |
|------|-------------|----------------|----------------------|
| E1 | Price breaks ORB + confirm bars close outside. Enter at NEXT bar's open (market order). | **~1.18x ORB width** (15-22% overshoot past boundary). More dollars at risk per trade. | **None.** No fill assumptions. Industry standard entry. What you see is what you get. |
| E2 | STOP-MARKET at ORB level + N ticks slippage. Triggers on ANY bar whose range crosses the ORB level (including fakeouts). No confirmation bars needed. CB1 in grid for infrastructure reuse. | ~1.0x ORB width + slippage (E2_SLIPPAGE_TICKS=1) | **Minimal.** Includes fakeout fills (honest). Slippage is configurable. Conservative: requires strict crossing (high > orb_high), not touching. |
| E3 | Price breaks ORB + confirm bars close outside. LIMIT ORDER at ORB edge, wait for retrace. Stop-breach guard prevents fill after stop hit. | 1.0x ORB width (exact boundary entry) | Fill-on-touch bias (no fill-bar ambiguity since retrace is separate bar). Fewer fills than E1 but better entry price. Adverse selection risk: strong runners never retrace. |

*E0 PURGED Feb 2026 — had 3 compounding optimistic biases (fill-on-touch, fakeout exclusion, fill-bar wins). Won 33/33 combos = artifact. Replaced by E2. See Entry Model Bias Audit below for historical record.*

### Entry Model Bias Audit (Feb 2026) — Historical Record

**E0 (PURGED) won every instrument/session combo in backtests.** This was a structural artifact from 3 optimistic biases, not evidence of universal superiority. E0 was fully replaced by E2 (stop-market) in Feb 2026. The biases:

1. **Fill-on-touch (Low-Medium):** Industry standard backtesting fills limit orders 1 tick BEYOND the limit level ("fill through"). E0 filled on exact touch. Small impact at 1m resolution.
2. **Fakeout fill exclusion (Medium):** A real limit order fills on ANY bar touching the boundary — including fakeout bars that touch then close back inside. E0 only counted fills where the break also confirmed (bar closed outside ORB). This excluded fakeout losers and inflated WR/ExpR.
3. **Fill-bar wins (Session-dependent):** E0 entry and target hit on the same 1m bar. Intra-bar sequence is unknown — the target may have been reached BEFORE the limit filled.

**E2 (replacement):** Stop-market at ORB level + 1 tick slippage. Triggers on ANY touch (fakeouts included). No confirmation bars needed. Entry price = ORB boundary + slippage, which is the honest industry standard for breakout backtesting.

**E1 chase distance:** Across all instruments/sessions, E1 enters 15-22% past the ORB boundary. Average risk = 1.18x ORB width. This is correctly accounted for in R-multiple computation (risk_points = |entry_price - stop_price|).

**E3 vs E1 (verified per-combo, not generalized):** E3 beats E1 in 20/33 combos at RR2.0 G4+, but 19 of those 20 are both negative (E3 just loses less). Only MGC CME_REOPEN has E3 positive AND beating E1 (+0.186 vs +0.137).

**Post-rebuild results (Feb 28):** 888 validated strategies with honest entries (E1+E2) across 5/15/30m ORB apertures (5m: 392, 15m: 312, 30m: 184). 241 FDR significant. 415 edge families (31 ROBUST, 53 WHITELISTED). E3 soft-retired (7 new E3 passed WF but retired per policy). Full rebuild with WF enabled for all instruments + MGC WF override (2022-01-01) + MGC cost model $8.40→$5.74.

### Confirm Bars (CB)
CB1-CB5 = 1-5 consecutive 1m bars must close outside ORB. If ANY bar closes back inside, count resets. For E3, CB1-CB5 produce identical entry prices (same limit order).

### Risk-Reward (RR)
Target = RR × risk. Risk = |entry_price - stop_price|. Stop = other side of ORB.
**For E2/E3:** entry = ORB boundary (+ slippage for E2), risk ≈ ORB size.
**For E1:** entry = next bar open (past boundary), risk = ORB size + overshoot (~1.18x ORB). More dollars at risk per trade.
Example (E2): 10pt ORB + 0.25pt slippage = 10.25pt risk, RR2.0 = 20.5pt target, 10.25pt stop.
Example (E1): 10pt ORB + 2pt overshoot = 12pt risk, RR2.0 = 24pt target, 12pt stop.

### ORB Size Filters
| Code | Meaning | Edge |
|------|---------|------|
| NO_FILTER | Take every trade | LOSES money. No edge. |
| L3/L4/L6/L8 | ORB < N points | LOSES money on 5m ORB. |
| G4+ | ORB >= 4 points | Breakeven starts here. |
| G5+ | ORB >= 5 points | Solid edge. |
| G6+ | ORB >= 6 points | Strongest per-trade edge, fewer trades. |
| G8+ | ORB >= 8 points | Very selective. US_DATA_830 only. |

### Other Terms
| Term | Meaning |
|------|---------|
| ORB | Opening Range Breakout. High/low of first 5 minutes of a session. |
| R-multiple | Profit/loss in units of risk. +1R = won 1× risk. -1R = full stop hit. |
| ExpR | Expected R per trade (average). Positive = profitable. |
| Sharpe | Risk-adjusted return. Above 0.15 = decent. |
| MaxDD | Maximum drawdown in R. Worst peak-to-trough. |
| MGC | Micro Gold futures. $10/point, $5.74 RT friction. |
| MNQ | Micro Nasdaq futures. $2/point, $2.74 RT friction. |
| MES | Micro S&P 500 futures. $5/point, $3.74 RT friction. |
| M2K | Micro Russell 2000 futures. $5/point, $3.24 RT friction. Source: RTY (E-mini Russell) for better coverage. |
| MCL | Micro Crude Oil futures. NO EDGE (0 validated). |
| M6E | Micro EUR/USD futures. 12,500 EUR contract. $12,500/point. $3.74 RT friction (~3 pips). Quarterly cycle (H/M/U/Z). Tick: 0.00005 = $0.625/tick. Size filters in pips (M6E_G4/G6/G8). **ORB breakout NO-GO** (0/2064 validated, Feb 2026). Data in DB for other research. |
| Edge family | Group of strategies with identical trade days. 1 head per family. |

### Reading a Strategy ID
`MGC_LONDON_METALS_E3_RR2.0_CB2_ORB_G4` = Micro Gold, London metals open, limit retrace entry, 2:1 target, 2 confirm bars, ORB >= 4pt filter.

---

## Edge Summary

### What Works
1. **ORB size IS the edge — not sessions, not parameters.** Cross-instrument stress test (Feb 2026) proved this: strip the size filter and ALL edges die. CB, RR, entry model are refinements on top of the one thing that matters: was the ORB big enough? See "ORB Size = The Edge" section below.
2. **E1 is the honest conservative entry, E2 is the honest aggressive entry.** E1 has no backtest biases (market on next bar). E2 (stop-market at ORB level + slippage) is the industry standard for breakout backtesting — includes fakeout fills, uses configurable slippage. E0 was purged Feb 2026 (3 optimistic biases).
3. **E3 retrace per-combo, not universal.** E3 beats E1 in 20/33 combos but 19/20 are both negative. Only MGC CME_REOPEN has E3 positive AND beating E1. LONDON_METALS E3 works for MGC (not MES).
4. **Kill losers early** at CME_REOPEN and TOKYO_OPEN per T80 research. See `config.py:EARLY_EXIT_MINUTES` for exact session kill times.
5. **Negative correlation between sessions** (TOKYO_OPEN vs LONDON_METALS: -0.39). Portfolio diversification is real.
6. **TOKYO_OPEN LONG-ONLY.** Short breakouts at TOKYO_OPEN are negative. Filter to longs.

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
| Systematic retrace dwell | Kills winners at TOKYO_OPEN. Discretionary only for LONDON_METALS E3. |
| Pyramiding | Structural NO-GO. Correlated intraday positions = tail risk. |
| Inside Day / Ease Day filters | All WORSE vs G4 baseline. |
| V-Box CLEAR filter (E1) | Filters nothing (98% trades pass). |
| Volume confirmation | Volume at break does NOT predict follow-through. |
| Session cascade | Prior session range does NOT predict next session quality. |
| Multi-day trend alignment | No improvement on established sessions. |
| Alt strategies (Gap Fade, VWAP Pullback, Value Area, Concretum) | ALL NO-GO. ORB breakout IS the edge in gold. |
| Non-ORB structural (Time-of-day, Range Expansion, Opening Drive) | ALL NO-GO. No risk structure, friction kills. |
| Cross-instrument TOKYO_OPEN LONG portfolio (MGC+MNQ+MES) | NO-GO for diversification. MNQ/MES correlation = +0.83 (same trade). MGC/equity = +0.40-0.44 (moderate, not free). Adding MNQ+MES to TOKYO_OPEN LONG worsens Sharpe. Pick ONE equity micro, don't stack. Need truly uncorrelated asset for portfolio diversification. |
| Crabel contraction/expansion (NR4/NR7 on session ORB history) | Expansion ratio = proxy for absolute size (confound r=0.35-0.86), adds zero within size bands. NR-contraction 2/24 tests significant = chance. Feb 2026. |
| 1015 session (as TOKYO_OPEN replacement) | NO-GO. Negative delta vs TOKYO_OPEN across all instruments/filters. Direction bias INVERTED on equities: MNQ/MES shorts win at 1015, longs win at TOKYO_OPEN. Mechanism: post-auction fade, not continuation. Feb 2026. |
| Transition ORBs (dead-zone periods) | NO-GO. Dead zones lack volume for breakout follow-through. Established session opens dominate. Feb 2026. |
| Wider ORB apertures (10-60min) | NO-GO. 5min aperture has highest per-trade avgR. Wider dilutes signal. Does NOT rescue DST-affected sessions. Feb 2026. |
| SIL (Micro Silver) ORB breakout | NO-GO. 0 of 432 strategies validated. All sessions negative Sharpe. $20 round-trip cost punishing vs silver's average move. NYSE_OPEN WINTER-DOM; US_DATA_830 E3 faint SUMMER-ONLY signal (S=+0.03-0.06) below threshold. Feb 2026. |
| M6E (Micro EUR/USD) ORB breakout | NO-GO. 0 of 2064 strategies validated (1463 trading days, 2021-02-21 to 2026-02-18). Best session TOKYO_OPEN avg ExpR=-0.176; best single strategy ExpR=+0.132 N=63 with chaotic year-by-year (3/6 negative). All 8 sessions negative on average. Mechanism: EUR/USD is a 24hr macro-driven market — no consistent intraday breakout structure at session opens. Data and schema remain in DB for future non-ORB research. Feb 2026. |
| E3 retrace timing optimization (delay buckets) | NO-GO. Break-anchored delay (0-240 min, G4+, RR2.0) tested across MGC+MES x 5 sessions. Zero cells survived CORE/PRELIMINARY sample + BH FDR correction with positive avg_r. MGC/US_DATA_830 0-2 min bucket N=521 avg_r=-0.55R (p~0); 30+ min N=290 avg_r=-0.40R (p~0). Edge in production E3 derives from confirm bars + size/session filters — not delay timing. Do not test confirm-bar-offset variants; that is micro-optimization. Allocate research capital elsewhere. Feb 2026. |
| **MES LONDON_METALS E3** | **NO-GO (MES-specific).** Year-by-year (2019-2025): 7/8 years both winter and summer negative. 2021 winter -0.332R, 2022 winter -0.281R, 2024 winter -0.676R. 2023 winter barely positive (+0.045R) — one outlier. Zero MES LONDON_METALS E3 strategies survive validation. **MGC and MNQ E3 LONDON_METALS are NOT affected** (10 and 42 active strategies respectively). Do not attempt MES LONDON_METALS E3. Mechanism: MES LONDON_METALS (London open) has 81% double-break rate; E3 retrace entry on a mean-reversion session is structurally wrong. Feb 2026. |

> **Key insight**: Gold trends intraday more than it mean-reverts. The only confirmed edges are momentum breakouts with size filters. Full research details: `docs/RESEARCH_ARCHIVE.md`.

### Calendar Effects (P2 — Feb 2026)

Three actionable calendar skip filters confirmed by `research/research_day_of_week.py`. Mechanisms are structural and stable across years.

**NFP_SKIP (Universal):** First-Friday Non-Farm Payrolls days are toxic for all instruments and sessions. MES TOKYO_OPEN G6+ delta = -0.75R. Random spike from 8:30 ET release destroys ORB signal. Skip all NFP days.

**OPEX_SKIP (Universal):** Third-Friday options expiration days degrade edge. Strongest at CME_REOPEN for MNQ (-0.28 delta) and MGC (-0.50 delta). Options pinning kills follow-through. Skip all OPEX days.

**FRIDAY_SKIP (CME_REOPEN only):** Friday CME_REOPEN underperforms other weekdays. MNQ G4+ Friday avgR = -0.376 vs +0.067 non-Friday. MGC G5+ Friday avgR = -0.292. Mechanism: weekend position-squaring. Do NOT apply to TOKYO_OPEN — all days are positive there.

**DOW at TOKYO_OPEN:** Day-of-week has no significant stable effect. Best day rotates year to year. Do NOT filter.

Calendar filters implemented in `pipeline/calendar_filters.py`. Filter classes and `CALENDAR_OVERLAYS` defined in `trading_app/config.py`. **Infrastructure complete** (columns in `daily_features`, filter classes, tests) but **not yet wired** into `portfolio.py` / `paper_trader.py` execution. CSVs: `research/output/day_of_week_*.csv`.

### DOW Alignment: Brisbane DOW vs Exchange DOW (Feb 2026)

The `day_of_week` column uses the Brisbane trading day. For sessions before midnight Brisbane (CME_REOPEN through US_DATA_830), Brisbane DOW = exchange DOW. For sessions after midnight Brisbane (NYSE_OPEN), Brisbane DOW ≠ exchange DOW.

**Investigation:** `research/research_dow_alignment.py`. Canonical mapping in `pipeline/dst.py` (`DOW_ALIGNED_SESSIONS`, `DOW_MISALIGNED_SESSIONS`).

| Session | Brisbane DOW = Exchange DOW? | Detail |
|---------|------------------------------|--------|
| CME_REOPEN | ALIGNED | Bris-Fri evening = CME Fri session (5PM Thu CT = start of CME Fri) |
| TOKYO_OPEN | ALIGNED | Bris-DOW = Tokyo DOW (no DST, UTC+9 same calendar day) |
| SINGAPORE_OPEN | ALIGNED | Bris-DOW = Singapore DOW (no DST) |
| LONDON_METALS | ALIGNED | Bris-DOW = London DOW (08:00 UTC = London morning same day) |
| US_DATA_830 | ALIGNED | Bris-DOW = US DOW (13:00 UTC = US morning same day) |
| NYSE_OPEN | MISALIGNED (-1) | Bris-Fri 00:30 = UTC Thu 14:30 = US Thursday 9:30 AM |

**Implication for NYSE_OPEN:** Brisbane-Friday at NYSE_OPEN is the US THURSDAY equity open. Any DOW research at NYSE_OPEN using Brisbane DOW is offset by -1 day relative to the US calendar. Currently harmless (NYSE_OPEN has no DOW filter in grid), but MUST be accounted for if DOW filters are ever added. `validate_dow_filter_alignment()` in `dst.py` enforces this at runtime.

**All three active DOW filters are correctly aligned:**
- NOFRI@CME_REOPEN → skips CME Friday
- NOMON@LONDON_METALS → skips London Monday
- NOTUE@TOKYO_OPEN → skips Tokyo Tuesday

---

## DST — FULLY RESOLVED (Feb 2026)

**All sessions are now event-based.** The old fixed-clock sessions (0900/1800/0030/2300) that blended winter+summer market contexts have been replaced with dynamic sessions (CME_REOPEN, LONDON_METALS, NYSE_OPEN, US_DATA_830) that resolve to the correct time per-day. DST contamination is no longer an issue for any session.

**Historical context (for reference):** The old fixed sessions drifted ±1 hour during DST transitions. Winter edges were stronger across instruments (MES +0.35R, MGC +0.18R, MNQ +0.09R). US_DATA_830 (formerly 2300) is WINTER-DOM on MGC G8+ — the quieter pre-data window produces better breakouts.

**Remediation history (Feb 2026):**
- Winter/summer split baked into `strategy_validator.py` (DST columns on both strategy tables)
- 1272 strategies re-validated: 275 STABLE, 155 WINTER-DOM, 130 SUMMER-DOM, 10 SUMMER-ONLY. No validated strategies broken.
- Volume analysis confirms event-driven edges (`research/output/volume_dst_findings.md`)
- 24-hour ORB time scan with winter/summer split (`research/research_orb_time_scan.py`)
- Red flags: all 10 were MES CME_REOPEN E1 experimental (never in production). MGC LONDON_METALS confirmed STABLE.

---

## Session Playbook

### CME_REOPEN — Momentum Breakout (formerly 0900)
**DST-clean:** Now event-based, always resolves to CME reopen time. MGC CME_REOPEN validated strategies are mostly STABLE or WINTER-DOMINANT. **MES CME_REOPEN is TOXIC in winter** — all 10 red-flag strategies (edge dies in one regime) are MES CME_REOPEN E1. Volume analysis confirms the edge is tied to the CME open event.

| Parameter | Value |
|-----------|-------|
| Entry model | **E1** (market on next bar) |
| Filter | **G4+** minimum (G5/G6 for higher per-trade edge) |
| RR target | **2.5** (sweet spot for CME_REOPEN) |
| Confirm bars | **CB2** (optimal on G5+ days) |
| Direction | Both (long stronger: +0.50R vs +0.23R short) |
| Early exit | **Kill losers at T80 threshold** (see `config.py:EARLY_EXIT_MINUTES`). +26% Sharpe, -38% MaxDD. |
| Do NOT | Use 30-min kill (tested, 41% of 30-min losers still win — too late). Use breakeven trail. |

**Full-period stats (E1 CB2 RR2.5 G4+):** N=125, WR=40%, TotalR=+38.2, AvgR=+0.31
**Rolling eval:** TRANSITIONING (score 0.42-0.54, passes 9-11/19 windows). High ExpR but inconsistent.

### TOKYO_OPEN — Momentum (LONG ONLY) + IB-Conditional Exit (formerly 1000)

| Parameter | Value |
|-----------|-------|
| Entry model | **E1** (market on next bar) |
| Filter | **G3+** (CORE tier, smallest stable filter for TOKYO_OPEN) |
| RR target | **2.5** (fixed target while IB pending) |
| Confirm bars | **CB2** |
| Direction | **LONG ONLY.** TOKYO_OPEN short is negative (WR=30%, AvgR=-0.04). |
| Early exit | **Kill losers at T80 threshold** (see `config.py:EARLY_EXIT_MINUTES`). 3.7x Sharpe improvement. |
| Exit mode | **Fixed target** (IB-conditional designed but disabled — not yet validated in outcome_builder) |
| Do NOT | Kill at 10 min (too early, actively harmful). |

**IB-Conditional Exit (TOKYO_OPEN — DISABLED, pending research validation):**
The IB-conditional exit mechanism is implemented in `execution_engine.py` but currently disabled
(`SESSION_EXIT_MODE["TOKYO_OPEN"] = "fixed_target"`). The design was never validated in `outcome_builder.py`,
creating a backtest-live parity gap. TOKYO_OPEN strategies execute with fixed target/stop + T80 early exit,
matching validated metrics. IB logic is preserved for future research as a signal/filter, not an exit override.

Original design (dormant):
1. On entry, trade starts in `ib_pending` mode with fixed target active.
2. IB = 120 minutes from CME_REOPEN. After IB forms, detect first break of IB high or low.
3. **IB aligned** (breaks same direction as ORB trade): cancel fixed target, hold for 7 hours. Stop still active.
4. **IB opposed** (breaks opposite direction): exit at market immediately.
5. **IB not yet broken**: keep fixed target active (limbo defense).
6. Opposed kill is **regime insurance** (stop usually fires before IB breaks), not active alpha.

Config: `SESSION_EXIT_MODE["TOKYO_OPEN"] = "fixed_target"`, `IB_DURATION_MINUTES = 120`, `HOLD_HOURS = 7` (dormant).

**Rolling eval:** STABLE (score 0.68-0.83, passes 13-16/19 windows). The MOST stable family.
**Nested 15m ORB:** TOKYO_OPEN is the ONLY session where 15m ORB beats 5m (+0.208R premium, 90% of pairs improve).

### LONDON_METALS — Retrace Entry (formerly 1800)
**DST-clean:** Now event-based, always resolves to London metals open time. MGC LONDON_METALS validated strategies confirmed **STABLE** using UK DST split. E1 RR2.0 CB1 G5: +1.59W(15) / +1.54S(13). E3 RR2.0 CB1 G5: +1.57W(15) / +1.48S(11). Edge is genuine year-round. London is the primary gold market — MGC shows the strongest London-open sensitivity.

| Parameter | Value |
|-----------|-------|
| Entry model | **E3** (limit at ORB edge) |
| Filter | **G4+** (G6 for tightest drawdown) |
| RR target | **2.0** (E3 gets better entry, closer target optimal) |
| Confirm bars | **CB2** (CB1-CB5 identical for E3) |
| Direction | Both |
| Early exit | **No systematic rule helps.** |
| Trade management | See LONDON_METALS E3 Management Rules below. |
| Do NOT | Use time-based cutoff (trades at 4h still +0.43R EV). Use breakeven trail. |

**Full-period stats (E3 CB5 RR2.0 G6+):** N=50, WR=52%, ExpR=+0.43, Sharpe=0.31, MaxDD=3.3R
**Rolling eval:** AUTO-DEGRADED (81% double-break rate). Only works in trending regimes.

### US_DATA_830 — Retrace Entry (formerly 2300)
**DST-clean:** Now event-based, always resolves to 8:30 AM ET. MGC US_DATA_830 validated strategies are WINTER-DOM. 4 validated strategies (all G8+). Edge works in the pre-data positioning window.

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

### SINGAPORE_OPEN Session — PERMANENTLY OFF (formerly 1100/1130, 2026-02-13)

No tradeable edge. 74% double-break rate (structurally mean-reverting). Hard exclusion in `execution_engine.py`, `portfolio.py`, `strategy_fitness.py`. Revisit only with fade/reversal model. Full research: `docs/RESEARCH_ARCHIVE.md`.

---

## LONDON_METALS E3 Management Rules

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
- **ALL 5 MNQ CME_REOPEN edges: SURVIVED or MOSTLY SURVIVED.** Positive all 3 years, bidirectional, ALL real parameter neighbors profitable.

The pattern across ALL instruments: strip the ORB size filter → edge dies. Small ORBs lose, large ORBs win. CB, RR, and entry model are second-order refinements. The size filter IS the strategy.

**Mechanism:** Friction as % of risk. A 2pt ORB has 42% friction (MGC). A 6pt ORB has 14%. Small ORBs don't have enough room for price to move beyond friction + noise. Large ORBs give the breakout space to work. This is arithmetic, not statistics — it persists as long as cost structure exists.

**Implication:** Stop chasing parameter combos. The G5+/G6+ gates are not "nice to have" — they ARE the strategy. Future research should focus on optimal size thresholds per session per instrument, not on CB/RR/EM tuning.

Win rate by CME_REOPEN ORB size (E1 RR2.5 CB2, MGC):

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
| MNQ | CME_REOPEN | G4+ | Deep dive suggested G3+ but hypothesis test (H1) disproved: 3-4pt band adds only 9 trades. |
| MNQ | TOKYO_OPEN | G4+ | Same — G3 band is 3 trades, all losses. |
| MES | CME_REOPEN | G3+ (NO cap) | Hypothesis test (H2) showed 12pt+ is BEST zone on CME_REOPEN (N=7, WR=85.7%). Do NOT cap. |
| MES | TOKYO_OPEN | G4+ with L12 cap | Hypothesis test (H2) confirmed: 12pt+ is TOXIC (N=9, avgR=-0.68). Band G4-L12 improves both avgR and totR. |
| MGC | CME_REOPEN | G5+ | Confirmed. Highest total R at G5. |
| MGC | TOKYO_OPEN | G5+ | Confirmed. |
| MGC | LONDON_METALS | G8+ | Only very large ORBs work in evening. |
| MGC | US_DATA_830 | G12+ or DEAD | Kill for MGC — consider removing from enabled sessions. |

Friction hypothesis (partially disproven):
- Theory: higher friction instruments need bigger ORBs. WRONG in absolute terms.
- MGC has LOWEST friction (0.84pt) but needs BIGGEST gate (G5+).
- MNQ has higher friction (1.37pt) but works from G4+.
- Reason: "points" mean different things. 5pt on MGC = 0.17% of price. 5pt on MNQ = 0.024%.
- TESTED & REJECTED: Percentage-based filter (H3) and ORB/ATR ratio (H4) do not beat instrument-specific fixed gates. See `docs/RESEARCH_ARCHIVE.md`.

Direction bias — hypothesis test confirmed for TOKYO_OPEN session (H5):
- **TOKYO_OPEN session: LONG-ONLY confirmed across all 3 instruments.** MGC shorts are negative (-0.09 avgR). MNQ shorts near zero (+0.03). LONG-ONLY doubles avgR.
- CME_REOPEN session: Both directions work. Difference too small to justify halving N. Keep bidirectional.
- LONDON_METALS session: Dead regardless of direction. No filter rescues it.

MES TOKYO_OPEN upper cap (hypothesis test H2 confirmed):
- 12pt+ ORBs on S&P TOKYO_OPEN session are TOXIC (N=9, avgR=-0.68, WR=11.1%). Mean-reversion trap.
- Band filter G4-L12 beats uncapped G4+ on BOTH avgR (+0.36 vs +0.28) AND totR (+40.3 vs +34.2).
- Band filter G5-L12 is best per-trade: avgR=+0.52, N=67.
- **Exception:** MES CME_REOPEN 12pt+ is the BEST zone (avgR=+1.47). Do NOT apply cap to CME_REOPEN.

**Hypothesis test results:** `scripts/tools/hypothesis_test.py` — full results in `docs/RESEARCH_ARCHIVE.md`.
| Hypothesis | Verdict | Action |
|-----------|---------|--------|
| H1: G3 for MNQ | NO-GO | Band 3-4pt adds 9 trades (noise). Stick with G4+. |
| H2: MES 12pt cap | PASS (TOKYO_OPEN only) | Add G4-L12, G5-L12 to MES TOKYO_OPEN grid. Do NOT cap CME_REOPEN. |
| H3: % of price filter | NO-GO | Not better than tuned fixed gates. |
| H4: ORB/ATR ratio | HARD NO-GO | Scale mismatch — ORBs are 5-15% of daily ATR. |
| H5: Direction filter | PASS (TOKYO_OPEN) | Add LONG-ONLY for TOKYO_OPEN across MGC, MNQ, MES. |

### Volume Filter (Relative Volume)
`rel_vol = break_bar_volume / median(same UTC minute, 20 prior days)`
Fail-closed: missing data = ineligible. CME_REOPEN rejects 34.7% of days. Config: `VOL_RV12_N20`.

### Direction Filter (Hypothesis Test H5 — confirmed Feb 2026)
**TOKYO_OPEN session: LONG ONLY across all instruments.**

| Instrument | LONG avgR | LONG N | SHORT avgR | SHORT N | BOTH avgR |
|-----------|----------|--------|-----------|---------|----------|
| MGC | +0.66 | 38 | -0.09 | 27 | +0.35 |
| MNQ | +0.26 | 178 | +0.03 | 203 | +0.14 |
| MES | +0.19 | 81 | +0.06 | 106 | +0.12 |

MGC TOKYO_OPEN shorts are actively negative. MNQ TOKYO_OPEN shorts are near-zero noise. LONG-ONLY doubles avgR for all three.

**CME_REOPEN session: KEEP BIDIRECTIONAL.** Both directions positive. Difference too small (MGC: +0.87 LONG vs +0.63 SHORT) to justify halving trade count.

**LONDON_METALS session: Dead regardless of direction.** MES LONDON_METALS is -0.23 LONG, -0.15 SHORT. No direction filter rescues a dead session.

### Double-Break Filter (discovery grid — SINGAPORE_OPEN session)

**Regime classifier:** Same-session double-break (both ORB high and low breached during
the full session) signals a mean-reversion day. ORB breakout is a momentum strategy.
Running momentum on mean-reversion days is structurally wrong.

- `NO_DBL_BREAK` = skip days where `orb_{label}_double_break` is True
- Applied as an optional discovery grid filter, not a blanket exclusion
- Composites: `ORB_G4_NODBL` through `ORB_G8_NODBL` (size + no-double-break)
- Currently wired for SINGAPORE_OPEN session only via `get_filters_for_grid()`
- Implementation: `build_daily_features.py:detect_double_break()` computes the flag

**Important:** `double_break` is computed over the FULL session (post-entry information).
As a discovery-level filter it classifies day types for research, but cannot be used as
a real-time pre-entry signal. Strategies validated with this filter trade on days where
the session's historical double-break rate is low — a structural property, not a
per-trade prediction.

### Filter Deduplication Rules
Many filter variants produce the SAME trade set.

- G2/G3 pass 99%+ of days on MGC — cosmetic label, not real filtering for gold
- G3 on MNQ was hypothesized as real but **disproved (H1)**: 3-4pt band adds only 9 trades on CME_REOPEN, all other sessions negative. G3 is cosmetic on MNQ too (different reason: instrument lives at 12pt+).
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
| CME_REOPEN_E2_ORB_G5 | CME_REOPEN | E2 | G5+ | Fixed target | None |
| CME_REOPEN_E2_ORB_G4 | CME_REOPEN | E2 | G4+ | Fixed target | None |
| CME_REOPEN_E1_ORB_G5 | CME_REOPEN | E1 | G5+ | Fixed target | None |
| TOKYO_OPEN_E2_ORB_G5 | TOKYO_OPEN | E2 | G5+ | Fixed target | None |
| TOKYO_OPEN_E2_ORB_G4 | TOKYO_OPEN | E2 | G4+ | Fixed target | None |
| TOKYO_OPEN_E1_ORB_G5 | TOKYO_OPEN | E1 | G5+ | Fixed target | None |
| LONDON_METALS_E2_ORB_G4_NOMON | LONDON_METALS | E2 | G4_NOMON | Fixed target | None |

### Tier 2: HOT (rolling-eval gated)
| Family | Session | EM | Filter | Gate |
|--------|---------|-----|--------|------|
| CME_REOPEN_E2_ORB_G4_NOFRI | CME_REOPEN | E2 | G4_NOFRI | Rolling stability >= 0.6 |
| TOKYO_OPEN_E2_ORB_G5_L12 | TOKYO_OPEN | E2 | G5_L12 | Rolling stability >= 0.6 |

### Tier 3: REGIME (fitness-gated)
*Currently empty. Pending portfolio re-assembly with current validated strategies.*

### Portfolio Parameters
- Max concurrent positions: 3
- Max daily loss: 5.0R
- Position sizing: 2% of equity per trade
- Account equity: $25,000 default

### Correlation
| Pair | Correlation | Interpretation |
|------|------------|---------------|
| CME_REOPEN vs TOKYO_OPEN | -0.04 | Near zero |
| CME_REOPEN vs LONDON_METALS | -0.12 | Mild hedging |
| TOKYO_OPEN vs LONDON_METALS | **-0.39** | Strong hedging |

---

## Reporting Rules

### Family-Level Reporting (MANDATORY)

**Never cite the performance of a single parameter variant.** Always report the family average.

A "family" = one unique combination of `(session, entry_model, filter_level)`. All RR/CB variants within a family share 85-100% of the same trade days.

**Correct**: "The CME_REOPEN E1 G5 family averages +0.34R ExpR across RR/CB variants."
**Wrong**: "MGC_CME_REOPEN_E1_RR3.0_CB2_ORB_G6 has +0.39R ExpR."

**Dedup rule**: Group by `(session, EM, filter_level)` first. Report unique family count, not parameter variant count.

---

## Regime Warnings

### The 2025 Shift
| Year | Median CME_REOPEN ORB | G5+ Days (CME_REOPEN) |
|------|----------------------|----------------------|
| 2021-2024 | 0.7-1.0pt | 2-9/year (1-3%) |
| 2025 | 2.5pt | 70/year (27%) |
| 2026 (Jan) | 14.8pt | 21/month (88%) |

- Validated strategies collapse to **2 truly STABLE families** under rolling evaluation
- **If gold returns to $1800 levels, G5+ days drop to ~5/year (untradeable frequency)**

### Rolling Evaluation Summary
| Family | Score | Windows Passed | Status |
|--------|-------|----------------|--------|
| TOKYO_OPEN_E1_G2 | 0.68-0.83 | 13-16/19 | STABLE |
| CME_REOPEN_G3+ families | 0.42-0.54 | 9-11/19 | TRANSITIONING |
| LONDON_METALS/US_DATA_830/SINGAPORE_OPEN/NYSE_OPEN | <0.30 | AUTO-DEGRADED | Double-break >67% |

### Double-Break Frequency
| Session | Rate | Implication |
|---------|------|------------|
| CME_REOPEN | 57% | Survives |
| TOKYO_OPEN | 57% | Survives |
| SINGAPORE_OPEN | 74% | Mean-reverting (OFF) |
| LONDON_METALS | 81% | Breakout fails most of the time |
| US_DATA_830 | 70% | Unreliable for breakout |
| NYSE_OPEN | 76% | Unreliable for breakout |

### ATR Regime Decision (2026-02-13)
- **NO hard ATR pre-trade gate.** Threshold-sensitive, family-inconsistent, inflated by prior lookahead.
- **KEEP ATR for position sizing:** Turtle-style vol normalization in `portfolio.py`.
- **KEEP rolling fitness:** `strategy_fitness.py` + `live_config.py` regime tier = the regime gate.
- Full ATR research: `docs/RESEARCH_ARCHIVE.md`.

### ATR Velocity + ORB Compression AVOID Signal (2026-02-20 — VALIDATED)

**Signal:** Skip when ATR is **Contracting** AND ORB compression is **Neutral or Compressed**.
- `atr_vel_ratio = atr_20 / mean(prior 5 days atr_20)`. Contracting = ratio < 0.95.
- `orb_compression_z` = z-score of (orb_size/atr_20) vs rolling prior 20-day mean/std. Compressed = z < -0.5, Neutral = -0.5..0.5.
- Both computed from prior days only — **no look-ahead**.

**Cross-instrument scan results:**
- Contracting×Neutral: 9/9 sessions 100% directional (all negative). Median delta = -0.372R.
- Contracting×Compressed: 10/10 sessions 100% directional (all negative). Median delta = -0.362R.
- BH-corrected anchor: MES TOKYO_OPEN Contracting x Neutral p_bh=0.0022. 5/5 years negative.

**Two-condition rule (apply to CME_REOPEN and TOKYO_OPEN sessions):** Skip when `atr_vel_regime == "Contracting"` AND `orb_{label}_compression_tier` in `{"Neutral", "Compressed"}`. Expanding is allowed even during contracting (MES TOKYO_OPEN Contracting x Expanded = +0.149R).

**Exceptions:**
- MNQ LONDON_METALS: positive in Contracting regime — explicitly excluded from filter.
- Sessions SINGAPORE_OPEN, US_DATA_830, NYSE_OPEN: not wired (insufficient data or borderline delta).
- Warm-up period (< 5 prior ATR days): fail-open → trade is allowed.

**Mechanism:** Contracting ATR = volatility de-volatilizing. ORB in neutral/compressed zone = no expansion catalyst. Market lacks energy to follow through on the breakout. Expanded ORB even in contracting ATR = compressed spring still has local tension.

**Implementation:**
- `ATRVelocityFilter` class in `trading_app/config.py`. `ATR_VELOCITY_OVERLAY = ATRVelocityFilter()`.
- Columns in `daily_features`: `atr_vel_ratio`, `atr_vel_regime`, `orb_{CME_REOPEN/TOKYO_OPEN/LONDON_METALS}_compression_z`, `orb_{CME_REOPEN/TOKYO_OPEN/LONDON_METALS}_compression_tier`.
- Wired into: `ExecutionEngine` (live), `build_strategy_daily_series()` + `correlation_matrix()` + `build_portfolio()` in `portfolio.py` (backtesting).
- Script: `research/research_avoid_crosscheck.py`.

---

## Cost Model

| Component | Amount |
|-----------|--------|
| Commission | $1.74/RT |
| Spread | $2.00/RT (1 tick × $10) |
| Slippage | $2.00/RT |
| **Total friction** | **$5.74/RT = 0.574pt** |

Friction as % of risk: 28.7% on 2pt ORB, 14.4% on 4pt, 9.6% on 6pt, 5.7% on 10pt.
This is why small ORBs lose — friction eats the edge.

---

## Confirmed Edges

| Finding | Evidence | Status |
|---------|----------|--------|
| ORB size G4+ breakout | Walk-forward positive, validated strategies | DEPLOYED |
| CME_REOPEN T80 early exit | +26% Sharpe, -38% MaxDD | DEPLOYED |
| TOKYO_OPEN T80 early exit | 3.7x Sharpe, -35% MaxDD | DEPLOYED |
| TOKYO_OPEN LONG-ONLY filter | Short WR=30%, AvgR=-0.04 | DEPLOYED |
| LONDON_METALS E3 30-min check + retrace dwell | In-loss 24% WR; dwell >10min = 33% WR | DEPLOYED (discretionary) |
| Nested 15m ORB for TOKYO_OPEN | +0.208R premium, 90% pairs improve | VALIDATED |
| IB 120m direction alignment | Opposed=3% WR. Cross-validated CME_REOPEN+TOKYO_OPEN. | DEPLOYED |
| SINGAPORE_OPEN permanent exclusion | 74% double-break, all signals failed | DEPLOYED |
| **C3: Skip slow breaks at TOKYO_OPEN (>3 min confirm)** | Fast (<=3 min) avg +0.213R vs slow (>3 min) -0.339R at TOKYO_OPEN (delta=+0.552R, p=0.020, BH-sig). Portfolio delta: +0.069R/trade. Pre-entry, clean mechanism. | **OPTIONAL (via BreakSpeedFilter in config.py)** — removed from hardcoded outcome_builder.py Feb 2026. Now discoverable filter, not mandatory. |
| **C8: Breakeven stop after 30-bar clean hold** | REVERTED. Corrected simulation (Feb 2026) shows +0.018R at CME_REOPEN, ~0.000R at TOKYO_OPEN, +0.017R at LONDON_METALS — noise level. Original +0.053-0.089R estimate was artifact: simulation used pnl_r < 0 filter but ignored Case B (winners scratched while trade still open). ~6 winners (avg +1.7R) scratched per session, nearly offsetting losers saved. | **NO-GO — removed from outcome_builder.py** |

## Pending / Inconclusive
| Idea | Status | Notes |
|------|--------|-------|
| **Time-based exit gate** | **NEXT STEP** | Winner speed confirms T80 <= 45m at RR1.0 and 90-170m at RR2.5-3.0. Sessions held for 480m carry 300-450m of dead exposure. Next: does pnl_r on positions still open past T80 justify continued hold? Script: `research/research_winner_speed.py`. Full data: `research/output/winner_speed_summary.csv`. |
| **MNQ NYSE_OPEN winter > summer pattern** | WATCH (2-year data) | Scan (Feb 2026) found W=+0.946R S=-0.332R on CB4 G6 (N=21). Broader E1 G8+ shows 2024 W=+0.391 -> 2025 W=+0.115 (declining). Mechanism plausible: winter NYSE_OPEN Bris = 14:30 UTC = 9:30 AM EST = US equity open. Summer NYSE_OPEN = 1hr post-open (momentum dissipated). NOT ready to trade — MNQ only has 2 years of data, declining trend. Revisit after 2026 winter season. |
| **MES LONDON_METALS (winter-ish)** | WATCH (inconsistent) | Historical note: was E0 CB2 G4 RR2.5 — E0 now purged. Needs re-evaluation with E2. Monitor 2026 season. |
| **C3 cross-session** | INVESTIGATE | CME_REOPEN delta=+0.257R, LONDON_METALS delta=+0.232R. Not BH-validated. Deployed TOKYO_OPEN-only for now. |
| **C9 (bar-30 exit rule)** | NO-GO | Bar-30 close avg = +0.017R at TOKYO_OPEN (expected -0.42 to -0.67R). Bar-30 position is near-neutral; forced exit would sacrifice residual wins. Do not implement. |
| **C5 (entry bar continues filter)** | PROMISING | True avg +0.360R vs False -0.113R at TOKYO_OPEN (delta +0.473R, N=64/78). Not yet wired as pre-trade filter. |
| **Prior-day outcome signal: MGC CME_REOPEN CB1 G4+ prev=LOSS** | **PROMISING HYPOTHESIS (REGIME)** | BH FDR survivor across 105-cell stress grid (RR x CB x EM x G) at q=0.10: **avgR=+0.585R, t=+3.34, p=0.0016, p_bh=0.088**. Baseline (all prev) = +0.273R; prev=LOSS >2x baseline. Grade: REGIME (N=49). Originally discovered under E0 — needs re-evaluation with E2/E1 entries. **Mechanism:** "failed break -> fresh break" momentum. **Action: WATCH. Needs E2 re-evaluation.** |
| 30m nested ORB | BUILT | 30m data populated in orb_outcomes + daily_features. Not yet validated. |
| RSI directional confirmation | Low confidence | RSI 60+ LONG = +0.81R but N=24. Monitor. |
| Percentage-based ORB filter | Not tested | 0.15% of price ≈ G5. Auto-adapts to price level. |
| E3 V-Box Inversion | Promising | CHOP +0.503 ExpR (N=127). Future E3-specific retest. |
| Low-vol counterbalance | NOT RESEARCHED | Mean-reversion partner for High Vol Breakout system. |
| Band filters (G4-G6, G6-G8 ranges) | INCONCLUSIVE | Q2 edge structure showed timing matters within size bands (avgR spread 0.884). Sensitivity test (+/-20% shifts): MGC SINGAPORE_OPEN G6-G8 genuinely failed (baseline avgR negative, N=76). MES TOKYO_OPEN G4-G6 promising (avgR positive 47/48 shifts, baseline +0.345, N=89) but boundary-unstable. 4 other candidates data-limited (N<20 in band cells). Revisit after 10-year rebuild. Feb 2026. |

---

## Reference Documents

| Document | Content |
|----------|---------|
| `docs/RESEARCH_ARCHIVE.md` | All research deep-dives, data tables, scripts |
| `MARKET_PLAYBOOK.md` | Full backtest results, yearly breakdowns |
| `artifacts/TRADE_MANAGEMENT_RULES.md` | LONDON_METALS E3 management rules with full data |
| `artifacts/EARLY_EXIT_RULES.md` | Early exit research per-session |
| `trading_app/live_config.py` | Declarative live portfolio (code) |
| `trading_app/config.py` | Filter definitions, entry models (code) |

# Bread & Butter Reference — Ground-Up Audit

**Generated:** 2026-02-25
**Data:** gold.db, orb_outcomes + daily_features (triple-join verified)
**ORB:** 5-minute, all 4 active instruments

---

## How to Read This Document

Every number below was computed from raw `orb_outcomes` joined to `daily_features` on
`(trading_day, symbol, orb_minutes)`. Join verified: raw count == joined count for all 4
instruments. No pre-computed metrics were trusted.

**Columns:**
- **N** = trades that hit target or stop (win + loss). Scratches excluded.
- **WR** = win rate = wins / N
- **ExpR** = average pnl_r across N trades (positive = edge, negative = bleeding)
- **TotR** = cumulative R earned across all N trades
- **Yrs+** = years with positive ExpR / total years with data

---

## Entry Model Definitions (Audited Against Industry Standards)

### E1 — Market-On-Confirm (INDUSTRY STANDARD)

After CB confirmation bars close outside the ORB, enter at the **next bar's open**.
This is a market order. No fill assumption issues.

- **CB1-CB5** = 1-5 consecutive 1m closes outside ORB before entry
- Entry price = next bar open (known, observable)
- Stop = opposite ORB boundary
- No intra-bar ambiguity
- **Risk per trade = ~1.18x ORB width** (15-22% overshoot past boundary — you're chasing)

**E1 Chase Distance (verified across all instruments/sessions at G4+):**

| Instrument | Avg overshoot as % of ORB |
|-----------|--------------------------|
| MGC | 15-20% |
| MNQ | 15-22% |
| MES | 15-19% |
| M2K | 13-17% |

This means E1 risks more dollars per trade than E0/E3. The R-multiple computation
correctly accounts for this (risk_points = |entry_price - stop_price|), but the
dollar risk is larger.

**Industry verdict:** This is the standard ORB entry taught by every source reviewed.
HowToTrade.com: *"we don't simply place a buy limit on the high of the range.
Instead, we wait for a full-body candle break."*

### E0 — Limit-On-Confirm (LIMIT ORDER BIASES — READ CAREFULLY)

A limit order sits at the ORB boundary. Fills when the confirm bar's range
**touches** the boundary level. For CB1, this is the break bar itself.

**How it actually works (what it skips):**
E0 skips trades where the break bar does NOT retrace to the ORB boundary.
If the bar gaps entirely past the boundary without touching it → no fill (scratch).
E0 does NOT skip "weak breaks" — it skips breaks that don't retrace.

**Known biases (audited Feb 2026):**

| Bias | Direction | Severity | Detail |
|------|-----------|----------|--------|
| Fill-on-touch | OPTIMISTIC | Low-Medium | Industry standard is "fill through" (1 tick beyond limit). We fill on exact touch. Impact is small at 1m resolution. |
| Fakeout fills excluded | OPTIMISTIC | Medium | Real limit would also fill on bars that touch boundary but close BACK INSIDE ORB (fakeouts → losers). Our backtest only counts confirmed-break fills. |
| Fill-bar wins | OPTIMISTIC | Session-dependent | Entry + target hit on same 1m bar. Intra-bar sequence unknown — target may have been reached BEFORE limit fill. Worst at CME_PRECLOSE RR1.0 (13-16% of all wins). Clean at TOKYO_OPEN/CME_REOPEN RR2.0+ (<1%). |
| Fill-bar both-hit | CONSERVATIVE | N/A | When fill bar hits both target and stop, scored as LOSS. This is correct industry practice. |

**Fill-bar win contamination by session (E0 CB1 RR1.0, all instruments):**

| Session | Fill-bar wins as % of all wins |
|---------|-------------------------------|
| CME_PRECLOSE | 12-16% — CONTAMINATED |
| NYSE_CLOSE | 5-8% — MODERATE |
| SINGAPORE_OPEN | 3-8% — MODERATE |
| TOKYO_OPEN | 4-5% — ACCEPTABLE |
| CME_REOPEN | 2-3% — CLEAN |
| NYSE_OPEN | 0-1% — CLEAN |
| US_DATA_830 | 2-4% — CLEAN |

At RR2.0+, fill-bar wins drop below 1% everywhere — the bias is an RR1.0 problem.

### E3 — Limit-At-ORB After Confirm (PROPERLY GUARDED)

After CB confirms, wait for price to retrace to the ORB boundary.
Checks that stop wasn't breached before the retrace bar fills the limit.

- Entry price = ORB boundary (same as E0)
- Risk per trade = 1.0x ORB width
- Same "fill on touch" bias as E0 but no fill-bar ambiguity (retrace is separate bar)
- Fewer fills than E1 (strong breakouts never retrace back)

**E3 vs E1 (verified per-combo at G4+ RR2.0, not generalized):**
E3 beats E1 in 20/33 combos. But 19 of those 20 are both negative — E3 just
loses less money. The only combo where E3 is positive AND beats E1:
**MGC CME_REOPEN** (E3: +0.186R vs E1: +0.137R). E3 positive combos total:
MGC CME_REOPEN (+0.186), MGC TOKYO_OPEN (+0.018, barely), MNQ NYSE_OPEN (+0.011, barely).

### Structural Finding: E0 Wins Every Combo (33/33) — Artifact, Not Edge

E0 beats E1 and E3 in every instrument/session combination tested. This is a
structural backtest artifact from E0's three optimistic biases, not evidence
that limit-on-confirm is universally superior. The ~0.10-0.15R per-trade
advantage matches the entry price improvement magnitude. Whether this survives
live execution (queue priority, fakeout fills, partial fills) is unknown.

---

## Section 1: E1 CB2 — The Honest Baseline

E1 is the "anyone can trade this" entry. No fill assumptions, no selection bias.
If it shows positive ExpR at E1, the session/instrument genuinely works.

### MGC (Micro Gold) — E1 CB2 G4+ — 2016-2026 (9-11 years)

| Session | RR | N | WR | ExpR | TotR | Yrs+ |
|---------|-----|-----|-----|-------|-------|------|
| **CME_REOPEN** | 1.5 | 112 | 46% | +0.055 | +6.1 | 4/9 |
| **CME_REOPEN** | **2.0** | **108** | **42%** | **+0.137** | **+14.8** | **5/9** |
| **CME_REOPEN** | **2.5** | **106** | **38%** | **+0.201** | **+21.3** | **5/9** |
| CME_REOPEN | 3.0 | 98 | 29% | +0.034 | +3.4 | 4/9 |
| TOKYO_OPEN | 1.5 | 145 | 47% | +0.052 | +7.5 | 6/8 |
| **TOKYO_OPEN** | **2.0** | **143** | **40%** | **+0.071** | **+10.1** | **5/8** |
| **TOKYO_OPEN** | **2.5** | **142** | **35%** | **+0.082** | **+11.6** | **5/8** |
| TOKYO_OPEN | 3.0 | 141 | 30% | +0.066 | +9.3 | 5/8 |

Everything else at E1 for MGC is negative or flat.

### MNQ (Micro Nasdaq) — E1 CB2 G4+ — 2021-2026 (6 years)

| Session | RR | N | WR | ExpR | TotR | Yrs+ |
|---------|-----|------|-----|-------|-------|------|
| NYSE_OPEN | 1.0 | 1185 | 52% | +0.011 | +13.0 | 4/6 |
| NYSE_OPEN | 1.5 | 1124 | 42% | +0.014 | +15.3 | 3/6 |

Barely positive. High N but tiny edge. Everything else negative.

### MES (Micro S&P) — E1 CB2 G4+

**Nothing positive.** All sessions negative at E1 for MES.

### M2K (Micro Russell) — E1 CB2 G4+

**Nothing positive with sufficient N.** M2K SINGAPORE_OPEN shows marginal positive but N=49 (too small).

### E1 Bottom Line

Only **MGC CME_REOPEN** and **MGC TOKYO_OPEN** produce real edge at E1.
MNQ NYSE_OPEN is technically positive but +0.01R per trade is not tradeable after costs.

---

## Section 2: E0 CB1 — Limit Entry (Better Price, Known Biases)

E0 gets a better entry price (at the boundary instead of after confirmation).
This mechanically improves WR and ExpR. See bias table above before trusting these numbers.

### E0 vs E1 vs E3 — Key Sessions Side by Side

**How to read:** E0 risk = 1.0x ORB width. E1 risk = ~1.18x ORB width (chasing).
E3 risk = 1.0x ORB width. N = trades (win+loss). E0 wins 33/33 combos — this is
a structural backtest artifact (see biases above), not proof of real superiority.

#### MGC CME_REOPEN (G4+)

| Entry | RR | N | WR | ExpR | TotR | Yrs+ |
|-------|-----|-----|-----|-------|-------|------|
| E1 CB2 | 2.5 | 106 | 38% | +0.201 | +21.3 | 5/9 |
| E3 CB1 | 2.5 | 101 | 39% | +0.208 | +21.1 | — |
| E0 CB1 | 2.5 | 109 | 43% | +0.351 | +38.3 | 7/9 |

E3 and E1 nearly identical here. E0 advantage is ~0.15R per trade (entry price improvement).

#### MGC TOKYO_OPEN (G4+)

| Entry | RR | N | WR | ExpR | TotR | Yrs+ |
|-------|-----|-----|-----|-------|-------|------|
| E1 CB2 | 2.0 | 687 | 39% | +0.047 | +32.5 | 5/8 |
| E3 CB1 | 2.0 | 656 | 38% | +0.012 | +7.6 | — |
| E0 CB1 | 2.0 | 697 | 42% | +0.118 | +82.4 | 7/8 |

E0 and E1 have similar N (~690) — E0 is not being selective. Pure price improvement.
E3 gets fewer fills (656) and worse ExpR than E1 at this session.

#### MES TOKYO_OPEN (G4+)

| Entry | RR | N | WR | ExpR | TotR | Yrs+ |
|-------|-----|-----|-----|-------|-------|------|
| E1 CB2 | 2.0 | 2757 | 34% | -0.075 | -205.7 | 2/8 |
| E3 CB1 | 2.0 | 2528 | 35% | -0.064 | -161.3 | — |
| E0 CB1 | 2.0 | 2750 | 39% | +0.038 | +104.3 | 5/8 |

E1 negative. E3 less negative. E0 flips to positive. Similar N for E0/E1.

#### MNQ TOKYO_OPEN (G4+)

| Entry | RR | N | WR | ExpR | TotR | Yrs+ |
|-------|-----|------|-----|-------|--------|------|
| E1 CB2 | 2.0 | 3040 | 33% | -0.074 | -224.2 | 3/6 |
| E3 CB1 | 2.0 | 2790 | 33% | -0.086 | -238.7 | — |
| E0 CB1 | 2.0 | 3033 | 38% | +0.055 | +166.7 | 6/6 |

E0 and E1 have nearly identical N (3033 vs 3040). E0 advantage is pure price improvement.
E3 gets fewer fills (2790) and is the worst of the three here.

#### MNQ NYSE_OPEN (G4+)

| Entry | RR | N | WR | ExpR | TotR | Yrs+ |
|-------|-----|------|-----|-------|--------|------|
| E1 CB2 | 2.0 | 1881 | 30% | -0.118 | -222.7 | — |
| E3 CB1 | 2.0 | 2041 | 32% | -0.064 | -129.7 | — |
| E0 CB1 | 2.0 | 2215 | 37% | +0.070 | +155.2 | 6/6 |

E3 has more fills than E1 here and loses less. E0 has even more fills.

#### M2K NYSE_OPEN (G4+)

| Entry | RR | N | WR | ExpR | TotR | Yrs+ |
|-------|-----|------|-----|-------|-------|------|
| E1 CB2 | 1.0 | 2624 | 51% | -0.024 | -62.6 | 1/6 |
| E3 CB1 | 1.0 | 2715 | 50% | -0.048 | -131.5 | — |
| E0 CB1 | 1.0 | 3000 | 57% | +0.066 | +199.3 | 6/6 |

E3 actually has MORE fills than E1 here but worse ExpR. E0 has the most fills and best ExpR.

---

## Section 3: G Filter Effect

G filters require ORB size >= N points. Larger ORBs = better cost absorption, more room
for the trade to work.

### MGC CME_REOPEN — E0 CB1 (RR2.0 sweet spot)

| Filter | N | WR | ExpR | TotR | Yrs+ |
|--------|-----|-----|-------|-------|------|
| NO_FLT | 1149 | 34% | -0.380 | -437.0 | 1/11 |
| G4 | 115 | 50% | +0.354 | +40.8 | 7/10 |
| G6 | 68 | 53% | +0.459 | +31.2 | 6/7 |
| G8 | 43 | 49% | +0.378 | +16.2 | 5/6 |

G4 is the critical threshold. Going to G6 improves ExpR but cuts N in half. G8 cuts too much.

### MES TOKYO_OPEN — E0 CB1 (RR1.0)

| Filter | N | WR | ExpR | TotR | Yrs+ |
|--------|-----|-----|-------|-------|------|
| NO_FLT | 1770 | 63% | -0.027 | -47.3 | 3/8 |
| G4 | 523 | 63% | +0.111 | +58.2 | 6/8 |
| G6 | 219 | 65% | +0.193 | +42.2 | 6/8 |
| G8 | 95 | 65% | +0.218 | +20.7 | 6/8 |

G4 flips negative to positive. G6 is better per-trade but lower N.

### MNQ NYSE_OPEN — E0 CB1 (RR2.0)

| Filter | N | WR | ExpR | TotR | Yrs+ |
|--------|------|-----|-------|--------|------|
| NO_FLT | 1203 | 39% | +0.148 | +177.6 | 6/6 |
| G4 | 1202 | 39% | +0.147 | +176.5 | 6/6 |
| G6 | 1201 | 39% | +0.146 | +175.2 | 6/6 |
| G8 | 1197 | 39% | +0.146 | +174.2 | 6/6 |

MNQ ORBs are almost always 4+ points. G filter has no effect — the edge is universal.

---

## Section 4: RR Tradeoff

Higher RR = lower WR but higher per-win payout. The sweet spot varies by session.

### Universal Pattern

| RR | Typical WR (E0 G4+) | Best For |
|-----|---------------------|----------|
| 1.0 | 55-65% | M2K NYSE_OPEN, short-session trades |
| 1.5 | 45-55% | Transitional — rarely optimal |
| 2.0 | 38-44% | MES/MNQ TOKYO_OPEN, MNQ NYSE_OPEN |
| 2.5 | 33-40% | MGC CME_REOPEN, MNQ TOKYO_OPEN |
| 3.0 | 28-34% | Only works for strongest sessions |
| 4.0 | 22-28% | Rarely justified — needs very strong momentum |

---

## Section 5: The Foundation (Bread & Butter Combos)

These are the combos with:
- Positive ExpR at E0 CB1 G4+
- N >= 100 (enough to be real)
- 5+ years of data
- 60%+ years positive

| # | Instrument | Session | RR | N | WR | ExpR | Yrs+ | Notes |
|---|-----------|---------|-----|------|-----|-------|------|-------|
| 1 | **MNQ** | **TOKYO_OPEN** | **2.5** | **1262** | **36%** | **+0.115** | **6/6** | Best overall. 100% fill rate. |
| 2 | **MNQ** | **NYSE_OPEN** | **2.0** | **1202** | **39%** | **+0.147** | **6/6** | Highest TotR (+176.5R). |
| 3 | **MNQ** | **TOKYO_OPEN** | **2.0** | **1262** | **42%** | **+0.106** | **6/6** | Same session as #1, higher WR. |
| 4 | **MNQ** | **NYSE_OPEN** | **1.5** | **1242** | **47%** | **+0.147** | **5/6** | Higher WR version. |
| 5 | **MES** | **TOKYO_OPEN** | **1.0** | **523** | **63%** | **+0.111** | **6/8** | Highest WR of core combos. |
| 6 | MGC | CME_REOPEN | 2.0 | 115 | 50% | +0.354 | 7/10 | Best ExpR per trade. Low N. |
| 7 | MGC | CME_REOPEN | 2.5 | 109 | 43% | +0.351 | 7/9 | Same session. |
| 8 | MGC | TOKYO_OPEN | 2.5 | 143 | 40% | +0.221 | 6/8 | 100% fill. |
| 9 | MES | TOKYO_OPEN | 3.0 | 520 | 31% | +0.098 | 5/8 | Lower WR but works. |
| 10 | MES | SINGAPORE_OPEN | 2.0 | 339 | 41% | +0.083 | 7/8 | 7/8 years positive. |
| 11 | MES | SINGAPORE_OPEN | 1.0 | 339 | 61% | +0.076 | 6/8 | High WR version. |
| 12 | M2K | NYSE_OPEN | 1.0 | 1195 | 57% | +0.053 | 6/6 | RR1.0 only. Decays at 2.0+. |
| 13 | M2K | NYSE_OPEN | 1.5 | 1131 | 46% | +0.060 | 5/6 | Slightly better ExpR. |
| 14 | MNQ | NYSE_OPEN | 1.0 | 1271 | 58% | +0.116 | 5/6 | If you want 58% WR. |

### NOT on this list (and why):

- **CME_PRECLOSE** — 12-16% of E0 wins are fill-bar wins (intra-bar sequence unknown). Not trustworthy.
- **NYSE_CLOSE** — Same fill-bar issue (5-8%). Also negative at E1.
- **LONDON_METALS at E1** — Negative everywhere except MGC G8 (N too small).
- **US_DATA_830** — Universally negative at both E0 and E1.
- **Anything at RR4.0** — Very few combos work. Not bread & butter.

---

## Section 6: The E1-Only Foundation

If you don't trust E0 biases and want only market-order entries:

| # | Instrument | Session | RR | N | WR | ExpR | TotR | Yrs+ |
|---|-----------|---------|-----|-----|-----|-------|-------|------|
| 1 | **MGC** | **CME_REOPEN** | **2.5** | **106** | **38%** | **+0.201** | **+21.3** | **5/9** |
| 2 | **MGC** | **CME_REOPEN** | **2.0** | **108** | **42%** | **+0.137** | **+14.8** | **5/9** |
| 3 | MGC | TOKYO_OPEN | 2.5 | 142 | 35% | +0.082 | +11.6 | 5/8 |
| 4 | MGC | TOKYO_OPEN | 2.0 | 143 | 40% | +0.071 | +10.1 | 5/8 |
| 5 | MGC | TOKYO_OPEN | 1.5 | 145 | 47% | +0.052 | +7.5 | 6/8 |

That's it. Only MGC. Only 2 sessions. Only RR1.5-2.5. Everything else bleeds.

---

## Methodology Notes

- All queries: `WHERE outcome IN ('win','loss') AND pnl_r IS NOT NULL`
- Join: `ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes`
- G filter: `d.orb_{session}_size >= N`
- No time-stop (ts_pnl_r) used — raw outcomes only
- WR = wins / (wins + losses). Scratches excluded from N.
- Industry sources reviewed: HowToTrade.com, LuxAlgo, Concretum Group, TradingView/MultiCharts fill docs

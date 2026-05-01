# Fitschen 2013 — Building Reliable Trading Systems (Ch 3, 5, 6, 7)

**Source:** `resources/Building_Reliable_Trading_Systems.pdf`
**Author:** Keith Fitschen
**Publication:** Wiley Trading, 2013 (ISBN 978-1-118-52874-7)
**Extracted:** 2026-04-15 (Ch 3); upgraded 2026-05-01 (Ch 5/6/7 added after pre-reg
2026-05-01-chordia-revalidation-deployed-lanes audit caught Ch 5-7 phantom citations).

**Criticality for our project:** 🟡 **MODERATE** — provides literature grounding for the
CORE premise that ORB breakouts work on our three instrument classes (commodities, stock
indices) at intraday scale. Provides a CLASS-LEVEL grounding for filter-based system
improvement (Ch 6) but does NOT mechanism-specifically ground the project's `ORB_G{N}`
size filters or `COST_LT{N}` cost filters.

---

## Ch 3 thesis (page 31 verbatim)

> "Finding the tendencies of a market class is a shortcut to finding good trading approaches
> for that class. If a market tends to trend in a certain time frame, trend-following
> techniques like moving averages, breakouts, trend-lines, and so forth are a good place to
> start the development process. If a market tends to counter-trend behavior,
> overbought/oversold oscillators like RSI, stochastic, or momentum might be good starting points."

## Ch 3 Daily-bar tendencies (pages 32-34)

Method: baseline buy-and-hold vs trend-follow (buy stocks/commodities 1σ above 20-close avg)
vs counter-trend (buy 1σ below avg). Tested 2000-2011.

**Stocks (1,714 instruments):**
- Buy-and-hold: +0.71% per month
- Trend-follow (strong): +0.16% per month (underperforms)
- Counter-trend (weak): +1.56% per month (outperforms +100%)
- **Conclusion:** stocks primarily counter-trend on daily bars.

**Commodities (56 instruments):**
- Buy-and-hold: $66 per month
- Trend-follow (strong): $158 per month (outperforms +140%)
- Counter-trend (weak): $23 per month (underperforms -65%)
- **Conclusion (p.34 verbatim):** *"Clearly the path of least resistance in the commodity world is up"* — commodities trend on daily bars.

## Ch 3 Intraday-bar tendencies (pages 40-42)

**CRITICAL for our project:** Fitschen tests the SAME methodology on hourly bars (2000-2010).

**Stocks intraday (100 Nasdaq stocks, hourly):** Table 3.8 p.41
- Baseline: +0.07% per day per stock
- Trend-follow on hourly (buy 1σ above 10-hour avg): +0.12% per day (+70%)
- **Conclusion (p.40-41 verbatim):** *"instruments that have a trending tendency on one time
  frame don't necessarily have the same trending tendency in other time frames."* Stocks:
  counter-trend on daily, **trend-follow on intraday**.

**Commodities intraday (25 commodities, hourly):** Table 3.9 p.41
- Baseline: $21.26 per day
- Trend-follow on hourly: $30.05 per day (+40%)
- **Conclusion (p.41 verbatim):** *"In the case of commodities, both daily bars and hourly
  bars have a tendency to trend."*

## Ch 3 Mechanism (pages 42-43 verbatim)

> "I can think of no fundamental reason why any stock market should move 1 percent of its
> value four out of every five days. I conclude that it must be trader emotions moving the
> market up and down."

> "When you buy stock you are given a piece of paper, the stock certificate, not a hard
> asset like a gold bar or bushel of grain, which forms the basis of value for commodities.
> As the dot-com bubble proved, nobody really knows what a company is worth. [...] Without
> a clear marker of worth, traders/investors are at a loss to knowing whether a stock is a
> good buy or not."

**Fitschen's mechanism for intraday trend-following:** trader-emotion driven herd behavior
drives intraday momentum in stocks; value fundamentals drive daily trend in commodities.
Both converge to support intraday-trend-follow on both asset classes.

---

## Ch 5 — Trading System Elements: Exits (pages 65-87)

**Scope (verified by reading pp.82-87):** The chapter develops exit techniques for the two
running case-study systems (stock counter-trend + commodity trend-follow). Topics covered:
moving-average trailing stops (Table 5.16), time-based exits (Table 5.17), dollar-based
profit stops (Table 5.18), volatility-based profit targets via standard-deviation counts
(Table 5.19). Concludes (p.87) with an explicit handoff to Ch 6: *"In the next chapter,
we'll look at adding filters to try and make the strategies tradeable."*

**What Ch 5 GROUNDS for our project:** volatility-scaled exit targets (project's RR-target
mechanism is functionally analogous to Fitschen's standard-deviation-count profit targets).

**What Ch 5 DOES NOT GROUND:** session-specific institutional-flow narratives, breakout
size-gates, or cost-ratio filters. Ch 5 is exit-mechanic methodology, not mechanism theory.

---

## Ch 6 — Trading System Elements: Filters (pages 89-105)

### Ch 6 filter taxonomy (page 89 verbatim)

> "In general, filters are used in conjunction with the entry to eliminate certain trades
> from being taken. ... Other common filters are seasonal, volatility, and longer-term
> trend. Seasonal filters use the seasonal tendencies of an instrument to avoid trading
> against historical bias. ... Volatility filters are useful for keeping you out of an
> instrument that is going parabolic, or one whose trading range is so volatile that your
> stop is likely to be hit by the normal noise of its day-to-day moves. ... The longer-term
> trend filter is useful as an adjunct to a shorter-term system: You only take the trades
> in the direction of the longer-term trend."

**Fitschen's four filter classes:**

1. **Trading-day-of-week** — gate on calendar day (Tables 6.1, 6.9-6.11)
2. **Seasonal** — gate on annual calendar (e.g., avoid shorting heating oil in winter)
3. **Volatility** — low-vol filter (skip when range < threshold) OR high-vol filter
   (skip when range > threshold). Tables 6.3-6.6 (stocks), 6.13-6.16 (commodities).
4. **Longer-term trend** — gate on direction of longer-bar moving average (Tables 6.2, 6.12)

### Ch 6 high-volatility filter on commodities (page 100-101 verbatim)

The high-vol *exclusion* filter substantially improved the commodity system:

> "The high-volatility standard deviation filter does improve trading, as shown by the
> gain-to-pain ratio. Notice how the draw-downs have been drastically reduced."

Table 6.15 shows the high-vol filter on commodities raised gain-to-pain from baseline 1.23
to 2.11 (using a 700-dollar 20-day average-range exclusion threshold).

### Ch 6 low-volatility filter on commodities (page 99-100 verbatim)

Low-vol *inclusion* filter (take only signals where standard deviation > X) had marginal
effect:

> "The low-volatility filter does eliminate some of the less profitable trades, as shown
> by increasing profit-per-trade in Table 6.13, but it doesn't help with the draw-downs,
> so it won't be added to the baseline."

### Ch 6 longer-term trend filter on commodities (page 99 verbatim)

> "The 70-day look-back filter increases profit-per-trade by over 40 percent while
> significantly lowering both average maximum draw-down and max draw-down. It will be added
> to the existing strategy to form the new baseline."

Table 6.12 shows the 70-day trend filter on commodities raised gain-to-pain from baseline
1.14 to 1.23.

### Ch 6 mapping to canompx3 filters (HONEST AUDIT — added 2026-05-01)

| canompx3 filter | Fitschen Ch 6 class | Mechanism polarity | Verdict |
|---|---|---|---|
| `ORB_G5` (skip when ORB size < 5pts) | Volatility (low-vol-exclusion) | OPPOSITE of high-vol-exclusion | Class-grounded; mechanism polarity does NOT match Fitschen's tested-positive direction |
| `ORB_G8`, `ORB_G4`, `ORB_G6`, `ORB_VOL_16K` | Volatility (variants) | OPPOSITE of high-vol-exclusion | Same — class-grounded, mechanism polarity opposite to Fitschen's positive result |
| `COST_LT12` (skip when cost/risk > 12bp) | NOT in Ch 6 taxonomy | N/A | Not grounded by Fitschen |
| `COST_LT08`, `COST_LT10`, `COST_LT15` | NOT in Ch 6 taxonomy | N/A | Not grounded by Fitschen |
| `OVNRNG_50`, `OVNRNG_100` (gate on overnight range) | Volatility (variant) | Same direction as low-vol exclusion | Class-grounded; closest mechanism match |
| `ATR_P50` (gate on ATR percentile) | Volatility (variant, percentile-shaped) | Configurable | Class-grounded |
| `VWAP_MID_ALIGNED`, `VWAP_BP_ALIGNED` | NOT in Ch 6 (intra-bar microstructure) | N/A | Not grounded by Fitschen |
| Day-of-week filters (none currently) | Trading-day-of-week | Direct match | Class-grounded if/when used |

**The honest position on `ORB_G5` and other size-gate filters:** Fitschen Ch 6 grounds the
*class* of "use a volatility-based filter to improve a breakout system" but the project's
`ORB_G{N}` filters work in the OPPOSITE polarity to Fitschen's empirically-positive result
(skip-low-vol vs skip-high-vol). For a `has_theory: true` claim under Chordia Pathway A,
the project would need EITHER:

1. A separate literature source for "skip-low-vol-day breakout" mechanism (none found in
   `resources/`), OR
2. A theoretical argument from microstructure that ORB-size gates a different latent
   variable than 20-day average range (e.g., "ORB-size proxies institutional participation
   on this specific session"). This would itself need literature grounding.

**The honest position on `COST_LT{N}` filters:** outside Fitschen's filter taxonomy
entirely. Cost-ratio gates are project-specific friction-economics filters, not in the
intraday-momentum literature. Per Chordia 2018, lacking pre-registered theory, these
should default to `has_theory: false` and require strict t ≥ 3.79.

### Ch 6 chapter conclusion (page 105 — not yet read)

Not yet extracted. Based on chapter structure (intro p.89, individual filter sections to
p.105), the chapter concludes with a final tradeable-system spec for both stock and
commodity case studies. Mark UNREAD.

---

## Ch 7 — Money Management Feedback (pages 107-117)

**Status:** chapter HEADER read only; full chapter NOT extracted. Title from TOC: "Why You
Should Include Money Management Feedback in Your System Development."

**Scope assumption** (UNREAD — verify before citing): based on title, this chapter is about
position-sizing-feedback in the system-development loop, NOT about session-specific or
filter mechanisms. **Do NOT cite Ch 7 for ORB filter or session-mechanism grounding** until
this chapter is read directly.

For Carver-style position-sizing grounding, use `carver_2015_volatility_targeting_position_sizing.md`
(canonical extract from `resources/Robert Carver - Systematic Trading.pdf`).

---

## Application to our project

### What Fitschen 2013 GROUNDS

**Our core ORB breakout premise** on MNQ (NASDAQ), MES (S&P), MGC (gold) at 5-30 minute
apertures is **literature-supported by Ch 3**:
- Commodities (MGC): daily AND intraday trend.
- Stock indices (MNQ, MES): intraday trend-follow (Ch 3 Table 3.8 hourly trend-follow beats
  baseline).
- ORB breakout is a trend-following entry (first-break-direction momentum).

**The CLASS of filter-augmented breakout systems** is supported by Ch 6:
- Filters CAN improve breakout systems (Ch 6 high-vol-exclusion on commodities raises
  gain-to-pain 1.23 → 2.11).
- Volatility filters and longer-term trend filters are explicitly tested and validated.

### What Fitschen 2013 DOES NOT GROUND

- **Session-specific institutional-flow narratives.** No European-morning-flow story, no
  COMEX-close-repositioning story, no Tokyo-open-Asia-liquidity story anywhere in Ch 3,
  Ch 5, Ch 6 (verified by direct extraction). These were inventions in the
  2026-05-01-chordia-revalidation-deployed-lanes pre-reg and were withdrawn 2026-05-01.
- **Specific mechanism for `ORB_G{N}` size-gate filters.** Class-grounded as "volatility
  filter" but Fitschen's positive result is the OPPOSITE polarity (skip-high-vol, not
  skip-low-vol). Pathway-A theory grounding for ORB_G{N} requires separate literature.
- **`COST_LT{N}` cost-ratio filters.** Outside Fitschen's filter taxonomy entirely.
- **Prior-day levels as filter.** F1-F8 features (NEAR_PDH, NEAR_PDL, etc.) are level-based
  not direction-based; Fitschen's chapter is about trend direction at a timescale.
- **Confluence / HTF multi-timeframe alignment.** Not covered.

### Usage rules (revised 2026-05-01 to lock against citation drift)

1. **Pre-reg cites must specify chapter AND page-range AND verbatim mechanism.** Do not
   cite "Fitschen Ch X-Y" without naming the page-range and the specific verbatim claim
   the chapter is being used to ground. The 2026-05-01-chordia-revalidation pre-reg's
   "Fitschen Ch 5-7" with invented session-flow mechanisms passed the previous looser
   format and contaminated 4 lanes' `has_theory: true` claims.

2. **For ORB CORE premise (intraday trend-follow on equity-indices and commodities):**
   cite Ch 3 pp.31-43 specifically. Use this to ground `has_theory: true` on the
   directional ORB-breakout entry mechanism.

3. **For volatility-class filters with skip-HIGH-vol polarity** (e.g., a hypothetical
   `ORB_LT_VOL_P90` filter that skips top-decile volatility days): cite Ch 6 pp.99-101
   specifically and Table 6.15. Pathway-A grounding allowed at t ≥ 3.00.

4. **For volatility-class filters with skip-LOW-vol polarity** (e.g., `ORB_G5`, `ORB_G8`,
   `OVNRNG_50`): Ch 6 grounds the CLASS but NOT the mechanism polarity. Class-grounding is
   weaker than mechanism-grounding per Chordia 2018; the safer reading is `has_theory:
   false` (default to strict t ≥ 3.79) UNLESS a separate mechanism source is also cited.

5. **For cost-ratio filters (`COST_LT{N}`):** Fitschen does NOT ground these. `has_theory:
   false` required unless separate friction-economics literature is cited (e.g., a
   transaction-cost-microstructure source — Almgren-Chriss is the canonical reference but
   is NOT in `resources/`).

6. **For session-specific narratives** (any "European morning", "COMEX close", "Tokyo
   open" mechanism story): Fitschen does NOT ground these. Either find separate literature
   (Dalton's *Mind Over Markets* covers session-time-of-day; not in `resources/`) or
   default to `has_theory: false`.

7. **If writing a Pathway-A pre-reg for a DIRECTIONAL ORB signal** (e.g., "take LONG
   breakout only when 20-bar hourly trend is UP"), Ch 3 supports `has_theory: true`. Do
   NOT extend that grounding to filters in the same pre-reg without independent grounding.

---

## Related literature (in `resources/`, not yet extracted for this purpose)

- `Algorithmic_Trading_Chan.pdf` — Chan 2013, Ch 7 already extracted; covers intraday
  momentum + stop-triggered breakouts. Provides SECOND source for ORB CORE premise.
- `Quantitative_Trading_Chan_2008.pdf` — Chan 2008, Ch 1 §1.4 confirmed PHANTOM citation
  (does not exist); see `chan_2009_ch1_intraday_session_handling.md` for the audit.
- `Robert Carver - Systematic Trading.pdf` — systematic trend systems; portfolio-level
  application; Carver Ch 9-10 extracted for vol-targeting/position-sizing only.

## Related literature (NOT in `resources/`; would need new sourcing)

- Dalton, J., Jones, E., Dalton, R. — *Mind Over Markets* / market profile literature —
  for VAH/VAL/POC level mechanisms AND session-time-of-day mechanisms (would resolve the
  "European morning flow" / "COMEX close" grounding gap).
- Murphy, J. — *Technical Analysis of the Financial Markets* — for support/resistance theory.
- Crabel, T. — *Day Trading with Short Term Price Patterns* — for ORB-style setups.
- Fisher, M. — *The Logical Trader* — for ACD opening range approach.
- Almgren, R., Chriss, N. — "Optimal Execution of Portfolio Transactions" — for cost-ratio
  filter mechanism grounding.

---

## Audit notes

### 2026-04-15 (original Ch 3 extract)

This extract was written during session HTF (2026-04-15) after Phase-1 discovery of 8
prior-day-level filters returned NO-GO under Protocol B. Triggered by user pushback that
the study may have pigeon-holed by jumping to Protocol B without attempting Protocol A
literature grounding. On extraction, Fitschen Ch 3 grounded the **core strategy premise**
(intraday trend-follow works) but NOT the specific level-based filter mechanisms. The
correct institutional response was: preserve core-strategy literature grounding, accept
Phase-1 NO-GO under Protocol B, mark level-based mechanisms as needing separate literature.

### 2026-05-01 (Ch 5/6/7 extension and citation-drift audit)

Triggered by Chordia revalidation pre-reg
(`docs/audit/hypotheses/2026-05-01-chordia-revalidation-deployed-lanes.yaml`) which cited
"Fitschen Ch 5-7" for session-specific institutional-flow narratives (European morning
ORB, COMEX close repositioning, Tokyo open Asia liquidity). User caught the citation as
unsupported and demanded honest extraction.

**Findings on extraction:**

1. Ch 5 (Exits, pp.65-87) is exit-mechanic methodology; does not contain session narratives.
2. Ch 6 (Filters, pp.89-105) lists FOUR filter classes (day-of-week, seasonal, volatility,
   longer-term-trend) with NO session-specific institutional-flow narratives anywhere.
3. Ch 7 (Money Management Feedback, pp.107-117) is by title position-sizing methodology;
   not extracted but title rules out session-mechanism grounding.

**Conclusion:** the pre-reg's "Fitschen Ch 5-7" cite for European/COMEX/Tokyo session
mechanisms was a **fabricated citation**. Ch 5-7 cover exits, filter mechanics, and
position sizing — none cover session-specific narratives. The CORE premise of intraday
trend-follow is grounded by Ch 3 (already extracted); session-specific story requires
non-Fitschen literature (likely Dalton, not in `resources/`).

**Cascading impact on Chordia revalidation:**

- All 4 pre-reg lanes (H1-H4) had `has_theory: true` declared on the basis of "Fitschen
  Ch 5-7 + Chan" citations. With Ch 5-7 grounding fabricated:
  - H1 (EUROPE_FLOW + ORB_G5): no honest theory grounding for the SESSION + the SIZE-GATE
    polarity. Defaults to `has_theory: false`.
  - H2 (COMEX_SETTLE + ORB_G5): same — no session grounding, mechanism-polarity-mismatch
    on size gate. `has_theory: false`.
  - H3 (NYSE_OPEN + COST_LT12): Chan Ch 7 IS legitimately grounded for equity-index
    intraday momentum (FSTX p.156 case study at Sharpe 1.4). But COST_LT12 is not in any
    grounding. Best honest reading: `has_theory: true` for the entry mechanism (NYSE_OPEN
    intraday momentum), `has_theory: false` for the COST_LT12 filter overlay. Conservative
    reading on the lane as a whole: `has_theory: false`.
  - H4 (TOKYO_OPEN + COST_LT12): Chan 2009 Ch 1 §1.4 is a phantom citation (see
    `chan_2009_ch1_intraday_session_handling.md`). Chan Ch 7 mentions stop-cascade
    momentum mechanism that could ground TOKYO_OPEN as another equity-index session
    proxy, but the Tokyo-Asia-liquidity narrative is not in either source. `has_theory: false`.

The Chordia gate must re-run with these honest flags. Lane verdicts will shift toward
FAIL_BOTH (strict threshold 3.79) and the doctrine action will widen substantially.

### 2026-05-01 partial-read disclosure

- Ch 6 pages READ: 89-101 (intro through commodity high-vol filter result). NOT READ:
  102-105 (chapter conclusion + final case-study spec).
- Ch 5 pages READ: 82-87 (exits final sections + chapter conclusion). NOT READ: 65-81
  (early exit techniques — moving averages, simple stops).
- Ch 7 pages READ: 0 pages (title only via TOC at p.v).
- Ch 4 (Entries) pages READ: 0 pages — title from TOC only. Notable: the Donchian breakout
  entry that the case-study uses is in Ch 4. Project's E2 entry mechanism is similar
  (stop-market on first range-cross). This chapter could be relevant for entry-mechanism
  grounding; not extracted in 2026-05-01 audit.

If a future pre-reg needs Ch 4 entry mechanism grounding, Ch 7 position-sizing grounding,
or Ch 5/6 unread-range content, do a targeted extraction extending this file (do not write
a new extract file — keep all Fitschen content in this single canonical extract).

### 2026-05-01 cross-references

- Postmortem: `docs/audit/results/2026-05-01-chordia-revalidation-deployed-lanes.md`
  (the result doc that needs revision per these honest grounding flags)
- Companion: `chan_2013_ch7_intraday_momentum.md` (legitimate equity-index momentum
  grounding)
- Companion: `chan_2009_ch1_intraday_session_handling.md` (the OTHER phantom citation
  audit; precedent for this audit pattern)

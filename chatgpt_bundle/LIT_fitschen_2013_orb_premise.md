# Fitschen 2013 — Path of Least Resistance (Ch 3, Building Reliable Trading Systems)

**Source:** `resources/Building_Reliable_Trading_Systems.pdf`
**Author:** Keith Fitschen
**Publication:** Wiley Trading, 2013 (ISBN 978-1-118-52874-7)
**Extracted:** 2026-04-15

**Criticality for our project:** 🟡 **MODERATE** — provides literature grounding for the CORE premise that ORB breakouts work on our three instrument classes (commodities, stock indices, by implication) at intraday scale. Does NOT ground level-based filter mechanisms (prior-day H/L, pivot, etc.) — those would require Dalton or Murphy, which are not in `resources/`.

---

## Ch 3 thesis (page 31 verbatim)

> "Finding the tendencies of a market class is a shortcut to finding good trading approaches for that class. If a market tends to trend in a certain time frame, trend-following techniques like moving averages, breakouts, trend-lines, and so forth are a good place to start the development process. If a market tends to counter-trend behavior, overbought/oversold oscillators like RSI, stochastic, or momentum might be good starting points."

## Daily-bar tendencies (pages 32-34)

Method: baseline buy-and-hold vs trend-follow (buy stocks/commodities 1σ above 20-close avg) vs counter-trend (buy 1σ below avg). Tested 2000-2011.

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

## Intraday-bar tendencies (pages 40-42)

**CRITICAL for our project:** Fitschen tests the SAME methodology on hourly bars (2000-2010).

**Stocks intraday (100 Nasdaq stocks, hourly):** Table 3.8 p.41
- Baseline: +0.07% per day per stock
- Trend-follow on hourly (buy 1σ above 10-hour avg): +0.12% per day (+70%)
- **Conclusion (p.40-41 verbatim):** *"instruments that have a trending tendency on one time frame don't necessarily have the same trending tendency in other time frames."* Stocks: counter-trend on daily, **trend-follow on intraday**.

**Commodities intraday (25 commodities, hourly):** Table 3.9 p.41
- Baseline: $21.26 per day
- Trend-follow on hourly: $30.05 per day (+40%)
- **Conclusion (p.41 verbatim):** *"In the case of commodities, both daily bars and hourly bars have a tendency to trend."*

## Mechanism (pages 42-43 verbatim)

> "I can think of no fundamental reason why any stock market should move 1 percent of its value four out of every five days. I conclude that it must be trader emotions moving the market up and down."

> "When you buy stock you are given a piece of paper, the stock certificate, not a hard asset like a gold bar or bushel of grain, which forms the basis of value for commodities. As the dot-com bubble proved, nobody really knows what a company is worth. [...] Without a clear marker of worth, traders/investors are at a loss to knowing whether a stock is a good buy or not."

**Fitschen's mechanism for intraday trend-following:** trader-emotion driven herd behavior drives intraday momentum in stocks; value fundamentals drive daily trend in commodities. Both converge to support intraday-trend-follow on both asset classes.

---

## Application to our project

### What Fitschen grounds

**Our core ORB breakout premise** on MNQ (NASDAQ), MES (S&P), MGC (gold) at 5-30 minute apertures is **literature-supported** by Fitschen Ch 3:
- Commodities (MGC): daily AND intraday trend.
- Stock indices (MNQ, MES): intraday trend-follow (Table 3.8 shows hourly trend-follow beats baseline).
- ORB breakout is a trend-following entry (first-break-direction momentum).

### What Fitschen does NOT ground

- **Prior-day levels as filter.** Fitschen's chapter is about trend *direction* at a timescale, not support/resistance levels. Our F1-F8 features (NEAR_PDH, NEAR_PDL, NEAR_PIVOT, ABOVE_PDH, BELOW_PDL, INSIDE_PDR, GAP_UP, GAP_DOWN) are level-based, not direction-based.
- **Prior-day direction as filter.** Fitschen tests 20-close rolling z-score as the trend proxy, not "was yesterday bullish or bearish." Different granularity.
- **Confluence / HTF multi-timeframe alignment.** Not covered.
- **Specific Chordia-supported mechanism claim for our 8 features.** Would require a different literature source (Dalton "Mind Over Markets" for market profile; Murphy "Technical Analysis of Financial Markets" for level theory; Crabel for day-trading short-term patterns). None of these are in `resources/`.

### Usage rules (derived from Extract-Before-Cite, `CLAUDE.md:79-81`)

1. Cite this extract when discussing ORB strategy CORE premise (intraday trend-follow).
2. Do NOT cite this extract for level-based, pivot-based, gap-directional, or prior-day-level filters. These require separate grounding OR Protocol B (no-theory, Chordia t≥3.79).
3. If writing a Protocol A pre-registration for a DIRECTIONAL ORB signal (e.g., "take LONG breakout only when 20-bar hourly trend is UP"), THIS extract supports t≥3.00 under the theory-supported Chordia pathway.

---

## Related literature (in `resources/`, not yet extracted)

- `Algorithmic_Trading_Chan.pdf` — may cover intraday momentum for futures
- `Quantitative_Trading_Chan_2008.pdf` — may cover mean-reversion vs momentum
- `Robert Carver - Systematic Trading.pdf` — systematic trend systems; portfolio-level application

## Related literature (NOT in `resources/`; would need new sourcing)

- Dalton, J., Jones, E., Dalton, R. — *Mind Over Markets* / market profile literature — for VAH/VAL/POC level mechanisms
- Murphy, J. — *Technical Analysis of the Financial Markets* — for support/resistance theory
- Crabel, T. — *Day Trading with Short Term Price Patterns* — for ORB-style setups with level context
- Fisher, M. — *The Logical Trader* — for ACD opening range approach

## Audit note

This extract was written during session HTF (2026-04-15) after Phase-1 discovery of 8 prior-day-level filters returned NO-GO under Protocol B. The extract was triggered by user pushback that the study may have pigeon-holed due to jumping to Protocol B without attempting Protocol A literature grounding. On extraction, Fitschen Ch 3 turned out to ground our **core strategy premise** (intraday trend-follow works) but NOT the specific level-based filter mechanisms we tested. The correct institutional response is:

1. This extract preserves the core-strategy literature grounding (previously informal).
2. The Phase-1 NO-GO under Protocol B remains correct.
3. Level-based filter mechanisms would require separate literature (Dalton/Murphy/Crabel), not yet in `resources/`.

See roadmap at `docs/audit/hypotheses/2026-04-15-htf-sr-untested-axes-roadmap.md` for the honest state of what remains open in the HTF/S-R question.

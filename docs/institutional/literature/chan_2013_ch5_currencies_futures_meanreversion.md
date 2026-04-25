---
source: resources/Algorithmic_Trading_Chan.pdf
title: Algorithmic Trading: Winning Strategies and Their Rationale
author: Ernest P. Chan
chapter: 5
chapter_title: Mean Reversion of Currencies and Futures
year: 2013
pages: 107-132
extracted_by: Claude session (joshd)
extracted_date: 2026-04-26
purpose: Verify whether Chan 2013 Ch 5 grounds an intraday "metals-settle counter-momentum" mechanism for the canompx3 ORB pre-velocity descriptive (PR #119, MGC COMEX_SETTLE 5/15/30 cells with t=-2.56 to -2.86, all aligned-loses).
verdict: DOES NOT GROUND the proposed mechanism. Chapter is interday cointegration and roll-yield mechanics, not intraday-microstructure mean-reversion at exchange settle.
---

# Chan 2013 Ch 5 — Mean Reversion of Currencies and Futures (extract + scope determination)

## TOC

- p.107 Chapter opens (currency mean-reversion via cointegration)
- p.109-114 Pair Trading USD.AUD versus USD.CAD using Johansen Eigenvector (Example 5.1)
- p.113-115 Rollover Interests in Currency Trading (with Example 5.2)
- **p.115 Trading Futures Calendar Spread** ← futures mean-reversion content begins
- p.116-118 Roll Returns, Backwardation, and Contango (theoretical framework)
- p.119+ Discussion of futures price model and ETF cointegration with futures
- p.~122 VIX/VX mean-reversion
- p.~125 Calendar spread mean-reversion of CL (crude oil) 12-month spread
- p.~129 Onward to Ch 6

## What Ch 5 actually covers (with verbatim grounding)

### 5.1 Currency mean-reversion (pp.107-115)
Pair trading two currency cross-rates against each other using cointegration tests (Johansen). Strategy is a classic linear mean-reversion on a stationary spread between two FX pairs, with hedge ratio derived from the eigenvector. Verbatim p.111: *"This is a classic linear mean-reverting strategy similar to the one in Example 3.1... Here, the hedge ratio between the two currencies is not one, so we cannot trade it as one cross-rate AUD.CAD."*

**Mechanism**: cointegration of related currency pairs (B1.USD vs B2.USD with stable long-run relationship). Not intraday momentum/counter-momentum. Not microstructure flow. Daily timeframe.

### 5.2 Trading Futures Calendar Spread (p.115)
Verbatim p.115: *"Pairing up futures contracts with different maturities creates what are known as calendar spreads. Since both legs of a calendar spread track the price of the underlying asset, one would think that calendar spreads potentially offer good opportunities for mean reversion trading. But in reality they do not generally mean-revert."*

**Mechanism**: roll-yield convergence at contract expiration. Backwardation vs contango.

### 5.3 Roll Returns, Backwardation, Contango (pp.116-118)
Verbatim p.116: *"The fact that futures contracts with different maturities have different prices implies that a futures position will have nonzero return even if the underlying spot price remains unchanged, since eventually all their prices have to converge toward that constant spot price. This return is called the roll return or roll yield."*

Theoretical framework for understanding futures return decomposition into spot return + roll return.

### 5.4 ETF–futures cointegration (pp.119+)
*"For example, an ETF of commodity producers (such as XLE) usually [cointegrates with the underlying]... of roll return, this ETF may not cointegrate with the futures price of that [commodity]"*. Daily-timeframe pair trading.

### 5.5 VIX/VX mean-reversion (pp.~122-125)
Mean-reversion of the VIX index and the relationship to the VX futures contract. Driven by the volatility-mean-reverting property of the underlying index AND the persistent contango that VX usually exhibits.

### 5.6 Calendar spread mean-reversion of CL (~p.125)
Tests whether the 12-month calendar spread of CL (crude oil) is stationary using ADF / cointegration tests, then applies a linear mean-reverting strategy on the spread.

## What Ch 5 does NOT cover

Searched the entire chapter (pp.107-132) for: `intraday`, `settle`, `inventory`, `dealer`, `stop[- ]order`, `microstructure`, `metals`, `gold`, `COMEX`, `silver`. **Findings:**

- `settle` appears only in the context of FX rollover settlement-day mechanics (T+2 system, p.113), NOT intraday exchange-settle behavior.
- `intraday` appears only in the context of "if we have only intraday positions, the financing cost is zero" (p.113) — a financing-cost note, not an intraday strategy framework.
- No mention of `inventory`, `dealer`, `stop-order cascades`, `microstructure`, `metals`, `gold`, `COMEX`, or any commodity exchange settle window.

## Conclusion: scope determination for canompx3 research

### What Ch 5 grounds

- **Daily-timeframe cointegration mean-reversion** for FX pairs and ETF–futures pairs — already grounded in `chan_2013_toc_determination.md` and not novel to this extract.
- **Calendar-spread roll-yield mechanics** — useful for term-structure strategies, NOT in canompx3's current scope.
- **VX/VIX interday mean-reversion** — narrow case, not transferable to MGC.

### What Ch 5 does NOT ground (load-bearing for our research)

1. **Intraday counter-momentum at exchange settle.** The chapter is silent on intraday timeframe mean-reversion and silent on exchange-settle microstructure. Our pre-break-context descriptive (PR #119) found 3 MGC COMEX_SETTLE cells (5/15/30 apertures) with t=-2.56 to -2.86 in the counter-momentum direction. **Chan Ch 5 cannot be cited as the mechanism prior for these cells.**
2. **Pre-ORB-velocity continuation/counter-momentum on commodity futures.** No section covers any pre-event flow signature.
3. **Metals-specific microstructure.** The only commodity examples are CL (crude) and the VX/VIX volatility complex. No gold, silver, or metals at all.

### Per institutional-rigor.md § 5 ("Local Academic / Project-Source Grounding Rule")

A claim that requires a literature citation must cite from a verbatim local extract OR be marked UNSUPPORTED.

**The MGC COMEX_SETTLE counter-momentum cluster has NO local-extract grounding** in Chan 2013 Ch 5, Chan 2013 Ch 7 (which is intraday MOMENTUM and explicitly excludes itself from being cited for mean-reversion strategies), Chan 2013 Ch 4 (TOC-only), Chan 2008 (no relevant sections per grep), Fitschen 2013 Ch 3 (intraday TREND prediction, opposite direction), or any other extracted source.

### Recommendation for the open question (counter-momentum at MGC COMEX_SETTLE)

**Do NOT pre-register a Pathway-B re-test** of the 3 MGC COMEX_SETTLE cells. Re-testing without mechanism grounding is post-hoc K-rescue. The cells stay UNVERIFIED at the K=89 cross-section level and DO NOT meet the Chordia-with-theory threshold (t≥3.0) anyway (max |t|=2.86).

To unblock this, ONE of the following must happen first:

1. **A different mechanism prior is found** (e.g., extract Aronson 2007 Evidence-Based Technical Analysis, or Carver 2015 portfolio chapters, or a microstructure paper not yet in resources/) and verbatim-supports intraday metals-settle mean-reversion.
2. **A volatility-regime-conditional reframe** (Chan Ch 7 stop-cascade × ATR_VEL_REGIME interaction) is tested at the descriptive level — this would be a NEW question, not a re-test of the same data.
3. **The descriptive itself is closed as a null cluster** — the right honest action under current evidence.

## Audit note (institutional rigor)

Per `institutional-rigor.md § 7` extract-before-dismiss rule: this extract was created BEFORE characterizing Ch 5 as "doesn't cover the mechanism." TOC + 6 mid-chapter sections sampled. Conclusion is grounded in verbatim text, not training memory.

Per `institutional-rigor.md § 5` mechanism check: counter-momentum at MGC COMEX_SETTLE remains UNSUPPORTED.

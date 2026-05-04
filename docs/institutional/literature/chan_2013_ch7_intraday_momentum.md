# Chan 2013 Ch 7 — Intraday Momentum Strategies

**Source:** `resources/Algorithmic_Trading_Chan.pdf` pp. 155-168
**Author:** Ernest P. Chan
**Publication:** Wiley Trading, 2013 (ISBN 978-1-118-46014-6)
**Extracted:** 2026-04-19

**Criticality for our project:** 🟢 **HIGH** — provides a SECOND literature source (alongside Fitschen Ch 3 at `fitschen_2013_path_of_least_resistance.md`) grounding intraday momentum / ORB breakout strategies on equity indices and futures. Also grounds stop-triggered breakout strategies (directly relevant to our E2 entry model) and news-driven intraday momentum.

---

## Thesis (p.155 verbatim)

> "In the preceding chapter we saw that most instruments, be they stocks or futures, exhibit cross-sectional momentum, and often time-series momentum as well. Unfortunately, the time horizon of this momentum behavior tends to be long — typically a month or longer. Long holding periods present two problems: They result in lower Sharpe ratios and backtest statistical significance because of the infrequent independent trading signals, and they suffer from underperformance in the aftermath of financial crises. In this chapter, we describe short-term, intraday momentum strategies that do not suffer these drawbacks."

## Causes of intraday momentum (p.155 verbatim)

Chan enumerates 5 causes of intraday momentum (4 from interday list + 1 intraday-specific):

> "We previously enumerated four main causes of momentum. We will see that all but one of them also operate at the intraday time frame. (The only exception is the persistence of roll return, since its magnitude and volatility are too small to be relevant intraday.)"

> "There is an additional cause of momentum that is mainly applicable to the short time frame: the triggering of stops. Such triggers often lead to the so-called breakout strategies. We'll see one example that involves an entry at the market open, and another one that involves intraday entry at various support or resistance levels."

And further (p.156):

> "Intraday momentum can also be triggered by the actions of large funds. I examine how the daily rebalancing of leveraged ETFs leads to short-term momentum."

> "Finally, at the shortest possible time scale, the imbalance of the bid and ask sizes, the changes in order flow, or the aforementioned nonuniform distribution of stop orders can all induce momentum in prices."

**Application to canompx3:** Chan ground the CORE project premise — that ORB breakouts (stop-triggered breakout strategies) produce intraday momentum — with a second source beyond Fitschen Ch 3. The specific mechanism is: the triggering of stops cascades into further momentum as stops placed further away also trigger. This is the causal explanation for why ORB breaks produce sustained directional moves past the break level.

## Opening Gap Strategy (p.156-157)

Chan describes an opening-gap-momentum strategy on Dow Jones STOXX 50 futures (FSTX) that goes long when the market gaps up beyond a z-score threshold and shorts when it gaps down:

> "After being tested on a number of futures, this strategy proved to work best on the Dow Jones STOXX 50 index futures (FSTX) trading on Eurex, which generates an annual percentage rate (APR) of 13 percent and a Sharpe ratio of 1.4 from July 16, 2004, to May 17, 2012."

Chan's rationale (p.157 verbatim):

> "What's special about the overnight or weekend gap that sometimes triggers momentum? The extended period without any trading means that the opening price is often quite different from the closing price. Hence, stop orders set at different prices may get triggered all at once at the open. The execution of these stop orders often leads to momentum because a cascading effect may trigger stop orders placed further away from the open price as well. Alternatively, there may be significant events that occurred overnight. As discussed in the next section, many types of news events generate momentum."

**Application:** Chan's gap-momentum result on FSTX (EU equity index future) is directly analogous to the project's MNQ/MES equity-index ORB breakout premise. A Sharpe 1.4 over 8 years on a simple gap-momentum strategy provides a literature precedent that equity-index intraday momentum is real and deployable. Our E2 entry model isn't strictly gap-based — it's ORB-break-based — but Chan's stop-cascade mechanism explains BOTH patterns.

## Post-Earnings Announcement Drift — PEAD (p.157-162)

Chan documents that PEAD persists: earnings-surprise-driven momentum continues into the next-open trading period:

> "There is no surprise that an earnings announcement will move stock price. It is, however, surprising that this move will persist for some time after the announcement, and in the same direction, allowing momentum traders to benefit. Even more surprising is that though this fact has been known and studied since 1968 (Bernard and Thomas, 1989), the effect still has not been arbitraged away, though the duration of the drift may have shortened."

> "For a universe of S&P 500 stocks, the APR from January 3, 2011, to April 24, 2012, is 6.7 percent, while the Sharpe ratio is a very respectable 1.5."

**Application:** Not directly a futures-ORB strategy but analogous: news-event momentum on an intraday scale. For canompx3, the analog is US_DATA_830 / US_DATA_1000 / US_DATA_1000-style sessions where macro data releases drive sustained directional moves. The VWAP_MID_ALIGNED filter that validated on MNQ US_DATA_1000 (N=436 Mode A, ExpR +0.18, t=4.2+) maps to Chan's "data-release-driven intraday momentum" mechanism.

## Pros and cons of momentum strategies (p.151-154 summary)

**Pros** (verbatim p.153):

> "Not only do momentum strategies survive risks well, they can thrive in them (though we have seen how poorly they did in the *aftermath* of risky events). For mean-reverting strategies, their upside is limited by their natural profit cap (set as the 'mean' to which the prices revert), but their downside can be unlimited. For momentum strategies, their upside is unlimited (unless one arbitrarily imposes a profit cap, which is ill-advised), while their downside is limited."

> "Momentum models thrive on 'black swan' events and the positive kurtosis of the returns distribution curve."

**Cons** (verbatim p.151):

> "First, as we have seen so far, many established momentum strategies have long look-back periods and holding periods. So clearly the number of independent trading signals is few and far in between."

> "Secondly, research by Daniel and Moskowitz on 'momentum crashes' indicates that momentum strategies for futures or stocks tend to perform miserably for several years after a financial crisis (Daniel and Moskowitz, 2011)."

**Application:** Chan's pros/cons framing directly informs the project's:
- **Stop-loss philosophy:** our ORB-break strategy uses fixed stops (part of E2 entry-model spec). Chan (p.153) notes "Stop losses are perfectly consistent with momentum strategies" — grounds the E2 stop design.
- **Tail-event benefit:** canompx3's positive-mean-floor gate + bootstrap-p gate in every pre-reg matches Chan's observation that momentum strategies thrive on positive-kurtosis distributions. Our block-bootstrap methodology in every scan is explicitly designed to preserve tail behavior.
- **Post-crisis regime risk:** the Phase 3 Mode A re-validation's finding that many MNQ lanes dropped 50-70% in Sharpe under Mode A partially reflects the 2022-2023 rate-hike regime shift. Chan's "momentum crashes" framing (Daniel-Moskowitz 2011) is consistent with that pattern.

## High-frequency strategies (p.164-167)

Chan discusses order-book-imbalance strategies (bid-ask size imbalance predicts short-term price direction) and tick-level strategies. These are NOT directly relevant to our O5/O15/O30 ORB scope — our time horizons are much longer than tick-level HFT.

**Application:** Out of scope for canompx3's current research direction. Noted for completeness.

---

## Key quotes for Pathway A citations

When writing a pre-reg that cites Chan Ch 7, use one of these verbatim quotes with page reference:

1. **For ORB entry-model as stop-triggered breakout:**
   "the triggering of stops. Such triggers often lead to the so-called breakout strategies. We'll see one example that involves an entry at the market open, and another one that involves intraday entry at various support or resistance levels." (p.155)

2. **For cascading-stop-cascade as the momentum mechanism:**
   "The execution of these stop orders often leads to momentum because a cascading effect may trigger stop orders placed further away from the open price as well." (p.157)

3. **For intraday momentum on equity-index futures:**
   "this strategy proved to work best on the Dow Jones STOXX 50 index futures (FSTX) trading on Eurex, which generates an annual percentage rate (APR) of 13 percent and a Sharpe ratio of 1.4 from July 16, 2004, to May 17, 2012." (p.156) — use for MNQ/MES Pathway A grounding alongside Fitschen Ch 3 Table 3.8.

4. **For stop-loss-as-re-entry consistency with momentum:**
   "Stop losses are perfectly consistent with momentum strategies. If momentum has changed direction, we should enter into the opposite position. Since the original position would have been losing, and now we have exited it, this new entry signal effectively served as a stop loss." (p.153) — grounds E2 stop-design.

---

## Usage rules

1. **Cite this extract** as a second source alongside Fitschen Ch 3 for intraday-trend-follow / intraday-momentum grounding.
2. **Cite this extract specifically** for stop-triggered breakout mechanism (p.155), which Fitschen Ch 3 does NOT explicitly cover — Fitschen is about trend direction at a timescale, Chan Ch 7 is about the cascading-stop mechanism. These are complementary.
3. **Cite this extract** for equity-index intraday momentum (FSTX Sharpe 1.4) as a near-peer benchmark for MNQ/MES ORB strategies.
4. **Do NOT cite this extract** for: mean-reversion strategies (use Chan Ch 4 instead, not extracted), high-frequency order-book strategies (out of canompx3 scope), stock-only stories (our scope is futures).
5. For any Pathway-A pre-reg on canompx3 ORB breakout strategies at O5/O15/O30 apertures, Fitschen Ch 3 AND Chan Ch 7 together provide robust double-source grounding for Chordia t ≥ 3.00 with-theory.

## Related local extracts

- `fitschen_2013_path_of_least_resistance.md` — primary source for intraday trend-follow on equity indices and commodities.
- `chan_2013_ch1_backtesting_lookahead.md` — same book, Ch 1, backtesting methodology + look-ahead bias.
- `carver_2015_volatility_targeting_position_sizing.md` — Carver's framework for volatility-targeted position sizing on momentum signals.
- `chan_2008_ch7_regime_switching.md` — earlier Chan book on regime switching.

## Audit note

This extract was written during the 2026-04-19 overnight session (Phase 9). Chan Ch 7 had been identified as legitimate supplement to Fitschen Ch 3 during the 2026-04-19 Chan TOC determination (`chan_2013_toc_determination.md`). That note flagged Ch 4 as a category error for HTF-level-break grounding but specifically called out Ch 7 as the correct intraday-momentum chapter. This formalization adds the second source. Chan Ch 4 (Mean Reversion of Stocks and ETFs) is NOT extracted — the TOC-determination note explains why (not relevant to our momentum/breakout scope).

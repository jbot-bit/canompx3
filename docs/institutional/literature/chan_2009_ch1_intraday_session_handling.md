# Chan 2009 Ch 1 §1.4 — Intraday Session Handling [UNSUPPORTED]

**Source:** `resources/Quantitative_Trading_Chan_2008.pdf` (Ernest P. Chan, *Quantitative Trading*, Wiley 2008/2009)
**Extracted:** 2026-04-27
**Pages read:** 23–30 (full Chapter 1) and 110–116 (Chapter 5 §"Why Does Actual Performance Diverge from Expectations?")

**Status: ⚠ UNSUPPORTED.** Per CLAUDE.md `Local Academic / Project-Source Grounding Rule`: *"PDFs: extract text (never cite from memory as if read); before dismissing as irrelevant, extract TOC + 3 mid-doc pages (terminology differs)."* Both the TOC scan and full-text reads were performed. Findings below.

---

## What the local PDF actually contains

### Chapter 1 of Chan 2008/2009 has NO §1.4

Chapter 1 ("The Whats, Whos, and Whys of Quantitative Trading") spans pp 1–8 (PDF pp 23–30) and contains exactly three sub-sections, none of which is §1.4:

1. **Who Can Become a Quantitative Trader?** (p 2, PDF p 24)
2. **The Business Case for Quantitative Trading** (p 4, PDF p 26) — contains sub-sub-sections "Scalability", "Demand on Time", "The Nonnecessity of Marketing"
3. **The Way Forward** (p 8, PDF p 30)

There is no §1.4 numbered section. The chapter is a high-level career-context introduction; it does not address backtest realism, session handling, or unfilled-order P&L attribution.

### Chapter 5 §"Why Does Actual Performance Diverge from Expectations?" (pp 89–94, PDF pp 111–116)

The closest topical match in Chan 2008/2009 is in Chapter 5 (Execution Systems). It enumerates:
- ATS software bugs
- Trade-vs-backtest divergence
- Higher-than-expected execution costs
- Market impact on illiquid stocks
- Data-snooping bias revealed by live trading
- Regime shifts (decimalization, plus-tick rule removal, hard-to-borrow shorts)

The section does NOT address session-end forced flat, daily MTM of open positions, or unfilled-order P&L attribution. The diagnostic checklist is for stat-arb equities trading divergence, not intraday-futures forced-flat realism.

### Chapter 3 §"Common Backtesting Pitfalls" (pp 50–60, PDF pp 72–82)

Covers look-ahead bias and data-snooping bias in detail with code-level examples. Topical, but again not session-end forced flat or unfilled-order MTM specifically.

## Verdict per CLAUDE.md UNSUPPORTED rule

The plan's citation "Chan 2009 Ch 1 §1.4 covering session-end forced flat and intraday backtest realism" is **NOT FOUND** in the local PDF. The institutional honest move per CLAUDE.md is to record this as UNSUPPORTED rather than fabricate a passage.

---

## Adjacent literature grounding (citations that ARE supported)

The institutional grounding for "backtest must price every simulated trade including forced-EOD scratches" is in fact provided by **Chan 2013** (a different Chan book, *Algorithmic Trading*, Wiley) and **Bailey-LdP 2014**, both of which are extracted from the local PDFs. Use those instead:

### From `chan_2013_ch1_backtesting_lookahead.md` (verbatim from Chan 2013 p 4):

> "If your backtesting and live trading programs are one and the same, and the only difference between backtesting versus live trading is what kind of data you are feeding into the program (historical data in the former, and live market data in the latter), then there can be no look-ahead bias in the program."

The contrapositive grounds the scratch fix: if the backtest does NOT compute the same P&L as the live execution path (which is forced flat at session end per TopStep prop-firm rules and futures exchange constraints), then they are NOT the same program, and the result is biased. Chan 2013 Ch 1 names this divergence-as-bias-class explicitly for look-ahead; the same principle applies to forced-exit handling.

### From `bailey_lopezdeprado_2014_dsr_sample_selection.md`:

Sample-selection bias from incomplete trade records is named as a first-class source of performance inflation, on equal footing with multiple-testing bias and non-Normality. Scratch-dropout via `WHERE pnl_r IS NOT NULL` is exactly this category of bias.

---

## Application to canompx3

### What this means for the canonical scratch-EOD-MTM fix

The fix proceeds. The literature backing is provided by:
1. **Chan 2013 Ch 1** — unified backtest/live program doctrine (the live path forces flat at session end; backtest must too).
2. **Bailey-LdP 2014** — sample-selection bias is a first-class inflation source.
3. **Carver 2015 Ch 12** — cost-realism prior; every round trip must be priced.
4. **Project-internal grounding** — empirical 10–45% ExpR inflation from `docs/audit/results/2026-04-27-mnq-unfiltered-high-rr-family-v1.md` analysis, quantifying the magnitude.

### What this file does NOT support

This file does **not** ground the fix on Chan 2009 (the citation in the plan was wrong). Any future work that claims "Chan 2009 §1.4 says ..." should cite this file showing the section does not exist. Future doctrine writes should reference `chan_2013_ch1_backtesting_lookahead.md` instead, which IS grounded.

---

## Cross-references

- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` — supported substitute citation.
- `docs/institutional/literature/bailey_lopezdeprado_2014_dsr_sample_selection.md` — supported sample-selection grounding.
- `docs/institutional/literature/carver_2015_ch12_speed_and_size.md` — supported cost-realism grounding.
- CLAUDE.md `Local Academic / Project-Source Grounding Rule` — the rule that mandates this honesty.

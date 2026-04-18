# Chan (2013) *Algorithmic Trading: Winning Strategies and Their Rationale* — Chapter 1 "Backtesting and Automated Execution"

**Source:** `resources/Algorithmic_Trading_Chan.pdf`, Wiley 2013, ISBN 978-1-118-46014-6.
**Extract scope:** book pages 1-10 ONLY (chapter 1 opening through the section "Venue Dependence of Currency Quotes"). Pages 11+ of this chapter are **NOT in this extract** — user should run a follow-up extraction if needed.
**Verification status:** read directly from PDF in the session that produced this file. Page citations below are book-page numbers as printed in the PDF margin.

**Why this chapter matters to our project:**
- p4 contains the invariant we cite in `pipeline/dst.py` to motivate the canonical `orb_utc_window()` single-source-of-truth design.
- p4 also grounds our look-ahead bias detection protocol in `backtesting-methodology.md` Rule 1.
- p4-5 discussion of data-snooping via tweaking-until-OOS-passes directly mirrors the failure mode our Mode A holdout policy defends against.

---

## Key verbatim passages (from the pages I actually read)

### Look-ahead Bias invariant — book p4

Chan defines look-ahead bias and then states the invariant we've adopted as project doctrine:

> "Look-ahead bias is essentially a programming error and can infect only a backtest program but not a live trading program because there is no way a live trading program can obtain future information. This difference between backtesting and a live trading program also points to an obvious way to avoid look-ahead bias. **If your backtesting and live trading programs are one and the same, and the only difference between backtesting versus live trading is what kind of data you are feeding into the program (historical data in the former, and live market data in the latter), then there can be no look-ahead bias in the program.**"
> — Chan 2013, Ch. 1, book page 4 (emphasis added)

**How we use it:** `pipeline/dst.py:37-43` cites this verbatim as the rationale for promoting `orb_utc_window()` to a single shared function between backtest (`outcome_builder`), live engine (`execution_engine`), and feature builder (`build_daily_features`). The E2 canonical-window refactor (2026-04-07) was triggered by a divergence between backtest and live in this exact sense — see `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`.

### Data-snooping via out-of-sample tweaking — book p4

> "The way to detect data-snooping bias is well known: We should test the model on out-of-sample data and reject a model that doesn't pass the out-of-sample test. But this is easier said than done. Are we really willing to give up on possibly weeks of work and toss out the model completely? Few of us are blessed with such decisiveness. Many of us will instead tweak the model this way or that so that it finally performs reasonably well on both the in-sample and the out-of-sample result. But voilà! By doing this we have just turned the out-of-sample data into in-sample data."
> — Chan 2013, Ch. 1, book page 4

**How we use it:** this is the failure mode that `HOLDOUT_SACRED_FROM = 2026-01-01` in `trading_app/holdout_policy.py` defends against. Mode A (sacred holdout) is specifically designed so that the holdout window physically cannot be touched during discovery. Also grounds the override-token warning language: *"ANY STRATEGY DISCOVERED WITH THIS OVERRIDE IS RESEARCH-PROVISIONAL — its OOS-clean property is destroyed."*

### Parsimony / linear-model preference — book p5

> "There is a general approach to trading strategy construction that can minimize data-snooping bias: make the model as simple as possible, with as few parameters as possible. Many traders appreciate the second edict, but fail to realize that a model with few parameters but lots of complicated trading rules are just as susceptible to data-snooping bias. Both edicts lead to the conclusion that nonlinear models are more susceptible to data-snooping bias than linear models because nonlinear models not only are more complicated but they usually have more free parameters than linear models."
> — Chan 2013, Ch. 1, book page 5

**How we use it:** implicit in the Seven Sins awareness (`.claude/rules/quant-agent-identity.md`) and in our preference for named filter families (G1-G9) with small parameter counts over complex multivariate scoring. The ML V3 post-mortem (`06_RD_GRAVEYARD.md`) is the concrete case: a multivariate RF with many effective parameters was more data-snoopable than the simple threshold filters it was meant to combine.

### Walk-forward + paper trading step — book p7

> "In the end, though, no matter how carefully you have tried to prevent data-snooping bias in your testing process, it will somehow creep into your model. So we must perform a walk-forward test as a final, true out-of-sample test. This walk-forward test can be conducted in the form of paper trading, but, even better, the model should be traded with real money (albeit with minimal leverage) so as to test those aspects of the strategy that eluded even paper trading. Most traders would be happy to find that live trading generates a Sharpe ratio better than half of its backtest value."
> — Chan 2013, Ch. 1, book page 7

**How we use it:** grounds our "signal-only shadow" recommendation (e.g., rel_vol HIGH_Q3 stress test 2026-04-15 — not deployed with capital, shadowed). The "half the backtest Sharpe" heuristic is the informal benchmark behind fitness classifier downgrades.

### Survivorship bias in stock databases — book pp 7-9

Chan explains that databases that drop delisted stocks inflate mean-reverting long-only backtests, deflate short-only, and distort long-short more mildly. He notes this is "less dangerous to momentum models" and recommends csidata.com or CRSP for survivorship-bias-free data.

**How we use it:** futures micros (MNQ, MES, MGC) do not have the same delisting problem as stocks. But the principle survives: our dead ORB instruments (MCL, SIL, M6E, MBT, M2K) are tracked explicitly in `DEAD_ORB_INSTRUMENTS` so we never conflate "didn't pass ORB discovery" with "didn't exist."

### Primary vs. consolidated prices (US stocks) — book pp 9-10

Chan warns that consolidated-feed closing prices reflect ANY venue (including dark pools and ECNs) and can be quite different from the primary exchange (NYSE, Nasdaq). Backtesting a mean-reverting model on consolidated prices "generates inflated backtest performance because a small number of shares can be executed away from the primary exchange at a price quite different from the auction price on the primary exchange." The same concern applies to consolidated highs and lows.

**How we use it:** not directly — we trade futures (single primary market per contract via CME Globex). But the principle generalizes to our cost model: we must use the actual execution venue's cost structure, not an average. This is why `COST_SPECS` in `pipeline/cost_model.py` uses TopStep Rithmic rates (our primary deployment path) rather than an IB/CME-average.

---

## Related Chan 2013 content NOT in this extract

These topics are in the broader book but were NOT read in the session that produced this file. If ChatGPT needs them, ask the user to run a focused PDF extraction:

- **Common pitfalls continued (book pp 11-30 of ch1):** almost certainly continues data-snooping discussion; likely covers transaction costs and statistical significance.
- **Hypothesis testing / Monte Carlo simulations for significance:** mentioned on book p1 as a topic of ch1; full treatment is somewhere pp 11-38.
- **Software platform choice / automated execution integration:** mentioned p1 as a ch1 topic.
- **Ch. 2: The Basics of Mean Reversion (book p 39).**
- **Ch. 7: Intraday Momentum Strategies (book p 155).** This is the chapter most directly adjacent to our ORB work — worth extracting explicitly.
- **Ch. 8: Risk Management (book p 169).**

---

## Bibliographic note

Chan, Ernest P. *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley Trading Series. Hoboken, NJ: John Wiley & Sons, 2013. ISBN 978-1-118-46014-6. Copyright © 2013 by Ernest P. Chan.

All verbatim passages quoted under fair use for research documentation in the `canompx3` project. Page citations use the PDF margin numbering (i.e., book-page numbers, not PDF-page indices).

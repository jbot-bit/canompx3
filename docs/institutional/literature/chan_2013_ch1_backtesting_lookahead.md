# Chan 2013 Ch 1 — Backtesting and Automated Execution

**Source:** `resources/Algorithmic_Trading_Chan.pdf` pp. 1-37
**Author:** Ernest P. Chan
**Publication:** Wiley Trading, 2013 (ISBN 978-1-118-46014-6)
**Extracted:** 2026-04-19

**Criticality for our project:** 🟢 **HIGH** — provides direct literature grounding for (a) look-ahead bias doctrine, (b) data-snooping-via-OOS-tweaking doctrine, (c) survivorship bias, (d) futures continuous-contract construction, (e) hypothesis-testing as a backtest framework. Already cited informally in `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`; this extract formalizes it.

---

## The backtesting premise (p.1 verbatim)

> "If one blithely goes ahead and backtests a strategy without taking care to avoid these pitfalls, the backtesting will be useless. Or worse — it will be misleading and may cause significant financial losses."

Chan frames the entire chapter around avoidable pitfalls that inflate backtest performance relative to live-trading reality. Every pitfall he enumerates has a direct canonical-source-of-truth check in our codebase.

## Look-ahead bias (p.4 verbatim, full definition)

> "As its name implies, look-ahead bias means that your backtest program is using tomorrow's prices to determine today's trading signals. Or, more generally, it is using future information to make a 'prediction' at the current time. A common example of look-ahead bias is to use a day's high or low price to determine the entry signal during the same day during backtesting. (Before the close of a trading day, we can't know what the high and low price of the day are.) Look-ahead bias is essentially a programming error and can infect only a backtest program but not a live trading program because there is no way a live trading program can obtain future information."

Chan's prescription (p.4 verbatim):

> "This difference between backtesting and a live trading program also points to an obvious way to avoid look-ahead bias. If your backtesting and live trading programs are one and the same, and the only difference between backtesting versus live trading is what kind of data you are feeding into the program (historical data in the former, and live market data in the latter), then there can be no look-ahead bias in the program."

**Application to canompx3:** this is EXACTLY the project's `pipeline.dst.orb_utc_window` unified-source doctrine. The 2026-04-08 E2 canonical-window-fix postmortem cites Chan Ch 1 p.4 for "any divergence between backtest and live execution ORB-window calculation is a look-ahead bias risk." The fix was to route BOTH paths through `pipeline.dst.orb_utc_window` — backtest (`outcome_builder`) and live (`execution_engine`) use the same function.

## Data-snooping bias via OOS tweaking (p.4 verbatim)

> "The way to detect data-snooping bias is well known: We should test the model on out-of-sample data and reject a model that doesn't pass the out-of-sample test. But this is easier said than done. Are we really willing to give up on possibly weeks of work and toss out the model completely? Few of us are blessed with such decisiveness. Many of us will instead tweak the model this way or that so that it finally performs reasonably well on both the in-sample and the out-of-sample result. But voilà! By doing this we have just turned the out-of-sample data into in-sample data."

**Application:** this is why `trading_app.holdout_policy.HOLDOUT_SACRED_FROM = 2026-01-01` is Mode A sacred. The research-truth-protocol.md § "2026 holdout is sacred" rule and every pre-reg's `tuning_against_oos: false` clause enforce Chan's discipline against our own worst human instinct. The 2026-04-07 Mode A decision doc (`docs/plans/2026-04-07-holdout-policy-decision.md`) explicitly retires the prior Mode B policy precisely to prevent this tweaking loop.

## The linearity / Occam's razor principle (p.4-5 verbatim)

> "There is a general approach to trading strategy construction that can minimize data-snooping bias: make the model as simple as possible, with as few parameters as possible. Many traders appreciate the second edict, but fail to realize that a model with few parameters but lots of complicated trading rules are just as susceptible to data-snooping bias. Both edicts lead to the conclusion that nonlinear models are more susceptible to data-snooping bias than linear models because nonlinear models not only are more complicated but they usually have more free parameters than linear models."

And (p.5 verbatim):

> "Linear models imply not only a linear price prediction formula, but also a linear capital allocation formula. [...] The most extreme form of linear predictive models is one in which all the coefficients are equal in magnitude (but not necessarily in sign). [...] Daniel Kahneman, the Nobel Prize–winning economist, wrote in his bestseller Thinking, Fast and Slow that 'formulas that assign equal weights to all the predictors are often superior, because they are not affected by accidents of sampling'."

**Application:** This is the institutional literature behind why canompx3's pre-reg criteria favor simple threshold filters (ORB_G5 = top-quintile size; OVNRNG_100 = overnight ≥ 100pts) rather than multi-parameter ML. It also grounds the repeated finding that ML V1/V2/V3 are DEAD per Blueprint §5. Chan Ch 1 p.5 is the institutional case that "formulas that assign equal weights to all predictors" (i.e., simple aggregation) are superior to over-parameterized nonlinear models.

## Walk-forward testing (p.7 verbatim)

> "In the end, though, no matter how carefully you have tried to prevent data-snooping bias in your testing process, it will somehow creep into your model. So we must perform a walk-forward test as a final, true out-of-sample test. This walk-forward test can be conducted in the form of paper trading, but, even better, the model should be traded with real money (albeit with minimal leverage) so as to test those aspects of the strategy that eluded even paper trading. Most traders would be happy to find that live trading generates a Sharpe ratio better than half of its backtest value."

**Application:** The project's WFE ≥ 0.50 gate in `trading_app.strategy_validator` + Criterion 6 from `pre_registered_criteria.md` directly implements Chan's "Sharpe ratio better than half of its backtest value" informal threshold. The formal version: `WFE = Sharpe_OOS / Sharpe_IS >= 0.50`. Chan also grounds the project's shadow-recorder + paper-trade logger infrastructure (see `trading_app/paper_trade_logger.py`, `trading_app/paper_trader.py`) — real-money low-leverage shadow observation as the honest gate before scaling.

## Survivorship bias (p.8-9 relevant passages)

> "If you are backtesting a stock-trading model, you will suffer from survivorship bias if your historical data do not include delisted stocks."

> "Survivorship bias is less dangerous to momentum models. The profitable short momentum trade will tend to be omitted in data with survivorship bias, and thus the backtest return will be deflated."

**Application:** For the canompx3 futures-trading scope, stock-level survivorship is not a direct concern. But the project's DEAD-instrument discipline (MCL/SIL/M6E/MBT/M2K removed from `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`) is a futures-analog survivor check — we retain the data for the dead instruments to avoid cherry-picking only the survivors (MNQ/MES/MGC). Chan's discussion informs the general principle.

## Primary vs. consolidated stock prices (p.9-10)

Not directly applicable to our futures scope but worth noting: Chan warns that consolidated prices inflate mean-reverting backtests because outlier trades on secondary venues are included. Our project uses single-venue CME tick data from Databento, so this pitfall is avoided by construction.

## Futures continuous contracts (p.12-15)

Chan's discussion on back-adjusted continuous contracts is directly relevant:

> "An additional difficulty occurs when we choose the price back-adjustment instead of the return back-adjustment method: the prices may turn negative in the distant past."

And (p.14 verbatim):

> "A data vendor can typically back-adjust the data series to eliminate the price gap [between contracts at rollover], so that the P&L on T+1 is p(T+1) − p(T). This can be done by adding the number (q(T+1) − p(T+1)) to every price p(t) on every date t on or before T, so that the price change and P&L from T to T+1 is correctly calculated as q(T+1) − (p(T) + (q(T+1) − p(T+1))) = p(T+1) − p(T). (Of course, to take care of every rollover, you would have to apply this back adjustment multiple times, as you go back further in the data series.)"

Chan's key caveat (p.14):

> "This subtlety in picking the right back-adjustment method is more important when we have a strategy that involves trading spreads between different contracts. [...] As you can see, when choosing a data vendor for historical futures prices, you must understand exactly how they have dealt with the back-adjustment issue."

**Application:** canompx3 uses Databento DBN files for futures; back-adjustment is handled at Databento's ingest side (see `pipeline/ingest.py` and related pipeline docs). The project doesn't trade inter-contract spreads, so the back-adjustment subtlety matters less for our strategy class. Chan's discussion grounds why we read `bars_1m` from Databento-processed continuous contracts without re-constructing them ourselves.

## Statistical significance via hypothesis testing (p.16 verbatim)

> "In any backtest, we face the problem of finite sample size: Whatever statistical measures we compute, such as average returns or maximum drawdowns, are subject to randomness. In other words, we may just be lucky that our strategy happened to be profitable in a small data sample. Statisticians have developed a general methodology called hypothesis testing to address this issue."

Chan then outlines the 4-step framework (p.16):
1. Compute test statistic (e.g., average daily return)
2. Suppose true mean is zero (null hypothesis)
3. Determine probability distribution of returns
4. Compute p-value

**Application:** This directly grounds the project's `trading_app.strategy_validator` + `pre_registered_criteria.md` Criterion 4 (Chordia t ≥ 3.00 with theory or ≥ 3.79 no theory) + RULE 4 of `.claude/rules/backtesting-methodology.md` (BH-FDR multi-framing). Chan Ch 1 provides the institutional case for why a backtest WITHOUT a hypothesis-test framework is insufficient — the project's every pre-reg locks a hypothesis, null, test statistic, and p-value threshold BEFORE the run.

---

## Usage rules

1. **Cite this extract** for the project's look-ahead doctrine (Chan p.4).
2. **Cite this extract** for the OOS-sacred doctrine (Chan p.4 on data-snooping-via-tweaking).
3. **Cite this extract** for the Pathway-A "with-theory" simple-filter preference (Chan p.4-5 Occam's razor + Kahneman equal-weights).
4. **Cite this extract** for WFE ≥ 0.50 gate (Chan p.7 "Sharpe better than half of backtest").
5. **Cite this extract** for the hypothesis-testing framework underlying the pre-reg system (Chan p.16).

Use as a secondary citation alongside Bailey et al 2013 (`bailey_et_al_2013_pseudo_mathematics.md`) for MinBTL and Bailey-LdP 2014 (`bailey_lopez_de_prado_2014_deflated_sharpe.md`) for DSR. Chan is the PRACTITIONER voice; Bailey-LdP and Chordia are the ACADEMIC-RIGOROUS voice. Pre-regs can cite both for defensibility.

## Related local extracts

- `fitschen_2013_path_of_least_resistance.md` — intraday trend-follow on commodities + equity indices (ORB premise). Chan Ch 7 (extracted separately, `chan_2013_ch7_intraday_momentum.md`) overlaps on intraday momentum and provides a second source.
- `chan_2008_ch7_regime_switching.md` — regime-switching (different book, different chapter; useful alongside).
- `harvey_liu_2015_backtesting.md` — more rigorous multiple-testing haircut (BHY) complementing Chan's hypothesis-testing framework.

## Audit note

This extract was written during the 2026-04-19 overnight session (Phase 9 of the 14-phase remediation plan). Chan Ch 1 had been informally cited in the project postmortem (`docs/postmortems/2026-04-07-e2-canonical-window-fix.md`) but not formally extracted to `docs/institutional/literature/`. The 2026-04-19 code-review findings identified this as queued literature extraction work. This formalization enables Pathway A pre-regs to cite Chan directly for backtest-methodology grounding.

# Chan 2008 — Quantitative Trading, Chapter 7: Regime Switching

**Source:** `resources/Quantitative_Trading_Chan_2008.pdf`
**Author:** Ernest P. Chan
**Chapter:** 7 — Special Topics in Quantitative Trading; Section: Regime Switching
**Book pages:** 119-126
**PDF pages:** 140-147
**Extracted:** 2026-04-15 (via pypdf, verbatim passages below)

## Why this extract matters

The 2026-04-15 volume-finding-exploitation research surfaced a gap: our system
has regime features (atr_vel_regime, garch_forecast_vol_pct, day_type) but no
unified regime classifier or conditional-activation infrastructure. Chan 2008
Ch 7 is the canonical institutional-grade treatment of regime switching in a
trading context, grounded in GARCH econometrics and Markov regime-switching
(hidden Markov models). This extract provides the theoretical foundation for
our proposed `trading_app/regime_classifier.py` module (see
`docs/institutional/regime-and-rr-handling-framework.md`).

## Verbatim passages

### Taxonomy of financial regimes (book p120, PDF p141)

> "Some of the other most common financial or economic regimes studied are
> inflationary vs. recessionary regimes, high- vs. low-volatility regimes,
> and mean-reverting vs. trending regimes. Among these, volatility regime
> switching seems to be most amenable to classical econometric tools such
> as the generalized autoregressive conditional heteroskedasticity (GARCH)
> model (See Klaassen, 2002). That is not surprising, as there is a long
> history of success among financial economists in modeling volatilities
> as opposed to the underlying stock prices themselves."

**Application to our system:** Our `garch_forecast_vol_pct` feature already
operationalizes GARCH-based volatility regime tracking. Chan's taxonomy
(inflationary vs recessionary, high vs low vol, mean-reverting vs trending)
maps cleanly to our proposed Tier R2 regime classifier labels:
`calm_trend`, `vol_expansion`, `range_day`, `event_day`, `reversion_setup`,
`unknown`.

### Regime-switching methodology (book p120-121, PDF p141-142)

> "Academic attempts to model regime switches in stock prices generally
> proceed along these lines:
> 1. Propose that the two (or more) regimes are characterized by different
>    probability distributions of the prices. In the simplest cases, the log
>    of the prices of both regimes may be represented by normal distributions,
>    except that they have different means and/or standard deviations.
> 2. Assume that there is some kind of transition probability among the
>    regimes.
> 3. Determine the exact parameters that specify the regime probability
>    distributions and the transition probabilities by fitting the model
>    to past prices, using standard statistical methods such as maximum
>    likelihood estimation.
> 4. Based on the fitted model above, find out the expected regime of the
>    next time step and, more importantly, the expected stock price.
>
> This type of approach is usually called Markov regime switching or hidden
> Markov models, and it is generally based on a Bayesian probabilistic
> framework."

### Chan's HARD caveat on Markov regime-switching (book p121, PDF p142)

> "Despite the elegant theoretical framework, such Markov regime-switching
> models are generally useless for actual trading purposes. The reason for
> this weakness is that they assume constant transition probabilities among
> regimes at all times. In practice, this means that at any time (as illustrated
> by the Nielsen and Olesen paper), there is always a very small probability
> for the stock to transition from a normal, quiescent regime to a volatile
> regime. But this is useless to traders who want to know when—and under what
> precise conditions—the transition probability will suddenly peak. This
> question is tackled by the turning points models."

**Critical implication for our framework:** do NOT build a pure Markov
regime-switching classifier. The constant-transition-probability assumption
makes the output useless for operational decisions. Instead, use
**turning-point / state-classification** approaches.

### Turning points models (book p121, PDF p142)

> "Turning points models take a data mining approach (Chai, 2007): Enter all
> possible variables that might predict a turning point or regime switch.
> Variables such as current volatility; last-period return; or changes in
> macroeconomic numbers such as consumer confidence, oil price changes, bond
> price changes, and so on can all be part of this input."

**Application:** our regime classifier should be a state-determination model
using current observable features (volatility percentile, vol-velocity,
day-type, calendar, overnight dynamics) — NOT a transition-probability
forecast model. Match current state, apply conditional lane activation.
Do not attempt to FORECAST regime transitions (Chan explicitly says this
fails).

### Stop loss in regime-dependent context (book p106-107, PDF p127-128)

> "For stop loss to be beneficial, we must believe that we are in a momentum,
> or trending, regime. In other words, we must believe that the prices will
> get worse within the expected lifetime of our trade. Otherwise, if the
> market is mean reverting within that lifetime, we will eventually recoup
> our losses if we didn't exit the position too quickly."

**Application to our ORB breakout strategies:** our E2 entry with ATR-scaled
stop is fundamentally a momentum-regime strategy. Chan's passage confirms
this is appropriate IF the regime is actually trending. In mean-reverting
regimes, our stop-loss behavior is anti-edge. This is exactly the case
for conditional-activation (Tier R2) — trade when regime is trending,
monitor-only when mean-reverting.

## Citation format for use in project docs

When citing this extract in future documents, use:
> "Chan 2008 Ch 7 §Regime Switching (book pp119-121); extracted in
> `docs/institutional/literature/chan_2008_ch7_regime_switching.md`"

## Related literature in project

- `pepelyshev_polunchenko_2015_cusum_sr.md` — Shiryaev-Roberts procedure for
  real-time regime-change detection. Complementary: Chan defines WHAT regimes
  to detect; SR defines HOW to detect them in real-time.
- `carver_2015_volatility_targeting_position_sizing.md` — how to SIZE
  positions within a regime (Carver Ch 9-10); Chan's regime identification
  feeds the size-scaling input.

## Gaps identified (training-memory-only until acquired)

- **Harris, "Trading and Exchanges"** — institutional order-flow mechanism
  for volume-as-confirmation. NOT in `resources/`. Acquisition pending.
- **O'Hara, "Market Microstructure Theory"** — same. NOT in `resources/`.
- **Dalton, "Mind Over Markets"** — market-profile regime framework for
  level-based trading. NOT in `resources/`. Required for Phase E SC2.1.
- **Murphy, "Technical Analysis of the Financial Markets"** — S/R framework
  for level trading. NOT in `resources/`. Required for Phase E SC2.1.

Until these are acquired, any claim about order-flow-based volume confirmation
or market-profile regimes must be labeled "from training memory — not verified
against local PDF."

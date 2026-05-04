# Anomalies and Multiple Hypothesis Testing: Evidence from Two Million Trading Strategies — Chordia, Goyal, Saretto (2018)

**Source:** `resources/Two_Million_Trading_Strategies_FDR.pdf`
**Authors:** Tarun Chordia (Emory), Amit Goyal (Swiss Finance Institute / Lausanne), Alessio Saretto (UT Dallas)
**Date:** May 2018
**Extracted:** 2026-04-07
**Pages read:** 1-20
**Status note on paper:** "PRELIMINARY AND INCOMPLETE. PLEASE DO NOT QUOTE."

**Criticality:** 🔴 **HIGH** — provides the empirical t-statistic threshold for finance data at 2.4M strategy scale and recommends specific MHT methods.

---

## Abstract (verbatim, page 1)

> "We construct a large laboratory of over two million trading strategies by data-mining the two most commonly used datasets in finance, viz. CRSP and COMPUSTAT. We use this very large sample for three purposes. First, we evaluate the properties of multiple hypothesis testing methods when applied to financial data. We find that only adaptive methods (for example, FDP-SetpM) should be employed in finance applications. Second, we provide an optimal thresholding for applications that evaluate trading strategies. Third, we quantify the proportion of false discoveries due to the failure to account for testing a multitude of hypotheses. Our estimates for the proportion of lucky discoveries is over 90%, which is considerably larger than that previously reported."

## On thresholds (page 5)

> "Imposing a tolerance of 5% of false discoveries (false discovery proportion) and a significance level of 5%, we find that the MHT threshold for alpha t-statistic (t_α) is **3.79** while that for FM coefficient t-statistic (t_λ) is **3.12**. While these thresholds are quite a bit higher than the conventional thresholds of 1.96, they are not far from the suggestion of Harvey, Liu, and Zhu (2015) to use a threshold of three. Our higher thresholds are due to our choice of a different MHT method, our sample of over two million strategies vis-à-vis 316 strategies in Harvey, Liu, and Zhu, and the fact that we fully account for dependence in the data. At these thresholds, 2.67% of strategies have significant alphas and 16.31% have significant FM coefficients."

## On false discovery proportion (page 6)

> "We conclude that, in our experiment, the great majority of the discoveries (i.e., rejections of the null of no predictability) that are made by relying on CHT and without accounting for the very large number of strategies that are never made public, are very likely false. In the case of alphas, that percentage can be as high as **91% (= 1 − 2.67/30.36)**, while the problem is less severe for FM coefficients, although it could still be as high as **59% (= 1 − 16.31/39.33)**."

## On consistency requirement (page 6)

> "In order to gauge some consistency between performance measures we ask of a trading signal to not only generate a high long-short portfolio alpha but also to explain the broader cross-section of returns in a regression setting. Eliminating strategies that have statistically significant t_α but insignificant t_λ, or vice-versa, drastically reduces the number of successful strategies to 806 (i.e., 0.04% of the total) under MHT and to 33,881 (i.e., 1.62% of the total) under CHT. **The lower bound on the proportion of lucky rejections remains very high, north of 95%.** Notably, the very large proportion of lucky discoveries is constant across many ways of classifying the trading strategies according to their strength: average return, alpha, Sharpe ratio or information ratio."

---

## Methodology — which MHT methods work (pages 3-4)

> "We consider the three most common approaches: family-wise error rate (FWER), false discovery ratio (FDR), and false discovery proportion (FDP). FWER controls the probability of making more than one false rejection, FDP controls the probability of a user-specified proportion of false rejections in a given sample, while FDR controls the expected (across different samples) proportion of false rejections. We concentrate on a total of five methods: two that control FWER (i.e., Bonferroni and Holm); two that control FDR (i.e., BH and BHY); and one that controls FDP (i.e., FDP-StepM)."

> "We find that, while all MHT methods have good size properties, they differ in their power and their ability to adapt to situations where the proportion of true alternatives might be high. In particular, we find that only one of the FDR methods (the FDR-BH procedure) and the FDP method (the FDP-StepM procedure) are reliable in terms of power properties; the other MHT methods have substantially low power (due to very high thresholds that they impose on the data). We also find that the magnitude of thresholds (and the corresponding rejection rates) are primarily dictated by the signal-to-noise ratio (magnitude of true alphas relative to return volatility) as well as the fraction of true rejections. The FDR-BH and FDP-StepM methods are adaptive (but the other MHT methods are not) in the sense that they produce lower critical values if the true number of rejections in the data is high."

## Correlation matters (pages 4-5)

> "Besides the conceptual distinction in what they are trying to control, the FDR and FDP methods also differ in their underlying assumptions. For our purposes, an important assumption is that of non-zero correlation between strategies. Trading strategies are not independent of each other, as there is cross-correlation in stock returns across different firms and in the information used to construct the signals, not only across different firms but also within a particular firm (i.e., total assets and profitability are not independent). The FDR methods make strong assumptions about the correlation structure of strategies whereas the FDP method delivers statistical cutoffs that account for the cross-correlations present in the data. **Therefore, we rely on the FDP method more heavily.**"

## FDR-BH foundation (page 2 footnote)

> "The BH procedure is from Benjamini and Hochberg (1995). The BHY procedure is a combination of Benjamini and Hochberg, and Benjamini and Yekutieli (2001). FDP-StepM was developed by Romano and Wolf (2007)."

---

## Bootstrap vs MHT (page 16-17 verbatim)

> "Overall, we find that actual t-statistics of percentiles in the tails do not appear to be drawn from the same percentile's distribution generated from zero alpha or zero FM coefficients. Yan and Zheng (2017) obtain very similar results based on their sample of 18,000 strategies. It is a little surprising that the bootstrap method 'rejects' so many strategies — we find that the likelihood/p-value of t-statistics appearing from the null is close to zero for all percentiles below 40 and above 60. These rejection rates are even higher than those reported using classical thresholds. For example, Panel A of Table 2 shows rejection rates of only 16.44% for t_α and 25.19% for t_λ at the high threshold of 2.57 for a significance level of 1%."

## The key comparison (implicit)

- **Classical 1.96 threshold (CHT):** rejects ~30% of strategies. Lower bound on false positives = 91%.
- **1% significance t=2.57 (still CHT):** rejects ~16-25% depending on metric. Still too lax.
- **HLZ proposed t=3.0:** recommended by Harvey-Liu-Zhu (2015), still under-correcting vs. Chordia's 2M sample.
- **Chordia MHT optimal: t=3.79** for alphas (FDP approach with correlation control).

---

## FWER Methods (Section 4.1, pages 18-19)

### Bonferroni (page 19)
> "The Bonferroni method, at level α, rejects H_m if p_m ≤ α/M. The Bonferroni method is a single-step procedure because all p-values are compared to a single critical value. This critical p-value is equal to α/M. For a very large number of strategies, this leads to an extremely small (large) critical p-value (t-statistic). While widely used for its simplicity, the biggest disadvantage of the Bonferroni method is that it is very conservative and leads to a loss of power. One of the main reasons for the lack of power is that the Bonferroni method implicitly treats all test statistics as independent and, consequently, ignores the cross-correlations that are bound to be present in most financial applications."

### Holm (page 19)
> "This is a stepwise method based on Holm (1979) and works as follows. The null hypothesis H_i is rejected at level α if p_i ≤ α/(M - i + 1) for i = 1, ..., M. In comparison with the Bonferroni method, the criterion for the smallest p-value is equally strict at α/M but it becomes less and less strict for larger p-values. Thus, the Holm method will typically reject more hypotheses and is more powerful than the Bonferroni method. However, because it also does not take into account the dependence structure of the individual p-values, the Holm method is also very conservative."

### Bootstrap reality check (page 19)
> "Bootstrap reality check (BRC) is based on White (2000). The idea is to estimate the sampling distribution of the largest test statistic taking into account the dependence structure of the individual test statistics, thereby asymptotically controlling FWER."

---

## Application to our project (synthesis)

### Where our FDR threshold sits

Our `validated_setups.fdr_adjusted_p` uses BH at q=0.05 (per code in `trading_app/strategy_discovery.py`). The implied t-statistic at that threshold is roughly **1.96 to 2.5**, depending on strategy size.

Deployed 5 MNQ lanes, approximate implied t-statistics from fdr_adjusted_p:
- MGC: fdr_p = 0.0259 → t ≈ 2.22
- MNQ COMEX: fdr_p = 0.00017 → **t ≈ 3.77** ← only one clearing Chordia's t ≥ 3.79
- MNQ EUR: fdr_p = 0.0353 → t ≈ 2.10
- MNQ SIN: fdr_p = 0.0248 → t ≈ 2.24
- MNQ TOK: fdr_p = 0.0411 → t ≈ 2.04

**Only MNQ COMEX_SETTLE OVNRNG_100 clears the Chordia threshold.** The other 4 are in the "likely false discovery" zone per Chordia's 91-95% false-positive estimate.

### Policy implication for our pre-registered criteria

- **Upgrade threshold:** require t ≥ 3.00 (HLZ) at minimum for discovery acceptance, t ≥ 3.79 (Chordia) for strategies without strong economic prior
- **Adaptive method:** use BH-FDR at q=0.05 AS A FIRST FILTER, with post-hoc FDP verification for the final deployed set
- **Consistency:** require both high t and high Sharpe (analog of Chordia's "both alpha AND FM coefficient significant")

---

## Related literature
- `bailey_et_al_2013_pseudo_mathematics.md` — MinBTL constraint upstream of this
- `bailey_lopez_de_prado_2014_deflated_sharpe.md` — DSR is an alternative to MHT corrections
- `harvey_liu_2015_backtesting.md` — HLZ threshold of 3 that Chordia references

# Backtesting — Harvey & Liu (2015)

**Source:** `resources/backtesting_dukepeople_liu.pdf`
**Authors:** Campbell R. Harvey (Duke University), Yan Liu (Texas A&M University)
**Publication:** *The Journal of Portfolio Management*, Fall 2015, pp. 12–28
**Extracted:** 2026-04-07
**Pages read:** 1-17 (full paper excerpted)

**Criticality:** 🔴 **HIGH** — provides the analytical Sharpe ratio haircut framework under multiple testing, including specific profitability hurdles we can apply directly.

---

## Opening (verbatim, page 12)

> "A common practice in evaluating backtests of trading strategies is to discount the reported Sharpe ratios by 50%. There are good economic and statistical reasons for reducing the Sharpe ratios. The discount is a result of data mining. This mining may manifest itself in academic researchers searching for asset-pricing factors that explain the behavior of equity returns, or by researchers at firms that specialize in quantitative equity strategies trying to develop profitable systematic strategies.
>
> The 50% haircut is only a rule of thumb. Our article's goal is to develop an analytical way to determine the haircut's magnitude."

## Why 50% is wrong (page 12)

> "We argue that it is a serious mistake to use the usual 50% haircut. Our results show that the multiple testing haircut is nonlinear. The highest Sharpe ratios are only moderately penalized, while the marginal Sharpe ratios are heavily penalized. This makes economic sense. The marginal strategies should be thrown out. The strategies with very high Sharpe ratios are probably true discoveries. In these cases, a 50% haircut is too punitive."

---

## Equation 4 — Multiple-testing-adjusted p-value (page 14 verbatim)

> "To quantitatively evaluate this overstatement, we assume that researchers have tried N strategies and present the most profitable one (that is, the one with the largest Sharpe ratio). Additionally, we assume (for now) that the test statistics for these N strategies are independent. Under these simplifying assumptions and under the null hypothesis that none of these strategies can generate non-zero returns, the multiple testing p-value, p^M, for observing a maximal t-statistic that is at least as large as the observed t-ratio is:
>
> **p^M = Pr(max{|r_i|, i = 1, ..., N} > t-ratio) = 1 - ∏_{i=1}^N Pr(|r_i| ≤ t-ratio) = 1 - (1 - p^S)^N**   (Equation 4)"

## Equation 5 — Haircut Sharpe ratio (page 14)

> "By equating the p-value of a single test to p^M, we obtain the defining equation for the multiple testing adjusted (haircut) Sharpe ratio HSR:
>
> **p^M = Pr(|r| > HSR·√T)**   (Equation 5)"

## Illustration (page 14)

> "For instance, assuming there are twenty years of monthly returns (T = 240), an annual Sharpe ratio of 0.75 yields a p-value of 0.0008 for a single test. When N = 200, p^M = 0.15, implying an adjusted annual Sharpe ratio of 0.32 through equation 5. Hence, multiple testing with 200 tests reduces the original Sharpe ratio by approximately **60% (=(0.75−0.32)/0.75)**."

---

## Three MHT methods compared (pages 15-17)

### Bonferroni (page 15)
> "Bonferroni's method adjusts each p-value equally. It inflates the original p-value by the number of tests M:
>
> **Bonferroni: p^Bonferroni_(i) = min[Mp_(i), 1], i = 1, ..., M**"

### Holm (page 15)
> "Holm's method relies on the sequence of p-values and adjusts each p-value by:
>
> **Holm: p^Holm_(i) = min[max_{j≤i}{(M-j+1)p_(j)}, 1], i = 1, ..., M**"

### Benjamini-Hochberg-Yekutieli BHY (page 16)
> "Benjamini, Hochberg and Yekutieli (BHY)'s procedure defines the adjusted p-values sequentially:
>
> **BHY: p^BHY_(i) = p_(M) if i = M, else min[ p^BHY_(i+1), (M·c(M)/i)·p_(i) ] if i ≤ M-1**
>
> where c(M) = ∑_{j=1}^M 1/j."

> "Hypothesis tests based on the adjusted p-values guarantee that the false discovery rate (FDR) does not exceed the pre-specified significance level. [...] In the original work by Benjamini and Hochberg [1995], c(M) is set equal to one and the test works when p-values are independent or positively dependent. We adopt the choice in Benjamini and Yekutieli [2001] by setting c(M) equal to ∑_{j=1}^M 1/j. This allows our test to work under arbitrary dependency for the test statistics."

## Conclusion on which method to use (page 20)

> "In the end, we advocate the BHY method. The FWER seems appropriate for applications where a false discovery brings a severe consequence. In financial applications, it seems reasonable to control for the rate of false discoveries, rather than the absolute number."

---

## Exhibit 4 — Minimum Profitability Hurdles (page 22 verbatim)

> "Average monthly return hurdles under single and multiple tests. At 5% significance, the table shows the minimum average monthly return for a strategy to be significant at 5% with 300 tests. All numbers are in percentage terms."
>
> | | σ = 5% | σ = 10% | σ = 15% |
> |---|---|---|---|
> | **Panel A: Observations = 120** | | | |
> | Single | 0.258 | 0.516 | 0.775 |
> | Bonferroni | 0.496 | 0.992 | 1.488 |
> | Holm | 0.486 | 0.972 | 1.459 |
> | BHY | 0.435 | 0.871 | 1.305 |
> | **Panel B: Observations = 240** | | | |
> | Single | 0.183 | 0.365 | 0.548 |
> | Bonferroni | 0.351 | 0.702 | 1.052 |
> | Holm | 0.344 | 0.688 | 1.031 |
> | BHY | 0.307 | 0.616 | 0.923 |
> | **Panel C: Observations = 480** | | | |
> | Single | 0.129 | 0.258 | 0.387 |
> | Bonferroni | 0.248 | 0.496 | 0.744 |
> | Holm | 0.243 | 0.486 | 0.729 |
> | BHY | 0.217 | 0.435 | 0.651 |
> | **Panel D: Observations = 1000** | | | |
> | Single | 0.089 | 0.179 | 0.268 |
> | Bonferroni | 0.172 | 0.344 | 0.516 |
> | Holm | 0.169 | 0.337 | 0.505 |
> | BHY | 0.151 | 0.302 | 0.452 |

## Key takeaway quote (page 22)

> "Exhibit 4 shows the large differences between the return hurdles for single testing and multiple testing. For example, in panel B (240 observations) and 10% volatility, the minimum required average monthly return for a single test is 0.365% per month, or 4.4% annually. However, for BHY, the return hurdle is much higher: 0.616% per month, or 7.4% annually."

## Caveats (page 13)

> "Our method does have a number of caveats, some of which apply to any use of the Sharpe ratio. First, high observed Sharpe ratios could be the results of non-normal returns, for instance, an option-like strategy with ex ante negative skew. In this case, Sharpe ratios should be viewed in the context of the skew.
>
> Second, Sharpe ratios do not necessarily control for risk. That is, the strategy's volatility may not reflect the true risk."

## In-sample multiple testing vs OOS validation (page 17)

> "Our multiple-testing adjustment is based on in-sample (IS) backtests. In practice, out-of-sample (OOS) tests are routinely used to select among many strategies.
>
> Despite its popularity, OOS testing has several limitations. First, an OOS test may not be truly out of sample. A researcher tries a strategy. After running an OOS test, she finds that the strategy fails. She then revises the strategy and tries again, hoping it will work this time. This trial and error approach is not truly OOS, but the difference is hard for outsiders to see.
>
> Second, an OOS test, like any other test in statistics, only works in a probabilistic sense. In other words, an OOS test's success can be due to luck for both the in-sample selection and the out-of-sample testing. Third, given that the researcher has experienced the data, there is no true OOS that uses historical data."

---

## Application to our project (synthesis)

### BHY hurdle at our horizon
Using Exhibit 4 Panel B (240 obs, 10% vol, 300 tests assumed): **BHY requires 0.616% monthly = 7.4% annual return** to be significant.

Our data horizon in months:
- 2.2 years clean MNQ = 26 months (closer to Panel A)
- 16 years proxy = 192 months (closer to Panel B)

At 26 months (roughly between Panels A and B), with assumed 300 tests and 10% vol, BHY hurdle is ~0.9-1.0% monthly = ~11-12% annual.

Our deployed lane ExpR range (converted monthly): a lane earning ~9 R/yr at median $40 risk/R × 5 contracts = $1,800/yr gross on a $50K account = **0.3% monthly**.

**We are below the BHY hurdle at 300-test MHT correction.** This aligns with the Bailey-LdP MinBTL finding: our test count is too high for our data horizon.

### The fix
Reduce tests to make BHY hurdle achievable. Per-registered 50-100 trials (not 300) would lower the hurdle to closer to single-test levels (~0.18-0.36% monthly ≈ 2-4% annual), which our deployed lanes can clear.

---

## Related literature
- `bailey_lopez_de_prado_2014_deflated_sharpe.md` — alternative (more general) adjustment via DSR
- `chordia_et_al_2018_two_million_strategies.md` — empirical validation at 2.4M strategy scale
- `bailey_et_al_2013_pseudo_mathematics.md` — data length constraint via MinBTL

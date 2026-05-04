# The Deflated Sharpe Ratio — Bailey & López de Prado (2014)

**Source:** `resources/deflated-sharpe.pdf`
**Authors:** David H. Bailey, Marcos López de Prado
**Publication:** *Journal of Portfolio Management*, Forthcoming, 2014
**First version:** April 15, 2014; this version: July 31, 2014
**Extracted:** 2026-04-07
**Pages read:** 1-20 (full paper)

**Criticality:** 🔴 **MAXIMUM** — provides the Deflated Sharpe Ratio formula that corrects for both selection bias under multiple testing AND non-normal returns. This is the stat we should be computing on every validated strategy.

---

## Abstract (verbatim, page 2)

> "With the advent in recent years of large financial data sets, machine learning and high-performance computing, analysts can backtest millions (if not billions) of alternative investment strategies. Backtest optimizers search for combinations of parameters that maximize the simulated historical performance of a strategy, leading to *backtest overfitting*.
>
> The problem of performance inflation extends beyond backtesting. More generally, researchers and investment managers tend to report only positive outcomes, a phenomenon known as *selection bias*. Not controlling for the number of trials involved in a particular discovery leads to over-optimistic performance expectations.
>
> The *Deflated Sharpe Ratio* (DSR) corrects for two leading sources of performance inflation: Selection bias under multiple testing and non-Normally distributed returns. In doing so, DSR helps separate legitimate empirical findings from statistical flukes."
>
> **Keywords:** Sharpe ratio, Non-Normality, Probabilistic Sharpe ratio, Backtest overfitting, Minimum Track Record Length, Minimum Backtest Length.

## Core quote on multiple testing (page 3)

> "Put bluntly, *a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless, regardless of how excellent the reported performance might be*. Investors and journal referees should demand this information whenever a backtest is submitted to them, although even this will not remove the danger completely."

---

## Equation 1 — Expected maximum Sharpe ratio (page 7)

> "Consider a set of N independent backtests or track records associated with a particular strategy class (e.g., Discretionary Macro). Each element of the set is called a *trial*, and it is associated with a SR estimate, ŜR_n, with n = 1, ..., N. Suppose that these trials' {ŜR_n} follow a Normal distribution, with mean E[{ŜR_n}] and variance V[{ŜR_n}]. [...] Appendix 1 proves that, under these assumptions, the expected maximum of {ŜR_n} after N ≫ 1 independent trials can be approximated as:
>
> **E[max{ŜR_n}] ≈ E[{ŜR_n}] + √V[{ŜR_n}] · ((1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(N·e)])**   (Equation 1)
>
> where γ (approx. 0.5772) is the Euler-Mascheroni constant, Z is the cumulative function of the standard Normal distribution, and e is Euler's number."

## Equation 2 — The Deflated Sharpe Ratio formula (page 8)

> "We propose a *Deflated Sharpe Ratio* (DSR) statistic that corrects for both sources of ŜR inflation, defined as:
>
> **DSR ≡ PŜR(ŜR_0) = Z[ ((ŜR - ŜR_0)·√(T-1)) / √(1 - γ̂₃·ŜR + (γ̂₄-1)/4 · ŜR²) ]**   (Equation 2)
>
> where ŜR_0 = √V[{ŜR_n}] · ((1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(Ne)]), V[{ŜR_n}] is the variance across the trials' estimated SR and N is the number of independent trials. We also use information concerning the selected strategy: ŜR is its estimated SR, T is the sample length, γ̂₃ is the skewness of the returns distribution and γ̂₄ is the kurtosis of the returns distribution for the selected strategy. Z is the cumulative function of the standard Normal distribution."

## Variables deflated by DSR (page 9)

> "Note that the standard ŜR is computed as a function of two estimates: Mean and standard deviation of returns. DSR deflates SR by taking into consideration five additional variables: The non-Normality of the returns (γ̂₃, γ̂₄), the length of the returns series (T), the variance of the SRs tested (V[{ŜR_n}]), as well as the number of independent trials involved in the selection of the investment strategy (N)."

---

## Numerical example (pages 9-10 verbatim)

> "Suppose that a strategist is researching seasonality patterns in the treasury market. He believes that the U.S. Treasury's auction cycle creates inefficiencies that can be exploited by selling off-the-run bonds a few days before the auction, and buying the new issue a few days after the auction. He backtests alternative configurations of this idea, by combining different pre-auction and post-auction periods, tenors, holding periods, stop-losses, etc. He uncovers that many combinations yield an annualized ŜR of 2, with a particular one yielding a ŜR of 2.5 over a daily sample of 5 years.
>
> Excited by this result, he calls an investor asking for funds to run this strategy, arguing that an annualized ŜR of 2.5 must be statistically significant. The investor, who is familiar with a paper recently published by the *Journal of Portfolio Management*, asks the strategist to disclose: i) The number of independent trials carried out (N); ii) the variance of the backtest results (V[{ŜR_n}]); iii) the sample length (T); and iv) the skewness and kurtosis of the returns (γ̂₃, γ̂₄). The analyst responds that N = 100, V[{ŜR_n}] = 1/2, T = 1250, γ̂₃ = -3 and γ̂₄ = 10.
>
> Shortly after, the investor declines the analyst's proposal. Why? Because the investor has determined that this is not a legitimate empirical discovery at a 95% confidence level. In particular, ŜR_0 = √(1/(2·250)) · ((1-γ)·Z⁻¹[1 - 1/100] + γ·Z⁻¹[1 - 1/(100·e⁻¹)]) ≈ 0.1132, non-annualized (with 250 observations per year), and DSR ≈ Z[((2.5/√250 - 0.1132)·√1249) / √(1 - (-3)·2.5/√250 + (10-1)/4·(2.5/√250)²)] ≈ 0.9004 < 0.95."

> "Exhibit 2 plots how the rejection threshold ŜR_0 increases with N, and consequently DSR decreases. The investor has recognized that there is only a 90% chance that the true SR associated with this strategy is greater than zero. Should the strategist have made his discovery after running only N=46 independent trials, the investor may have allocated some funds, as DSR would have been 0.9505, above the 95% confidence level.
>
> Non-Normality also played a role in discarding this investment offer. If the strategy had exhibited Normal returns (γ̂₃ = 0, γ̂₄ = 3), DSR = 0.9505 after N=88 independent trials. If non-Normal returns had not inflated the performance so much, the investor would have been willing to accept a much larger number of trials. This example illustrates that it is critical for investors to account for both sources of performance inflation jointly, as DSR does."

---

## Exhibit 1 — Expected Max SR vs number of trials (page 16)

Chart showing E[{ŜR_n}] = 0, V[{ŜR_n}] ∈ {1, 4}, N ∈ [10, 1000]. At V=1, expected max SR grows from ~1.5 at N=10 to ~3.3 at N=1000. At V=4, expected max SR grows from ~3 at N=10 to ~6.5 at N=1000.

## Exhibit 2 — SR0 and DSR vs N (page 17)

Two panels showing the rejection threshold SR0 (annualized) and DSR as N grows:
- At N=10, SR0 ≈ 1.0 annualized for DSR to equal 0.95
- At N=100, SR0 ≈ 1.5
- At N=1000, SR0 ≈ 1.8
- Beyond N=100, the DSR curve drops below 0.95 quickly

## Exhibit 4 — Implied independent trials N̂ from M correlated trials (page 20)

Chart showing N̂ as a function of average correlation ρ̂ and number of trials M. For ρ̂ = 0.5 and M = 10,000, N̂ ≈ 5,000. For ρ̂ = 0.9, N̂ ≈ 1,000.

---

## Appendix A.3 — Estimating the number of independent trials (pages 14-15)

> "It is critical to understand that the N used to compute E[max{SR_n}] corresponds to the number of *independent* trials. Suppose that we run M trials, where only N trials are independent, N<M. Clearly, using M instead of N will overstate E[max{SR_n}]. So given M dependent trials we need to derive the number of 'implied independent trials', N̂.
>
> One path to accomplish that is by taking into account the average correlation between the trials, ρ. [...] The implication is that the average correlation is bounded by ρ ∈ (-1/(M-1), 1], with M>1 for a correlation to exist. The larger the number of trials, the more positive the average correlation is likely to be, and for a sufficiently large M we have -1/(M-1) ≈ 0 < ρ ≤ 1.
>
> Third, we know that as ρ → 1, then N → 1. Similarly, as ρ → 0, then N → M. Given an estimated average correlation ρ̂, we could therefore interpolate between these two extreme outcomes to obtain:
>
> **N̂ = ρ̂ + (1 - ρ̂)·M**   (Equation 9)"

---

## When should we stop testing — "secretary problem" (page 10)

> "An elegant answer to this critical question can be found in the theory of optimal stopping, more concretely the so called 'secretary problem', or 1/e-law of optimal choice, see Bruss [1984]. [...] In the context of our discussion, it translates as follows: From the set of strategy configurations that are theoretically justifiable, sample a fraction 1/e of them (roughly 37%) at random and measure their performance. After that, keep drawing and measuring the performance of additional configurations from that set, one by one, until you find one that beats all of the previous. That is the optimal number of trials, and that 'best so far' strategy the one that should be selected."

---

## Conclusions (page 11)

> "In this paper we have proposed a test to determine whether an estimated SR is statistically significant after correcting for two leading sources of performance inflation: Selection bias and non-Normal returns. The Deflated Sharpe Ratio (DSR) incorporates information about the unselected trials, such as the number of independent experiments conducted and the variance of the SRs, as well as taking into account the sample length, skewness and kurtosis of the returns' distribution."

---

## Application to our project (CLEARLY SEPARATE FROM VERBATIM EXTRACTS)

### Variables we need to compute DSR for each deployed lane

1. **ŜR** — annualized Sharpe ratio of the deployed strategy (we have this stored)
2. **T** — sample length (number of return observations, likely trades per day × trading days)
3. **V[{ŜR_n}]** — variance of Sharpe across all discovery trials
4. **N** — number of independent trials (via Eq. 9 with ρ̂)
5. **γ̂₃, γ̂₄** — skewness and kurtosis of strategy returns

### Implementation gap

**As of 2026-04-07, we do not compute DSR in our pipeline.** `validated_setups` stores `dsr_score` and `sr0_at_discovery` columns but the computation logic needs to be verified. A drift check should confirm DSR > 0.95 for any strategy claimed as FDR-significant.

### The corrective

- Compute variance of Sharpe across discovery trials (sample-level, not per-instrument)
- Compute N̂ via correlation adjustment (we have 35,000 raw M but likely <500 effective N̂)
- Compute per-strategy DSR using Eq. 2
- Reject strategies with DSR < 0.95

**Locked in `pre_registered_criteria.md` as the primary strategy-selection gate.**

---

## Related literature

- `bailey_et_al_2013_pseudo_mathematics.md` — the prior paper establishing the overfitting problem this paper solves
- `lopez_de_prado_bailey_2018_false_strategy.md` — formal proof of the expected max SR formula
- `harvey_liu_2015_backtesting.md` — alternative framework (Bonferroni/Holm/BHY haircuts)
- `chordia_et_al_2018_two_million_strategies.md` — empirical test of MHT thresholds at scale

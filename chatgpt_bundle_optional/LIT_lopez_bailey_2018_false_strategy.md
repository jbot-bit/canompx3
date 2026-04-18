# The False Strategy Theorem — López de Prado & Bailey (2018)

**Source:** `resources/false-strategy-lopez.pdf`
**Authors:** Marcos López de Prado, David H. Bailey
**Publication:** *American Mathematical Monthly*, Vol. 125, No. 7 (August–September 2018)
**Extracted:** 2026-04-07
**Pages read:** 1-7 (full paper)

**Criticality:** 🔴 **MAXIMUM** — formalizes the mathematical proof that N >> 1 guarantees false positives.

---

## Abstract (verbatim, page 1)

> "The late Jonathan M. Borwein excelled at a wide range of mathematical fields. He is perhaps best known for his work in experimental mathematics and optimization. But his interests extended far beyond these two arenas, to often unexpected topics. Unbeknownst to many, he also made important contributions to mathematical finance, and additionally published studies concerned with the reproducibility of scientific discoveries in numerous different fields. In this paper we have attempted to merge some of these seemingly unrelated topics, elucidating a common thread connecting them all."

## Section 1 — The Reproducibility Crisis in Finance (page 1)

> "The reproducibility crisis in finance is primarily the consequence of three factors. First, once a theory is published, it may take many decades to collect the future ('out-of-sample') information needed to evaluate the accuracy of the proposed theory. It takes one day of markets activity to produce one day's worth of data. Second, finance is not a static system. Even if we collect enough information to debunk a theory, it is possible that the theory was correct at the time of publication, but that changes in the system since publication render it no longer valid. Or perhaps the theory was false to begin with — we may never know. Third, we cannot repeat an experiment over and over while controlling for specific environmental variables, so as to identify a precise cause-effect mechanism. All we have is a single historical path."

## Section 2 — Selection Bias Under Multiple Testing (page 2)

> "Even at the present date most academic finance journals do not require authors to declare the number of trials involved in a discovery, even though the authors may well have performed an extensive computer search for optimal parameters, effectively iterating over millions of possibilities. Journals operate under the demonstrably false assumption that researchers have carried out a single test. The sobering implication of this fact is that many discoveries in finance may be false, and it may be very difficult to determine which are true."

---

## Theorem 1 — False Strategy Theorem (page 3 verbatim)

> **Theorem 1 (False strategy theorem).** *Given a sample of estimated performance statistics {ŜR_k}, k = 1, ..., K, with independent and identically distributed Gaussian distribution, i.e., {ŜR_k} ~ N[0, V[{ŜR_k}]], then*
>
> **E[max_k {ŜR_k}] · (V[{ŜR_k}])^(-1/2) ≈ (1-γ)·Z⁻¹[1 - 1/K] + γ·Z⁻¹[1 - 1/(Ke)]**   (Equation 1)

### Proof sketch (page 3)

> "It is known that the maximum value in a sample of independent random variables following an exponential distribution converges asymptotically to a Gumbel distribution. For a proof, see [5, pp. 138–147]. As a particular case, the Gumbel distribution covers the maximum domain of attraction of the Gaussian distribution, and therefore it can be used to estimate the expected value of the maximum of several independent random Gaussian variables."
>
> "[After applying Fisher-Tippet-Gnedenko:] For a sufficiently large K, the mean of the sample maximum of standard normally distributed random variables can be approximated by:
>
> **E[max_k {y_k}] ≈ α + γβ = (1-γ)·Z⁻¹[1 - 1/K] + γ·Z⁻¹[1 - 1/(Ke)]**   (Equation 4)
>
> where K ≫ 1."

## Figure 1 (page 5 verbatim)

> "Comparison of experimental and theoretical results from the False Strategy theorem. Experimental maximum Sharpe ratios (for Monte Carlo trials where the true Sharpe ratio is zero) are plotted in color. Results with higher probability are cast in a lighter color. The dashed line are the results predicted by the False Strategy theorem."

### Key data points from Figure 1
- At K=10, expected max Sharpe ≈ 1.5
- At K=100, expected max Sharpe ≈ 2.3
- At K=1000, expected max Sharpe ≈ 3.26 (explicit quote below)
- At K=1,000,000, expected max Sharpe ≈ 4.8

## Experimental verification quote (page 4)

> "For instance, if we conduct K = 1000 trials, the expected maximum Sharpe ratio E[max_k {ŜR_k}] is 3.26, even though the true Sharpe ratio of the strategy is zero. As expected, there is a rising hurdle that the researcher must beat as he or she conducts more backtests."

## Approximation error (page 5)

> "From this result, it appears that the False Strategy theorem produces asymptotically unbiased estimates. Only at K ≈ 50, the theorem's estimate exceeds the experimental value by approximately 0.7%. [...] From this experiment, we can deduce that the standard deviations are relatively small, below 0.5% of the values forecasted by the theorem, and they become smaller as the number of trials raises. These error estimates constitute upper boundaries, because the estimated errors would be smaller if we increased the number of Monte Carlo simulations."

---

## Section 6 — Conclusion (page 6 verbatim)

> "The main conclusion from the False Strategy theorem is that, unless max_k {ŜR_k} >> E[{ŜR_k}], the discovered strategy is likely to be a *false positive*. The result can be used in connection with other techniques, such as the deflated Sharpe ratio [3], which estimates the probability of a false positive.
>
> In real-world finance, the False Strategy theorem tells us that the optimal outcome of an unknown number of historical simulations is right-unbounded — with enough trials, there is no Sharpe ratio sufficiently large enough to reject the hypothesis that a strategy is false. Given the ease with which one can use a computer to explore many trials or variations of given strategies and only select the optimal variation, it follows that it is very easy to find impressive-looking strategy variations that are nothing more than false positives. This is the essence of *selection bias under multiple testing*."

## Policy implications (page 6)

> "First, academic journals should cease to accept or publish articles that do not disclose the number of trials involved in a discovery. Second, financial regulators should withdraw their license from asset managers who publicize financial products that have not been rigorously vetted for selection bias under multiple testing. Third, investors should demand that the probability of a false discovery be reported with every product offering."

---

## Application to our project (synthesis, not verbatim)

**Our April 2026 discovery ran ~35,616 MNQ trials.** Applying Eq. 4:
- Z⁻¹[1 - 1/35616] ≈ Z⁻¹[0.99997] ≈ 4.01
- Z⁻¹[1 - 1/(35616·e)] ≈ 3.75
- E[max_k {ŜR_k}] ≈ 0.423·4.01 + 0.577·3.75 = **3.87**

**Interpretation:** Under the null hypothesis that our filter set has zero true edge, we would expect the best observed Sharpe to be ~3.87 just from random selection. Our best deployed strategy (COMEX OVNRNG_100) has annualized Sharpe 1.23. **We are well below the noise floor.**

Note this assumes normality (which our returns are not — fat-tailed) and independence (which our correlated filters are not). DSR extends this to non-normal, correlated cases per `bailey_lopez_de_prado_2014_deflated_sharpe.md`.

---

## Related literature
- `bailey_lopez_de_prado_2014_deflated_sharpe.md` — DSR extends this to non-normal, non-independent trials
- `bailey_et_al_2013_pseudo_mathematics.md` — MinBTL Theorem derived from same foundation

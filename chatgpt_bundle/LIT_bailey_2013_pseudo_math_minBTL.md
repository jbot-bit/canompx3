# Pseudo-Mathematics and Financial Charlatanism — Bailey, Borwein, López de Prado, Zhu (2013)

**Source:** `resources/Pseudo-mathematics-and-financial-charlatanism.pdf`
**Authors:** David H. Bailey, Jonathan M. Borwein, Marcos López de Prado, Qiji Jim Zhu
**Publication:** *Notices of the American Mathematical Society*, Vol. 61, No. 5 (2014), pp. 458–471
**First version:** September 2013
**Extracted:** 2026-04-07

**Criticality for our project:** 🔴 **MAXIMUM** — this paper contains the **Minimum Backtest Length (MinBTL) Theorem** which directly bounds how many strategies we can test on our 6-year MNQ dataset. Our prior discovery methodology (35,000+ combinations) violates the bound by ~600x.

---

## Abstract (verbatim, page 2)

> "Recent computational advances allow investment managers to search for profitable investment strategies. In many instances, that search involves a pseudo-mathematical argument, which is spuriously validated through a simulation of its historical performance (also called backtest).
>
> We prove that high performance is easily achievable after backtesting a relatively small number of alternative strategy configurations, a practice we denote 'backtest overfitting.' The higher the number of configurations tried, the greater is the probability that the backtest is overfit. Because financial analysts rarely report the number of configurations tried for a given backtest, investors cannot evaluate the degree of overfitting in most investment proposals.
>
> The implication is that investors can be easily misled into allocating capital to strategies that appear to be mathematically sound and empirically supported by an outstanding backtest. This practice is particularly pernicious, because due to the nature of financial time series, backtest overfitting has a detrimental effect on the future strategy's performance."

## Feynman epigraph (page 3)

> "Another thing I must point out is that you cannot prove a vague theory wrong. […] Also, if the process of computing the consequences is indefinite, then with a little skill any experimental result can be made to look like the expected consequences." — Richard Feynman [1964]

---

## Proposition 1 — Expected maximum Sharpe Ratio under zero true edge (page 7)

> **PROPOSITION 1:** Given a sample of IID random variables, x_n ~ Z, n = 1, …, N, where Z is the CDF of the Standard Normal distribution, the expected maximum of that sample, E[max_N] = E[max{x_n}], can be approximated for N ≫ 1 as:
>
> **E[max_N] ≈ (1 - γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(Ne)]**   (Equation 4)
>
> where γ (approx. 0.5772156649) is the Euler-Mascheroni constant.

**Plain reading:** If you run N independent backtests on data where the true Sharpe is zero, the expected maximum Sharpe you'll observe GROWS with N. For N=10 trials, expected max SR IS is **1.57 annualized** despite all strategies having zero true edge.

---

## Theorem 1 — Minimum Backtest Length (MinBTL) (page 8)

> **THEOREM 1:** The Minimum Backtest Length (MinBTL, in years) needed to avoid selecting a strategy with an IS Sharpe ratio of E[max_N] among N independent strategies with an expected OOS Sharpe ratio of zero is:
>
> **MinBTL ≈ ((1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(Ne)])² / E[max_N]² < 2·Ln[N] / E[max_N]²**   (Equation 6)

**Plain reading:** If you want your best observed IS Sharpe to NOT be the result of random luck among N trials, you need MinBTL years of data. The formula scales with **ln(N)**, not linearly.

## The 5-year / 45-trials rule (page 8)

> "Eq. (6) tells us that MinBTL must grow as the researcher tries more independent model configurations (N), in order to keep constant the expected maximum Sharpe ratio at a given level E[max_N]. Figure 2 shows how many years of backtest length (MinBTL) are needed so that E[max_N] is fixed at 1. **For instance, if only 5 years of data are available, no more than 45 independent model configurations should be tried, or we are almost guaranteed to produce strategies with an annualized Sharpe ratio IS of 1, but an expected Sharpe ratio OOS of zero.** Note that Proposition 1 assumed the N trials to be independent, which leads to a quite conservative estimate."

---

## Proposition 2 — Compensation effect under global constraint (page 12)

> **PROPOSITION 2:** Given two alternative configurations (A and B) of the same model, where σ²_IS^A = σ²_OOS^A = σ²_IS^B = σ²_OOS^B, imposing a global constraint μ^A = μ^B implies that:
>
> **SR_IS^A > SR_IS^B ⇔ SR_OOS^A < SR_OOS^B**

**Plain reading:** When returns have memory (common in financial series), **optimizing IS actively degrades OOS** — a strategy that looks best in-sample is literally expected to underperform out-of-sample.

## Proposition 4 — Compensation effect under AR(1) (page 13)

> **PROPOSITION 4:** Given two alternative configurations (A and B) of the same model where σ²_IS^A = σ²_OOS^A = σ²_IS^B = σ²_OOS^B and the performance series follows the same first-order autoregressive stationary process,
>
> **SR_IS^A > SR_IS^B ⇔ SR_OOS^A < SR_OOS^B**   (Equation 13)

## Half-life of AR(1) (page 12)

> **PROPOSITION 3:** The half-life period of a first-order autoregressive process with autoregressive coefficient φ ∈ (0,1) occurs at:
>
> **τ = -Ln[2] / Ln[φ]**   (Equation 12)

---

## Key quotes (verbatim)

### On hiding trial counts (page 13)
> "Not reporting the number of trials (N) involved in identifying a successful backtest is a similar kind of fraud. The investment manager only publicizes the model that works, but says nothing about all the failed attempts, which as we have seen can greatly increase the probability of backtest overfitting."

### On good backtests being bad news (page 14)
> **"Hiding trials appears to be standard procedure in financial research. As an aggravating factor, we know from Section 6 that backtest overfitting typically has a detrimental effect on future performance, due to the compensation effects present in financial series. Indeed, the customary disclaimer 'past performance is not an indicator of future results' is too optimistic in the context for backtest overfitting. When investment advisers do not control for backtest overfitting, good backtest performance is an indicator of negative future results."**

### On overfitting model complexity (page 9)
> "A relatively simple strategy with just 7 binomial independent parameters offers N = 2⁷ = 128 trials, with an expected maximum Sharpe ratio above 2.6."

---

## Application to our project

### Our situation
- **Clean MNQ data:** Feb 2024 - present = ~2.2 years of actual MNQ micro
- **Proxy-extended (NQ parent + MNQ):** ~16 years for price-based filters only
- **Strategies tested in April 2026 discovery:** ~35,616 MNQ 5m combinations + ~26,000 MGC 5m
- **Best deployed annualized Sharpe:** MNQ COMEX OVNRNG_100 = 1.23

### Applying MinBTL to us

Using Equation 6 (MinBTL < 2·ln[N] / E[max_N]²):

| Effective N | Max SR we can trust with 2.2 years clean data | Max SR we can trust with 16 years proxy data |
|---|---|---|
| 45 (Bailey's "5-year rule") | 1.88 | 0.70 |
| 100 | 2.05 | 0.76 |
| 500 | 2.49 | 0.92 |
| 5,000 | 2.92 | 1.08 |
| 35,000 | 3.24 | 1.20 |

**Reading the table:** With our actual 35,000-trial discovery, on 2.2 years of clean MNQ data, the maximum IS Sharpe that's NOT expected to be pure noise is **3.24 annualized**. Our best strategy has Sharpe 1.23. **Our best strategy is below the noise floor.**

Even at the generous 16-year proxy-extended horizon and assuming only 500 effective independent trials (tight correlation between combinations), we need IS SR > 0.92 to clear the noise floor. **Only COMEX OVNRNG_100 clears this bar.** The other 3 MNQ lanes (SIN, EUR, TOK) with SR ~0.6-0.8 are below.

### The corrective action

From page 8: **"no more than 45 independent model configurations should be tried"** with 5 years of data. If we pre-register discovery to ~45-200 economically-justified hypotheses (not brute-force 35,000), the MinBTL constraint becomes achievable.

### Practical binding rule for our project

**BEFORE any discovery run, compute MinBTL using Equation 6 with our intended N and the threshold SR we require. If MinBTL > years of available data, reduce N until it fits. This is the locked constraint in `pre_registered_criteria.md`.**

---

## Related literature

- `bailey_lopez_de_prado_2014_deflated_sharpe.md` — extends this by correcting the Sharpe ratio for the number of trials and non-normality
- `lopez_de_prado_bailey_2018_false_strategy.md` — formal proof of the False Strategy Theorem used throughout
- `lopez_de_prado_2020_ml_for_asset_managers.md` — theory-first approach that fixes the root cause (brute-force search without economic prior)
- `harvey_liu_2015_backtesting.md` — alternative framework via BHY haircut Sharpe

## Audit note

**Before 2026-04-07, this project's discovery pipeline did not enforce MinBTL.** The April 3 rebuild tested ~35,000 combinations per instrument, exceeding the bound by ~600x. The 20,574 "FDR significant" strategies at q=0.05 should be interpreted under the lens of this paper: a large majority are expected to be false positives from selection bias, even after BH correction. Phase 4+ of the 2026-04-07 audit plan addresses this via pre-registered hypothesis discovery.

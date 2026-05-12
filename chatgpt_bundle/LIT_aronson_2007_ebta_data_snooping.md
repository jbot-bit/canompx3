# Evidence-Based Technical Analysis — Aronson (2007), Chapters 5-6 & 8-9

**Source:** `resources/Evidence_Based_Technical_Analysis_Aronson.pdf` (544 PDF pages)
**Author:** David R. Aronson
**Publication:** *Evidence-Based Technical Analysis: Applying the Scientific Method and Statistical Inference to Trading Signals*, John Wiley & Sons (2007)
**Chapters extracted:** Ch 5 (Hypothesis Tests, bootstrap & Monte Carlo), Ch 6 (Data-Mining Bias), Ch 8 (S&P 500 Case Study), Ch 9 (Case Study Results)
**Extracted:** 2026-05-12 (post-Harris-2002 ingestion, closing the data-snooping/TA-rigor citation gap)

**Criticality for our project:** 🔴 **HIGH** — Aronson is the canonical "integrated treatment of data-snooping bias applied to TA rule discovery." Our prior corpus had Bailey 2013 (finance-overfitting theory), Harvey-Liu 2015 (Sharpe haircut), Chordia 2018 (BH-FDR application). Aronson supplies the **direct empirical mapping to TA breakouts** — exactly the regime ORB lives in.

**Why this extract was added:** plan called out (Gap 2) that Aronson is "the canonical 'data-snooping bias in TA / breakouts' treatment — directly germane to ORB. Currently we lean on Bailey + Harvey-Liu for the statistical side and Harris/Chan for the mechanism, but the integrated TA-rigor argument lives in Aronson."

**Extraction method:** keyword-scan first (via `pymupdf`) to locate load-bearing passages (`data-mining bias`, `bootstrap`, `Monte Carlo`, `White's Reality Check`, `in-sample`/`out-of-sample`), then targeted reads of the surrounding pages. NOT a cover-to-cover read — and this extract honestly scopes itself to the four chapters listed above. Other chapters (psychology of subjective TA, theories of nonrandom price motion) are out-of-scope for ORB grounding and are NOT summarized here.

---

## Book structure (TOC, PDF p. 11-12)

PART I — Methodological, Psychological, Philosophical, and Statistical Foundations
- Ch 1: Objective Rules and Their Evaluation (p. 15)
- Ch 2: The Illusory Validity of Subjective Technical Analysis (p. 33)
- Ch 3: The Scientific Method and Technical Analysis (p. 103)
- Ch 4: Statistical Analysis (p. 165)
- Ch 5: **Hypothesis Tests and Confidence Intervals (p. 217)** — bootstrap & Monte Carlo permutation
- Ch 6: **Data-Mining Bias: The Fool's Gold of Objective TA (p. 255)**
- Ch 7: Theories of Nonrandom Price Motion (p. 331)

PART II — Case Study: Signal Rules for the S&P 500 Index
- Ch 8: **Case Study of Rule Data Mining for the S&P 500 (p. 389)** — 6,402 rules on S&P 1980-2005
- Ch 9: **Case Study Results and the Future of TA (p. 441)** — none significant after WRC/MCP

Appendix, Notes, Index (pp. 475-517).

---

## Core definitions (verbatim, Ch 6 opening, PDF pp. 271-272)

The chapter opens with a glossary that must be cited exactly:

> "**Expected performance:** the expected return of a rule in the immediate practical future. This can also be called the true performance of the rule, which is attributable to its legitimate predictive power.
>
> **Observed performance:** the rate of return earned by a rule in a back test.
>
> **Data-mining bias:** the expected difference between the observed performance of the best rule and its expected performance. Expected difference refers to a long-run average difference that would be obtained by numerous experiments that measure the difference between the observed return of the best rule and the expected return of the best rule.
>
> **Data mining:** the process of looking for patterns, models, predictive rules, and so forth in large statistical databases.
>
> **Best rule:** the rule with the best observed performance when many rules are back tested and their performances are compared.
>
> **In-sample data:** the data used for data mining (i.e., rule back testing).
>
> **Out-of-sample data:** data not used in the data mining or back-testing process.
>
> **Rule universe:** the full set of rules back tested in a data mining venture.
>
> **Universe size:** the number of rules comprising the rule universe."

**Plain reading and project mapping:** Aronson's "best rule" = our best survivor cell in a `comprehensive_deployed_lane_scan` run. His "data-mining bias" = the expected gap between IS ExpR (selected) and OOS ExpR (deployed). The HOLDOUT_SACRED_FROM = 2026-01-01 split is precisely his "in-sample vs out-of-sample" partition. The 324-combo lane grid is his "rule universe."

---

## The data-mining bias — central thesis (verbatim, Ch 6 p. 255)

> "In rule data mining, many rules are back tested and the rule with the best observed performance is selected. That is to say, data mining involves a performance competition that leads to a winning rule being picked. The problem is that the winning rule's observed performance that allowed it to be picked over all other rules systematically overstates how well the rule is likely to perform in the future. This systematic error is the data-mining bias.
>
> Despite this problem, data mining is a useful research approach. It can be proven mathematically that, out of all the rules tested, the rule with the highest observed performance is the rule most likely to do the best in the future, provided a sufficient number of observations are used to compute performance statistics. In other words, it pays to data mine even though the best rule's observed performance is positively biased."

**Plain reading:** mining is necessary (the best observed rule IS the best bet on future performance), but its OBSERVED performance is upward-biased relative to its true future performance. This is the foundation under our entire validated-setups → live-allocator pipeline: we use IS performance to RANK candidates, then expect a haircut (per Harvey-Liu 2015) on OOS / deployed performance.

### Aronson's two-villain decomposition (PDF p. 279)

> "A more complete account of out-of-sample performance deterioration is based on the data-mining bias. It names two villains: (1) randomness, which is a relatively large component of observed performance and (2) the logic of data mining, in which a best-performing rule is selected after the back-tested performances of all tested rules are available for the data miner's examination. When these two effects combine, they cause the observed performance of the best rule to overstate its future (expected) performance."

This is the same insight Bailey-LdP 2014 (Deflated Sharpe Ratio) formalizes as `SR_deflated = (SR_observed - E[max SR | N independent backtests]) / σ(SR_observed)`. Aronson provides the qualitative story; Bailey provides the closed-form deflation.

---

## Aronson's single-rule vs multiple-rule distinction (verbatim, PDF pp. 286-287)

This is the part our doctrine should cite most often:

> "In single-rule back testing, observed performance serves as an estimator of future performance. In data mining, observed performance serves as a selection criterion. Problems arise for the data miner when observed performance is asked to play both roles.
>
> [...] In data mining, back-tested performance serves as a selection criterion. That is to say, it is used to identify the best rule. The mean returns of all back-tested rules are compared and the one with the highest return is selected. This, too, is a perfectly legitimate use of the back test (observed) performance statistic.
>
> It is legitimate in the sense that the rule with the highest back-tested mean return is in fact the rule that is most likely to perform the best in the future. This is certainly not guaranteed, but it is the most reasonable inference that can be made. A formal mathematical proof of this statement is offered by White."

### The Data Miner's Mistake (verbatim, PDF p. 287)

> "The data miner's mistake is using the best rule's back-tested performance to estimate its expected performance. This is not a legitimate use of back-tested performance because the back-tested performance of the best-performing rule is positively biased. That is, the level of performance that allowed the rule to win the performance competition overstates its true predictive power and its expected performance. **This is the data-mining bias.**"

**Direct mapping to our project:**
- Selecting the top-ranked cell from `validated_setups` based on ExpR_IS = LEGITIMATE (this is the right candidate).
- Reporting that IS ExpR as the expected deployed ExpR = the data miner's mistake.
- Our remedy: Pathway-B K=1 paired-ΔR on holdout, Harvey-Liu Sharpe haircut, and a deployed-lane SR monitor with Shiryaev-Roberts (Pepelyshev-Polunchenko 2015) to detect when the haircut wasn't enough.

---

## OOS deterioration is not "regime change" (verbatim, PDF pp. 278-279)

Aronson dismantles a popular trader excuse:

> "Out-of-sample performance deterioration is a well-known problem. Objective technicians have proposed a number of explanations. I propose a relatively new one, based on data-mining bias.
>
> The least plausible explanation attributes the problem to random variation. Although it is true that a rule's performance will vary from one sample of history to another due to sampling variability, this explanation does not even fit the evidence. If random variation were responsible, then out-of-sample performance would be higher than in-sample performance about as frequently as it is lower. Anyone experienced with back testing knows that out-of-sample performance is inferior far more often.
>
> A second rationale is that the market's dynamics changed when the out-of-sample period began. It is reasonable to assume that financial markets are nonstationary systems. However, it is not reasonable to assume that each time a rule fails out of sample it is because market dynamics have changed. It would be odd, almost fiendish, for the market to always change its ways just when a rule moves from the technician's laboratory to the real world of trading."

**Plain reading and project mapping:** when an OOS check shows performance drop, our default explanation MUST be data-mining bias (Aronson's primary villain), NOT "the regime changed". This is consistent with our `pre_registered_criteria.md` posture — we do not relax a kill verdict by claiming "the regime is different now" without a pre-registered regime hypothesis.

---

## The Monte Carlo permutation method (Masters) — Ch 5 (verbatim, PDF pp. 255-256)

Aronson popularizes Timothy Masters's MCP method as an alternative to White's Reality Check:

> "Although the Monte Carlo method has been in existence for a long time, it had not been previously applied to rule testing. This was made possible by Dr. Masters' insight that the Monte Carlo method could generate the sampling distribution of a rule with no predictive power. This is accomplished by randomly pairing or permuting the detrended daily returns of the market (e.g., S&P 500) with the ordered time series representing the sequence of daily rule output values. […]
>
> The random pairing of the rule output values with market changes destroys any predictive power that the rule may have had. I refer to this random pairing as a noise rule."

### The MCP procedure (verbatim, PDF p. 256)

> "1. Obtain a sample of one-day market-price changes for the period of time over which the TA rule was tested, detrended as described in Chapter 1.
> 2. Obtain the time series of daily rule output values over the back-test period. […]
> 3. Place the market's detrended one-day-forward price changes on a piece of paper. Place them in a bin and stir.
> 4. Randomly select a market-price change from the bin and pair it with the first (earliest) rule output value. Do not put the price change back in the bin. In other words this sampling is being done without replacement.
> 5. Repeat step 4 until all the returns in the bin have been paired with a rule output value. […]
> 6. Compute the return for each of the 1,231 random pairings. […]
> 7. Compute the average return for the 1,231 returns obtained in step 6.
> 8. Repeat steps 4 through 7 a large number of times (e.g., 5,000).
> 9. Form the sampling distribution of the 5,000 values obtained in step 8.
> 10. Place the tested rule's rate of return on the sampling distribution and compute the p-value (the fraction of random rule returns equal to or greater than the tested rule's return)."

**Plain reading and project mapping:** MCP is the canonical block-bootstrap-without-replacement procedure that decouples rule output from market path. Our `research/comprehensive_deployed_lane_scan.py` uses a circular-block bootstrap of the trade-return vector with a comparable null structure. The procedural skeleton is identical: pair (signal × forward return), permute the pairing many times, build the null distribution of average return, compute one-sided p.

---

## The S&P 500 case study — 6,402 rules, NONE significant (Ch 9, verbatim, PDF p. 441)

This is the empirical centerpiece and the single most-citable passage in the book:

> "With respect to the primary objective, the case study resoundingly demonstrated the importance of using significance tests designed to cope with data-mining bias. With respect to the second objective, **no rules with statistically significant returns were found. Specifically, none of the 6,402 rules had a back-tested mean return that was high enough to warrant a rejection of the null hypothesis, at a significance level of 0.05.** In other words, the evidence was insufficient to reject a presumption that none of the rules had predictive power.
>
> The rule with the best performance, E-12-28-10-30, generated a mean annualized return of 10.25 percent, on detrended market data. […] **The p-value of the return is 0.8164, far above the 0.05 level set as the significance threshold.**"

### What the naive test would have said (verbatim, PDF p. 459)

> "Ironically, the failure of any rule to generate statistically significant returns, after adjustment for data-mining bias, underscores the huge importance of using statistical inference methods that take the biasing effects of data mining into consideration. **Had I used an ordinary significance test, which pays no attention to data-mining bias, the mean return of the best rule would have appeared to be highly significant (a p-value of 0.0005).**"

### The headline number (verbatim, PDF p. 460)

> "Had a conventional test of significance been used, **about 320 of the 6,402 rules would have appeared to be significant at the 0.05 level. This is exactly what would be predicted to occur by chance.** The naive data miner using a conventional test of significance would have concluded that many rules with predictive power had been discovered. In reality, mining operations conducted in this fashion would have produced nothing but fool's gold."

**Plain reading and project mapping:** 6,402 rules × α = 0.05 → expected 320 "significant" by chance under the global null. Aronson found 320, exactly the null prediction. This is the direct empirical demonstration of why ORB rule discovery WITHOUT BH-FDR multi-framing and Chordia-strict t ≥ 3.79 is a "fool's gold" exercise. Our prior brute-force scans of >300 trials are now bound under Bailey-Borwein-LdP-Zhu 2013 MinBTL theorem precisely because Aronson empirically validated the regime.

---

## Data snooping vs data mining (Ch 8, verbatim, PDF p. 406)

A critical distinction Aronson draws but our doctrine had not explicitly anchored:

> "In addition to data mining bias, rule studies can also suffer from an even more serious problem, the **data-snooping bias**. Data snooping refers to using the results of prior rule studies reported by other researchers. Because these studies typically do not disclose the amount of data mining that led to the discovery of whatever it was that was discovered, there is no way to take its effects into account and hence no way to properly evaluate the statistical significance of the results."

**Plain reading and project mapping:** when we adopt a feature from outside research (e.g., Yordanov's value-area filter, Fitschen's intraday breakout claims, Chan's mean-reversion patterns), we inherit their N_trials, which we cannot fully audit. This is why our pre-reg requires literature grounding (`institutional-rigor.md` § 7) and why we re-test on our own holdout. Adopting a published rule without re-validation IS data snooping in Aronson's sense.

---

## What Aronson does NOT solve (honest limits)

These are scope limits for our citation usage, NOT critiques of the book:

1. **White's Reality Check is described, not derived.** For the math, we need White 1997 / 2000 directly (not in `resources/`). Aronson is an applied treatment.
2. **No closed-form deflation factor for Sharpe.** Bailey-LdP 2014 (Deflated Sharpe Ratio) is the analytical counterpart; Aronson's empirical case study is the pre-Bailey version of the same insight.
3. **No FDR / BH treatment.** Aronson uses single-statistic WRC/MCP family-wise control. He does not discuss the FDR philosophy (Benjamini-Hochberg 1995) which dominates modern strategy validation per Chordia 2018.
4. **Case study scope = S&P 500 daily 1980-2005.** Not commodity futures, not intraday, not session-conditioned. The mechanism (data-mining bias) generalizes; the specific 11%-centered null distribution does not.
5. **Aronson tested individual rules, not lane/aperture × filter × RR matrices.** Our discovery search space is structurally more complex; the bias is correspondingly larger and the corrections more delicate.

---

## How our project uses this (canonical cross-references)

| Where | Use |
|-------|-----|
| `research/comprehensive_deployed_lane_scan.py::bootstrap_block_resample()` | Block-bootstrap null distribution per Aronson Ch 5 / Masters MCP |
| `docs/institutional/pre_registered_criteria.md` § 1-3 | "Selected best survivor ≠ expected deployed performance" formalized as Harvey-Liu Sharpe haircut + Bailey DSR |
| `.claude/rules/backtesting-methodology.md` § RULE 12 (red flags) | "Every top survivor references the same feature class" = data-mining-bias smell test in Aronson's language |
| `docs/institutional/literature/benjamini_hochberg_1995_fdr.md` | Modern FDR alternative to Aronson's WRC/MCP family-wise framework |
| `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` | Closed-form MinBTL bound on data-mining bias (Aronson is the empirical demonstration; Bailey is the theorem) |
| `docs/institutional/literature/harvey_liu_2015_backtesting.md` | Sharpe-haircut framework that quantifies Aronson's "data miner's mistake" |
| `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` | Direct counterpart to Aronson's 6,402-rule case study, scaled to 2.4M strategies |

---

## Key quotes for citation in audit results / pre-reg docs

### On selection bias being the dominant effect (PDF p. 278)

> "Out-of-sample performance is inferior far more often [than higher]. […] It is not reasonable to assume that each time a rule fails out of sample it is because market dynamics have changed."

### On the necessity of mining despite the bias (PDF p. 255)

> "It pays to data mine even though the best rule's observed performance is positively biased."

### On the central concept (PDF p. 287)

> "The data miner's mistake is using the best rule's back-tested performance to estimate its expected performance."

### On the case study null distribution centering (PDF p. 458-459)

> "Both sampling distributions are centered at approximately an 11 percent mean return. […] The expected return for the best rule of 6,402 competing rules with no predictive power under these specific conditions is approximately 11 percent. It is not 0 percent."

This is the empirical analog of Bailey 2013 Proposition 1: `E[max_N] ≈ (1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(Ne)]`. Aronson's 11% null-best return is the simulated draw from this distribution for the S&P case study.

---

## Provenance

- PDF extracted via `pymupdf` 2026-05-12.
- Keyword-search-then-targeted-read methodology (NOT a cover-to-cover read). Search terms: `data-mining bias`, `data mining bias`, `bootstrap`, `Monte Carlo`, `in-sample`, `out-of-sample`, `White's Reality Check`, `permutation`.
- Pages read in full: 255-256 (Ch 5 MCP procedure), 271-272 (Ch 6 opening + glossary), 277-279 (data-mining-bias core argument), 286-288 (single vs multiple rule distinction), 306 (data-mining-bias variance factors), 405-406 (Ch 8 case study setup + data snooping distinction), 409 (case study population & H0), 457-460 (Ch 9 results).
- Every verbatim block above was copy-pasted from extracted text and de-OCR-spaced (the scan inserts spurious whitespace).
- The 6,402-rule case study punchline (none significant after WRC/MCP, naive testing would have found ~320) was triple-confirmed across pp. 441, 459, and 460.
- Chapters NOT extracted (Ch 1-4, 7, Part II Chs other than 8-9, Appendix): not load-bearing for current ORB grounding gaps; would be flagged in future audit and read at that point if needed.

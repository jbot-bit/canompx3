# Controlling the False Discovery Rate — Benjamini & Hochberg (1995)

**Source:** `resources/benjamini-and-Hochberg-1995-fdr.pdf`
**Authors:** Yoav Benjamini (Tel Aviv University), Yosef Hochberg (Tel Aviv University)
**Publication:** *Journal of the Royal Statistical Society. Series B (Methodological)*, Vol. 57, No. 1 (1995), pp. 289–300
**Extracted:** 2026-04-19
**Pages read:** 289-294 (pp 1-7 of the PDF)

**Criticality:** 🔴 **HIGH** — original definition of FDR and the step-up procedure now universally used across finance/genomics MHT. Cited by every downstream FDR extract in this project (`harvey_liu_2015_backtesting.md`, `chordia_et_al_2018_two_million_strategies.md`, `harvey_liu_zhu_2015_cross_section.md`).

---

## The FDR definition (verbatim, page 291)

> "The proportion of errors committed by falsely rejecting null hypotheses can be viewed through the random variable Q = V/(V + S) — the proportion of the rejected null hypotheses which are erroneously rejected. Naturally, we define Q = 0 when V + S = 0, as no error of false rejection can be committed. Q is an unobserved (unknown) random variable, as we do not know v or s, and thus q = v/(v + s), even after experimentation and data analysis. We define the FDR Q_e to be the expectation of Q,
>
> **Q_e = E(Q) = E{V/(V + S)} = E(V/R)**."

Where the table of possible outcomes (page 291) is:

| | Declared non-significant | Declared significant | Total |
|---|---|---|---|
| True null hypotheses | U | **V (false positives)** | m₀ |
| Non-true null hypotheses | T | S | m − m₀ |
| | m − R | R | m |

## The step-up procedure (verbatim, page 293)

> "Consider testing H₁, H₂, . . ., H_m based on the corresponding p-values P₁, P₂, . . ., P_m. Let P(1) ≤ P(2) ≤ . . . ≤ P(m) be the ordered p-values, and denote by H(i) the null hypothesis corresponding to P(i). Define the following Bonferroni-type multiple-testing procedure:
>
> **let k be the largest i for which P(i) ≤ (i/m) · q*,**
>
> **then reject all H(i), i = 1, 2, . . ., k.** (equation 1)
>
> **Theorem 1.** For independent test statistics and for any configuration of false null hypotheses, the above procedure controls the FDR at q*."

## The independence / positive-regression-dependency caveat

The 1995 original proof requires independence. Benjamini-Yekutieli (2001) later extended the result to allow arbitrary dependency by multiplying the critical value by c(M) = ∑_{j=1}^M 1/j — this is the BHY procedure advocated by Harvey-Liu 2015.

Under BH (1995) as written: valid when test statistics are independent OR satisfy "positive regression dependency on subsets of true nulls" (Benjamini-Yekutieli 2001 Theorem 1.2).

In finance applications, strategies on the same underlying (e.g., two lanes on the same session-year) typically have POSITIVELY correlated test statistics — same bars drive both — which falls inside the PRDS regime. Standard BH is defensible. For arbitrary-dependency safety, BHY is strictly more conservative.

## Power gain vs FWER methods (verbatim, page 294)

> "The series of linearly decreasing constants of the FDR controlling method is always larger than the hyperbolically decreasing constants of Hochberg, and the extreme ratio is as large as 4m/(m + 1)² at i = (m + 1)/2. **This shows that the suggested procedure rejects samplewise at least as many hypotheses as Hochberg's method and therefore has also greater power than other FWER controlling methods such as Holm's (1979).**"

## When FDR control is appropriate vs FWER (verbatim, page 291)

> "Often the control of the FWER is not quite needed. The control of the FWER is important when a conclusion from the various individual inferences is likely to be erroneous when at least one of them is. [...] However, a treatment group and a control group are often compared by testing various aspects of the effect (different end points in clinical trials terminology). The overall conclusion that the treatment is superior need not be erroneous even if some of the null hypotheses are falsely rejected."

Financial application: when running 38 lanes × 7 years = 266 cells, we are asking a set of separate questions ("does lane L have a year-regime effect?") rather than one joint yes/no question. FDR is the appropriate family-error rate, not FWER. Harvey-Liu 2015 page 20 makes this argument explicitly ("In financial applications, it seems reasonable to control for the rate of false discoveries, rather than the absolute number").

---

## Application to our project (canonical algorithm reference)

Canonical BH step-up pseudocode (from the paper's equation 1):

```
Input: p[1..m] test p-values, q* target FDR
Sort: P(1) ≤ P(2) ≤ ... ≤ P(m)
k = max { i : P(i) ≤ (i/m) · q* }   # largest rank clearing its critical value
Reject H(i) for all i ≤ k
```

Implementation in this project: `research/phase_2_9_comprehensive_multi_year_stratification.py::bh_fdr(pvalues, q)` implements exactly this procedure. Design choice: the denominator `n` is set to the full input length (including None/NaN cells) rather than the count of valid p-values. This is strictly more conservative than standard BH when some cells fail to produce p-values; it caps the effective K at the pre-committed 266 regardless of data availability.

## Related literature

- `harvey_liu_2015_backtesting.md` — advocates BHY (Benjamini-Yekutieli 2001) for financial applications, but at least for PRDS-compatible pairs (same-underlying strategies) standard BH is also valid
- `chordia_et_al_2018_two_million_strategies.md` — empirical calibration at 2.4M strategies; finds BH adaptive and reliable (pages 3-4) whereas Bonferroni/Holm lose power from treating test statistics as independent
- `bailey_et_al_2013_pseudo_mathematics.md` — MinBTL constraint on the MAXIMUM K that data can support
- `bailey_lopez_de_prado_2014_deflated_sharpe.md` — alternative adjustment via DSR (no MHT needed if using DSR instead)

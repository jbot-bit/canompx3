# Controlling the False Discovery Rate — Benjamini & Hochberg (1995)

**Source:** `resources/benjamini-and-Hochberg-1995-fdr.pdf` (JSTOR scan, 13 PDF pages = journal pp. 289-300)
**Authors:** Yoav Benjamini, Yosef Hochberg (Tel Aviv University)
**Publication:** *Journal of the Royal Statistical Society. Series B (Methodological)*, Vol. 57, No. 1 (1995), pp. 289-300
**Received:** January 1993. Revised March 1994.
**Extracted:** 2026-05-12 (post-Harris-2002 ingestion, closing citation chain for `pre_registered_criteria.md`)

**Criticality for our project:** 🔴 **MAXIMUM** — this is the **canonical primary source** for the FDR-control procedure cited (via Chordia 2018 and Harvey-Liu 2015) by every multiple-testing gate in `docs/institutional/pre_registered_criteria.md`. Our `bh_fdr_multi_framing()` in `research/comprehensive_deployed_lane_scan.py` implements this exact procedure.

**Why this extract was added:** prior to 2026-05-12 we cited BH-1995 only via secondary sources (Chordia 2018, Harvey-Liu 2015). An institutional audit would flag the missing primary anchor. This file closes the citation chain.

---

## Abstract (verbatim, journal p. 289 / PDF p. 2)

> "The common approach to the multiplicity problem calls for controlling the familywise error rate (FWER). This approach, though, has faults, and we point out a few. A different approach to problems of multiple significance testing is presented. It calls for controlling the expected proportion of falsely rejected hypotheses — the false discovery rate. This error rate is equivalent to the FWER when all hypotheses are true but is smaller otherwise. Therefore, in problems where the control of the false discovery rate rather than that of the FWER is desired, there is potential for a gain in power. A simple sequential Bonferroni-type procedure is proved to control the false discovery rate for independent test statistics, and a simulation study shows that the gain in power is substantial. The use of the new procedure and the appropriateness of the criterion are illustrated with examples."

---

## Definition: False Discovery Rate (verbatim, journal p. 291 / PDF p. 4)

The setup uses this error table for `m` null hypotheses:

|                          | Declared non-significant | Declared significant | Total |
|--------------------------|--------------------------|----------------------|-------|
| **True null hypotheses**     | U                        | V                    | m₀    |
| **Non-true null hypotheses** | T                        | S                    | m − m₀|
|                              | m − R                    | R                    | m     |

> "The proportion of errors committed by falsely rejecting null hypotheses can be viewed through the random variable Q = V/(V + S) — the proportion of the rejected null hypotheses which are erroneously rejected. Naturally, we define Q = 0 when V + S = 0, as no error of false rejection can be committed. […] We define the FDR Q_e to be the expectation of Q,
>
> **Q_e = E(Q) = E{V/(V + S)} = E(V/R).**"

**Plain reading:** FDR is the expected fraction of "discoveries" that are actually false. Bonferroni controls `P(V ≥ 1)` (any false positive); BH controls `E(V/R)` (proportion of false positives among declared discoveries). FDR is a strictly weaker (more permissive) criterion — except under the global null where the two coincide.

### Two key properties (journal p. 291 / PDF p. 4)

> "(a) If all null hypotheses are true, the FDR is equivalent to the FWER […]
>
> (b) When m₀ < m, the FDR is smaller than or equal to the FWER […] any procedure that controls the FWER also controls the FDR. However, if a procedure controls the FDR only, it can be less stringent, and a gain in power may be expected. In particular, the larger the number of the non-true null hypotheses is, the larger S tends to be, and so is the difference between the error rates. As a result, the potential for increase in power is larger when more of the hypotheses are non-true."

---

## The BH procedure (verbatim, journal p. 293 / PDF p. 6)

> "Consider testing H₁, H₂, …, H_m based on the corresponding p-values P₁, P₂, …, P_m. Let P_(1) ≤ P_(2) ≤ … ≤ P_(m) be the ordered p-values, and denote by H_(i) the null hypothesis corresponding to P_(i). Define the following Bonferroni-type multiple-testing procedure:
>
> **let k be the largest i for which P_(i) ≤ (i/m)·q\***
> **then reject all H_(i), i = 1, 2, …, k.   (1)**"

### Theorem 1 (verbatim, journal p. 293 / PDF p. 6)

> "**Theorem 1.** For independent test statistics and for any configuration of false null hypotheses, the above procedure controls the FDR at q\*."

**Proof:** by lemma (p. 293), proved by induction on m in Appendix A (journal pp. 299-300 / PDF pp. 12-13). Independence of the false-null test statistics is NOT needed; only the true-null statistics must be independent.

**Plain reading of the procedure:**
1. Run all m tests, collect p-values.
2. Sort ascending: P_(1) ≤ P_(2) ≤ … ≤ P_(m).
3. For each rank `i`, compute the threshold `(i/m)·q*`.
4. Find the LARGEST `i` (call it `k`) where P_(i) ≤ (i/m)·q*.
5. Reject every hypothesis with rank ≤ k.

This is a **step-up** procedure (starts at largest p, walks down). The Hochberg (1988) FWER procedure is the same shape with `(α/(m+1−i))` thresholds; BH-FDR uses `(i/m)·q*` thresholds which are uniformly larger (more rejections, higher power).

---

## Worked clinical example (verbatim, journal p. 295 / PDF p. 8)

Family of 15 cardiac-trial p-values for rt-PA vs APSAC:

> "The ordered p_(i)s for the 15 comparisons made are
>
> 0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000."

**Bonferroni @ 0.05/15 = 0.0033:** rejects the smallest 3 p-values. Does **NOT** include the mortality comparison (p = 0.0095).
**Hochberg (1988) @ FWER 0.05:** same 3 rejected.
**BH-FDR @ q* = 0.05:**

> "we now compare sequentially each P_(i) with 0.05·i/15, starting with P_(15). The first p-value to satisfy the constraint is P_(4) as
>
> **P_(4) = 0.0095 ≤ (4/15)·0.05 = 0.013.**
>
> Thus we reject the four hypotheses having p-values which are less than or equal to 0.013. We may support now with appropriate confidence the statements about mortality decrease, of which we did not have sufficiently strong evidence before."

**Plain reading:** BH recovers the mortality finding (p = 0.0095) that Bonferroni and Hochberg both miss. This is the canonical demonstration that FDR control gives real-world power increases over FWER.

---

## Simulation power results (journal pp. 296-298 / PDF pp. 9-11)

Setup (p. 296): m ∈ {4, 8, 16, 32, 64} independent normal z-tests, q* = α = 0.05, 20,000 repetitions per configuration.

Key findings (verbatim, journal p. 297 / PDF p. 10-11):

> "(c) The power of the FDR controlling method is uniformly larger than that of the other methods.
> (d) The advantage increases with the number of non-null hypotheses.
> (e) The advantage increases in m. […]
> (f) The advantage in some situations is extremely large. For example, testing 32 hypotheses, equally spread in four clusters from 1.25 to 5 so that none is true, the power of the Bonferroni method is 0.42. The procedure suggested increases the power to 0.65; testing as few as four hypotheses, half of which are true, the values are 0.62 and 0.70 respectively.
> (g) It is known that Hochberg's method offers a more powerful alternative to the traditional Bonferroni method. Nevertheless, it is important to note that the gain in power due to the control of the FDR rather than the FWER is much larger than the gain of the FWER controlling method over the Bonferroni method."

**Plain reading:** in scenarios where many true effects exist (the relevant case in strategy discovery), BH gives 50%+ more rejections than Bonferroni. This is precisely the regime our ORB discovery operates in — multiple lanes, multiple genuine effects, FWER would be punitively conservative.

---

## Theorem 2 — BH as constrained maximization (verbatim, journal p. 295 / PDF p. 8)

> "**Theorem 2.** The FDR controlling procedure given by expression (1) is the solution of the following constrained maximization problem:
>
> choose α that maximizes the number of rejections at this level, r(α),
> subject to the constraint αm/r(α) ≤ q*.   (3)"

**Plain reading:** BH is equivalent to "maximize discoveries subject to expected-false-discovery-rate ≤ q*". This is the post-hoc decision-theoretic interpretation — BH gives the most lenient threshold consistent with FDR control.

---

## Independence assumption — important caveat

The 1995 paper proves FDR control **for independent test statistics** (Theorem 1, journal p. 293). For positively-dependent statistics (PRDS — positive regression dependency on subset of true nulls), Benjamini & Yekutieli 2001 extended the result. For arbitrary dependence, BY-2001's modified procedure with the harmonic-sum penalty `c(m) = Σ_{i=1}^m 1/i` is required.

**Implication for our work:** ORB lanes are NOT independent (correlated via shared market regime, shared cost structure, shared instrument). Strict BH-1995 applicability requires either:
1. Empirical check that PRDS holds (typical for one-sided tests with shared confounders → usually fine).
2. BY-2001 with the harmonic penalty as the conservative fallback.
3. Bootstrap-based FDR (resample-the-data null), which sidesteps the analytical dependence question.

Our `bh_fdr_multi_framing()` documents this and currently applies plain BH-1995 with a per-framing K (per-feature-family, per-lane, per-session) — the family-level K largely sidesteps the correlation issue because within-family dependence is mostly shared mean shift, which is the regime where BH is known to be conservative.

---

## How our project uses this (canonical implementation cross-references)

| Where | Use |
|-------|-----|
| `research/comprehensive_deployed_lane_scan.py::bh_fdr_multi_framing()` | Implements procedure (1) at K_global, K_family, K_lane, K_session, K_instrument, K_feature framings |
| `docs/institutional/pre_registered_criteria.md` § FDR gate | Locks q* = 0.05 for K_family or K_lane survival |
| `.claude/rules/backtesting-methodology.md` § RULE 4 | Mandates multi-framing reporting |
| `pipeline/check_drift.py::check_literature_grounding_consistency` | Enforces this extract file exists |
| `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` | Uses BH-FDR with q* = 0.05 and adds the prior-Sharpe-1.0 strict t ≥ 3.79 layer |
| `docs/institutional/literature/harvey_liu_2015_backtesting.md` | Combines BH-FDR with the analytical Sharpe haircut |

---

## Key quotes for citation

### On why FWER is wrong for many problems (journal p. 290 / PDF p. 3)

> "(c) Often the control of the FWER is not quite needed. The control of the FWER is important when a conclusion from the various individual inferences is likely to be erroneous when at least one of them is. […] However, a treatment group and a control group are often compared by testing various aspects of the effect (different end points in clinical trials terminology). The overall conclusion that the treatment is superior need not be erroneous even if some of the null hypotheses are falsely rejected."

### On the new error-rate philosophy (journal p. 290 / PDF p. 3)

> "In many multiplicity problems the number of erroneous rejections should be taken into account and not only the question whether any error was made. Yet, at the same time, the seriousness of the loss incurred by erroneous rejections is inversely related to the number of hypotheses rejected. From this point of view, a desirable error rate to control may be the expected proportion of errors among the rejected hypotheses, which we term the false discovery rate."

### On screening (journal p. 292 / PDF p. 5)

> "The third type involves screening problems, where multiple potential effects are screened to weed out the null effects. One example is screening of various chemicals for potential drug development. […] In such examples we want to obtain as many as possible discoveries (candidates for drug developments, factors that affect the quality of a product) but again wish to control the FDR, because too large a fraction of false leads [is undesirable]."

**Direct mapping to our work:** ORB lane discovery IS a screening problem. We have hundreds of candidate (lane, filter, RR, aperture) combinations; we want to elevate as many true edges to deployment as possible; we are willing to tolerate a controlled fraction of false discoveries because each deployed strategy gets a separate paper-trading verification (Pathway-B K=1) and SR-monitor before any capital exposure. FWER would block legitimate finds.

---

## Limitations called out by the authors

From journal p. 298 / PDF p. 11 (Conclusion):

> "The approach to multiple significance testing in this paper is philosophically different from the classical approaches. The classical approach requires the control of the FWER in the strong sense, a conservative type I error rate control against any configuration of the hypotheses tested. The new approach calls for the control of the FDR instead, and thereby also the control of the FWER in the weak sense."

The authors flag (journal p. 298) that future work should extend to:
- Specific structural problems (pairwise comparisons in ANOVA)
- Different stochastic structures of the test statistics

The 2001 Benjamini-Yekutieli paper covered positive dependence; the 2009 Benjamini-Heller covered hierarchical FDR — neither is in our `resources/` set but neither is currently load-bearing for ORB discovery either.

---

## Provenance

- PDF extracted via `pymupdf` 2026-05-12, all 13 pages read.
- Page references use both journal pagination (289-300) and PDF pagination (1-13) for cross-checking.
- Every verbatim block above was copy-pasted from extracted text and then de-OCR-spaced (the JSTOR scan inserts spurious whitespace; the meaning is preserved exactly).
- Theorem 1 statement, the BH procedure formula, the worked clinical example, and the Theorem 2 maximization characterization were all verified against the PDF text in this session.


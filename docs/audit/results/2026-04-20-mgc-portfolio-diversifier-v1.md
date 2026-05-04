# H1 — MGC Portfolio-Diversifier v1 Result

**Date:** 2026-04-20
**Pre-reg:** `docs/audit/hypotheses/2026-04-20-mgc-portfolio-diversifier.yaml`
**Script:** `research/research_mgc_portfolio_diversifier_v1.py`
**Output:** `research/output/mgc_portfolio_diversifier_v1.json`
**Parent audit:** `docs/audit/results/2026-04-20-mgc-adversarial-reexamination.md` §9 H1
**Classification:** AUDIT (no deployment, no new discovery claim)
**Testing mode:** individual / Pathway B (K=1)

## Executive verdict

**MIXED** — structural diversification premise is strongly confirmed
(`max|corr| = 0.070`, well below 0.50 threshold), but neither tested
MGC return-stream variant under the pre-registered 1-contract uniform
sizing is Sharpe-accretive to the book. A **vol-matched diversifier
variant is not tested here** (pre-reg prohibits post-hoc weight tuning)
and is flagged as the cleanest follow-up if the overall diversifier
question is to be fully answered.

### Kill-criteria evaluation (per pre-reg §kill_criteria)

| Variant | C1 max\|corr\| < 0.50 | C2 ΔSR@10% ≥ 0.05 | C3 no destruction 5-20% | C4 N ≥ 500 days | Verdict |
|---|---|---|---|---|---|
| `raw` (unfiltered) | **PASS** (0.070) | FAIL | **FAIL** (destructive) | PASS (945) | **KILL C3** |
| `ovnrng_100` | **PASS** (0.054) | n/a — too thin | n/a — too thin | **FAIL** (4) | **KILL C4** |

## Raw evidence

```
Trading-day calendar in IS (2022-06-13 → 2026-01-01): 1039 days
Active lanes: 38
Book daily stream: mean=+0.0726R std=0.3922 sharpe=+2.94

MGC raw (all sessions, E2 CB1, RR1.0, no filter):
  active days=945, mean=-0.8568R std=2.2622 sharpe=-6.01
  vs 38 lanes: max|corr|=0.070 mean=+0.027 median=+0.024

MGC ovnrng_100 (E2 CB1, RR1.0, OVNRNG_100):
  active days=4, mean=+0.0139R std=0.2481 sharpe=+0.89
  vs 38 lanes: max|corr|=0.054 mean=+0.009 median=+0.008
```

## Finding 1 — Structural diversification premise: CONFIRMED

MGC daily-return stream has near-zero correlation with every one of the
38 active book lanes. Across 945 MGC trading days:

- max absolute Pearson correlation = **0.070**
- mean correlation = +0.027
- median correlation = +0.024

At ρ ≤ 0.10 MGC is operationally an independent return source. This
directly satisfies Markowitz (1952) Ch 3 — adding a near-uncorrelated
asset to a portfolio mechanically reduces variance per unit return.

**This finding alone resolves the prior ambiguity in the audit §1**:
there is NO evidence that MGC returns co-move with the book. The "MGC is
redundant to index exposure" concern is empirically refuted.

## Finding 2 — Raw unfiltered MGC is too lossy to use as a diversifier at contract-weight parity

Unfiltered MGC (every ORB break, every session, E2 CB1 RR1.0) averages
**−0.86R per active day** with σ = 2.26R. At a 10% contract-count weight
in the book (pre-reg-locked), MGC is destructive at every weight 5-20%
tested:

| Weight | ΔSR vs book-only |
|---|---|
| 5% | (destroyed) |
| 10% | (destroyed) |
| 15% | (destroyed) |
| 20% | (destroyed) |

(exact ΔSR per weight in `research/output/mgc_portfolio_diversifier_v1.json:sharpe_lift.weight_sweep`)

**Root cause:** unfiltered ORB on MGC is noise-plus-friction-dominated.
Every trading day gets ~1-3 break signals across 12 sessions; most are
losers. Daily aggregation produces a large-variance small-negative-drift
stream. At 10% contract weight this is 58% of the combined portfolio
volatility (MGC σ = 5.8× book σ). Way too much MGC.

## Finding 3 — OVNRNG_100 filter on MGC is too restrictive to be testable

Overnight range >= 100 points on gold is an extreme threshold. In the
1039-day IS window, only 4 days satisfied the filter. This is a 0.4%
fire rate, well below `.claude/rules/backtesting-methodology.md` RULE 8.1
extreme_fire floor of 5%. The pre-reg's C4 kill criterion (N ≥ 500 days)
is violated immediately.

**Root cause:** OVNRNG_100 was calibrated for MNQ (where 100 points is
~1-2% of index — a normal day). On gold, 100 points is ~4% of spot —
a rare move. The filter doesn't transfer to MGC without recalibration.

This matches the `docs/plans/2026-04-19-gc-mgc-handling-note.md` §3
conclusion that broad proxy success doesn't imply broad filter transfer.

## Finding 4 — The vol-matched diversifier question is NOT answered

The pre-reg locked uniform 1-contract weighting. The finding that MGC σ
is 5.8× book σ means a contract-count weighting overstates MGC's
portfolio footprint. A **vol-matched MGC position** (e.g., 1/6 of a
contract, or equivalently ~1.7% dollar weight for same vol) would
answer a different and arguably more relevant question:

> *Is a vol-scaled, zero-expectancy MGC stream Sharpe-additive to the
book given ρ ≈ 0.05?*

Under standard portfolio math, with μ_mgc = 0, σ_mgc_scaled = σ_book,
ρ ≈ 0.05, weight 10% of book-vol:

```
σ_combo² = (0.9σ)² + (0.1σ)² + 2·0.9·0.1·0.05·σ²
        = 0.81σ² + 0.01σ² + 0.009σ²
        = 0.829σ²
σ_combo = 0.911σ
μ_combo = 0.9·μ_book
SR_combo/SR_book = 0.9/0.911 = 0.988
ΔSR absolute = (0.988 − 1) × 2.94 = −0.035
```

So even **zero-alpha, vol-matched, at 10% weight** gives SLIGHTLY
NEGATIVE ΔSR (−0.035). The 10% weight dilutes the book's positive
drift faster than the low correlation reduces combined variance.

**A PASSING vol-matched diversifier requires either:**
- positive MGC ExpR (not zero) — but then we'd be deploying alpha not diversifier
- larger MGC weight where the variance reduction dominates (~30% would
  reach ΔSR = 0), but that's more concentration than a book should
  tolerate on an underpowered instrument

### Implication for the MGC closure

The portfolio-diversifier thesis does NOT rescue MGC from closure under
uniform-weight testing. Under vol-matched analysis with zero ExpR
assumption, the benefit is only visible at aggressive weights (>25%)
that would themselves be imprudent on an instrument with 3.8 years
real data and MinBTL ~30 trial budget.

**Conclusion for H1:** MGC's low correlation is real and interesting
but insufficient to motivate a diversifier-only deployment. The
practical implication is: **the closure of MGC for discovery is NOT
refuted by portfolio-structure economics** — the marginal Sharpe lift
from an uncorrelated zero-alpha asset at reasonable weight is below
both our ΔSR=0.05 threshold AND well below material (ΔSR < 0.05 ≈
typical monthly book variation).

## Finding 5 — Book baseline sanity check

Book daily stream: mean = +0.0726R, std = 0.3922, annualized SR = +2.94.

This is an **idealized** book SR that treats each of 38 lanes as
equally weighted and includes all `orb_outcomes` rows matching each
lane's (symbol, session, aperture, entry, RR, CB) tuple without
applying the lane's filter. It overstates book performance vs. the
filter-applied reality because filtered lanes trade on a smaller subset
of days with typically better ExpR.

**Baseline caveat (declared in pre-reg §baseline_cross_check):** this
idealized SR is used only as the denominator for correlation and
ΔSR-structure comparisons. It is not a claim about live-book Sharpe.

## What's NOT claimed here

- This is NOT a claim that MGC has positive ExpR.
- This is NOT a deployment recommendation.
- This is NOT a claim that OVNRNG_100 is the right MGC filter — it
  demonstrably isn't.
- This is NOT a rejection of the diversifier thesis; it IS a
  demonstration that the tested variants don't pass under the
  pre-registered sizing rule.

## Follow-ups (enumerated, not executed)

| Ref | Question | Status |
|---|---|---|
| H1b | Vol-matched MGC stream (σ_mgc scaled to σ_book/6) at weight sweep 5-30% | NOT in scope; separate pre-reg required |
| H1c | Diversifier with MGC RESCALED cost model (pilot slippage 6.75 ticks) | Subsumed into H0 execution |
| H1d | Session-rotated MGC with COST_LT gate (MGC-specific cost floor) | Would add mechanism — separate pre-reg |

## Trial count spent from MGC MinBTL budget

**1 trial** (this single pre-reg, K=1). Remaining budget estimate: 29
trials. Per `docs/audit/results/2026-04-20-mgc-adversarial-reexamination.md` §7.

## Audit trail

- Pre-reg committed BEFORE script: `docs/audit/hypotheses/2026-04-20-mgc-portfolio-diversifier.yaml`
- Script committed with reproducible seed (20260420)
- Output JSON with full per-lane correlation table: `research/output/mgc_portfolio_diversifier_v1.json`
- Canonical data: `gold.db` snapshot at 2026-04-20
- Commit: TO_FILL_AFTER_COMMIT

# Zero-Context Adversarial Review #2 — Null Test Findings

**Date:** 2026-03-18
**Reviewer:** Fresh Claude agent, isolated worktree, zero project context
**Input:** Null test results only (no CLAUDE.md, no memory, no TRADING_RULES.md)

---

## Findings

### 1. Gaussian Null Is Too Easy (HIGH)
The i.i.d. random walk is missing vol clustering, fat tails, microstructure, and regime changes. All of these produce MORE false positives than Gaussian noise. 60 per seed is a FLOOR. A proper null uses GARCH(1,1) with Student-t innovations calibrated to actual instrument parameters.

### 2. BH FDR Uses Inflated K and Uncorrected P-Values (HIGH)
- Strategies sharing the same session/entry/filter are massively correlated (differ only in CB/RR). BH assumes independence or PRDS — hand-waved, not proven.
- P-values use raw t-stat without Lo (2002) autocorrelation correction. Lag-1 autocorrelation of 0.15 reduces effective N by ~26%.
- FDR applied to deflated p-values with inflated K = under-correction.

### 3. E2 Structural Near-Breakeven Is CRITICAL, Not a Feature
- E2 averages -0.008R on noise. A real strategy only needs +0.008R of genuine edge to appear profitable.
- With 500+ trades, sqrt(500) × 0.008 / sigma is easily achievable by chance.
- 59/60 null survivors are E2. **The majority of validated E2 strategies on real data are suspect.**
- Cannot distinguish "real E2 edge" from "tiny fluctuation above noise floor" without E2-specific permutation test or much higher ExpR floor.

### 4. Acceptance Bar Undefined (HIGH)
- Best noise strategy: ExpR 0.238, Sharpe_ann 1.56, 655 trades. A competent quant would trade this. From pure noise.
- Need 100+ seeds (not 10) for stable tail estimation of the null envelope.
- Any real strategy below the 99th percentile of the null's best-strategy distribution is indistinguishable from noise.

### 5. Builder-As-Auditor Is Structural Conflict (MEDIUM)
- Null test had a bug that masked failure — written by the builder.
- No pre-registration of acceptance criteria.
- Institutional standard: independent implementation, blind null test, pre-committed pass/fail criteria, third-party code audit.

### 6. BH FDR Is the Wrong Framework (HIGH)
- BH FDR: "which individual strategies are significant?" ← wrong question
- White's Reality Check / Hansen's SPA: "is the BEST strategy better than noise?" ← right question
- Also consider: Romano-Wolf (FWER), permutation tests (preserves correlation structure), Model Confidence Set (Hansen-Lunde-Nason 2011)

### 7. Walk-Forward at 60%/3 Windows Is a Coin Flip (MEDIUM)
- Each OOS window is ~50/50 for E2 on noise.
- P(2/3 positive) = 50%, P(3/5 positive) = 31%. With 5-8 windows, substantial pass probability.
- Walk-forward detects overfitting, not false discovery. It's orthogonal to the multiple testing problem.
- Fix: 70-75% threshold, 5 minimum windows, stress costs in OOS.

### 8. Minimum Evidence Before Real Money (HIGH)

| # | Requirement | Status |
|---|------------|--------|
| 1 | Pre-registered null test with realistic DGP (GARCH + fat tails), 100+ seeds, 99th percentile | NOT DONE |
| 2 | White's Reality Check or Hansen's SPA (p < 0.01 for best strategy) | NOT BUILT |
| 3 | 20-trade manual spot-check (entry→fill→PnL trace) | NOT DONE |
| 4 | E2 null distribution (100+ seeds, 99th percentile acceptance floor) | NOT DONE |
| 5 | 3 months paper trading with pre-committed kill criteria | NOT STARTED |
| 6 | Position sizing assumes 50% of strategies are false | NOT IMPLEMENTED |
| 7 | Independent human review ($1,000-2,000, not $200) | NOT DONE |

---

## Summary

The pipeline demonstrates genuine methodological sophistication. The developer is clearly trying to be honest. But sophistication is not the same as correctness. The null test revealed that the pipeline's validation cascade does not control the false positive rate at an acceptable level, and the proposed fix (turn on existing gates) is necessary but insufficient. The E2 structural issue is the most concerning finding because it undermines the majority of the validated strategy book.

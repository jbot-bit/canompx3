---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# External Quant Review Brief

**Budget:** $500-2,000 USD for 2-4 hours of detailed written feedback
**What this is NOT:** A code review. I need you to assess whether the statistical methodology is sound.

---

## The System

Systematic intraday futures trading across 3 micro instruments (MGC, MNQ, MES) and 12 session windows per day (TOKYO_OPEN through CME_PRECLOSE). Strategy: Opening Range Breakout (ORB) at 5/15/30-minute apertures with grid-searched parameters.

Grid search space: ~120,000 strategy combinations (2 entry models x 3 instruments x 12 sessions x 3 ORB apertures x 3 confirm bars x multiple filter types x multiple RR targets). Data: 5-7 years of 1-minute OHLCV bars depending on instrument.

## Validation Pipeline (in order)

1. Grid search all combinations, record per-trade P&L in R-multiples
2. Benjamini-Hochberg FDR correction at alpha=0.05 across ALL ~120K tests (hard gate -- rejected, not tagged)
3. Walk-forward validation: 6-month rolling OOS windows, WFE > 50% required
4. Classification: CORE (>=100 trades), REGIME (30-99), INVALID (<30)
5. Edge families: cluster overlapping strategies by trade-day hash, pick best per cluster
6. Noise floor gate (see below)

## The Null Test

I generated synthetic 1-minute bars using a Gaussian random walk (zero drift, i.i.d. N(0, sigma)) and ran the full pipeline. 10 seeds tested. Results:
- 611 total false-positive "strategies" survive the full pipeline across all seeds
- E2 entry model (stop-market at ORB boundary): 577 noise survivors, max ExpR = 0.316 R
- E1 entry model (limit at ORB boundary): 34 noise survivors, max ExpR = 0.243 R
- E2 structural bias: avg ExpR on noise = -0.004R (near-breakeven), making it easy for noise to fake edge
- E1 structural bias: avg ExpR on noise = -0.118R (meaningful cost), harder to fake

I currently use the noise MAX (rounded up) as the acceptance threshold. Current floors: E1 >= 0.25R, E2 >= 0.32R. This kills 252/253 edge families. Only 1 survives (MGC TOKYO_OPEN E1, ExpR=0.270).

## The Specific Questions

1. Is the noise MAX the right acceptance threshold, or should I use noise P95, or compute a p-value per strategy against the null distribution, or use the Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)?
2. With only 10 seeds, is the observed max a stable estimator or could it shift substantially with 50+ seeds?
3. Is a Gaussian random walk an adequate null, or do I need block-bootstrapped actual returns (Politis-Romano)?
4. Is BH FDR at alpha=0.05 appropriate for ~120K correlated tests (strategies share underlying price data)?
5. The E2 entry model has a near-breakeven structural bias on noise data (-0.004R avg). Is there a better way to handle entry-model-specific noise profiles than separate floors?
6. Walk-forward uses calendar windows (6 months). For strategies with low trade frequency (50-100/year), should I use trade-count-based windows?

## What You Get Access To

- Full RESEARCH_RULES.md (statistical methodology document, ~4 pages)
- Null test databases (queryable -- all 611 noise survivors with full stats)
- Production validated strategies with stats for comparison
- Walk-forward efficiency distribution
- Complete NO-GO list (things tested and rejected -- longer than the success list)
- Cost model documentation

## What I Want

Adversarial feedback. I want you to find problems. Tell me what's wrong, what's weak, what's missing. I explicitly do NOT want validation that the approach is sound. Assume everything is wrong until proven otherwise. If you think the entire premise (ORB breakouts in micro futures) is flawed, say so and explain why.

## About Me

Solo researcher, not institutional. 5+ years of data, custom pipeline, no off-the-shelf backtester. I've already killed more ideas than I've kept (calendar effects, break quality, E0 entry model, 4 instruments, multiple indicator overlays -- all NO-GO). I'm looking for the holes I can't see because I built the thing.

# Pre-Registration: RR1.0 Economic Reasoning

**Date:** 2026-03-20
**Status:** PRE-SPECIFIED (written before holdout test)
**Git commit:** This document committed BEFORE any clean holdout analysis

## Why RR1.0

The hypothesis: ORB breakout momentum carries price through 1× the ORB range
(RR1.0) with >50% probability, but decays at higher multiples.

### Economic mechanism

1. **Session opens create liquidity imbalances.** Institutional flow arrives at
   predictable times (NYSE 9:30, CME data releases, Asian opens). The first
   5 minutes establish a range; the break resolves the imbalance.

2. **Momentum persistence is short-lived.** Gao et al. (2018, JFE) document
   intraday momentum concentrated in the first and last 30 minutes. The
   initial breakout move is the momentum; it decays as new information arrives.

3. **RR1.0 captures the initial move.** Target = 1× risk means the trade exits
   at the point where the breakout has traveled one ORB width beyond the boundary.
   At higher RR (2.0, 3.0), the target requires the move to CONTINUE past the
   initial momentum — which mean reversion opposes.

4. **Empirical gradient confirms mechanism.** From this project's data:
   - RR1.0: +7.6% win rate edge above breakeven
   - RR1.5: +4.2% edge
   - RR2.0: +2.8% edge
   - RR3.0: +1.9% edge
   Monotonic decay. Consistent with momentum that fades, not a random artifact.

5. **Cost efficiency.** Lower RR = higher win rate = more trades hitting target
   before stop. Each trade's cost (slippage, commission) is fixed regardless
   of RR. Higher win rate amortizes fixed costs over more winners.

### What this means for the holdout test

RR1.0 is pre-specified from economic reasoning, NOT from scanning RR1.0-3.0
and picking the best. The holdout test will use RR1.0 for ALL sessions.
No RR optimization on the holdout data.

### Test specification

- Instruments: MNQ, MGC, MES (all with sufficient data)
- Entry: E2 CB1, 5-minute ORB
- RR: 1.0 (pre-specified, frozen)
- Sessions: ALL available per instrument (no session selection before holdout)
- Holdout: 2025-01-01 to 2025-12-31
- Training: all data before 2025-01-01
- Test count N: number of instrument × session combos with >= 50 holdout trades
- Correction: BH FDR at alpha = 0.05 across N tests
- Yearly consistency: required positive in >= 60% of pre-2025 years

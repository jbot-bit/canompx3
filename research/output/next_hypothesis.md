# Next Hypothesis (queued)

## Hypothesis: Relay Chain Confirmation (3-leg)

Instead of single leader->follower confirmation, require a relay:

1. `M6E_US_EQUITY_OPEN` sets direction, then
2. `MES_US_DATA_OPEN` confirms same direction, then
3. trade `M2K_US_POST_EQUITY` only if both align.

### Why this hypothesis
- Builds on the strongest current shinies.
- May reduce false positives from one noisy leader.
- Should trade less frequently but with higher quality.

### Fast test spec
- Baseline: `M2K_US_POST_EQUITY` strategy slice (`E1/CB5/RR1.5`).
- Compare:
  - no filter
  - single leader filter (best current)
  - relay chain filter (new)
- Metrics: avgR, WR, uplift, sample size, yearly stability, quick OOS.

### Keep/Kill gate
- KEEP only if relay chain beats single-leader filter on avgR and OOS uplift without collapsing sample too far.

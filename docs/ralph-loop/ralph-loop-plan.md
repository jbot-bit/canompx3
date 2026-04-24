## Iteration: 171
## Target: trading_app/lane_allocator.py:506 (finding rooted in lane_correlation.py audit)
## Finding: CORRELATION_REJECT_RHO = 0.70 in lane_allocator.py duplicates RHO_REJECT_THRESHOLD = 0.70 from lane_correlation.py — parallel constants that can silently diverge. Comment on line 506 explicitly acknowledges it ("Same as lane_correlation.RHO_REJECT_THRESHOLD").
## Classification: [mechanical]
## Blast Radius: 2 files (lane_allocator.py, test_lane_allocator.py), 0 callers change behavior
## Invariants:
##   1. Threshold value 0.70 must remain unchanged (research-derived)
##   2. build_allocation correlation gate behavior (rho > threshold) must not change
##   3. test_corr_threshold_boundary semantics (at threshold = pass, NOT above = reject) must not change
## Diff estimate: 4 lines
## Doctrine cited: integrity-guardian.md § 2 (never hardcode — import from canonical source); institutional-rigor.md § 4 (delegate to canonical sources, never re-encode)

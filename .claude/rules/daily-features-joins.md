---
paths:
  - "pipeline/**"
  - "trading_app/**"
  - "research/**"
  - "scripts/**"
---
# Daily Features JOIN Rules

## The Triple-Join Trap
`daily_features` has 3 rows per (trading_day, symbol) — one per `orb_minutes` (5, 15, 30).
`orb_outcomes` also has `orb_minutes` = 5, 15, and 30 (nested ORB outcomes).

**ALWAYS join on all three columns:**
```sql
ON o.trading_day = d.trading_day
AND o.symbol = d.symbol
AND o.orb_minutes = d.orb_minutes
```

Missing the `orb_minutes` join triples row count and creates fake correlations.

## Look-Ahead Columns
- `double_break` is LOOK-AHEAD — checks if both ORB sides were hit during the FULL session (after trade entry). Cannot be used as a real-time filter.
- NODBL filter was REMOVED Feb 2026 after 6 validated strategies proved to be artifacts.

## CTE / Subquery Guard (added Apr 2026 — caught 3x N-inflation bug)
When a CTE or subquery reads daily_features for non-ORB-specific columns (prev_day_close, prev_day_range, overnight_range_pct, atr_vel_ratio, etc.), ALWAYS add `WHERE d.orb_minutes = 5` to deduplicate. Without this, the CTE produces 3 rows per (day, symbol) and any subsequent JOIN to orb_outcomes triples N, inflating t-stats by sqrt(3)=1.73x.

This is NOT the same as the triple-join rule above. The triple-join applies to direct JOINs between daily_features and orb_outcomes. This CTE guard applies when daily_features is read in isolation (e.g., to compute IBS, NR7, or any derived feature) and THEN joined to orb_outcomes.

**The bug:** `verify_external_ibs_nr7.py` initially reported t=6.78 (N=222). After adding `orb_minutes=5` to all CTEs, corrected to t=3.89 (N=74). Commit `94546ccf`.

## LAG() Queries
For LAG() or any window function on daily_features, ALWAYS filter `WHERE d.orb_minutes = 5` inside the CTE to prevent cross-orb-minutes contamination.

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

## LAG() Queries
For LAG() or any window function on daily_features, ALWAYS filter `WHERE d.orb_minutes = 5` inside the CTE to prevent cross-orb-minutes contamination.

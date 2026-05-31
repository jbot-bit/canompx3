# 2026-06-01 Bounded Benchmark Source Notes

This artifact exercises the EV-3 bounded benchmark harness against the shared
canonical `gold.db` surfaced by `pipeline.paths.GOLD_DB_PATH`.

Fixed split:
- Train: rows with `trading_day <= 2025-12-31`
- Holdout: rows with `trading_day >= 2026-01-01`
- Selection date for all baselines: `2025-12-31`

Fixed baseline definitions:
- `naive`: MNQ COMEX_SETTLE 5-minute E1, confirm_bars=1, RR=1.0
- `trend`: MNQ COMEX_SETTLE 5-minute E2, confirm_bars=1, RR=1.5
- `core_lane`: raw ORB parameter family for the current `topstep_50k_mnq_auto`
  MNQ lanes: COMEX_SETTLE 5-minute E2 RR=1.5 CB1, US_DATA_1000
  15-minute E2 RR=1.5 CB1, and TOKYO_OPEN 5-minute E2 RR=1.5 CB1

Boundary:
- The current `orb_outcomes` table does not encode the lane-level filters
  (`OVNRNG_100`, `VWAP_MID_ALIGNED`, `COST_LT08`). This report is therefore
  a bounded raw-parameter-family benchmark, not a post-filter production lane
  expectancy claim.

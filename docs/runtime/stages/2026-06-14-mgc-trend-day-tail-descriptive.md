task: MGC Trend-Day Runner — Phase A descriptive scan (read-only, no prereg, no LOCK)
mode: IMPLEMENTATION

## Scope Lock
- research/mgc_trend_day_tail_descriptive.py
- research/mgc_trend_day_exit_sweep.py
- tests/test_research/test_mgc_trend_day_trail.py
- tests/test_research/test_mgc_exit_sweep.py
- docs/audit/results/2026-06-14-mgc-trend-day-tail-descriptive.md

## Blast Radius
- research/mgc_trend_day_tail_descriptive.py — NEW file, zero callers, read-only descriptive scan.
- Imports/calls research/research_trend_day_mfe.py::compute_true_session_mfe (reuse, no copy).
- Reads gold.db READ_ONLY (orb_outcomes, daily_features, bars_1m canonical layers only).
- Writes: NONE to gold.db. CSV output under research/output/ + one result MD under docs/audit/results/.
- No pipeline/ or trading_app/ production logic touched. No schema change. No validated_setups write.
- Sanctioned without prereg per backtesting-methodology RULE 10 (pure truth-finding on canonical layers).

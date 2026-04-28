# Fix E2 break_dir enforcement gap

mode: IMPLEMENTATION
**Date:** 2026-04-28

## Task

Two mechanical fixes surfaced by code review of commit `dfe1d5cd`:
1. Add `orb_\w+_break_dir` to `tainted_feature_re` in drift check 124
2. Add taint warning to `LEDGER_HEADER` in shadow script

## Scope Lock

- pipeline/check_drift.py
- research/shadow_htf_mes_europe_flow_long_skip.py
- research/carry_encoding_exploration.py
- research/mnq_comex_settle_f6_inside_pdr_v1.py
- research/mnq_us_data_1000_f5_below_pdl_v1.py
- research/mnq_us_data_830_f3_near_pivot_50_v1.py
- research/mnq_usdata1000_directional_state_study_v1.py
- research/nyse_open_skip_stress_test.py
- research/r5_sizer_cross_lane_replication.py
- research/research_mnq_nyse_close_failure_mode_audit.py

## Blast Radius

`check_drift.py`: advisory-only check (no production logic). Adding `break_dir` to regex means future scripts using `orb_*_break_dir` as an E2 selector will be caught by drift check 124.

`shadow_htf_mes_europe_flow_long_skip.py`: `LEDGER_HEADER` string change only — affects newly written ledger rows, not existing rows already in the `.md` ledger file.

## Acceptance Criteria

- `tainted_feature_re` includes `orb_\w+_break_dir`
- `LEDGER_HEADER` carries a taint warning block after the canonical predicate line
- `python pipeline/check_drift.py` passes (advisory still ok)
- shadow script continues to pass drift check 124 via its `tainted` annotation

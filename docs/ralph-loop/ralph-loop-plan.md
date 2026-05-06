## Iteration: 184
## Target: trading_app/prop_profiles.py:1030 (_LANE_NAMES stale hardcode)
## Finding: _LANE_NAMES dict maps sessions to old strategy-specific names (e.g., COMEX_SETTLE->'COMEX_G8') while all current paper_trades records use the dynamically-derived format '{orb_label}_{filter_type}' from paper_trade_logger.py. get_profile_lane_definitions returns stale lane_names causing contract drift with DB contents.
## Classification: [judgment]
## Blast Radius: 1 production file (prop_profiles.py), 1 test file (test_prop_profiles.py). All 8 consumers of get_profile_lane_definitions that use lane_name (only log_trade.py) will now produce names consistent with paper_trades DB.
## Invariants: [1] parse_strategy_id logic unchanged; [2] fail-closed behavior on missing lanes unchanged; [3] existing paper_trades data NOT modified
## Diff estimate: 9 lines (prop_profiles.py: 1 line change + 7 lines of deprecated _LANE_NAMES comment; test: add assertion for dynamic lane_name)
## Doctrine cited: integrity-guardian.md § 2 (canonical violation — stale hardcoded mapping instead of dynamically derived), Ralph-specific Contract Drift extension

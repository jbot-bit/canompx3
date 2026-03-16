## Iteration: 120
## Target: trading_app/db_manager.py:verify_trading_app_schema (lines 602-708)
## Finding: verify_trading_app_schema missing 10+ migration-added columns from expected_cols sets — incomplete schema verification gate silently passes DBs missing production columns (p_value, sharpe_ann_adj, autocorr_lag1, n_trials_at_discovery, fst_hurdle, dollar agg cols, fdr cols)
## Classification: [mechanical]
## Blast Radius: 3 callers (test_db_manager.py, test_app_sync.py, db_manager.py CLI) — all test/verification only
## Invariants: [1] production init path (migrations) must not change; [2] no column additions to CREATE TABLE; [3] only add to expected_cols verification sets
## Diff estimate: 19 lines

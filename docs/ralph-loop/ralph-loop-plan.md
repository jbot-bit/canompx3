## Iteration: 129
## Target: trading_app/ml/config.py:267-274
## Finding: compute_config_hash() omits GLOBAL_FEATURES, SESSION_FEATURE_SUFFIXES, ATR_NORMALIZE, CATEGORICAL_FEATURES, and LOOKAHEAD_BLACKLIST — changes to these primary feature-engineering lists will NOT be detected as "retrain needed" by the drift check (silent failure in safety mechanism)
## Classification: [judgment]
## Blast Radius: 1 file changed (config.py); callers meta_label.py, predict_live.py, check_drift.py call compute_config_hash() unchanged. Hash value changes → stored model bundles will fail drift check (correct — they were trained without full coverage)
## Invariants:
##   1. Hash must be deterministic (sort LOOKAHEAD_BLACKLIST set before hashing)
##   2. compute_config_hash() remains the single source of truth for both training and prediction
##   3. test_config_hash_deterministic must still pass (calls function twice, asserts equal)
## Diff estimate: 2 lines added

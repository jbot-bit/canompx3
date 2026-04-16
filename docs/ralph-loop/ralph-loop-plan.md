## Iteration: 166
## Target: trading_app/consistency_tracker.py:111,213,349
## Finding: CAST(entry_time AS DATE) used for trade-day grouping instead of canonical trading_day column — UTC-cast date differs from Brisbane trading day for trades near midnight UTC
## Classification: [mechanical]
## Blast Radius: 1 production file, 1 test file, 2 read-only callers (pre_session_check.py, weekly_review.py — no signature change)
## Invariants: [1] check_consistency returns same result for same input data; [2] tests still pass; [3] no schema changes
## Diff estimate: 3 production lines

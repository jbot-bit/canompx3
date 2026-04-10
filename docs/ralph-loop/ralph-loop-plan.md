## Iteration: 165
## Target: trading_app/sprt_monitor.py:129, trading_app/sr_monitor.py:117
## Finding: Hardcoded date(2026, 1, 1) for holdout boundary instead of canonical HOLDOUT_SACRED_FROM
## Classification: [mechanical]
## Blast Radius: 2 production files (sprt_monitor.py, sr_monitor.py); 0 callers of affected lines; tests don't mock the constant
## Invariants: No behavior change. Value stays date(2026, 1, 1). Only source of truth changes to canonical import.
## Diff estimate: 4 lines (2 import lines + 2 usage substitutions)

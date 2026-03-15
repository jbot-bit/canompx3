## Iteration: 51
## Target: trading_app/live_config.py:499
## Finding: Bare `except Exception` in `_check_dollar_gate` — should be narrow exception types (ValueError, TypeError)
## Blast Radius: 1 file (live_config.py); _check_dollar_gate is private, only called at lines 587 and 726 in same file
## Invariants: [function remains fail-closed; tuple return signature unchanged; dollar gate blocking behavior unchanged]
## Diff estimate: 1 line

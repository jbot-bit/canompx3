## Iteration: 146
## Target: trading_app/pre_session_check.py:328
## Finding: check_signal_exists() hardcodes entry_model='E2' in SQL instead of using lane["entry_model"]
## Classification: [mechanical]
## Blast Radius: 1 file, 1 call site (pre_session_check.py:408), 1 test file (test_pre_session_check.py — no direct test for this function)
## Invariants:
##   1. Function signature unchanged (con, session, lane, today)
##   2. SQL query logic unchanged except parameterized entry_model
##   3. Return tuple format unchanged (bool, str)
## Diff estimate: 2 lines

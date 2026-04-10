## Iteration: 163
## Target: trading_app/pre_session_check.py:314
## Finding: check_lane_lifecycle() returns (True, "WARN: ...") on exception — fail-open when lifecycle state is unreadable, permits lane to trade
## Classification: [judgment]
## Blast Radius: 1 production file, 1 test file
## Invariants:
##   1. Main orchestration path (line 473-478) must NOT be changed
##   2. Fix only changes exception return value from True to False in check_lane_lifecycle()
##   3. All existing test assertions for blocked/stale/pass cases must still pass
## Diff estimate: 1 line production + ~8 lines test = 9 lines total

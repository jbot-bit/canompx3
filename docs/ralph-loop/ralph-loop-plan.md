## Iteration: 151
## Target: trading_app/account_hwm_tracker.py:322
## Finding: Poll failure counter not persisted — process restart between poll failures resets counter, allowing indefinite poll failures without halting (fail-open)
## Classification: [judgment]
## Blast Radius: 1 production file, 1 test file (no API change, just saves state more frequently on poll failures)
## Invariants:
##   1. After N consecutive poll failures the halt IS still triggered
##   2. _save_state() on non-threshold failure must not raise — should be same call as existing path
##   3. No behavioral change to post-threshold path or any other code path
## Diff estimate: 1 production line + ~8 test lines

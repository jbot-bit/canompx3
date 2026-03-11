## Iteration: 26
## Phase: implement
## Target: trading_app/live/position_tracker.py:189
## Finding: best_entry_price() uses `or` chain — fill_entry_price=0.0 silently falls through to engine_entry_price
## Decision: implement
## Rationale: Same pattern as OR1 (iter 21) / OR2 (iter 25). Low severity but well-understood fix, blast radius = 1 file + 1 test. Consistent with established is-None guard pattern.
## Blast Radius: position_tracker.py (fix), tests/test_trading_app/test_position_tracker.py (add zero-fill test)
## Diff estimate: 6 lines changed, 6 lines added (test)

## Iteration: 15
## Phase: implement
## Target: trading_app/walkforward.py:162,242
## Finding: IS guard (15) and window imbalance ratio (5.0) missing @research-source annotations
## Decision: implement
## Rationale: safe — comments only, no logic change, blast radius = 0 (no callers affected)
## Blast Radius: walkforward.py only
## Diff estimate: 4 lines (2 comment additions)

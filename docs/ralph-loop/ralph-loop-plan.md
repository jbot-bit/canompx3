## Iteration: 134
## Target: pipeline/stats.py:4
## Finding: Module docstring lists meta_label.py as a caller but meta_label.py does not import pipeline.stats (computes Sharpe inline instead). Real callers: evaluate.py, evaluate_validated.py, select_family_rr.py.
## Classification: [mechanical]
## Blast Radius: 0 callers affected (docstring-only change)
## Invariants: [1] No behavior change; [2] All three real callers remain correct; [3] Function signatures and return values unchanged
## Diff estimate: 1 line

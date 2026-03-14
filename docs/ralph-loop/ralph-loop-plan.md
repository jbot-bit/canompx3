## Iteration: 46
## Target: trading_app/outcome_builder.py:22
## Finding: Dead `PROJECT_ROOT` assignment — assigned but never referenced anywhere in the file or imported by any caller. Same orphan-risk pattern fixed in rolling_portfolio.py (iter 43, RP1).
## Blast Radius: 1 file (outcome_builder.py). No callers import PROJECT_ROOT from this module. Confirmed with grep across trading_app/, pipeline/, scripts/, tests/.
## Invariants:
1. All imports from outcome_builder (CONFIRM_BARS_OPTIONS, RR_TARGETS, compute_single_outcome) remain unchanged.
2. `Path` import stays — used by function signatures and GOLD_DB_PATH.
3. Zero behaviour change — removal of an unreferenced module-level assignment.
## Diff estimate: 1 line deleted

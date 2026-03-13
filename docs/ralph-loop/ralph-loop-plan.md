## Iteration: 37
## Target: trading_app/cascade_table.py:17
## Finding: CT1 — Dead `PROJECT_ROOT` assignment (defined but never used in file) — orphan risk; module docstring usage example uses relative `Path("gold.db")` instead of canonical `GOLD_DB_PATH`
## Blast Radius: 1 file changed (cascade_table.py); 3 importers (paper_trader.py, test_cascade_table.py, test_market_state.py) — none reference PROJECT_ROOT; no callee changes
## Invariants:
##   1. build_cascade_table signature unchanged
##   2. lookup_cascade signature unchanged
##   3. No logic changes — delete dead assignment only
## Diff estimate: 2 lines (delete PROJECT_ROOT line; update docstring path note)

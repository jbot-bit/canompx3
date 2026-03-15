## Iteration: 50
## Target: trading_app/execution_engine.py:457-465
## Finding: Unknown filter_type in _arm_strategies silently falls through and arms the strategy (fail-open) instead of logging an error and skipping (fail-closed). Consistent with how every other caller handles ALL_FILTERS.get() returning None.
## Blast Radius: 1 file (execution_engine.py). No callers to update — _arm_strategies is private, only called from on_bar() internally.
## Invariants:
##   1. NO_FILTER must still arm (filt is not None for NO_FILTER — it returns a NoFilter() instance)
##   2. Valid filter_types from ALL_FILTERS must still pass through to filt.matches_row()
##   3. The logger.error + continue path must not arm the strategy
## Diff estimate: 6 lines (add else: logger.error + continue after the if filt is not None block)

## Iteration: 35
## Target: trading_app/strategy_validator.py:32
## Finding: SV1 — PROJECT_ROOT defined at module level but never referenced anywhere in the file (Orphan Risk)
## Blast Radius: 1 line, 0 callers, 0 importers — each trading_app module defines its own PROJECT_ROOT independently; test file: tests/test_trading_app/test_strategy_validator.py
## Invariants:
##   1. Path import stays (used at lines 641, 1287)
##   2. No logic change — deletion only
##   3. All 49 strategy_validator tests pass post-fix
## Diff estimate: 1 line deleted

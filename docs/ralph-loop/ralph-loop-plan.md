## Iteration: 194
## Target: trading_app/strategy_fitness.py:607,720,815
## Finding: `min_rolling_trades=15` default duplicated at 3 function signatures instead of referencing `MIN_ROLLING_FIT` (canonical constant at line 95). If `MIN_ROLLING_FIT` is ever updated for research reasons (annotated @research-source @sensitivity-tested), the function defaults silently diverge.
## Classification: [mechanical]
## Blast Radius: 1 file (strategy_fitness.py), 0 external callers pass min_rolling_trades (all rely on default)
## Invariants: [1] classify_fitness() logic unchanged [2] default value remains 15 (same as MIN_ROLLING_FIT) [3] no callers broken (none pass the kwarg)
## Diff estimate: 3 lines
## Doctrine cited: integrity-guardian.md § 2 (canonical sources), institutional-rigor.md § 4 (delegate to canonical sources — never re-encode)

## Iteration: 163
## Target: trading_app/strategy_fitness.py:489, trading_app/rolling_portfolio.py:308
## Finding: Two dead parameters — `con` unused in `_compute_fitness_from_cache`, `train_months` unused in `compute_day_of_week_stats`
## Classification: [mechanical]
## Blast Radius: 2 production files, 2 call sites (strategy_fitness.py:786 + rolling_portfolio.py:599), tests not directly affected
## Invariants:
##   1. _compute_fitness_from_cache behavior MUST NOT change (only signature)
##   2. compute_day_of_week_stats behavior MUST NOT change
##   3. Call sites must be updated to not pass the removed arguments
## Diff estimate: 4 lines (2 signature lines + 2 call-site lines)

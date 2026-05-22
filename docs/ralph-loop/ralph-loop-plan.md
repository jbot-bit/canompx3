## Iteration: 195
## Target: trading_app/prop_portfolio.py:291
## Finding: Silent `except (ValueError, duckdb.Error): pass` on live fitness gate swallows compute_fitness exceptions; operator sees "Fitness: UNKNOWN" HOLD with zero diagnostic context.
## Classification: [mechanical]
## Blast Radius: 1 file (prop_portfolio.py), 0 behavior change (still HOLD on exception; only logger.warning added)
## Invariants: [1] fitness_status remains "UNKNOWN" on exception [2] line 311 gate unchanged — UNKNOWN still causes HOLD [3] no callers broken
## Diff estimate: 2 lines
## Doctrine cited: integrity-guardian.md § 6 (No silent failures — every except must record the exception)

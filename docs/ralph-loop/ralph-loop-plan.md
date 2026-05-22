## Iteration: 189
## Target: pipeline/build_daily_features.py:851-852
## Finding: GARCH convergence/model exceptions swallowed at DEBUG level — operator-invisible silent failure
## Classification: [mechanical]
## Blast Radius: 1 file (build_daily_features.py), zero callers change behavior (None return preserved)
## Invariants:
##   1. Return value stays None on exception (no behavior change)
##   2. ImportError path unchanged (already at WARNING)
##   3. No new exception types introduced
## Diff estimate: 1 line (logger.debug → logger.warning, keep exc in message)
## Doctrine cited: integrity-guardian.md § 3 (fail-closed), institutional-rigor.md § 6 (no silent failures)
## Invariants: All current callers pass instrument= explicitly; CLI default at L1784 untouched; no SQL changes
## Diff estimate: 4 lines
## Doctrine cited: integrity-guardian.md § 2 (institutional-rigor.md § 10) — never hardcode canonical sources as defaults

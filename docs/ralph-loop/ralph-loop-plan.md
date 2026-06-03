## Iteration: 225
## Target: trading_app/live/projectx/auth.py:123
## Cluster: 1 finding, types=[silent_failure], severity=[LOW]
## Classification: [mechanical]
## Blast Radius: 1 file, 1 internal caller (refresh_if_needed:80), zero external callers
## Invariants: fallback to _login() must remain; no logic change; warning log must remain
## Diff estimate: 2 lines
## Doctrine cited: institutional-rigor.md § 6 (no silent failures — every except Exception must record the exception)
## Findings deferred: none

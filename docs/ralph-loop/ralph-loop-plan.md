## Iteration: 205
## Target: trading_app/live/tradovate/http.py + order_router.py:64-99,101-149
## Finding: Tradovate submit() uses READ_POLICY (5 retries) for order mutations; build_order_spec has no clOrdId; combined risk of duplicate order on transient failure
## Classification: [judgment]
## Blast Radius: 2 production files (tradovate/http.py, tradovate/order_router.py), 1 test file
## Invariants: [existing cancel/positions READ_POLICY calls must not change, backwards-compat default READ_POLICY in shim, clOrdId must be stripped from wire_spec internal field name if present]
## Diff estimate: 8-10 lines
## Doctrine cited: integrity-guardian.md § 3 (fail-closed — never report success after exception/timeout), integrity-guardian.md § 6 (no silent failures)

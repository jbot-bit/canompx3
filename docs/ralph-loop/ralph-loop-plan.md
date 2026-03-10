## Iteration: 21
## Phase: implement
## Target: tradovate/order_router.py:136,140,202,206 + projectx/order_router.py:74,88,171
## Finding: Fill price `or` pattern uses falsy check — 0.0 fill price would be treated as None. 7 instances across 2 broker routers.
## Decision: implement
## Rationale: Wrong on principle — numeric None-checking via truthiness is a known Python antipattern. Futures never trade at 0.0 so theoretical, but would break test mocks and violates fail-closed principle. Callers already use `is not None` correctly. Fix is mechanical: `or` → `if is None` fallback, `if x` → `if x is not None`. Also normalizes ProjectX query_order_status to use float() cast (consistency with submit path).
## Blast Radius: submit() and query_order_status() in both routers → session_orchestrator.py (already uses `is not None`), webhook_server.py (uses result dict), position_tracker.py (already correct). No caller behavior changes.
## Diff estimate: ~14 lines changed (7 locations, each 1-2 lines)

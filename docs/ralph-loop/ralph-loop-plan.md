## Iteration: 126
## Target: trading_app/live/tradovate/contract_resolver.py:16-17, order_router.py:26-27, positions.py:11-12
## Finding: LIVE_BASE/DEMO_BASE URL constants duplicated across 4 tradovate modules — auth.py is canonical source, others should import from it
## Classification: [mechanical]
## Blast Radius: 3 files modified (contract_resolver.py, order_router.py, positions.py); auth.py untouched; test files: test_order_router.py, test_tradovate_positions.py, test_broker_factory.py
## Invariants: [1] LIVE_BASE value "https://live.tradovateapi.com/v1" unchanged; [2] DEMO_BASE value "https://demo.tradovateapi.com/v1" unchanged; [3] all runtime behavior identical — only deduplication
## Diff estimate: 6 lines removed + 3 lines added = 9 lines total

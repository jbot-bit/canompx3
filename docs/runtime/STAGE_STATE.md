---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Bloomey remaining fixes — bracket collar, secondary cancel, token renewal test, device ID, cost model
updated: 2026-04-03T16:30:00Z
scope_lock:
  - trading_app/live/tradovate/order_router.py
  - trading_app/live/tradovate/auth.py
  - trading_app/live/broker_dispatcher.py
  - tests/test_trading_app/test_tradovate.py
  - docs/plans/2026-03-06-multi-broker-design.md
  - pipeline/cost_model.py
blast_radius:
  - order_router.py: expand price collar to bracket prices
  - auth.py: pin device ID warning
  - broker_dispatcher.py: cancel delegation to secondaries
  - test_tradovate.py: add token renewal + query_open_orders tests
  - cost_model.py: READ ONLY check (verify numbers, may update comment)
acceptance:
  - All tests pass
  - Drift clean
  - Bracket prices validated by collar
  - Token renewal path tested
---

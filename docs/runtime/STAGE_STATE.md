---
mode: IMPLEMENTATION
task: ProjectX API compliance audit — 6 fixes against canonical spec
scope_lock:
  - trading_app/live/projectx/auth.py
  - trading_app/live/projectx/order_router.py
  - trading_app/live/projectx/data_feed.py
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_projectx_router.py
acceptance:
  - Check 8: Default base URL matches spec
  - Check 11: query_order_status handles integer status
  - Check 6: fill price field name matches spec (filledPrice)
  - Check 10: signalrcore on_open re-subscribes
  - Check 3: bracket verification uses searchOpen + type/price matching
  - Check 4: customTag set on orders, bracket cleanup uses type/price not tag
  - All existing tests pass
  - e2e sim test passes
---

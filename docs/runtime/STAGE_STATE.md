---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Fix bracket collar rejecting RR2.5+ trades + update stale doc
updated: 2026-04-03T17:00:00Z
scope_lock:
  - trading_app/live/tradovate/order_router.py
  - tests/test_trading_app/test_tradovate.py
acceptance:
  - MGC RR2.5 bracket order does NOT get collar-rejected
  - Entry stop price still collared at 0.5%
  - All tests pass
---

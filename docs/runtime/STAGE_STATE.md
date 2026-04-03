---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Fix bracket collar rejecting RR2.5+ trades
updated: 2026-04-03T17:00:00Z
scope_lock:
  - trading_app/live/tradovate/order_router.py
  - tests/test_trading_app/test_tradovate.py
blast_radius:
  - order_router.py: revert bracket collar to entry-stop-only (matches ProjectX). No caller changes needed.
  - test_tradovate.py: update bracket collar test to verify entry-only behavior.
acceptance:
  - MGC RR2.5 bracket order does NOT get collar-rejected
  - Entry stop price still collared at 0.5%
  - All tests pass
---

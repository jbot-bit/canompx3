---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Bloomey review fixes — stop_multiplier, cancel_bracket_orders, auth docstring
updated: 2026-04-03T16:00:00Z
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/live/tradovate/order_router.py
  - trading_app/live/tradovate/auth.py
  - trading_app/live/broker_dispatcher.py
  - tests/test_trading_app/test_prop_profiles.py
  - tests/test_trading_app/test_tradovate.py
blast_radius:
  - prop_profiles.py: stop_multiplier 0.75 -> 1.0 (self-funded only)
  - order_router.py: add cancel_bracket_orders method
  - auth.py: fix docstring token lifetime
  - broker_dispatcher.py: add cancel_bracket_orders delegation
acceptance:
  - python -m pytest tests/test_trading_app/ -x -q passes
  - python pipeline/check_drift.py passes
  - stop_multiplier == 1.0 on self_funded_tradovate
  - cancel_bracket_orders callable on TradovateOrderRouter
---

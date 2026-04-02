---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Tradovate broker integration + BrokerDispatcher (multi-firm deployment)
updated: 2026-04-03T10:00:00Z
scope_lock:
  - trading_app/live/tradovate/__init__.py
  - trading_app/live/tradovate/auth.py
  - trading_app/live/tradovate/order_router.py
  - trading_app/live/tradovate/contracts.py
  - trading_app/live/tradovate/positions.py
  - trading_app/live/broker_dispatcher.py
  - trading_app/live/broker_factory.py
  - trading_app/live/projectx/order_router.py
blast_radius:
  - All NEW files except broker_factory.py
  - broker_factory.py: add create_tradovate_components() alongside existing create_broker_components()
  - SessionOrchestrator NOT touched (uses BrokerRouter ABC)
acceptance:
  - TradovateAuth authenticates with demo endpoint
  - TradovateOrderRouter places market/stop/bracket orders
  - BrokerDispatcher routes to N CopyOrderRouters
  - Existing ProjectX integration unaffected
  - Drift clean
---

---
mode: IMPLEMENTATION
task: Adversarial audit Round 2 — 12 fixes for race conditions, timing, state machine hardening
scope_lock:
  - trading_app/live/session_orchestrator.py
  - trading_app/live/bar_aggregator.py
  - trading_app/live/projectx/data_feed.py
  - trading_app/live/projectx/order_router.py
  - trading_app/live/projectx/contract_resolver.py
  - trading_app/live/projectx/positions.py
  - trading_app/live/position_tracker.py
  - trading_app/execution_engine.py
  - trading_app/live/trade_journal.py
  - trading_app/live/bot_state.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - trading_app/portfolio.py
  - scripts/run_live_session.py
  - scripts/e2e_sim_test.py
  - tests/test_trading_app/test_projectx_router.py
  - tests/test_trading_app/test_execution_engine.py
  - tests/test_trading_app/test_bar_aggregator.py
acceptance:
  - R2-C1/C2: signalrcore callbacks routed via call_soon_threadsafe
  - R2-C4: _emergency_flatten cancels brackets before exit
  - R2-C5: _on_feed_stale flattens when positions exist
  - R2-C6: PENDING_EXIT stuck recovery with re-close attempt
  - R2-C3: engine.cancel_trade() removes ghost trades
  - R2-C7: Rollover close failure does not reset engine
  - R2-H2/H3: Position tracker state guards on transitions
  - R2-H4: Strategy-scoped bracket cancellation
  - R2-H5: Engine circuit breaker flattens before pausing
  - R2-H6: Consistent _total_pnl_r() for all entry models
  - R2-M1: _publish_state moved after engine processing
  - R2-M5: Preflight uses Brisbane trading day, not date.today()
  - All existing tests pass
  - e2e sim test passes
---

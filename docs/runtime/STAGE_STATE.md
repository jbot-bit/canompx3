---
mode: IMPLEMENTATION
task: Live trading hardening — 7 fixes from adversarial audit (kill scenarios)
scope_lock:
  - trading_app/live/session_orchestrator.py
  - trading_app/live/projectx/order_router.py
  - trading_app/live/projectx/contract_resolver.py
  - trading_app/live/projectx/positions.py
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
acceptance:
  - Bracket legs verified after fill via searchOpen
  - Orphan AutoBracket orders cleaned on startup
  - Unrealized PnL included in risk check
  - Contract re-resolved on feed reconnect
  - Post-market 10min buffer enforced
  - Session end force-flatten at close_time_et - 5min
  - Position tracker restored from journal on restart
  - All 15 kill scenarios re-verified
  - 67 unit tests + e2e sim pass
---

---
mode: IMPLEMENTATION
task: Round 3 adversarial audit — 10 fixes (7 CRITICAL, 7 HIGH)
scope_lock:
  - trading_app/live/session_orchestrator.py
  - trading_app/live/instance_lock.py
  - trading_app/live/bar_aggregator.py
  - trading_app/live/trade_journal.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/projectx/auth.py
  - trading_app/live/projectx/data_feed.py
  - trading_app/live/tradovate/auth.py
  - trading_app/cascade_table.py
  - trading_app/dsr.py
  - trading_app/market_state.py
  - trading_app/prop_profiles.py
  - trading_app/portfolio.py
  - scripts/run_live_session.py
  - scripts/tools/session_preflight.py
  - pyproject.toml
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_webhook_server.py
  - tests/test_trading_app/test_bar_aggregator.py
  - tests/test_trading_app/test_instance_lock.py
acceptance:
  - 18 session_orchestrator tests pass (was 74/92)
  - 12 webhook server tests pass (was 0/12)
  - tzdata/signalrcore/httpx importable
  - Multi-instance lock prevents double-start
  - Bar validation rejects bad data
  - Token refresh retries 3x with backoff
  - Fill price sanity check blocks bad fills
  - DuckDB connections use context managers
  - Trading day re-checked on restart
  - /tmp/ replaced with tempfile.gettempdir()
  - Session name validated at startup
  - e2e sim 7/7 pass
---

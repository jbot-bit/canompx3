---
mode: IMPLEMENTATION
task: TopstepX bot hardening — pre-live safety audit fixes
scope_lock:
  - trading_app/execution_engine.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/projectx/order_router.py
  - trading_app/live/trade_journal.py
  - trading_app/portfolio.py
  - scripts/run_live_session.py
  - scripts/e2e_sim_test.py
  - tests/test_trading_app/test_projectx_router.py
  - docs/plans/topstepx-preflight-checklist.md
acceptance:
  - max_contracts enforced at execution layer
  - Restart dedup seeds from journal
  - Naked position protection with loud alerts
  - Max DD tracking wired from profile
  - 6 broken unit tests fixed
  - Order payload logged before submission
  - All drift checks pass
  - e2e sim test passes
---

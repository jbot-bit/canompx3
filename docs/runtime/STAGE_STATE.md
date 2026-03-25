---
mode: IMPLEMENTATION
task: ORB risk management — max ORB cap on NYSE_OPEN, account upgrade prep, weekly monitoring
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/live/session_orchestrator.py
  - trading_app/weekly_review.py
  - trading_app/pre_session_check.py
  - trading_app/log_trade.py
  - tests/test_trading_app/test_prop_profiles.py
  - tests/test_trading_app/test_weekly_review.py
  - tests/test_trading_app/test_session_orchestrator.py
acceptance:
  - DailyLaneSpec has max_orb_size_pts field
  - NYSE_OPEN lane has max_orb_size_pts=150.0, others None
  - Lane registry propagates max_orb_size_pts
  - SessionOrchestrator checks ORB cap before entry
  - Weekly review section_8 shows ORB size monitor
  - Tests: 149pt passes, 150pt skips, 151pt skips, None=no cap
  - 75/75 drift, 0 regressions
---

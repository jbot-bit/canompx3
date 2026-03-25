---
mode: IMPLEMENTATION
task: Review fixes + dashboard UX tweaks
scope_lock:
  - trading_app/live/bot_state.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - trading_app/live/session_orchestrator.py
  - trading_app/live/trade_journal.py
  - trading_app/portfolio.py
  - scripts/run_live_session.py
acceptance:
  - 3 critical review findings fixed
  - Dashboard UX tweaks applied
  - All tests pass
---

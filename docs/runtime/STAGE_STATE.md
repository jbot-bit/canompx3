---
mode: IMPLEMENTATION
task: Bot operations dashboard — dark theme, mobile, ADHD-optimized
scope_lock:
  - trading_app/live/bot_state.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/session_orchestrator.py
  - scripts/run_live_session.py
  - docs/plans/topstepx-preflight-checklist.md
acceptance:
  - Dashboard serves at http://localhost:8080
  - Shows bot mode, heartbeat, 4 lane cards, trade log
  - Control buttons work (paper, kill, preflight)
  - Auto-launches with bot as daemon thread
  - Standalone launch works: python -m trading_app.live.bot_dashboard
  - Dark theme, mobile responsive
  - Bot unaffected if dashboard crashes
---

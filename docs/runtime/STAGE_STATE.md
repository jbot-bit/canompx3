---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Fix 5 dashboard issues from audit
updated: 2026-04-01T15:00:00Z
scope_lock:
  - trading_app/live/bot_dashboard.html
  - trading_app/live/bot_dashboard.py
  - trading_app/live/session_orchestrator.py
blast_radius:
  - HTML UI only + one backend endpoint change
  - No trading logic, no pipeline, no profiles
acceptance:
  - fetchAccounts on 60s interval (not just once)
  - Per-profile STOP button (not just global KILL)
  - Copies count shown on account cards
  - Per-account equity display (when copy trading active)
  - Tailwind bundled or fallback
---

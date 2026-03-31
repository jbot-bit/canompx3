---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: One-click bot launcher — mode selection in dashboard, auto-confirm for live
updated: 2026-04-01T11:00:00Z
scope_lock:
  - scripts/run_live_session.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - START_BOT.bat
blast_radius:
  - run_live_session.py: entry point for all live trading. --auto-confirm is additive only.
  - bot_dashboard.py: /api/action/start gains mode param. Existing signal-only flow unchanged.
  - bot_dashboard.html: UI-only changes to account cards.
  - START_BOT.bat: launcher behavior change (no more taskkill).
acceptance:
  - Dashboard START shows 3 modes (signal/demo/live)
  - Live requires typing LIVE in text input
  - --auto-confirm flag works in run_live_session.py
  - START_BOT.bat opens browser automatically
  - Existing signal-only flow unchanged (backward compat)
  - Preflight passes
---

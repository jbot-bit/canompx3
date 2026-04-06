# STAGE_STATE

mode: IMPLEMENTATION
task: Dashboard UI redesign — typography, hierarchy, progressive disclosure, declutter
updated: 2026-04-06T11:20+10:00
scope_lock:
  - trading_app/live/bot_dashboard.html
blast_radius: display-only, no backend changes
acceptance:
  - System font for body, monospace only for values
  - Next Session card visually dominant
  - Collapsible sections with localStorage persistence
  - Inactive profiles hidden behind toggle
  - All description/copy text removed
  - Toolbar merged into header
  - All existing JS functionality preserved (API calls, rendering, actions)
  - Dashboard loads and renders correctly at localhost:8080

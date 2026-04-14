---
mode: IMPLEMENTATION
task: Dashboard polish + usability upgrades — Tier 3 (foundation) → Tier 1 (trader-value) → Tier 2 (advanced)
created: 2026-04-15
updated: 2026-04-15
scope_lock:
  - trading_app/live/bot_dashboard.html
  - trading_app/live/bot_dashboard.py
blast_radius:
  - trading_app/live/bot_dashboard.html (CSS additive block + markup additions for Tier 1/2)
  - trading_app/live/bot_dashboard.py (Tier 2 only — additive API fields if required, no logic change)
acceptance:
  - Each tier ships with HTTP 200 at localhost:8080
  - Zero console errors on reload
  - All existing data ids present in DOM
  - Tab switch works (Dashboard / Connections)
  - Responsive breakpoints (1100px, 820px) still collapse correctly
  - Existing JS state classes untouched (.pulse, .on, .active, .hidden, .compat-hidden, .stale-indicator, .collapsed)
  - User approves visual result per tier
design_spec: docs/plans/2026-04-15-dashboard-polish-design.md
---

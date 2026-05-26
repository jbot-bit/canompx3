---
task: |
  UI declutter — make bot_dashboard.html a glanceable single-screen cockpit.
  Operator feedback: "WAY TOO MUCH STUFF, tiny text, AI-template look, doesn't
  mean anything to me." Priorities (operator-stated): (1) am I making money,
  (2) what fires next & when, (3) is it actually working.

  Approach: do NOT delete elements (77 companion pytests + JS query these IDs).
  Instead extend the existing CSS-only FOCUS-mode hide list to demote the 12
  dev-telemetry sections, enlarge the value typography on the ~4 panels that
  stay (chart, P&L strip, hero balance/PnL/DD/next-session), and remove the
  AI-template flourishes (rounded pulse START FEED button, duplicate connection
  summary). All client-side data for "next session" already exists (next-session
  + getCountdownText, lines 4224-4228) — it is buried, not missing.

mode: IMPLEMENTATION
status: DONE
priority: P2

scope_lock:
  - trading_app/live/bot_dashboard.html

## Blast Radius
- trading_app/live/bot_dashboard.html — CSS + markup only. No JS logic change,
  no element removal (IDs preserved for the 77 dashboard pytests + binding JS).
  FOCUS-mode is a CSS `:has(#focus-toggle:checked)` hack (no JS) per
  feedback_dashboard_html_focus_mode_css_only_checkbox_hack — extending the
  hide list and bumping font-size is zero-logic-risk.
- Reads: none (static asset). Writes: none. No DB, no allocator file, no
  trading-path behavior. SHOW ALL toggle still reveals every demoted section.
- Tests: trading_app/.../test_*dashboard* (77) must still pass — they assert
  on element presence/IDs which are preserved (display:none ≠ removed).

---
task: Chart cockpit — Brisbane TZ axis + time-bounded ORB rectangle + live price-relative badge
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - tests/test_trading_app/test_orb_window_payload.py
  - tests/test_trading_app/test_bot_dashboard_sse.py
---

## Blast Radius

- trading_app/live/bot_dashboard.py — patches `_orb_levels_for_instrument` only.
  Adds a stable-shape null payload helper and projects canonical
  `pipeline.dst.orb_utc_window` output into the `/api/bars-recent`
  response. No new endpoint, no schema change. Backwards-compatible:
  legacy clients that ignore the new fields keep working.
- trading_app/live/bot_dashboard.html — surgical edits to the ChartCockpit
  module: (a) Brisbane Intl-based timeFormatter + tickMarkFormatter on the
  chart instance, (b) time-bounded ORB rectangle from `timeToCoordinate`
  using the new window epoch seconds, (c) live close-vs-ORB badge in the
  header, (d) session/direction adornments derived from existing state.
  No new chart engine, no new SSE event, no new HTTP fetch.
- tests/test_trading_app/test_orb_window_payload.py — new pytest covering
  the backend patch (happy path + 2 mutation probes).
- Reads: bot_state.json (read-only), gold.db (read-only).
- Writes: none.
- Canonical-source consumers: `pipeline.dst.orb_utc_window` (per
  institutional-rigor.md § 10 + postmortem 2026-04-07).

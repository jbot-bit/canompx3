---
task: Chart cockpit — replace CSS ORB overlay with ISeriesPrimitive rectangle (canonical v5 path)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.html
  - tests/test_trading_app/test_bot_dashboard_sse.py
---

## Blast Radius

- trading_app/live/bot_dashboard.html — replaces `_renderOrbBox` CSS overlay
  with an `ISeriesPrimitive` (TradingView Lightweight Charts v5 canonical
  rectangle path per official docs:
  https://tradingview.github.io/lightweight-charts/docs/5.0/plugins/intro).
  Drops the sibling `<div id="chart-orb-box">` and its CSS class. Rectangle
  is painted inside the chart's own canvas in chart-coordinate space via
  `target.useBitmapCoordinateSpace` + `positionsBox` helper — eliminates the
  full-width fallback bug (box rendered edge-to-edge when window epochs
  unavailable) and the coordinate-translation bug (CSS pixels relative to
  `.chart-cockpit-body` vs canvas pixels relative to the price pane).
  Primitive auto-redraws on every chart resize/scroll/zoom via the chart's
  own render lifecycle, so the existing `subscribeVisibleTimeRangeChange` +
  `ResizeObserver` re-render hooks for the box become redundant.
- tests/test_trading_app/test_bot_dashboard_sse.py — no payload-shape change,
  but the ORB box DOM id is removed; if any test asserts on it, relax.

- Reads: bot_state.json (read-only), gold.db (read-only).
- Writes: none.
- Canonical-source consumers: still `pipeline.dst.orb_utc_window` (unchanged
  backend), now consumed by the primitive instead of the CSS overlay.
- No backend file touched (bot_dashboard.py payload shape unchanged).

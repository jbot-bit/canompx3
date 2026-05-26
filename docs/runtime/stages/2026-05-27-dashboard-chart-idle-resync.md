---
task: Self-heal the dashboard candlestick chart after long idle / tab-switch — add visibilitychange re-sync + client-side staleness watchdog
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.html
blast_radius: "Client JS only (bot_dashboard.html SseClient module). No Python, no schema, no order/risk path. Running --demo session untouched. Worst-case bug = a redundant read-only /api/bars-recent snapshot (idempotent). Reuses existing server recovery: Last-Event-ID replay_since() + /api/bars-recent (same path bars_source_changed already calls)."
---

## Problem

Candlestick chart freezes when the operator leaves the dashboard idle / backgrounds
the tab for a long time and returns. Data layer is healthy (verified: /api/bars-recent
returns 240 valid OHLC bars). Root cause is purely client-side:

1. No `visibilitychange` handler anywhere in bot_dashboard.html (confirmed zero matches).
   On tab re-visible nothing re-syncs the chart.
2. Browsers throttle/suspend background EventSource connections; on return the SSE can be
   in a zombie state where `onerror` never fires (MDN: error only fires on failed-open/CLOSED),
   so the existing manual-reconnect path (es.readyState === CLOSED) never triggers.
3. The SSE replay ring is only `_SSE_RING_SIZE = 100` events — a long idle overflows it,
   so even a clean Last-Event-ID reconnect leaves a hole in the bars.

## Approach (grounded in official docs)

- MDN Page Visibility API: listen for `visibilitychange` on `document`, gate on
  `document.visibilityState === "visible"`. (Canonical re-sync pattern.)
- MDN EventSource / Using server-sent events: EventSource auto-reconnects on drop; server
  uses `id:` field + `Last-Event-ID` header for replay. Server already implements
  `_SSEBroker.replay_since()` (bot_dashboard.py:2599) + `/api/events/stream` Last-Event-ID
  (bot_dashboard.py:3363) — no server change needed.

Two client-side additions inside the `SseClient` IIFE:

1. `visibilitychange` handler: on visible, if `es.readyState !== OPEN` force `connect()`;
   always re-snapshot via `ChartCockpit.loadInitial(currentInstrument)` to repaint from
   canonical history (covers ring-overflow). Same recovery path `bars_source_changed` uses.
2. Staleness watchdog: a `setInterval` that, while the tab is visible, checks time since the
   last `bar`/`heartbeat`; if it exceeds a threshold, forces reconnect + snapshot. Closes the
   zombie-stall case even without a tab switch.

ChartCockpit must expose its `currentInstrument` (read-only) so SseClient can re-snapshot
the right instrument.

## Acceptance

- visibilitychange handler present, gated on visibilityState === "visible".
- watchdog interval present, only acts while document is visible, idempotent.
- ChartCockpit exposes current instrument for the snapshot.
- In-browser proof (Playwright): background the tab, advance, foreground → chart repaints
  (bar count grows / last-bar time advances), no console errors.
- 77/77 dashboard pytests pass.
- python pipeline/check_drift.py passes.

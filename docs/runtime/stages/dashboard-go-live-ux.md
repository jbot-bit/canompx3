---
task: Dashboard UX — replace chart "no feed" wall-of-text with one START FEED button + replace per-account type-LIVE gate with double-click-and-hold GO LIVE
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.html
---

## Blast Radius

- `trading_app/live/bot_dashboard.html` — UX-only. Replaces the chart fallback message with a one-button "NO LIVE FEED → START FEED" panel that calls existing `launchSession(..., "signal")`. Replaces per-account `Live` button + type-LIVE gate with a hold-to-confirm `GO LIVE` button (pointerdown/pointerup, 2-second ring, fires existing `launchSession(..., "live")`).
- No `pipeline/` or `trading_app/*.py` changes. No schema. No canonical sources touched. No new endpoints — uses existing `/api/action/start` via `launchSession()`.
- Reads: none (DOM-only). Writes: none.
- Affects: dashboard operator UX. Reverting = revert the diff; zero state migration.

## Done criteria

- Chart fallback shows big NO LIVE FEED + START FEED button (no wall of text).
- Per-account row shows a single GO LIVE button (hold 2s to fire) instead of `Live` + text-input gate.
- Open dashboard in browser; both interactions work; no console errors.

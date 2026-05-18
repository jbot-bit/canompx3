---
task: Add visible button-click feedback (toast + busy spinner) to bot_dashboard so operator can tell when Signal/Preflight/Refresh actually fire
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.html
blast_radius: "Presentational HTML/CSS/JS only inside trading_app/live/bot_dashboard.html. No backend, no API, no schema, no production logic, no Python. Adds #toast-stack container, toast/busy CSS, showToast()/setBusy() helpers, and wraps existing button click handlers to set .is-busy + emit toast. Reads none; writes none; affects dashboard rendering only."
---

## Why
Operator (user) reports buttons appear unresponsive — clicking Signal/Preflight/Refresh
gives no visible confirmation. The existing `activity-panel` is hidden by default
(`display:none`) and lives below the fold; even when `showActivity()` opens it, the
operator can miss it. Result: feels broken even when the bot is fine.

## What changes
1. Add CSS for `.toast`, `#toast-stack`, `button.is-busy` (spinner + opacity).
2. Add `<div id="toast-stack"></div>` near `<body>` open.
3. Add `showToast(title, msg, tone, opts)` + `setBusy(btnId, label)` / `clearBusy(btnId)` helpers.
4. Wire `btn-preflight`, `btn-refresh`, `btn-kill`, and per-profile Start Signal button
   into the new helpers so every click shows an immediate toast + busy state.

## Blast Radius
- trading_app/live/bot_dashboard.html — CSS block (~25 lines), one new DOM element, ~30 lines JS helpers, ~6 small handler wrap edits
- No Python files touched
- No DB, no API endpoint, no schema, no canonical source
- No tests required (presentational UI only; no production logic)
- Downstream consumers: zero — this is a leaf rendering file served by `bot_dashboard.py`

## Acceptance
- Click Preflight → toast "Preflight - Running checks..." appears top-right within 50ms.
- Button shows spinner + dimmed state until response.
- On response, busy clears + toast updates to result (success / warning / error).
- Toast auto-dismisses after 6s for success/info; sticks until clicked for error.

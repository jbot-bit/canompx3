---
task: START_BOT dashboard cockpit rewrite v3 — signal-history endpoint + SSE push + chart + drawers
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - trading_app/live/bot_dashboard_legacy.html
  - pipeline/check_drift.py
  - tests/test_trading_app/test_bot_dashboard_sse.py
  - tests/test_pipeline/test_check_drift.py
---

## Blast Radius

- bot_dashboard.py — additive endpoints + SSE infra (~200 lines), single-worker process, localhost-only bind
- bot_dashboard.html — UI changes only; existing endpoints + DOM IDs preserved; polling retained as SSE fallback
- bot_dashboard_legacy.html — verbatim snapshot of pre-change HTML, single-file rollback
- pipeline/check_drift.py — 2 new checks (localhost-bind, single-worker) plus tests
- Reads: live_signals_YYYY-MM-DD.jsonl (tail), bot_state.json (mtime poll), gold.db bars_1m (read-only)
- Writes: NONE — dashboard never mutates state, all kill/start go through existing /api/action/* POSTs
- IPC: dashboard runs in a separate Python subprocess from the orchestrator (line 2270 `launch_dashboard_background`). JSONL + bot_state.json are the IPC bus; no orchestrator changes needed.
- START_BOT.bat: UNTOUCHED.
- 21 existing endpoints: UNTOUCHED.

## Stages

### Stage 1 (LOW severity — no audit required)

Goal: signal-history JSON endpoint + legacy snapshot.

Files:
- WRITE `trading_app/live/bot_dashboard_legacy.html` (verbatim copy)
- EDIT `trading_app/live/bot_dashboard.py`:
  - `_read_recent_signals(limit, since_ts) -> list[dict]` using `signal_log_rotator.signals_file_for_day`
  - `@app.get("/api/signals-recent")` returning `{"signals":[...], "server_ts":"..."}`
  - SIGNALS_DIR derivation inline-commented to pin coupling to `session_orchestrator.py:296`

Verify:
- `curl http://localhost:8080/api/signals-recent` returns `{"signals":[], ...}` when no file
- Append synthetic record → query returns it
- `since_ts` filter works
- `python pipeline/check_drift.py` passes

Audit: NOT required (LOW judgment, additive endpoint, no truth-layer write path).

### Stage 2 (HIGH severity — audit required after)

Goal: live-bars endpoint + SSE stream + 2 new drift checks.

Files:
- EDIT `trading_app/live/bot_dashboard.py`:
  - `@app.get("/api/bars-recent")` — reads `bars_1m` read-only via `pipeline.paths.GOLD_DB_PATH`; returns Lightweight-Charts-shaped JSON; ORB high/low/complete sourced via existing `/api/lane-status` data, NOT re-derived
  - `@app.get("/api/events/stream")` — SSE via `sse_starlette.EventSourceResponse`:
    - on_subscribe: emit `state` snapshot (current bot_state.json content)
    - heartbeat 1s with `{ts, bot_alive, session_state, last_tick_age_s}`
    - bar event on each closed minute (mtime-poll bars_1m at 1s; emit new rows)
    - signal event on each new line in today's `live_signals_YYYY-MM-DD.jsonl` (tail at 500ms via offset tracker)
    - state event on `bot_state.json` mtime change (poll 500ms)
    - alert event on new line in `operator_alerts.jsonl`
    - max 4 concurrent subscribers → 5th = 429 + Retry-After
    - `Last-Event-ID` replay from bounded ring buffer (last 100 events)
  - Module-level subscriber set with asyncio.Queue refs; cleanup via try/finally
  - Assertion: refuse to start if `host not in {"127.0.0.1", "localhost"}`
- EDIT `pipeline/check_drift.py`:
  - `check_dashboard_localhost_only_binding` — grep `host=` in bot_dashboard.py, assert localhost default
  - `check_dashboard_sse_single_worker` — grep `uvicorn.run` for explicit `workers=` >1
- WRITE `tests/test_trading_app/test_bot_dashboard_sse.py`:
  - test_signals_recent_returns_empty_when_no_file
  - test_signals_recent_filters_since_ts
  - test_sse_heartbeat_emits_every_second (fakeclock)
  - test_sse_signal_event_fires_on_record_append
  - test_sse_subscriber_cap_returns_429
- EDIT `tests/test_pipeline/test_check_drift.py` — add tests for the 2 new checks

Verify:
- `curl -N http://localhost:8080/api/events/stream` shows `event: heartbeat` every 1s
- Append line to today's signals JSONL → `event: signal` within ≤200ms
- `python pipeline/check_drift.py` passes (with new checks)
- `pytest tests/test_trading_app/test_bot_dashboard_sse.py -v` all pass

Audit: REQUIRED — dispatch `evidence-auditor` before Stage 3. Block Stage 3 until PASS/CONDITIONAL.

### Stage 3 (MEDIUM severity — UI only)

Goal: cockpit-default + Lightweight Charts + SSE wiring + notifications + hold-to-kill.

Files:
- EDIT `trading_app/live/bot_dashboard.html`:
  - `<head>`: `<script src="https://unpkg.com/lightweight-charts@5.2.0/dist/lightweight-charts.standalone.production.js" integrity="sha384-..." crossorigin="anonymous">` (SRI hash computed before commit)
  - Promote cockpit-mode to default (remove conditional toggle)
  - Heartbeat dot: cyan/green/yellow/red per `last_tick_age_s` thresholds
  - 5-element layout: status banner + hero card + chart + P&L strip + drawer tabs
  - ChartCockpit module: candlestick + ORB price-lines + CSS-overlay ORB box + entry/SL/TP price-lines + v5 createSeriesMarkers
  - SSE EventSource wiring with onerror → polling fallback after 10s CLOSED
  - Notification API: persistent click-to-enable button; localStorage 24h dismiss
  - Hero state machine: idle / watching / armed / fired / in-trade
  - Hold-to-kill 1500ms progress ring; if position open → secondary modal "Flatten N now? OK/Cancel"

Verify:
- 1920×1080: only 5 cockpit elements above fold
- DevTools: 1 SSE connection, heartbeat every 1s
- Synthetic signal append → notification + chart arrow + 3 price-lines + hero flip within ≤200ms
- All existing pollers still callable as fallback when SSE.readyState === CLOSED

Audit: OPTIONAL (UI-only, no truth-layer touch).

### Stage 4 (LOW severity)

Goal: drawer wrapping + retire redundant pollers.

Files:
- EDIT `trading_app/live/bot_dashboard.html`:
  - Right-side `.drawer` (480px, slide-in) wrapping `#broker-accounts-section`, `#alerts-shell`, `#activity-panel`, blotter, profiles, account specs, trade book, connections
  - Retire 3 setInterval pollers (status, alerts, lane-status); gate fetchTrades/fetchAccounts/fetchBrokerList on drawer-open

Verify:
- DevTools network shows 1 SSE conn + 0 polls with all drawers closed
- Opening Trades drawer triggers one fetchTrades + resumes interval
- Closing drawer pauses interval

Audit: OPTIONAL.

## Rollback

`copy trading_app/live/bot_dashboard_legacy.html trading_app/live/bot_dashboard.html` + restart dashboard.

## Honesty section — what this plan does NOT promise

- Pixel-perfect ORB box (Lightweight Charts has no native filled rect — CSS overlay approximates).
- Works on every browser/OS (Brave/Firefox stricter Notification policy → degrade gracefully).
- Works behind every corporate proxy (SSE may be blocked → polling fallback kicks in at 10s).
- Multi-process / multi-worker (single-uvicorn-worker assumption, asserted).
- Multi-operator (single-operator-tab assumption, dedup is per-tab).
- Offline (CDN load required; failure degrades chart to text-only — operator-visible).
- "No bugs" (claims: tested rollback + drift checks + audit gate before Stage 3).

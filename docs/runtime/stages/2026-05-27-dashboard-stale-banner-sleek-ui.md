---
task: Add flashing stale-bot banner + manual restart + sleek visual polish to live dashboard
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.html
blast_radius: "Frontend-only. bot_dashboard.html (single-page cockpit) gains: (1) a full-width pulsing stale banner driven by existing heartbeat_age_s from /api/status, (2) a one-shot WebAudio beep on stale-onset, (3) a [RESTART] button calling existing launchSession()/(POST /api/action/start) which the server already arbitrates for running sessions (bot_dashboard.py:2434), (4) elevation/shadow + radius polish via existing --shadow* design tokens currently set to none, (5) chart grows to fill ~38% dead vertical space. Reads: /api/status (no new endpoints). Writes: none. No bot_dashboard.py change. No schema. No canonical-source change."
---

## What

Live dashboard (localhost:8080) currently: heartbeat 47m stale shown as quiet text; 38% dead vertical space on 1080p; flat (all `--shadow*` tokens = `none`). User (ADHD/ASD) wants: calm by default, LOUD only when the bot needs attention; sleek like a modern consumer app; practical restart.

## Decisions (user-confirmed 2026-05-27)

- Stale alert = flashing bar + one-shot beep (fires once per stale event).
- Restart = manual button only (no auto-fire on live-capital path).
- Desktop only (mobile deferred).
- Do NOT surface all hidden panels — less noise, not more.

## Scope Lock

- trading_app/live/bot_dashboard.html

## Blast Radius

- bot_dashboard.html — single-page cockpit; adds stale banner element + CSS keyframes, one-shot audio, restart handler, shadow-token values, chart flex-grow. ~150-250 net lines.
- Reads /api/status `heartbeat_age_s` + `mode` (already fetched in fetchStatus / updateHeartbeatWarning at line 3741). No new endpoint.
- Restart calls existing `launchSession()` → POST /api/action/start; server arbitrates running session via handoff (bot_dashboard.py:2434-2445). No double-launch risk.
- No bot_dashboard.py, no schema, no canonical source touched.

## Verify

- Playwright screenshot: healthy state (no banner, filled layout, depth), forced-stale state (pulsing banner + restart visible).
- Console clean (no JS errors).
- Existing dashboard pytests still pass.
- check_drift.py passes.

# Signal Monitor — Live ORB Status Strip

**Date:** 2026-04-06
**Status:** Approved
**Depends on:** Dashboard Connections tab (done), bot_state.json lane cards (existing)

## Purpose

When trading manually, there's no way to see if the ORB system is generating signals. The bot knows (signal-only mode exists) but it only runs when explicitly started. The dashboard should show live signal status for every session, always, without the user having to start anything.

## Architecture

**Auto-start signal-only session on dashboard startup.**

The session orchestrator's `--signal-only` mode already:
- Connects to ProjectX market data feed (SignalR WebSocket)
- Monitors all lanes in the active profile
- Detects ORB formation (5/15/30 min)
- Evaluates filters (COST_LT, ORB_G, OVNRNG, etc.)
- Detects breakouts and generates signals
- Writes lane status to `bot_state.json`

The dashboard already reads `bot_state.json` every 5s and renders lane cards.

**Zero new computation code.** We reuse the entire existing pipeline. The only new things are:
1. Auto-start signal-only session in dashboard lifespan
2. Signal strip UI component on the Dashboard tab
3. Replace signal-only with real session when user clicks Demo/Live

## Signal Strip Layout

Lives on Dashboard tab between metrics strip and session timeline. Compact table — one row per lane.

### States per lane:

| Status | Dot | Meaning |
|---|---|---|
| WAITING | ○ grey | Session hasn't started yet, showing countdown |
| ORB_FORMING | ◐ yellow | Inside ORB window, bars accumulating |
| ARMED | ● cyan | ORB formed, watching for breakout |
| SIGNAL | ● green | Breakout detected, filters passed — would enter |
| SIGNAL_REJECTED | ● red | Breakout detected but filters failed — no trade |
| IN_TRADE | ● blue | Position open (only in demo/live mode) |
| DONE | ○ dim | Session complete for today |

### Columns:

| Column | Content |
|---|---|
| Time | Brisbane session time |
| Session | Session name |
| Instrument | MNQ / MGC |
| Status | Dot + label (from table above) |
| ORB Range | High — Low (once formed) |
| Direction | LONG / SHORT / — |
| Entry | E2 stop-market price (once signal fires) |
| Stop | Stop loss price |
| Target | RR target price |
| Filters | Which filters passed/failed with checkmarks |
| Setup | Entry model, RR, filter type |

### Behavior:

- Rows sorted by session time (next session first)
- Past sessions (DONE) collapsed/dimmed at bottom
- Active session (ORB_FORMING or ARMED) highlighted with accent border
- SIGNAL row pulses green — "this is tradeable right now"
- SIGNAL_REJECTED row shows which filter failed and why
- Countdown timer for WAITING sessions

## Auto-Start Signal Session

In `_lifespan` startup (bot_dashboard.py):
1. After connecting broker, check if a session is already running (mode != STOPPED)
2. If not running, auto-start signal-only for the active profile
3. Use existing `/api/action/start` with mode=signal internally (or direct subprocess)
4. When user clicks Demo/Live button, the existing session is killed and replaced

**Edge cases:**
- Dashboard restarts: signal session was already running → detect via bot_state.json, don't double-start
- No active profile: don't auto-start, show "No profile configured" in signal strip
- Market data connection fails: show error in signal strip, retry on next poll
- User starts live session: signal session stops, live takes over, signal strip shows live data

## Files Changed

| File | Change |
|---|---|
| `bot_dashboard.py` | Auto-start signal session in lifespan. Detect existing session. |
| `bot_dashboard.html` | Signal strip HTML + CSS + JS. Renders from existing lane_cards in /api/status. |

## What We Do NOT Build

- No new computation engine (reuse session orchestrator)
- No new market data connection (reuse existing feed)
- No new data format (reuse bot_state.json lane_cards)
- No separate signal API endpoint (reuse /api/status)

## Test Plan

| Test | Expected |
|---|---|
| Dashboard starts → signal session auto-starts | bot_state shows mode=SIGNAL, lanes populate |
| Signal strip shows all 5 lanes | Correct session times, instruments, setups |
| ORB forms → status updates | WAITING → ORB_FORMING → ARMED |
| Breakout detected → signal shows | Entry/stop/target prices visible, direction shown |
| Filter fails → rejection shown | Red status, which filter failed |
| User clicks Demo → signal replaced | Signal session killed, demo session starts |
| Dashboard restart while session running | Detects existing session, doesn't double-start |

# Dashboard V2 Phases 4-6 — State Panels, Sidebar, Integration

> **Status:** Design complete. Ready for implementation.
> **Date:** 2026-03-07
> **Depends on:** Phases 1-3 (committed)

## Goal

Fill the 3 placeholder main panels (ORB_FORMING, IN_SESSION, DEBRIEF), populate the 2 skeleton sidebar sections (session history, discipline), wire remaining SSE event handlers and keyboard shortcuts, add audio alerts, and connect session start/stop.

## Architecture

Component-based JS modules. Each component exports `init()` and `render(data)`. `app.js` orchestrates by importing components and routing SSE events to renderers. Server-side: new `orb_tracker.py` for ORB state tracking + SSE push.

## Phase 4: State Panels

### 4A: JS Components

New files in `ui_v2/static/js/components/`:

| File | Purpose | SSE Event | Keyboard |
|------|---------|-----------|----------|
| `debrief-form.js` | Fast-path (Space) + expandable form (D) | `debrief_required` | Space, D |
| `orb-tracker.js` | SVG progress ring, high/low/size, filter badges | `orb_update` | — |
| `trade-status.js` | Entry/stop/target levels, direction, live R, time-in-trade | `trade_update` | — |
| `signal-log.js` | Scrolling color-coded event log (max 50 entries) | `signal` | — |
| `cooling.js` | Breathing animation overlay, letter from past self | `cooling` | — |
| `manual-trade.js` | Signal-Only entry/exit form | — | M |

### 4B: HTML Panel Replacement

Replace placeholder divs in `index.html` for:
- `#panel-orb-forming` — ORB progress ring, high/low/size, instrument/session header, filter badges
- `#panel-in-session` — trade status card, signal log area
- `#panel-debrief` — trade summary + fast debrief + expandable form

### 4C: SSE + Keyboard Wiring

In `app.js`:
- Import all components
- Add SSE handlers: `orb_update`, `signal`, `trade_update`, `cooling`, `debrief_required`, `clock_tick`
- Register shortcuts: Space, D, C, 1, 2, S, M, ←, →

### Debrief Fast-Path Design

```
[Trade Summary: MGC LONG +1.2R | 08:55-09:23 | 28m]

[Space] Clean Trade          [D] Details...

Expanded form (hidden by default):
  Adherence: [FOLLOWED_PLAN ▼]
  Deviation: [text input]
  Notes: [text input]
  Letter to future self: [text input]
  [Submit]
```

Space → POST `/api/debrief` with `adherence="FOLLOWED_PLAN"`, auto-filled fields.

## Phase 5: Sidebar + Audio

### Sidebar Sections

- **Session History** (`#sidebar-session-history`): Fetch `/api/session-history/{next_session}`, render last 10 outcomes with W/L/R
- **Discipline** (`#sidebar-adherence`): Fetch `/api/adherence-stats/{session}`, render adherence %, latest letter, coaching note

### Audio Alerts (`audio.js`)

Web Audio API oscillator tones (no MP3 files):
- APPROACHING: 440Hz sine, 200ms, soft
- ALERT: 660Hz+880Hz square, 150ms×2
- ORB complete: 440→660→880Hz ascending, 100ms each
- Entry signal: 880Hz ping, 100ms
- Exit signal: 660→440Hz descending, 150ms×2
- Cooling: 220Hz sustained sine, 500ms

Muted by default. localStorage persistence. Triggered from SSE events.

## Phase 6: Integration

### Server-Side

- `orb_tracker.py`: Stateful ORB tracker — receives bar updates from session_monitor, computes high/low/size, checks filter qualification against briefings, broadcasts `orb_update` SSE
- Wire `/api/session/start` and `/api/session/stop` in server.py
- Session start: clock-based (ORB_FORMING when session time reached), NOT subprocess-dependent. Signal-Only is the primary mode.

### End-to-End Flow

```
Clock hits session time → state_broadcaster sends state_change(ORB_FORMING)
  → app.js switches to ORB panel
  → Manual: trader enters ORB high/low from chart
  → OR: session_monitor detects ORB bars from live_signals.jsonl
  → orb_tracker broadcasts orb_update
Break occurs → signal event → app.js switches to IN_SESSION
  → trade-status shows levels
Exit signal → debrief_required event → app.js switches to DEBRIEF
  → Space for clean trade, D for details
  → POST /api/debrief → cooling if loss → back to IDLE/APPROACHING
```

## Files Touched

### New files (14):
- `ui_v2/static/js/components/debrief-form.js`
- `ui_v2/static/js/components/orb-tracker.js`
- `ui_v2/static/js/components/trade-status.js`
- `ui_v2/static/js/components/signal-log.js`
- `ui_v2/static/js/components/cooling.js`
- `ui_v2/static/js/components/manual-trade.js`
- `ui_v2/static/js/audio.js`
- `ui_v2/orb_tracker.py`
- `tests/test_ui_v2/test_orb_tracker.py`

### Modified files (4):
- `ui_v2/static/index.html` — replace 3 placeholders, update sidebar
- `ui_v2/static/js/app.js` — import components, SSE handlers, keyboard
- `ui_v2/static/js/keyboard.js` — register all shortcuts
- `ui_v2/server.py` — session start/stop wiring

### Not modified:
- pipeline/, trading_app/, existing ui/, pyproject.toml, schema

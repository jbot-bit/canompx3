# Dashboard V2 — HTML/FastAPI Trading Cockpit

> **Status:** Design approved. Ready for implementation planning.
> **Date:** 2026-03-07
> **Replaces:** `ui/` (Streamlit app)
> **New directory:** `ui_v2/`

## 1. Problem Statement

The current Streamlit dashboard has fundamental gaps:
- No state for active trading sessions (state machine stops at "session approaching")
- Signal-Only mode (primary use case) is a second-class citizen — no order details, no P&L, broken discipline system
- IDLE state (60-70% of the day) shows nothing useful
- Debrief forms are too heavy (5 fields × N trades)
- No daily/weekly P&L tracking
- No overnight session recap
- Cooling period feels punitive, not restorative
- `time.sleep() → st.rerun()` polling model causes flicker and blocks interaction

## 2. Architecture

**Stack:** Static HTML/CSS/JS + FastAPI backend + Server-Sent Events (SSE)
**Why not Streamlit:** Layout control ceiling, no real-time without polling, CSS hacks are fragile
**Why not React:** Overkill for single-user tool. No build step needed.

```
Browser (index.html + JS)
    ↕ SSE (real-time events)
    ↕ REST (initial load, actions)
FastAPI server (ui_v2/server.py)
    → state_machine.py (clock-driven state transitions)
    → data_layer.py (DuckDB read-only queries)
    → discipline_api.py (JSONL I/O for debriefs/cooling)
    → sse_manager.py (broadcast to connected clients)
    → session_monitor.py (watches live_signals.jsonl)
```

## 3. State Machine (8 States)

| State | Condition | Main Panel | Sidebar |
|-------|-----------|------------|---------|
| **WEEKEND** | Sat/Sun | Week summary, last day P&L, next Monday schedule | Fitness warnings, adherence stats |
| **OVERNIGHT** | Next session outside awake hours, >60m away | Overnight session list (dimmed), day summary, rolling P&L chart | Next morning preview, coaching note |
| **IDLE** | >60m to next, awake hours | Rolling P&L sparkline (20 days), completed sessions with outcomes, next session time (NOT countdown) | Next session briefing (collapsed), session history (last 10), adherence |
| **APPROACHING** | 15-60m to next | Countdown (MM:SS), briefing cards, pre-session checklist | ATR context, session history sparkline, fitness badges, letter from past self |
| **ALERT** | <15m to next | Urgent countdown (large, red), briefing cards, commit checklist, start controls | Same as APPROACHING. Audio chime. |
| **ORB_FORMING** | Session started, aperture accumulating | Live ORB tracker: high/low/size, progress ring, filter qualification badges | Active strategies with qualified/disqualified status, signal log |
| **IN_SESSION** | ORB complete, watching for break | Trade status: levels (entry/stop/target), direction, live R, time-in-trade | Signal log, position state, day P&L, cooling if active |
| **DEBRIEF** | Trade exited | Fast debrief (1-click) + expandable full form, trade summary (entry/exit/R/time) | Signal log, day P&L, cooling timer |

### State Transitions
```
WEEKEND → IDLE (Monday 09:00 Brisbane)
IDLE → APPROACHING (60m to next session)
IDLE → OVERNIGHT (next session outside awake hours)
APPROACHING → ALERT (15m to next)
ALERT → ORB_FORMING (session start time reached)
ORB_FORMING → IN_SESSION (aperture complete)
IN_SESSION → DEBRIEF (trade exit) | IDLE/APPROACHING (no trade, session expires)
DEBRIEF → IDLE/APPROACHING (debrief submitted)
OVERNIGHT → IDLE/APPROACHING (next session in awake hours)
```

### Multi-Session Overlap
Track a **session stack** — list of active SessionState objects each with own sub-state.
Global UI state = highest-priority active session:
`DEBRIEF > IN_SESSION > ORB_FORMING > ALERT > APPROACHING > IDLE`

## 4. Layout

```
┌──────────────────────────────────────────────────────────┐
│ TOPBAR (48px)                                            │
│ [10:25 AM BRIS] [8:25 PM ET] │ ●●●○○ Session Strip │ 🔇 │
├──────────────────────────────┬───────────────────────────┤
│                              │                           │
│   MAIN PANEL (65%)           │   SIDEBAR (35%)           │
│   State-dependent content    │   Peripheral context      │
│   (see state table above)    │   (always-visible)        │
│                              │                           │
├──────────────────────────────┴───────────────────────────┤
│ STATUSBAR (32px)                                         │
│ CME_REOPEN: ORB forming 3/5 │ MGC │ Daily: +1.2R        │
└──────────────────────────────────────────────────────────┘
```

### UX Principles (from UX Critic)
1. **Cockpit instrument** — every element answers "what/when/did I do it right?" or it's noise
2. **Emotional density** — show LESS after losses, MORE during idle. Match trader's optimal state.
3. **Time is navigation** — session strip replaces page navigation. No "pages", only the timeline.
4. **Commitment loops close** — every discipline mechanic reads back in a future state
5. **Sidebar = peripheral vision** — NEVER put action buttons in sidebar. Actions in main panel only.
6. **Degrade gracefully, fail loudly** — show what's unavailable, never show stale data without indicator
7. **Dark theme only** — all color decisions against dark background first

## 5. Key Design Decisions

### 5.1 Signals Show Actionable Order Details
Every signal displays: **direction** (LONG/SHORT), **entry price**, **stop price**, **target price**, **R-risk in points**, **instrument**. Not just strategy_id + price. The trader should be able to place the order from the signal alone.

### 5.2 IDLE Shows Time, Not Countdown
IDLE displays "Next session at 10:00 AM" — NOT "in 3h 42m". Countdowns create anxiety and make it hard to disengage. Reserve countdowns for APPROACHING/ALERT only.

### 5.3 Debrief Fast Path
- **1-click "Clean Trade"** (keyboard: `Space`) — auto-fills: followed plan, calm, no notes
- **Expandable "Details..."** (keyboard: `D`) — full form for deviations
- **Batch mode** — "All trades followed plan" button when multiple pending debriefs
- Trade summary (entry, exit, R, time held) always visible above the form

### 5.4 Cooling = Breathing, Not Punishment
- Replace progress bar with breathing animation (inhale/exhale pacing guide)
- Show trader's own letter from past self (if available) instead of random quotes
- Show data: "Loss streaks of 3 happen X% of the time. Your edge is across 200+ trades."
- Soft mode override: **long-press** (not timed delay) — friction without arbitrary wait

### 5.5 Commitment = Checklist, Not Button
Replace "I commit to following the plan" with a pre-flight checklist:
- `[ ] Chart open` `[ ] Order ready` `[ ] Risk sized`
- Resets daily (sessionStorage). No persistent logging. Implicit commitment when all checked.

### 5.6 Signal-Only Mode: Manual Trade Logger
When no orchestrator running, sidebar shows manual trade log:
- "Log Entry" → instrument, direction, entry price, ORB size
- "Log Exit" → exit price, auto-computes R
- Writes same JSONL format → debrief system works identically
- State machine still transitions by clock (ORB_FORMING, IN_SESSION)
- ORB tracker shows manual input fields ("enter ORB high/low from your chart")

### 5.7 Overnight "What You Missed" Recap
Morning IDLE state shows overnight session outcomes:
- Per-session: which instruments broke, direction, R result
- Total overnight R
- Sourced from `orb_outcomes` table for previous trading day's overnight sessions

### 5.8 Weekly P&L (Monday + Friday + Weekend)
- Monday morning: "Last week: +4.2R across 23 trades (14W 9L)"
- Friday evening: "This week: +2.8R, best session: TOKYO_OPEN (+3.1R)"
- Weekend: Full week summary with session breakdown, instrument breakdown, adherence stats

### 5.9 Session History in Briefings
Each briefing card includes: "Last 5 [SESSION_NAME]: 3W 2L, +2.1R, avg ORB 11.2pts"
Sourced from `orb_outcomes` table filtered by session label.

### 5.10 Multi-Aperture Wave Guidance
Sessions with both 5m and 15m strategies show:
"Wave 1: 5m ORB (forming now) → Wave 2: 15m ORB (in 10 min)"
Clear indicator of how many signal waves to expect.

## 6. CSS Design System

```css
:root {
  /* Surfaces */
  --surface-bg:         #0a0e14;
  --surface-card:       #131820;
  --surface-card-hover: #1a2230;
  --surface-border:     #1e2a3a;
  --surface-topbar:     #0d1117;

  /* Text */
  --text-primary:       #e6edf3;
  --text-secondary:     #8b949e;
  --text-tertiary:      #484f58;

  /* Trading semantics */
  --color-long:         #3fb950;   /* Profit / Buy */
  --color-short:        #f85149;   /* Loss / Sell */
  --color-neutral:      #58a6ff;   /* Info / Selected */
  --color-warning:      #d29922;   /* Warning / Amber */

  /* State accents */
  --state-idle:         #8b949e;
  --state-approaching:  #58a6ff;
  --state-alert:        #f85149;
  --state-orb-forming:  #d29922;
  --state-in-session:   #3fb950;
  --state-debrief:      #bc8cff;

  /* Fitness regimes */
  --regime-fit:         #3fb950;
  --regime-watch:       #d29922;
  --regime-decay:       #f85149;

  /* Typography */
  --font-mono:  'JetBrains Mono', 'Fira Code', monospace;
  --font-sans:  'Inter', -apple-system, sans-serif;

  /* Spacing: 4px base */
  --space-1: 0.25rem;  --space-2: 0.5rem;  --space-3: 0.75rem;
  --space-4: 1rem;     --space-6: 1.5rem;  --space-8: 2rem;

  /* Layout */
  --topbar-height:    48px;
  --statusbar-height: 32px;
  --main-width:       65%;
  --sidebar-width:    35%;
  --border-radius:    6px;
}
```

**Typography rules:**
- Prices, R-values, times, counts → `--font-mono` (tabular alignment)
- Labels, instructions, prose → `--font-sans`
- All contrast ratios ≥ 4.5:1 (WCAG AA)

## 7. API Endpoints

### REST
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve index.html |
| `/api/state` | GET | Full current state snapshot |
| `/api/briefings` | GET | Briefing cards with fitness data |
| `/api/session-history/{name}` | GET | Last 10 occurrences of a session |
| `/api/day-summary` | GET | Today's completed sessions + outcomes |
| `/api/rolling-pnl` | GET | Daily/weekly/monthly R totals + sparkline data |
| `/api/overnight-recap` | GET | Overnight session outcomes |
| `/api/fitness` | GET | Fitness regime for all live strategies |
| `/api/adherence-stats/{name}` | GET | Discipline stats for session |
| `/api/debrief/pending` | GET | Pending debriefs |
| `/api/debrief` | POST | Submit debrief |
| `/api/trade-log` | POST | Manual trade entry/exit (Signal-Only) |
| `/api/session/start` | POST | Start live session subprocess |
| `/api/session/stop` | POST | Stop live session |
| `/api/commitment` | POST | Record checklist completion |
| `/api/cooling/override` | POST | Override cooling (soft mode) |

### SSE (`/api/events`)
| Event | Payload | Trigger |
|-------|---------|---------|
| `state_change` | state, next_session, minutes | Adaptive timer |
| `clock_tick` | bris_time, et_time, seconds_to_next | 1s (ALERT+), 5s otherwise |
| `orb_update` | session, instrument, high, low, size, bars, qualifications | Each bar during ORB window |
| `signal` | type, strategy_id, instrument, **direction**, **price**, **stop**, **target**, pnl_r | JSONL file watcher |
| `trade_update` | entry, current, unrealized_r, time_in_trade, stop, target | Each bar during IN_SESSION |
| `pnl_update` | daily_r, weekly_r, monthly_r | After each trade exit |
| `cooling` | active, remaining, mode | On trigger or tick |
| `debrief_required` | strategy_id, exit_ts, pnl_r | On trade exit |
| `session_dot_update` | sessions with statuses | Session start/complete |
| `fitness_alert` | strategy_id, regime, message | On regime change |

## 8. Keyboard Shortcuts

| Key | Action | Context |
|-----|--------|---------|
| `Space` | Clean Trade debrief (1-click) | DEBRIEF state |
| `D` | Expand/collapse debrief form | DEBRIEF state |
| `C` | Mark all checklist items | APPROACHING/ALERT |
| `1` | Start Signal-Only | ALERT, no session running |
| `2` | Start Demo | ALERT, no session running |
| `S` | Stop session | Session running |
| `M` | Toggle manual trade log | Signal-Only, IN_SESSION |
| `Esc` | Close expanded panel / cancel | Any |
| `←/→` | Cycle session history | IDLE |
| `?` | Show shortcut overlay | Any |

Disabled when input fields focused. Audio toggle via speaker icon in topbar.

## 9. Audio Alerts

| Sound | Trigger |
|-------|---------|
| Soft chime | → APPROACHING |
| Two-tone ascending | → ALERT |
| Three-note ascending | ORB complete |
| Sharp ping | Entry signal |
| Descending two-note | Exit signal |
| Low sustained tone | Cooling triggered |

Muted by default. Click speaker icon to enable (persisted in localStorage).
Browser autoplay: "Click to enable audio" overlay on first load.

## 10. File Structure

```
ui_v2/
  __init__.py
  server.py              # FastAPI app, REST + SSE endpoints
  state_machine.py       # 8-state machine (port + extend session_helpers.py)
  data_layer.py          # DuckDB read-only (port + extend db_reader.py)
  discipline_api.py      # Discipline data ops (port discipline_data.py)
  sse_manager.py         # SSE broadcast, client connection management
  session_monitor.py     # File watcher on live_signals.jsonl
  orb_tracker.py         # Live ORB state, filter qualification, SSE push
  static/
    index.html           # Single-page app shell
    css/
      design-system.css  # CSS custom properties, resets, base
      components.css     # Component styles
      states.css         # State-specific panel styles
      animations.css     # Transitions, pulse, breathing, fade
    js/
      app.js             # Main: SSE, state routing, panel swap
      sse-client.js      # EventSource wrapper, auto-reconnect
      clock.js           # Client-side clock (Brisbane + ET via ZoneInfo)
      keyboard.js        # Shortcut handler
      audio.js           # Audio alert manager
      components/
        topbar.js        # Session strip, clocks, mode badge
        briefing-card.js # Instrument briefing with fitness badge
        orb-tracker.js   # ORB progress ring, filter qualification
        trade-status.js  # Live trade panel (entry/stop/target/R)
        signal-log.js    # Scrolling signal log, color-coded
        debrief-form.js  # Fast-path + expanded form
        pnl-chart.js     # Rolling P&L sparkline (inline SVG)
        countdown.js     # Countdown display
        cooling.js       # Breathing animation overlay
        manual-trade.js  # Signal-Only manual entry/exit
        session-history.js # Historical session stats
    audio/
      approaching.mp3
      alert.mp3
      orb-complete.mp3
      signal-entry.mp3
      signal-exit.mp3
      cooling.mp3
```

## 11. Implementation Phases

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **1: Backend Core** | state_machine.py, data_layer.py, discipline_api.py, server.py (REST only) | None |
| **2: SSE Infra** | sse_manager.py, session_monitor.py, wire into server.py | Phase 1 |
| **3: Frontend Shell** | index.html, CSS design system, app.js, sse-client.js, clock.js | Phase 1 |
| **4: State Panels** | IDLE, APPROACHING/ALERT, ORB_FORMING, IN_SESSION, DEBRIEF panels | Phase 2+3 |
| **5: Sidebar + Polish** | Session history, signal log, fitness warnings, keyboard, audio, manual trade log | Phase 4 |
| **6: Integration** | Run alongside live session, verify SSE, debrief JSONL compat, multi-session overlap | Phase 5 |

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| DuckDB lock during pipeline write | Already read_only=True. Add retry with backoff (3 attempts). |
| SSE drops on sleep/wake | Auto-reconnect in sse-client.js + full state snapshot on reconnect |
| File watching on Windows | Use `watchfiles` (Rust). Fallback: poll mtime every 500ms |
| No live bars in Signal-Only | Manual ORB input fields. State transitions by clock, not bar receipt. |
| Browser audio autoplay block | "Click to enable audio" overlay on first load |
| ET time calculation wrong | Use `ZoneInfo("US/Eastern")` not month-based heuristic |

## 13. What This Does NOT Include (V2+)

- Chart embedding (use TradingView on separate monitor)
- Mobile/responsive layout (desktop-only, dedicated monitor)
- Multi-user support (single trader)
- Historical trade journal browser (future IDLE content)
- AI coaching content generation (existing JSONL-based notes continue)
- Light theme toggle (dark only, per UX principle #7)

# Bot Operations Dashboard — Design Doc

**Date:** 2026-03-25
**Status:** APPROVED (4TP auto-proceed)
**Purpose:** Replace terminal log monitoring with a visual web dashboard for the trading bot. ADHD-optimized: dark theme, high contrast, big status badges, mobile-friendly.

## Architecture

```
SessionOrchestrator ─writes→ data/bot_state.json (atomic, every bar)
                                    │
BotDashboard (FastAPI :8080) ─reads→ bot_state.json + live_journal.db
    GET /              → serves single HTML page
    GET /api/status    → reads bot_state.json (bot mode, lanes, heartbeat)
    GET /api/trades    → queries live_journal.db for today's trades
    POST /api/action/* → shell out to existing CLI commands
```

## Files

| File | Action | Lines (est) |
|------|--------|-------------|
| `trading_app/live/bot_state.py` | CREATE | ~60 |
| `trading_app/live/bot_dashboard.py` | CREATE | ~250 |
| `trading_app/live/session_orchestrator.py` | MODIFY | +5 lines |
| `scripts/run_live_session.py` | MODIFY | +10 lines |

## State File Format (data/bot_state.json)

```json
{
  "mode": "PAPER|LIVE|STOPPED",
  "account_id": 19858923,
  "account_name": "50KTC-V2-451890-20967121",
  "instrument": "MNQ",
  "contract": "CON.F.US.MNQ.M26",
  "heartbeat_utc": "2026-03-25T00:12:00+00:00",
  "daily_pnl_r": -0.5,
  "daily_loss_limit_r": -5.0,
  "bars_received": 42,
  "strategies_loaded": 4,
  "lanes": {
    "NYSE_CLOSE": {
      "strategy_id": "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20_O15",
      "status": "WAITING|ARMED|IN_TRADE|FLAT",
      "direction": null,
      "entry_price": null,
      "current_pnl_r": null
    }
  }
}
```

## Frontend Stack

- Tailwind CSS v4 Play CDN (dark theme, responsive, zero build)
- Vanilla JS setInterval(fetch, 5000)
- No HTMX, no React, no npm
- Single HTML served inline from FastAPI HTMLResponse

## Control Buttons

| Button | Color | Action | Safety |
|--------|-------|--------|--------|
| Start Paper | Green | `python -m scripts.run_live_session --profile apex_50k_manual --signal-only` | None |
| Start Live | Yellow | `python -m scripts.run_live_session --profile apex_50k_manual --live --account-id X` | Type CONFIRM popup |
| Kill All | Red | `echo stop > live_session.stop` | Second confirmation click |
| Preflight | Grey | `python -m scripts.run_live_session --profile apex_50k_manual --preflight` | None, output shown inline |

## Safety

- Dashboard runs in daemon thread — crash does NOT affect bot
- DuckDB reads use read_only=True with retry-backoff
- Control buttons use subprocess, never send orders directly
- State file uses atomic write (tmp + rename)
- No auth needed (localhost only, bind to 0.0.0.0 for LAN)

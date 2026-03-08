"""Dashboard V2 — FastAPI server with REST endpoints.

Serves the trading cockpit HTML frontend and provides REST API for:
- State machine (clock-driven state transitions)
- Session briefings and history
- Daily P&L tracking
- Discipline system (debriefs, cooling, adherence)

Port 8766 (avoids conflict with webhook_server on 8765).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time as _time
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ui_v2.data_layer import (
    get_fitness_regimes,
    get_overnight_recap,
    get_previous_trading_day,
    get_rolling_pnl,
    get_session_history,
    get_today_completed_sessions,
)
from ui_v2.discipline_api import (
    ADHERENCE_VALUES,
    append_debrief,
    compute_adherence_stats,
    cooling_remaining_seconds,
    get_latest_letter,
    get_pending_debriefs,
    is_cooling_active,
    load_coaching_note,
    override_cooling,
    trigger_cooling,
)
from ui_v2.session_monitor import SessionMonitor
from ui_v2.sse_manager import SSEManager
from ui_v2.state_broadcaster import StateBroadcaster
from ui_v2.state_machine import (
    AppState,
    SessionBriefing,
    SessionState,
    StateName,
    build_session_briefings,
    current_trading_day,
    get_app_state,
    get_et_time,
    get_refresh_seconds,
    resolve_global_state,
)
from ui_v2.state_persistence import (
    load_commitment_state,
    load_cooling_state,
    save_commitment_state,
    save_cooling_state,
)

log = logging.getLogger(__name__)

PORT = int(os.environ.get("DASHBOARD_PORT", "8766"))

# ── Server-side shared state ─────────────────────────────────────────────────

# Cooling state — loaded from disk on startup, persisted on change
_cooling_state: dict[str, Any] = load_cooling_state()

# Session stack — active trading sessions
_session_stack: list[SessionState] = []

# Session start lock — prevents concurrent session starts
_session_lock = asyncio.Lock()

# Commitment checklist state — loaded from disk on startup
_commitment: dict[str, Any] = load_commitment_state()

# Server start time for /api/health uptime
_server_start_time: float = _time.monotonic()

STATIC_DIR = Path(__file__).parent / "static"
SIGNALS_PATH = Path(__file__).parent.parent / "live_signals.jsonl"

# ── SSE infrastructure ────────────────────────────────────────────────────────

sse_manager = SSEManager()
session_monitor = SessionMonitor()
state_broadcaster = StateBroadcaster(sse_manager)


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Dashboard V2 server starting on port %d", PORT)
    # Start SSE infrastructure
    await sse_manager.start()
    session_monitor.start(sse_manager, SIGNALS_PATH)
    state_broadcaster.start()
    yield
    # Shutdown SSE infrastructure
    state_broadcaster.stop()
    session_monitor.stop()
    await sse_manager.shutdown()
    log.info("Dashboard V2 server shutting down")


app = FastAPI(title="ORB Trading Dashboard V2", lifespan=lifespan)

# CORS for dev (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request / Response models ────────────────────────────────────────────────


class DebriefRequest(BaseModel):
    strategy_id: str
    signal_exit_ts: str
    adherence: str
    pnl_r: float | None = None
    deviation_trigger: str | None = None
    deviation_cost_dollars: float | None = None
    notes: str | None = None
    letter_to_future_self: str | None = None


class TradeLogRequest(BaseModel):
    instrument: str
    direction: str  # "long" | "short"
    action: str  # "entry" | "exit"
    price: float
    orb_high: float | None = None
    orb_low: float | None = None
    session: str | None = None


class CommitmentRequest(BaseModel):
    items: dict[str, bool]  # {"chart_open": True, "order_ready": True, "risk_sized": True}


class CoolingOverrideRequest(BaseModel):
    reason: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _serialize_date(val: date | datetime | None) -> str | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return val.isoformat()


def _state_to_dict(state: AppState) -> dict:
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Australia/Brisbane"))
    global_state = resolve_global_state(state)

    return {
        "name": global_state.value,
        "clock_state": state.name.value,
        "next_session": state.next_session,
        "next_session_dt": _serialize_date(state.next_session_dt),
        "minutes_to_next": round(state.minutes_to_next, 1) if state.minutes_to_next else None,
        "then_session": state.then_session,
        "then_session_dt": _serialize_date(state.then_session_dt),
        "next_monday": _serialize_date(state.next_monday),
        "trading_day": _serialize_date(state.trading_day),
        "active_sessions": [
            {
                "session_name": s.session_name,
                "instrument": s.instrument,
                "sub_state": s.sub_state.value,
                "orb_minutes": s.orb_minutes,
            }
            for s in state.active_sessions
        ],
        "bris_time": now.strftime("%I:%M %p BRIS"),
        "et_time": get_et_time(now),
        "refresh_seconds": get_refresh_seconds(
            state.minutes_to_next or 999,
            is_weekend=(state.name == StateName.WEEKEND),
        ),
        "cooling_active": is_cooling_active(_cooling_state),
        "cooling_remaining": round(cooling_remaining_seconds(_cooling_state), 0),
    }


def _briefing_to_dict(b: SessionBriefing) -> dict:
    return {
        "session": b.session,
        "instrument": b.instrument,
        "conditions": b.conditions,
        "rr_target": b.rr_target,
        "entry_instruction": b.entry_instruction,
        "direction_note": b.direction_note,
        "session_hour": b.session_hour,
        "session_minute": b.session_minute,
        "orb_minutes": b.orb_minutes,
        "strategy_count": b.strategy_count,
    }


# ── Routes: HTML ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return HTMLResponse("<html><body><h1>Dashboard V2</h1><p>Place index.html in ui_v2/static/</p></body></html>")


# ── Routes: State ────────────────────────────────────────────────────────────


@app.get("/api/state")
async def get_state():
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Australia/Brisbane"))
    state = get_app_state(now)
    state.active_sessions = list(_session_stack)
    return _state_to_dict(state)


# ── Routes: Briefings ───────────────────────────────────────────────────────


@app.get("/api/briefings")
async def get_briefings():
    try:
        briefings = build_session_briefings()
        result = [_briefing_to_dict(b) for b in briefings]

        # First-run detection: if no briefings, check if LIVE_PORTFOLIO is empty
        first_run = False
        if not result:
            try:
                from trading_app.live_config import LIVE_PORTFOLIO

                if len(LIVE_PORTFOLIO) == 0:
                    first_run = True
            except ImportError:
                first_run = True

        return {"briefings": result, "first_run": first_run}
    except Exception as e:
        log.error("Failed to build briefings: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to build briefings: {e}") from e


# ── Routes: Session History ──────────────────────────────────────────────────


@app.get("/api/session-history/{name}")
async def get_session_history_endpoint(name: str, limit: int = 10):
    from pipeline.dst import SESSION_CATALOG

    if name not in SESSION_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown session: {name}")
    records = get_session_history(name, limit=min(limit, 50))
    return {"session": name, "history": records}


# ── Routes: Day Summary ─────────────────────────────────────────────────────


@app.get("/api/day-summary")
async def get_day_summary():
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Australia/Brisbane"))
    trading_day = current_trading_day(now)
    sessions = get_today_completed_sessions(trading_day)

    prev_day = get_previous_trading_day(trading_day)
    prev_sessions = get_today_completed_sessions(prev_day) if prev_day else []

    return {
        "trading_day": trading_day.isoformat(),
        "sessions": sessions,
        "previous_day": prev_day.isoformat() if prev_day else None,
        "previous_sessions": prev_sessions,
    }


# ── Routes: Rolling P&L ─────────────────────────────────────────────────────


@app.get("/api/rolling-pnl")
async def get_rolling_pnl_endpoint(days: int = 20):
    return get_rolling_pnl(days=min(days, 60))


# ── Routes: Overnight Recap ─────────────────────────────────────────────────


@app.get("/api/overnight-recap")
async def get_overnight_recap_endpoint():
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Australia/Brisbane"))
    trading_day = current_trading_day(now)
    recap = get_overnight_recap(trading_day)
    return {"trading_day": trading_day.isoformat(), "recap": recap}


# ── Routes: Fitness ──────────────────────────────────────────────────────────


@app.get("/api/fitness")
async def get_fitness():
    regimes = get_fitness_regimes()
    return {"strategies": regimes}


# ── Routes: Adherence Stats ─────────────────────────────────────────────────


@app.get("/api/adherence-stats/{name}")
async def get_adherence_stats(name: str):
    stats = compute_adherence_stats(session=name)
    letter = get_latest_letter(name)
    coaching = load_coaching_note()
    return {
        "session": name,
        "stats": stats,
        "latest_letter": letter,
        "coaching_note": coaching,
    }


# ── Routes: Debrief ─────────────────────────────────────────────────────────


@app.get("/api/debrief/pending")
async def get_pending():
    pending = get_pending_debriefs(signals_path=SIGNALS_PATH)
    return {"pending": pending}


@app.post("/api/debrief")
async def submit_debrief(req: DebriefRequest):
    if req.adherence not in ADHERENCE_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f"adherence must be one of {ADHERENCE_VALUES}, got '{req.adherence}'",
        )

    record = {
        "ts": datetime.now(UTC).isoformat(),
        "strategy_id": req.strategy_id,
        "signal_exit_ts": req.signal_exit_ts,
        "adherence": req.adherence,
        "pnl_r": req.pnl_r,
        "deviation_trigger": req.deviation_trigger,
        "deviation_cost_dollars": req.deviation_cost_dollars,
        "notes": req.notes,
        "letter_to_future_self": req.letter_to_future_self,
    }
    success = append_debrief(record)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to write debrief")

    # Trigger cooling if loss
    if req.pnl_r is not None and req.pnl_r < 0:
        trigger_cooling(
            _cooling_state,
            pnl_r=req.pnl_r,
            consecutive_losses=1,
            session_pnl_r=req.pnl_r,
        )
        save_cooling_state(_cooling_state)

    return {"status": "ok", "cooling_active": is_cooling_active(_cooling_state)}


# ── Routes: Trade Log (Signal-Only mode) ────────────────────────────────────


@app.post("/api/trade-log")
async def log_trade(req: TradeLogRequest):
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "type": f"MANUAL_{req.action.upper()}",
        "instrument": req.instrument.upper(),
        "direction": req.direction.lower(),
        "price": req.price,
        "orb_high": req.orb_high,
        "orb_low": req.orb_low,
        "session": req.session,
    }
    # Write to signals JSONL for compatibility with debrief system
    import json

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(SIGNALS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to log trade: {exc}") from exc

    return {"status": "ok", "record": record}


# ── Routes: Session Start/Stop ───────────────────────────────────────────────


@app.post("/api/session/start")
async def start_session():
    async with _session_lock:
        # Placeholder — Phase 2 will wire to SessionOrchestrator
        return {"status": "not_implemented", "message": "Session start requires Phase 2 wiring"}


@app.post("/api/session/stop")
async def stop_session():
    # Placeholder — Phase 2
    return {"status": "not_implemented", "message": "Session stop requires Phase 2 (SSE + session_monitor)"}


# ── Routes: Commitment ───────────────────────────────────────────────────────


@app.post("/api/commitment")
async def record_commitment(req: CommitmentRequest):
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("Australia/Brisbane")).date().isoformat()

    # Reset if new day
    if _commitment.get("date") != today:
        _commitment["items"] = {}
        _commitment["date"] = today

    _commitment["items"].update(req.items)
    save_commitment_state(_commitment)
    return {"status": "ok", "items": _commitment["items"], "date": today}


# ── Routes: Cooling Override ─────────────────────────────────────────────────


@app.post("/api/cooling/override")
async def cooling_override(req: CoolingOverrideRequest):
    if not is_cooling_active(_cooling_state):
        raise HTTPException(status_code=400, detail="No active cooling period")

    override_cooling(_cooling_state)
    save_cooling_state(_cooling_state)
    return {"status": "ok", "cooling_active": False}


# ── Routes: SSE ─────────────────────────────────────────────────────────────


@app.get("/api/events")
async def sse_events(request: Request):
    """SSE endpoint — streams real-time events to the browser."""

    async def event_generator():
        client_id = sse_manager.connect()
        try:
            async for event in sse_manager.subscribe(client_id):
                if await request.is_disconnected():
                    break
                yield event
        finally:
            sse_manager.disconnect(client_id)

    return EventSourceResponse(event_generator())


@app.get("/api/sse-status")
async def sse_status():
    """Return SSE connection count for monitoring."""
    return {"connections": sse_manager.connection_count}


# ── Routes: Health ──────────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check():
    """Health endpoint — DB connectivity, SSE status, uptime, state."""
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Australia/Brisbane"))
    uptime_seconds = _time.monotonic() - _server_start_time

    # DB connectivity check
    db_ok = False
    db_error = None
    try:
        from ui_v2.data_layer import get_connection

        con = get_connection()
        con.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception as exc:
        db_error = str(exc)

    # LIVE_PORTFOLIO check (first-run detection)
    portfolio_status = "ok"
    portfolio_count = 0
    try:
        from trading_app.live_config import LIVE_PORTFOLIO

        portfolio_count = len(LIVE_PORTFOLIO)
        if portfolio_count == 0:
            portfolio_status = "empty"
    except ImportError:
        portfolio_status = "unavailable"
    except Exception:
        portfolio_status = "error"

    # Current state
    state = get_app_state(now)
    global_state = resolve_global_state(state)

    return {
        "status": "healthy" if db_ok else "degraded",
        "uptime_seconds": round(uptime_seconds),
        "db_connected": db_ok,
        "db_error": db_error,
        "sse_clients": sse_manager.connection_count,
        "state": global_state.value,
        "portfolio_strategies": portfolio_count,
        "portfolio_status": portfolio_status,
        "cooling_active": is_cooling_active(_cooling_state),
        "server_time": now.isoformat(),
    }

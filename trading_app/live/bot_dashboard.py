"""Bot Operations Dashboard — local web UI for monitoring and control.

FastAPI server on port 8080. Single HTML page with Tailwind dark theme.
Reads bot_state.json + live_journal.db. Control buttons shell out to CLI.

Usage:
    Standalone: python -m trading_app.live.bot_dashboard
    Auto-launch: started as daemon thread by run_live_session.py
"""

import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from pipeline.db_config import configure_connection
from trading_app.live.bot_state import read_state

log = logging.getLogger(__name__)

PORT = int(os.environ.get("BOT_DASHBOARD_PORT", "8080"))
PROJECT_ROOT = Path(__file__).parent.parent.parent
JOURNAL_PATH = PROJECT_ROOT / "live_journal.db"
STOP_FILE = PROJECT_ROOT / "live_session.stop"

app = FastAPI(title="Bot Dashboard")


# ── API Endpoints ─────────────────────────────────────────────────────────────


@app.get("/api/status")
async def api_status():
    """Read bot state from JSON file."""
    state = read_state()
    if not state:
        return {"mode": "STOPPED", "lanes": {}, "bars_received": 0}
    # Check heartbeat staleness
    hb = state.get("heartbeat_utc")
    if hb:
        try:
            hb_dt = datetime.fromisoformat(hb)
            state["heartbeat_age_s"] = (datetime.now(UTC) - hb_dt).total_seconds()
        except (ValueError, TypeError):
            state["heartbeat_age_s"] = 9999
    else:
        state["heartbeat_age_s"] = 9999
    return state


@app.get("/api/trades")
async def api_trades():
    """Today's trades from live_journal.db."""
    if not JOURNAL_PATH.exists():
        return {"trades": [], "note": "No journal DB found"}
    try:
        with duckdb.connect(str(JOURNAL_PATH), read_only=True) as con:
            configure_connection(con)
            rows = con.execute(
                """
                SELECT trading_day, strategy_id, direction, entry_model,
                       engine_entry, fill_entry, fill_exit, pnl_r, pnl_dollars,
                       exit_reason, contracts, session_mode,
                       created_at, exited_at
                FROM live_trades
                WHERE trading_day >= CURRENT_DATE - INTERVAL '1 DAY'
                ORDER BY created_at DESC
                LIMIT 50
                """
            ).fetchall()
            cols = [
                "trading_day",
                "strategy_id",
                "direction",
                "entry_model",
                "engine_entry",
                "fill_entry",
                "fill_exit",
                "pnl_r",
                "pnl_dollars",
                "exit_reason",
                "contracts",
                "session_mode",
                "created_at",
                "exited_at",
            ]
            trades = [dict(zip(cols, r, strict=False)) for r in rows]
            return {"trades": trades}
    except Exception as e:
        return {"trades": [], "error": str(e)}


@app.post("/api/action/kill")
async def action_kill():
    """Write stop file to kill the bot gracefully."""
    try:
        STOP_FILE.write_text("stop", encoding="utf-8")
        return {"status": "ok", "message": "Stop file created — bot will shut down within 5 seconds"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/action/preflight")
async def action_preflight():
    """Run preflight checks and return output."""
    try:
        # Read profile from bot state if available, default to apex_50k_manual
        state = read_state()
        profile = "apex_50k_manual"
        portfolio_name = state.get("account_name", "")
        if portfolio_name.startswith("profile_"):
            profile = portfolio_name.removeprefix("profile_")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_live_session", "--profile", profile, "--preflight"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "output": result.stdout + result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "output": "Preflight timed out after 30s"}
    except Exception as e:
        return {"status": "error", "output": str(e)}


# ── HTML Frontend ─────────────────────────────────────────────────────────────


DASHBOARD_HTML = Path(__file__).parent / "bot_dashboard.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    if DASHBOARD_HTML.exists():
        return DASHBOARD_HTML.read_text(encoding="utf-8")
    return "<h1>Dashboard HTML not found</h1>"


# ── Server launch ─────────────────────────────────────────────────────────────


def run_dashboard(host: str = "127.0.0.1", port: int = PORT) -> None:
    """Run the dashboard server (blocking). For standalone use or subprocess."""
    uvicorn.run(app, host=host, port=port, log_level="warning")


def launch_dashboard_background(port: int = PORT) -> subprocess.Popen | None:
    """Launch dashboard as a background subprocess (Windows-safe).

    Returns the Popen handle so the caller can terminate it on exit.
    Using subprocess instead of threading avoids asyncio event loop
    conflicts between uvicorn and the main bot's event loop on Windows.
    """
    import webbrowser

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "trading_app.live.bot_dashboard", "--port", str(port)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("Bot dashboard launched as subprocess (PID %d) at http://localhost:%d", proc.pid, port)
    except Exception as e:
        log.warning("Dashboard subprocess launch failed: %s", e)
        return None
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    return proc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bot Operations Dashboard")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="127.0.0.1")
    _args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"Bot Dashboard: http://localhost:{_args.port}")
    print("Press Ctrl+C to stop")
    run_dashboard(host=_args.host, port=_args.port)

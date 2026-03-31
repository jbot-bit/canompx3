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
import threading
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.live.bot_state import read_state

log = logging.getLogger(__name__)

PORT = int(os.environ.get("BOT_DASHBOARD_PORT", "8080"))
PROJECT_ROOT = Path(__file__).parent.parent.parent
JOURNAL_PATH = PROJECT_ROOT / "live_journal.db"
STOP_FILE = PROJECT_ROOT / "live_session.stop"
LOG_DIR = PROJECT_ROOT / "logs"

app = FastAPI(title="Bot Dashboard")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_profile() -> str:
    """Read active profile from bot state, fallback to topstep_50k_mnq_auto.

    IMPORTANT: logs a warning when falling back so silent wrong-profile launches
    are detectable. The returned profile name is also shown in the API response
    so the UI can display which profile will be started.
    """
    state = read_state()
    name = state.get("account_name", "")
    if name.startswith("profile_"):
        return name.removeprefix("profile_")
    log.warning(
        "No profile in bot_state (account_name=%r) — falling back to topstep_50k_mnq_auto",
        name,
    )
    return "topstep_50k_mnq_auto"


# Track background processes. Guarded by _bg_lock to prevent race conditions
# (e.g., double-click spawning two concurrent DB writers — violates CLAUDE.md
# "NEVER run two write processes against the same DuckDB file simultaneously").
_bg_processes: dict[str, subprocess.Popen] = {}
_bg_lock = threading.Lock()


def _ensure_log_dir() -> Path:
    """Create logs/ directory if it doesn't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


# ── Shutdown handler — prevent orphaned child processes ──────────────────────


@app.on_event("shutdown")
async def _shutdown_children():
    """Terminate all background subprocesses and close file handles on server shutdown.

    Without this, orphaned session or refresh processes continue running
    after dashboard restart — a live bot with no dashboard monitoring it.
    """
    for name, val in list(_bg_processes.items()):
        # Skip non-Popen entries (log paths, file handles stored as metadata)
        if not isinstance(val, subprocess.Popen):
            # Close any file handles
            if hasattr(val, "close"):
                try:
                    val.close()
                except Exception:
                    pass
            continue
        if val.poll() is None:
            log.warning("Shutdown: terminating orphaned %s process (PID %d)", name, val.pid)
            try:
                val.terminate()
                val.wait(timeout=10)
            except Exception:
                try:
                    val.kill()
                except Exception:
                    pass


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
        # Also terminate session subprocess and close log handle
        with _bg_lock:
            if "session" in _bg_processes:
                proc = _bg_processes["session"]
                if isinstance(proc, subprocess.Popen) and proc.poll() is None:
                    proc.terminate()
            # Close session log file handle to release Windows lock
            log_file = _bg_processes.pop("_session_logfile", None)
            if log_file and hasattr(log_file, "close"):
                try:
                    log_file.close()
                except Exception:
                    pass
        return {"status": "ok", "message": "Stop file created — bot will shut down within 5 seconds"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/action/preflight")
async def action_preflight():
    """Run preflight checks and return output."""
    try:
        profile = _resolve_profile()
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_live_session", "--profile", profile, "--preflight"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "output": result.stdout + result.stderr,
            "returncode": result.returncode,
            "profile": profile,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "output": "Preflight timed out after 60s"}
    except Exception as e:
        return {"status": "error", "output": str(e)}


@app.get("/api/data-status")
async def api_data_status():
    """Data freshness for all active instruments."""
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        results = {}
        with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
            configure_connection(con)
            for inst in ACTIVE_ORB_INSTRUMENTS:
                row = con.execute("SELECT MAX(ts_utc)::DATE FROM bars_1m WHERE symbol = ?", [inst]).fetchone()
                last_date = row[0] if row and row[0] else None
                if last_date is None:
                    gap = 999
                else:
                    gap = max(0, (datetime.now(UTC).date() - last_date).days)
                results[inst] = {
                    "last_bar_date": str(last_date) if last_date else None,
                    "gap_days": gap,
                    "stale": gap > 2,
                }
        any_stale = any(r["stale"] for r in results.values())
        return {"instruments": results, "any_stale": any_stale}
    except Exception as e:
        return {"instruments": {}, "any_stale": True, "error": str(e)}


@app.post("/api/action/refresh")
async def action_refresh():
    """Download fresh data from Databento + rebuild pipeline. Runs in background.

    Output goes to logs/refresh.log (not a pipe — prevents deadlock on Windows
    where the 64KB pipe buffer fills and blocks the child process forever).
    """
    with _bg_lock:
        if "refresh" in _bg_processes:
            proc = _bg_processes["refresh"]
            if proc.poll() is None:
                return {"status": "running", "message": "Data refresh already in progress"}

        instrument = "MNQ"
        state = read_state()
        if state.get("instrument"):
            instrument = state["instrument"]

        log_file = None
        try:
            log_path = _ensure_log_dir() / "refresh.log"
            log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
            proc = subprocess.Popen(
                [sys.executable, "scripts/tools/refresh_data.py", "--instrument", instrument],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            _bg_processes["refresh"] = proc
            _bg_processes["_refresh_log"] = log_path  # type: ignore[assignment]
            _bg_processes["_refresh_logfile"] = log_file  # type: ignore[assignment]
            return {"status": "started", "message": f"Refreshing {instrument} data..."}
        except Exception as e:
            if log_file is not None:
                log_file.close()
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/action/refresh-status")
async def action_refresh_status():
    """Check data refresh progress. Reads from log file (not pipe — avoids deadlock)."""
    with _bg_lock:
        if "refresh" not in _bg_processes:
            return {"status": "idle", "output": ""}
        proc = _bg_processes["refresh"]
        log_path = _bg_processes.get("_refresh_log")
        running = isinstance(proc, subprocess.Popen) and proc.poll() is None

        # Read log file (can be read multiple times, unlike pipe)
        output = ""
        if log_path and Path(str(log_path)).exists():
            try:
                output = Path(str(log_path)).read_text(encoding="utf-8", errors="replace")
            except Exception:
                output = "(log read failed)"

        if running:
            return {"status": "running", "output": output}

        # Process finished — close log file handle
        log_file = _bg_processes.pop("_refresh_logfile", None)
        if log_file and hasattr(log_file, "close"):
            try:
                log_file.close()
            except Exception:
                pass

        status = "done" if proc.returncode == 0 else "failed"
        return {"status": status, "returncode": proc.returncode, "output": output}


@app.post("/api/action/start")
async def action_start():
    """Launch signal-only trading session from the dashboard.

    Output goes to logs/session.log (not a pipe — live sessions run for hours
    and would deadlock on a 64KB pipe buffer within minutes).
    """
    with _bg_lock:
        if "session" in _bg_processes:
            proc = _bg_processes["session"]
            if proc.poll() is None:
                return {"status": "running", "message": "Session already running"}

        profile = _resolve_profile()
        log_file = None
        try:
            log_path = _ensure_log_dir() / "session.log"
            log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_live_session",
                    "--profile",
                    profile,
                    "--signal-only",
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            _bg_processes["session"] = proc
            _bg_processes["_session_logfile"] = log_file  # type: ignore[assignment]
            return {
                "status": "started",
                "message": f"Signal-only session started: {profile}",
                "pid": proc.pid,
                "profile": profile,
            }
        except Exception as e:
            if log_file is not None:
                log_file.close()
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


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

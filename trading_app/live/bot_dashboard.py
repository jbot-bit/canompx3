"""Bot Operations Dashboard — local web UI for monitoring and control.

FastAPI server on port 8080. Single HTML page with Tailwind dark theme.
Reads bot_state.json + live_journal.db. Control buttons shell out to CLI.

Usage:
    Standalone: python -m trading_app.live.bot_dashboard
    Auto-launch: started as daemon thread by run_live_session.py
"""

import logging
import os
import re
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from pipeline.db_config import configure_connection
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH, LIVE_JOURNAL_DB_PATH
from trading_app.live.bot_state import read_state

log = logging.getLogger(__name__)

PORT = int(os.environ.get("BOT_DASHBOARD_PORT", "8080"))
PROJECT_ROOT = Path(__file__).parent.parent.parent
JOURNAL_PATH = LIVE_JOURNAL_DB_PATH
STOP_FILE = PROJECT_ROOT / "live_session.stop"
LOG_DIR = PROJECT_ROOT / "logs"
BRISBANE_TZ = ZoneInfo("Australia/Brisbane")

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: clean stale locks/state. Shutdown: terminate child processes."""
    # ── Startup ──
    import tempfile

    lock_dir = Path(tempfile.gettempdir()) / "canompx3"
    if lock_dir.exists():
        for lock_file in lock_dir.glob("bot_*.lock"):
            try:
                lock_file.unlink()
                log.info("Startup: removed stale lock %s", lock_file.name)
            except PermissionError:
                log.warning("Startup: lock %s still held — process may be running", lock_file.name)
            except Exception:
                pass

    from trading_app.live.bot_state import STATE_FILE, clear_state

    if STATE_FILE.exists():
        state = read_state()
        hb = state.get("heartbeat_utc")
        if hb:
            try:
                age = (datetime.now(UTC) - datetime.fromisoformat(hb)).total_seconds()
                if age > 300:  # 5 minutes stale = definitely dead
                    clear_state()
                    log.info("Startup: cleared stale bot_state (heartbeat %ds old)", int(age))
            except Exception:
                log.warning("Startup: failed to parse heartbeat %r — keeping state as-is", hb)

    # ── Connect brokers ──
    import asyncio as _aio

    from dotenv import load_dotenv as _ld
    _ld()  # Ensure .env loaded before broker_connections reads os.environ
    from trading_app.live.broker_connections import connection_manager
    connection_manager.load()
    await _aio.to_thread(connection_manager.connect_all_enabled)

    yield

    # ── Shutdown ──
    for name, val in list(_bg_processes.items()):
        if not isinstance(val, subprocess.Popen):
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


app = FastAPI(title="Bot Dashboard", lifespan=_lifespan)


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


def _get_session_time_brisbane(session_name: str | None, trading_day: date | None = None) -> str | None:
    """Resolve a session's Brisbane clock time for a specific trading day."""
    if not session_name:
        return None
    entry = SESSION_CATALOG.get(session_name)
    if not entry:
        return None
    resolver = entry.get("resolver")
    if not resolver:
        return None
    hour, minute = resolver(trading_day or datetime.now(BRISBANE_TZ).date())
    return f"{hour:02d}:{minute:02d}"


def _sort_time_key(time_text: str | None) -> tuple[int, int]:
    if not time_text:
        return (99, 99)
    try:
        hour_text, minute_text = time_text.split(":", maxsplit=1)
        return int(hour_text), int(minute_text)
    except (AttributeError, ValueError):
        return (99, 99)


def _strategy_meta(strategy_id: str | None, trading_day: date | None = None) -> dict[str, object]:
    """Extract human-readable strategy metadata for dashboard display."""
    strategy_id = strategy_id or ""
    session_name = None
    instrument = None

    for candidate in sorted(SESSION_CATALOG, key=len, reverse=True):
        token = f"_{candidate}_"
        if token in strategy_id:
            prefix, _, _suffix = strategy_id.partition(token)
            session_name = candidate
            instrument = prefix.split("_", maxsplit=1)[0] if prefix else None
            break

    entry_match = re.search(r"_E(\d+)", strategy_id)
    rr_match = re.search(r"_RR([0-9.]+)", strategy_id)
    cb_match = re.search(r"_CB(\d+)", strategy_id)
    filter_match = re.search(r"_CB\d+_(.+)$", strategy_id)

    session_time = _get_session_time_brisbane(session_name, trading_day)
    label = strategy_id
    if instrument and session_name:
        label = f"{instrument} {session_name}"

    return {
        "instrument_label": instrument,
        "session_name": session_name,
        "session_time_brisbane": session_time,
        "entry_model": f"E{entry_match.group(1)}" if entry_match else None,
        "rr_target": float(rr_match.group(1)) if rr_match else None,
        "confirm_bars": int(cb_match.group(1)) if cb_match else None,
        "filter_type": filter_match.group(1) if filter_match else None,
        "lane_label": label,
    }


def _legacy_lanes_to_lane_cards(
    lanes: dict[str, dict] | None,
    trading_day: date | None = None,
    account_name: str | None = None,
) -> list[dict[str, object]]:
    """Backfill lane_cards from legacy session-keyed bot_state payloads.

    Old bot_state snapshots only stored one row per session, so shared-session
    strategies cannot be recovered exactly. This function reconstructs all
    profile lanes when possible and only applies live runtime fields to the
    exact strategy rows still recoverable from the legacy payload.
    """
    raw_lanes = lanes or {}
    lane_cards: list[dict[str, object]] = []

    strategy_runtime: dict[str, dict] = {}
    session_runtime: dict[str, list[dict]] = {}
    for _lane_key, lane in raw_lanes.items():
        strategy_id = lane.get("strategy_id")
        if strategy_id:
            strategy_runtime[strategy_id] = lane
        sname = lane.get("session_name", _lane_key)
        session_runtime.setdefault(sname, []).append(lane)

    profile_id = None
    if account_name and account_name.startswith("profile_"):
        profile_id = account_name.removeprefix("profile_")

    if profile_id:
        try:
            from trading_app.prop_profiles import ACCOUNT_PROFILES

            profile = ACCOUNT_PROFILES.get(profile_id)
            if profile is not None:
                from trading_app.prop_profiles import effective_daily_lanes

                for lane in effective_daily_lanes(profile):
                    runtime = strategy_runtime.get(lane.strategy_id)
                    session_rows = session_runtime.get(lane.orb_label, [])
                    shared_session_collision = runtime is None and len(session_rows) > 0
                    meta = _strategy_meta(lane.strategy_id, trading_day)
                    lane_cards.append(
                        {
                            "lane_key": lane.strategy_id,
                            "strategy_id": lane.strategy_id,
                            "instrument": lane.instrument,
                            "session_name": lane.orb_label,
                            "session_time_brisbane": _get_session_time_brisbane(lane.orb_label, trading_day),
                            "filter_type": runtime.get("filter_type") if runtime else meta.get("filter_type"),
                            "rr_target": runtime.get("rr_target")
                            if runtime and runtime.get("rr_target") is not None
                            else meta.get("rr_target"),
                            "orb_minutes": runtime.get("orb_minutes") if runtime else None,
                            "entry_model": runtime.get("entry_model") if runtime else meta.get("entry_model"),
                            "confirm_bars": runtime.get("confirm_bars")
                            if runtime and runtime.get("confirm_bars") is not None
                            else meta.get("confirm_bars"),
                            "status": runtime.get("status", "WAITING")
                            if runtime
                            else ("UNKNOWN" if shared_session_collision else "WAITING"),
                            "direction": runtime.get("direction") if runtime else None,
                            "entry_price": runtime.get("entry_price") if runtime else None,
                            "current_pnl_r": runtime.get("current_pnl_r") if runtime else None,
                            "status_detail": (
                                "Legacy session-keyed state cannot disambiguate this lane until the bot restarts."
                                if shared_session_collision
                                else None
                            ),
                        }
                    )
        except Exception:
            log.warning("Failed to load profile %r for lane reconstruction — falling back to raw lanes",
                        profile_id, exc_info=True)

    if not lane_cards:
        for session_name, lane in raw_lanes.items():
            strategy_id = lane.get("strategy_id")
            meta = _strategy_meta(strategy_id, trading_day)
            card = {
                "lane_key": strategy_id or session_name,
                "strategy_id": strategy_id,
                "instrument": lane.get("instrument") or meta.get("instrument_label"),
                "session_name": lane.get("session_name") or meta.get("session_name") or session_name,
                "session_time_brisbane": lane.get("session_time_brisbane")
                or meta.get("session_time_brisbane")
                or _get_session_time_brisbane(session_name, trading_day),
                "filter_type": lane.get("filter_type") or meta.get("filter_type"),
                "rr_target": lane.get("rr_target") if lane.get("rr_target") is not None else meta.get("rr_target"),
                "orb_minutes": lane.get("orb_minutes"),
                "entry_model": lane.get("entry_model") or meta.get("entry_model"),
                "confirm_bars": lane.get("confirm_bars")
                if lane.get("confirm_bars") is not None
                else meta.get("confirm_bars"),
                "status": lane.get("status", "WAITING"),
                "direction": lane.get("direction"),
                "entry_price": lane.get("entry_price"),
                "current_pnl_r": lane.get("current_pnl_r"),
                "status_detail": None,
            }
            lane_cards.append(card)

    lane_cards.sort(
        key=lambda item: (
            _sort_time_key(item.get("session_time_brisbane")),
            str(item.get("instrument") or ""),
            str(item.get("strategy_id") or ""),
        )
    )
    return lane_cards


# Track background processes. Guarded by _bg_lock to prevent race conditions
# (e.g., double-click spawning two concurrent DB writers — violates CLAUDE.md
# "NEVER run two write processes against the same DuckDB file simultaneously").
_bg_processes: dict[str, subprocess.Popen] = {}
_bg_lock = threading.Lock()


def _ensure_log_dir() -> Path:
    """Create logs/ directory if it doesn't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


# ── API Endpoints ─────────────────────────────────────────────────────────────


@app.get("/api/status")
async def api_status():
    """Read bot state from JSON file."""
    state = read_state()
    if not state:
        return {"mode": "STOPPED", "lanes": {}, "lane_cards": [], "bars_received": 0}
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
    trading_day = None
    raw_trading_day = state.get("trading_day")
    if isinstance(raw_trading_day, str):
        try:
            trading_day = date.fromisoformat(raw_trading_day)
        except ValueError:
            trading_day = None
    if not state.get("lane_cards") and state.get("lanes"):
        state["lane_cards"] = _legacy_lanes_to_lane_cards(
            state.get("lanes"),
            trading_day,
            state.get("account_name"),
        )
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
                SELECT trading_day, instrument, strategy_id, direction, entry_model,
                       engine_entry, fill_entry, fill_exit, actual_r, pnl_dollars,
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
                "instrument",
                "strategy_id",
                "direction",
                "entry_model",
                "engine_entry",
                "fill_entry",
                "fill_exit",
                "actual_r",
                "pnl_dollars",
                "exit_reason",
                "contracts",
                "session_mode",
                "created_at",
                "exited_at",
            ]
            trades = [dict(zip(cols, r, strict=False)) for r in rows]
            for trade in trades:
                meta = _strategy_meta(trade.get("strategy_id"), trade.get("trading_day"))
                trade.update(meta)
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


@app.get("/api/accounts")
async def api_accounts():
    """All trading profiles with human-readable names, firm info, and lane summaries."""
    try:
        from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes, get_account_tier, get_firm_spec

        accounts = []
        for pid, p in ACCOUNT_PROFILES.items():
            tier = get_account_tier(p.firm, p.account_size)
            firm = get_firm_spec(p.firm)
            p_lanes = effective_daily_lanes(p)
            lanes_summary = []
            for lane in p_lanes:
                meta = _strategy_meta(lane.strategy_id)
                session_time = _get_session_time_brisbane(lane.orb_label)
                rr_target = meta.get("rr_target")
                rr_label = f"RR{rr_target:g}" if isinstance(rr_target, float) else None
                setup_parts = [part for part in (meta.get("entry_model"), rr_label, meta.get("filter_type")) if part]
                lanes_summary.append(
                    {
                        "session": lane.orb_label,
                        "session_time_brisbane": session_time,
                        "filter": meta.get("filter_type"),
                        "instrument": lane.instrument,
                        "entry_model": meta.get("entry_model"),
                        "rr_target": meta.get("rr_target"),
                        "confirm_bars": meta.get("confirm_bars"),
                        "label": f"{lane.instrument} {lane.orb_label}",
                        "schedule_label": (f"{session_time} Brisbane" if session_time else "Time unknown"),
                        "setup_label": " | ".join(setup_parts) if setup_parts else "Setup unknown",
                    }
                )
            lanes_summary.sort(
                key=lambda item: (
                    _sort_time_key(str(item.get("session_time_brisbane") or "")),
                    str(item.get("instrument") or ""),
                    str(item.get("session") or ""),
                )
            )
            accounts.append(
                {
                    "profile_id": pid,
                    "firm": firm.display_name,
                    "firm_key": p.firm,
                    "account_size": p.account_size,
                    "copies": p.copies,
                    "max_dd": tier.max_dd,
                    "dll": tier.daily_loss_limit,
                    "active": p.active,
                    "auto_trading": firm.auto_trading,
                    "platform": firm.platform,
                    "lane_count": len(p_lanes),
                    "lanes": lanes_summary,
                    "instruments": sorted(p.allowed_instruments) if p.allowed_instruments else [],
                    "sessions": sorted(p.allowed_sessions) if p.allowed_sessions else [],
                    "stop_multiplier": p.stop_multiplier,
                    "notes": getattr(p, "notes", None) or "",
                }
            )
        return {"accounts": accounts}
    except Exception as e:
        return {"accounts": [], "error": str(e)}


# ── Broker connection management ─────────────────────────────────────


@app.get("/api/broker/list")
async def api_broker_list():
    from trading_app.live.broker_connections import BROKER_TYPES, connection_manager

    return {
        "connections": connection_manager.list_connections(),
        "broker_types": {k: {"display": v["display"], "fields": v["fields"]} for k, v in BROKER_TYPES.items()},
    }


@app.post("/api/broker/add")
async def api_broker_add(request_body: dict | None = None):
    import asyncio

    from trading_app.live.broker_connections import connection_manager

    if not request_body:
        return JSONResponse(status_code=400, content={"error": "Request body required"})
    broker_type = request_body.get("broker_type", "")
    display_name = request_body.get("display_name", broker_type.title() if broker_type else "")
    credentials = request_body.get("credentials", {})
    if not broker_type or not credentials:
        return JSONResponse(status_code=400, content={"error": "broker_type and credentials required"})

    test_result = await asyncio.to_thread(connection_manager.test_connection, broker_type, credentials)
    if not test_result["success"]:
        return JSONResponse(status_code=400, content={"error": f"Test failed: {test_result['message']}"})

    try:
        conn = connection_manager.add_connection(broker_type, display_name, credentials)
        await asyncio.to_thread(connection_manager.connect, conn["id"])
        _equity_cache["data"] = None
        return {"success": True, "connection": conn}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/broker/remove")
async def api_broker_remove(request_body: dict | None = None):
    from trading_app.live.broker_connections import connection_manager

    if not request_body or not request_body.get("id"):
        return JSONResponse(status_code=400, content={"error": "id required"})
    if not connection_manager.remove_connection(request_body["id"]):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    _equity_cache["data"] = None
    return {"success": True}


@app.post("/api/broker/toggle")
async def api_broker_toggle(request_body: dict | None = None):
    import asyncio

    from trading_app.live.broker_connections import connection_manager

    if not request_body or not request_body.get("id"):
        return JSONResponse(status_code=400, content={"error": "id required"})
    result = connection_manager.toggle_connection(request_body["id"])
    if result is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    if result["enabled"]:
        try:
            await asyncio.to_thread(connection_manager.connect, result["id"])
        except Exception as e:
            return {"success": True, "enabled": True, "connect_error": str(e)}
    _equity_cache["data"] = None
    return {"success": True, "enabled": result["enabled"]}


@app.post("/api/broker/test")
async def api_broker_test(request_body: dict | None = None):
    import asyncio

    from trading_app.live.broker_connections import connection_manager

    if not request_body:
        return JSONResponse(status_code=400, content={"error": "Request body required"})
    broker_type = request_body.get("broker_type", "")
    credentials = request_body.get("credentials", {})
    if not broker_type or not credentials:
        return JSONResponse(status_code=400, content={"error": "broker_type and credentials required"})
    return await asyncio.to_thread(connection_manager.test_connection, broker_type, credentials)


# ── Live equity ──────────────────────────────────────────────────────
_equity_cache: dict = {"data": None, "ts": 0.0}
_equity_lock = threading.Lock()
_EQUITY_TTL = 30  # seconds — extended to 120 when a live session is detected
_EQUITY_TTL_LIVE = 120

_HWM_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "account_hwm.json"


def _load_hwm() -> dict[str, float]:
    """Load high-water-mark balances from persistent JSON."""
    try:
        if _HWM_PATH.exists():
            import json as _json

            return _json.loads(_HWM_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_hwm(hwm: dict[str, float]) -> None:
    """Persist HWM balances to JSON."""
    try:
        import json as _json

        _HWM_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HWM_PATH.write_text(_json.dumps(hwm, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Failed to save HWM: %s", e)


def _classify_account(can_trade: bool, is_visible: bool) -> str:
    """Classify account status from API fields (no simulated — REST only)."""
    if can_trade and is_visible:
        return "tradeable"
    if not can_trade and is_visible:
        return "restricted"  # outside hours, pending review, etc.
    return "archived"  # blown or hidden


def _fetch_accounts_for_connection(conn_id: str, broker_type: str, auth: object) -> dict:
    """Fetch accounts for a single broker connection using its auth singleton.

    Single API call per broker — balance included in response (no per-account loop).
    HWM tracking for correct trailing DD.
    """
    import requests as _requests

    if broker_type == "projectx":
        from trading_app.live.projectx.auth import BASE_URL

        resp = _requests.post(
            f"{BASE_URL}/api/Account/search",
            json={"onlyActiveAccounts": False},
            headers=auth.headers(),  # type: ignore[union-attr]
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"ProjectX Account/search failed: {data.get('errorMessage', data)}")

        raw_accounts = data if isinstance(data, list) else data.get("accounts") or []
        hwm = _load_hwm()
        accounts = []
        for acct in raw_accounts:
            acct_id = acct.get("id") or acct.get("accountId")
            if acct_id is None:
                continue
            balance = acct.get("balance")
            can_trade = acct.get("canTrade", False)
            is_visible = acct.get("isVisible", False)
            name = acct.get("name", f"account_{acct_id}")

            hwm_key = str(acct_id)
            stored_hwm = hwm.get(hwm_key, 0.0)
            if balance is not None and balance > stored_hwm:
                hwm[hwm_key] = balance
                stored_hwm = balance

            accounts.append({
                "id": int(acct_id),
                "name": name,
                "balance": balance,
                "hwm": stored_hwm if stored_hwm > 0 else balance,
                "can_trade": can_trade,
                "is_visible": is_visible,
                "status": _classify_account(can_trade, is_visible),
                "broker": broker_type,
                "broker_display": "TopStepX",
                "balance_type": "realized",
                "connection_id": conn_id,
            })

        _save_hwm(hwm)
        return {"accounts": accounts, "count": len([a for a in accounts if a["status"] == "tradeable"])}

    return {"accounts": [], "count": 0}


@app.get("/api/equity")
async def api_equity():
    """Live account balances from all configured brokers.

    Architecture (audit-corrected):
    - Singleton auth (token cache survives between requests)
    - Single Account/search call per broker (no per-account loop)
    - asyncio.to_thread (no event loop blocking)
    - Per-broker error isolation (one failure doesn't kill others)
    - HWM tracking for correct trailing DD
    - fetched_at timestamp for stale detection
    """
    import asyncio
    import time as _time

    from trading_app.live.broker_connections import connection_manager

    now = _time.time()

    # Check if live session is running — use longer TTL to preserve rate limit headroom
    ttl = _EQUITY_TTL
    try:
        state_data = read_state()
        if state_data.get("mode") not in (None, "STOPPED"):
            ttl = _EQUITY_TTL_LIVE
    except Exception:
        pass

    with _equity_lock:
        if _equity_cache["data"] is not None and now - _equity_cache["ts"] < ttl:
            cached = _equity_cache["data"].copy()
            cached["cached"] = True
            return cached

    brokers = []

    for conn in connection_manager.get_enabled_connections():
        conn_id = conn["id"]
        broker_type = conn["broker_type"]
        display_name = conn.get("display_name", broker_type)
        auth = connection_manager.get_auth(conn_id)

        if auth is None:
            try:
                await asyncio.to_thread(connection_manager.connect, conn_id)
                auth = connection_manager.get_auth(conn_id)
            except Exception as e:
                brokers.append({"name": broker_type, "display": display_name, "connection_id": conn_id, "error": str(e), "accounts": []})
                continue

        if auth is None:
            brokers.append({"name": broker_type, "display": display_name, "connection_id": conn_id, "error": "Auth not available", "accounts": []})
            continue

        try:
            fetch_result = await asyncio.to_thread(_fetch_accounts_for_connection, conn_id, broker_type, auth)
            connection_manager.update_account_count(conn_id, fetch_result["count"])
            brokers.append({"name": broker_type, "display": display_name, "connection_id": conn_id, "error": None, "accounts": fetch_result["accounts"]})
        except Exception as e:
            log.warning("Equity fetch failed for %s: %s", display_name, e)
            brokers.append({"name": broker_type, "display": display_name, "connection_id": conn_id, "error": str(e), "accounts": []})

    eq_result = {
        "brokers": brokers,
        "fetched_at": datetime.now(UTC).isoformat(),
        "cached": False,
    }

    with _equity_lock:
        _equity_cache["data"] = eq_result
        _equity_cache["ts"] = _time.time()

    return eq_result


@app.get("/api/sessions")
async def api_sessions():
    """Server-side DST-correct session schedule with next-session computation."""
    try:
        from pipeline.dst import SESSION_CATALOG

        now_bris = datetime.now(ZoneInfo("Australia/Brisbane"))
        today = now_bris.date()  # Use Brisbane date — not system local (matters at NYSE_OPEN midnight crossing)
        sessions = []
        for name, info in sorted(SESSION_CATALOG.items()):
            resolver = info.get("resolver")
            if not resolver:
                continue
            try:
                h, m = resolver(today)
            except Exception:
                continue
            session_time = now_bris.replace(hour=h, minute=m, second=0, microsecond=0)
            diff_min = (session_time - now_bris).total_seconds() / 60
            # Wrap to next day if >1hr past
            if diff_min < -60:
                diff_min += 1440
            sessions.append(
                {
                    "name": name,
                    "hour": h,
                    "minute": m,
                    "minutes_away": round(diff_min),
                    "status": "PASSED" if diff_min < -5 else ("NOW" if diff_min < 5 else "UPCOMING"),
                }
            )
        # Sort by minutes_away so "next" is first upcoming
        sessions.sort(key=lambda s: s["minutes_away"])
        # Find the next upcoming session
        next_session = next((s for s in sessions if s["status"] == "UPCOMING"), None)
        return {"sessions": sessions, "next": next_session}
    except Exception as e:
        return {"sessions": [], "next": None, "error": str(e)}


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

        # Close any stale log handle from a previous run that was never polled
        old_log = _bg_processes.pop("_refresh_logfile", None)
        if old_log and hasattr(old_log, "close"):
            try:
                old_log.close()
            except Exception:
                pass

        # Refresh ALL active instruments (MGC + MNQ + MES) — no --instrument flag
        log_file = None
        try:
            log_path = _ensure_log_dir() / "refresh.log"
            log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
            proc = subprocess.Popen(
                [sys.executable, "-m", "scripts.tools.refresh_data"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            _bg_processes["refresh"] = proc
            _bg_processes["_refresh_log"] = log_path  # type: ignore[assignment]
            _bg_processes["_refresh_logfile"] = log_file  # type: ignore[assignment]
            return {"status": "started", "message": "Refreshing all instruments..."}
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
async def action_start(profile: str | None = None, mode: str = "signal"):
    """Launch trading session from the dashboard.

    Args:
        profile: Profile ID to start (e.g. 'topstep_50k_mnq_auto').
                 Passed explicitly from the account card's START button.
                 Falls back to _resolve_profile() if not provided.
        mode: Execution mode — "signal" (default), "demo", or "live".
              Live mode uses --auto-confirm (safety gate is in the UI).

    Output goes to logs/session.log (not a pipe — live sessions run for hours
    and would deadlock on a 64KB pipe buffer within minutes).
    """
    if mode not in ("signal", "demo", "live"):
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid mode: {mode}"})

    with _bg_lock:
        if "session" in _bg_processes:
            proc = _bg_processes["session"]
            if proc.poll() is None:
                return {"status": "running", "message": "Session already running"}

        # Use explicit profile from card button, or fallback
        if not profile:
            profile = _resolve_profile()

        # Close any stale log handle from a previous session
        old_log = _bg_processes.pop("_session_logfile", None)
        if old_log and hasattr(old_log, "close"):
            try:
                old_log.close()
            except Exception:
                pass

        # Build command based on mode
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_live_session",
            "--profile",
            profile,
        ]
        if mode == "signal":
            cmd.append("--signal-only")
            mode_label = "SIGNAL-ONLY"
        elif mode == "demo":
            cmd.append("--demo")
            mode_label = "DEMO"
        else:  # live
            cmd.extend(["--live", "--auto-confirm"])
            mode_label = "LIVE"

        log_file = None
        try:
            log_path = _ensure_log_dir() / "session.log"
            log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            _bg_processes["session"] = proc
            _bg_processes["_session_logfile"] = log_file  # type: ignore[assignment]
            return {
                "status": "started",
                "message": f"{mode_label} session started: {profile}",
                "pid": proc.pid,
                "profile": profile,
                "mode": mode,
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

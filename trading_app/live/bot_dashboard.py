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
import tempfile
import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import duckdb
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from pipeline.db_config import configure_connection
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH, LIVE_JOURNAL_DB_PATH
from trading_app.live.alert_engine import read_operator_alerts, summarize_operator_alerts
from trading_app.live.bot_state import read_live_health, read_state
from trading_app.live.instance_lock import is_pid_alive

log = logging.getLogger(__name__)

PORT = int(os.environ.get("BOT_DASHBOARD_PORT", "8080"))
PROJECT_ROOT = Path(__file__).parent.parent.parent
JOURNAL_PATH = LIVE_JOURNAL_DB_PATH
STOP_FILE = PROJECT_ROOT / "live_session.stop"
LOG_DIR = PROJECT_ROOT / "logs"
BRISBANE_TZ = ZoneInfo("Australia/Brisbane")
# Rationale: 2x the ~60s bar cadence (SessionOrchestrator._publish_state runs
# once per 1m bar) — two consecutive missed writes flips the dashboard to STALE.
HEARTBEAT_STALE_AFTER_S = 120
LIVE_PILOT_PROFILE = "topstep_50k_mnq_auto"
LIVE_PILOT_INSTRUMENT = "MNQ"
LIVE_PILOT_COPIES = 1


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(k): v for k, v in value.items()}


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: clean stale locks/state. Shutdown: terminate child processes."""
    # ── Startup ──
    lock_dir = Path(tempfile.gettempdir()) / "canompx3"
    if lock_dir.exists():
        for lock_file in lock_dir.glob("bot_*.lock"):
            try:
                content = lock_file.read_text(encoding="utf-8").strip()
                pid = int(content) if content else None
                if pid and is_pid_alive(pid):
                    log.info("Startup: keeping live lock %s (PID %d)", lock_file.name, pid)
                    continue
                lock_file.unlink()
                log.info("Startup: removed stale lock %s", lock_file.name)
            except ValueError:
                lock_file.unlink(missing_ok=True)
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
    from dotenv import load_dotenv as _ld

    _ld(PROJECT_ROOT / ".env")
    from trading_app.live.broker_connections import connection_manager

    connection_manager.load()
    _start_broker_connect_background(connection_manager)

    yield

    # ── Shutdown ──
    # Cancel SSE watcher coroutines first — they own no resources but block
    # graceful asyncio teardown if left running. Audit Critical #2 (Stage 2,
    # commit cff1efcd verdict): _sse_tasks were orphaned at lifespan exit.
    try:
        await _sse_cancel_watchers()
    except Exception as exc:
        log.warning("Shutdown: SSE watcher cancellation failed: %s", exc)
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


def _extra_origins_from_env() -> tuple[str, ...]:
    raw = os.environ.get("DASHBOARD_ALLOWED_ORIGINS", "")
    return tuple(o.strip() for o in raw.split(",") if o.strip())


class OriginAllowlistMiddleware(BaseHTTPMiddleware):
    """Gate mutating requests to same-origin only.

    Localhost binding (run_dashboard host assertion) stops LAN attackers;
    this stops CSRF from other browser tabs. Mutating methods
    (POST/PUT/DELETE/PATCH) require Origin or Referer to match the
    dashboard's own origin. GET/HEAD/OPTIONS pass through.
    """

    SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

    def __init__(self, app: ASGIApp, *, port: int, extra_origins: tuple[str, ...] = ()) -> None:
        super().__init__(app)
        self._allowed = frozenset(
            (
                f"http://localhost:{port}",
                f"http://127.0.0.1:{port}",
                f"http://[::1]:{port}",
                *extra_origins,
            )
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method in self.SAFE_METHODS:
            return await call_next(request)
        origin = request.headers.get("origin")
        if origin is not None:
            if origin in self._allowed:
                return await call_next(request)
            return Response(status_code=403, content="cross-origin request blocked")
        referer = request.headers.get("referer")
        if referer is not None:
            for allowed in self._allowed:
                if referer.startswith(allowed + "/") or referer == allowed:
                    return await call_next(request)
            return Response(status_code=403, content="cross-origin request blocked")
        # No Origin and no Referer. pytest TestClient omits both — allow under
        # pytest ONLY when pytest is actually loaded in this interpreter. The
        # env-var alone is operator-mutable; requiring `pytest in sys.modules`
        # forces an attacker to also achieve code execution (at which point
        # CSRF protection is already moot), closing the env-var-only bypass.
        if os.environ.get("PYTEST_CURRENT_TEST") and "pytest" in sys.modules:
            return await call_next(request)
        return Response(status_code=403, content="missing Origin/Referer on mutating request")


app.add_middleware(OriginAllowlistMiddleware, port=PORT, extra_origins=_extra_origins_from_env())


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
            log.warning(
                "Failed to load profile %r for lane reconstruction — falling back to raw lanes",
                profile_id,
                exc_info=True,
            )

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
            _sort_time_key(str(item.get("session_time_brisbane") or "")),
            str(item.get("instrument") or ""),
            str(item.get("strategy_id") or ""),
        )
    )
    return lane_cards


# Track background processes. Guarded by _bg_lock to prevent race conditions
# (e.g., double-click spawning two concurrent DB writers — violates CLAUDE.md
# "NEVER run two write processes against the same DuckDB file simultaneously").
_bg_processes: dict[str, Any] = {}
_bg_lock = threading.Lock()

# _state_lock guards _preflight_cache + _handoff_state. Reentrant so helpers can
# nest, but lock hierarchy rule: NEVER hold _state_lock while acquiring _bg_lock
# (or vice versa). Cross-lock nesting would deadlock under threaded access.
_state_lock = threading.RLock()
_preflight_cache: dict[str, dict[str, object]] = {}
_handoff_state: dict[str, object] = {
    "active": False,
    "status": "idle",
    "target_profile": None,
    "target_mode": None,
    "requested_at": None,
    "message": "",
}


def _ensure_log_dir() -> Path:
    """Create logs/ directory if it doesn't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def _lock_dir() -> Path:
    return Path(tempfile.gettempdir()) / "canompx3"


def _close_bg_handle(key: str) -> None:
    handle = _bg_processes.pop(key, None)
    if handle and hasattr(handle, "close"):
        try:
            handle.close()
        except Exception:
            pass


def _prune_bg_processes() -> None:
    with _bg_lock:
        for name in ("session", "refresh"):
            proc = _bg_processes.get(name)
            if not isinstance(proc, subprocess.Popen) or proc.poll() is None:
                continue
            _bg_processes.pop(name, None)
            suffix = "session" if name == "session" else "refresh"
            _close_bg_handle(f"_{suffix}_logfile")
            _bg_processes.pop(f"_{suffix}_log", None)


def _heartbeat_age_s(state: dict[str, object]) -> float:
    age = _as_float(state.get("heartbeat_age_s"), 9999.0)
    if state and "heartbeat_age_s" not in state:
        hb = state.get("heartbeat_utc")
        if hb:
            try:
                age = (datetime.now(UTC) - datetime.fromisoformat(str(hb))).total_seconds()
            except (TypeError, ValueError):
                age = 9999
    return age


def _session_snapshot() -> dict[str, object]:
    _prune_bg_processes()
    state = read_state()
    raw_mode = str(state.get("mode") or "STOPPED").upper()
    heartbeat_age_s = _heartbeat_age_s(state)
    account_name = str(state.get("account_name") or "")
    profile = account_name.removeprefix("profile_") if account_name.startswith("profile_") else None
    with _bg_lock:
        tracked = _bg_processes.get("session")
        tracked_alive = isinstance(tracked, subprocess.Popen) and tracked.poll() is None
    running = tracked_alive or (raw_mode in {"SIGNAL", "DEMO", "LIVE"} and heartbeat_age_s < HEARTBEAT_STALE_AFTER_S)
    return {
        "running": running,
        "raw_mode": raw_mode,
        "heartbeat_age_s": heartbeat_age_s,
        "profile": profile,
        "tracked_alive": tracked_alive,
    }


def _refresh_snapshot() -> dict[str, object]:
    _prune_bg_processes()
    with _bg_lock:
        proc = _bg_processes.get("refresh")
        running = isinstance(proc, subprocess.Popen) and proc.poll() is None
    return {"running": running}


def _journal_lock_status() -> dict[str, object]:
    if not JOURNAL_PATH.exists():
        return {"locked": False, "detail": "journal absent"}
    try:
        con = duckdb.connect(str(JOURNAL_PATH), read_only=True)
        con.close()
        return {"locked": False, "detail": "journal available"}
    except duckdb.IOException as exc:
        return {"locked": True, "detail": str(exc)}


def _instance_lock_status() -> dict[str, object]:
    active: list[dict[str, object]] = []
    lock_dir = _lock_dir()
    if not lock_dir.exists():
        return {"locked": False, "locks": active}
    for lock_file in lock_dir.glob("bot_*.lock"):
        try:
            content = lock_file.read_text(encoding="utf-8").strip()
            pid = int(content) if content else None
        except (OSError, ValueError):
            pid = None
        if pid and is_pid_alive(pid):
            active.append({"path": str(lock_file), "pid": pid})
        elif pid is None:
            active.append({"path": str(lock_file), "pid": None})
    return {"locked": bool(active), "locks": active}


def _clear_handoff() -> None:
    with _state_lock:
        _handoff_state.update(
            {
                "active": False,
                "status": "idle",
                "target_profile": None,
                "target_mode": None,
                "requested_at": None,
                "message": "",
            }
        )


def _set_handoff(profile: str, mode: str, message: str) -> None:
    with _state_lock:
        _handoff_state.update(
            {
                "active": True,
                "status": "stopping",
                "target_profile": profile,
                "target_mode": mode,
                "requested_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "message": message,
            }
        )


def _handoff_snapshot(
    *,
    data_summary: dict[str, object] | None = None,
    preflight_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    # Read _handoff_state under _state_lock, but release it before calling the
    # snapshot helpers that acquire _bg_lock (hierarchy rule at module top).
    with _state_lock:
        if not bool(_handoff_state.get("active")):
            return {"active": False, "status": "idle", "reason": "", "action": None}
        target_profile = str(_handoff_state.get("target_profile") or "")
        target_mode = str(_handoff_state.get("target_mode") or "")

    session = _session_snapshot()
    refresh = _refresh_snapshot()
    journal = _journal_lock_status()
    instance_locks = _instance_lock_status()

    if session["running"]:
        status = "stopping"
        reason = f"Stopping current session before switching to {target_mode.upper()}."
        action = {"id": "stop_session", "label": "Stop Session"}
    elif journal["locked"] or instance_locks["locked"]:
        status = "waiting_cleanup"
        reason = "Waiting for runtime locks to clear before continuing."
        action = {"id": "wait_cleanup", "label": "Waiting For Cleanup"}
    elif refresh["running"]:
        status = "waiting_refresh"
        reason = "Data refresh is running. Wait for it to finish before continuing."
        action = {"id": "wait_refresh", "label": "Refresh Running"}
    else:
        summary = data_summary or _collect_data_status()
        if bool(summary.get("any_stale", True)):
            status = "needs_refresh"
            reason = "Data is stale. Refresh before continuing the handoff."
            action = {"id": "refresh_data", "label": "Refresh Data"}
        else:
            status = "ready_to_start"
            reason = f"Cleanup finished. Ready to start {target_mode.upper()}."
            action = {
                "id": "continue_handoff",
                "label": f"Start {target_mode.upper()}",
            }

    with _state_lock:
        _handoff_state["status"] = status
        _handoff_state["message"] = reason
        requested_at = _handoff_state.get("requested_at")
    return {
        "active": True,
        "status": status,
        "target_profile": target_profile,
        "target_mode": target_mode,
        "requested_at": requested_at,
        "reason": reason,
        "action": action,
    }


def _normalize_check_status(raw_status: str) -> str:
    status = raw_status.strip().upper()
    if status == "OK":
        return "pass"
    if status.startswith("WARN"):
        return "warn"
    if status == "SKIPPED":
        return "warn"
    if status == "FAILED":
        return "fail"
    return "info"


def _parse_preflight_output(output: str) -> dict[str, object]:
    """Parse run_live_session --preflight output into structured checks."""
    checks: list[dict[str, str]] = []
    passed = None
    total = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^\[(\d+)/(\d+)\]\s+(.+?)\.\.\.\s+(OK|FAILED|WARN(?:INGS?)?|SKIPPED)\s*(.*)$", line)
        if match:
            _idx, total_text, name, raw_status, tail = match.groups()
            total = int(total_text)
            detail = tail.strip()
            if detail.startswith(":"):
                detail = detail[1:].strip()
            checks.append(
                {
                    "name": name.strip(),
                    "status": _normalize_check_status(raw_status),
                    "detail": detail or raw_status.title(),
                }
            )
            continue

        summary_match = re.match(r"^Preflight:\s+(\d+)/(\d+)\s+passed$", line)
        if summary_match:
            passed = int(summary_match.group(1))
            total = int(summary_match.group(2))

    has_warn = any(check["status"] == "warn" for check in checks)
    has_fail = any(check["status"] == "fail" for check in checks)
    overall = "pass"
    if has_fail:
        overall = "fail"
    elif has_warn:
        overall = "warn"

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "overall": overall,
        "has_warnings": has_warn,
        "has_failures": has_fail,
    }


def _cache_preflight_entry(profile: str, cache_entry: dict[str, object]) -> None:
    with _state_lock:
        _preflight_cache[profile] = cache_entry


def _live_pilot_cli_args(profile: str) -> list[str]:
    """Return server-side live pilot routing args for the funded MNQ pilot."""
    if profile != LIVE_PILOT_PROFILE:
        return []
    return ["--instrument", LIVE_PILOT_INSTRUMENT, "--copies", str(LIVE_PILOT_COPIES)]


def _run_preflight_subprocess(profile: str, mode: str = "live") -> dict[str, object]:
    cmd = [sys.executable, "-m", "scripts.run_live_session", "--profile", profile, "--preflight"]
    # Match preflight mode to the requested session mode so signal-only's
    # auto-pass on the telemetry-maturity gate (run_live_session.py:369-378)
    # applies. Without this, Start Signal blocks at the very gate signal-only
    # is meant to clear.
    if mode == "signal":
        cmd.append("--signal-only")
    elif mode == "live":
        cmd.append("--live")
        cmd.extend(_live_pilot_cli_args(profile))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    output = result.stdout + result.stderr
    parsed = _parse_preflight_output(output)
    status = "pass" if result.returncode == 0 else "fail"
    if status == "pass" and bool(parsed.get("has_warnings")):
        status = "warn"
    cache_entry = {
        "status": status,
        "profile": profile,
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "returncode": result.returncode,
        **parsed,
        "output": output,
    }
    _cache_preflight_entry(profile, cache_entry)
    return cache_entry


def _run_control_refresh_subprocess(profile: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tools.refresh_control_state", "--profile", profile],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    output = result.stdout + result.stderr
    return {
        "status": "pass" if result.returncode == 0 else "fail",
        "profile": profile,
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "returncode": result.returncode,
        "output": output,
    }


def _combine_prepare_output(control: dict[str, object], preflight: dict[str, object] | None = None) -> str:
    parts: list[str] = []
    control_output = str(control.get("output") or "").strip()
    if control_output:
        parts.append("=== Control Refresh ===")
        parts.append(control_output)
    if preflight is not None:
        preflight_output = str(preflight.get("output") or "").strip()
        if preflight_output:
            parts.append("=== Preflight ===")
            parts.append(preflight_output)
    return "\n\n".join(parts).strip()


async def _prepare_profile_for_start(profile: str, mode: str = "live") -> dict[str, object]:
    import asyncio

    try:
        control = await asyncio.to_thread(_run_control_refresh_subprocess, profile)
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "profile": profile,
            "message": "Control-state refresh timed out.",
            "output": "Control-state refresh timed out after 120s.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "profile": profile,
            "message": f"Control-state refresh failed: {exc}",
            "output": str(exc),
        }

    if str(control.get("status")) != "pass":
        return {
            "status": "fail",
            "profile": profile,
            "message": "Control-state refresh failed.",
            "control": control,
            "output": _combine_prepare_output(control),
        }

    try:
        preflight = await asyncio.to_thread(_run_preflight_subprocess, profile, mode)
    except subprocess.TimeoutExpired:
        cache_entry = {
            "status": "timeout",
            "profile": profile,
            "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "checks": [],
            "passed": 0,
            "total": None,
            "overall": "fail",
            "output": "Preflight timed out after 60s",
        }
        _cache_preflight_entry(profile, cache_entry)
        return {
            "status": "timeout",
            "profile": profile,
            "message": "Automatic preflight timed out.",
            "control": control,
            "preflight": cache_entry,
            "output": _combine_prepare_output(control, cache_entry),
        }
    except Exception as exc:
        cache_entry = {
            "status": "error",
            "profile": profile,
            "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "checks": [],
            "passed": 0,
            "total": None,
            "overall": "fail",
            "output": str(exc),
        }
        _cache_preflight_entry(profile, cache_entry)
        return {
            "status": "error",
            "profile": profile,
            "message": f"Automatic preflight failed: {exc}",
            "control": control,
            "preflight": cache_entry,
            "output": _combine_prepare_output(control, cache_entry),
        }

    return {
        "status": str(preflight.get("status") or "error"),
        "profile": profile,
        "message": "Automatic readiness checks completed.",
        "control": control,
        "preflight": preflight,
        "output": _combine_prepare_output(control, preflight),
    }


def _profile_session_ambiguity(profile_id: str | None) -> dict[str, object]:
    if not profile_id:
        return {"status": "info", "detail": "No profile selected"}

    try:
        from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes

        profile = ACCOUNT_PROFILES.get(profile_id)
        if profile is None:
            return {"status": "warn", "detail": f"Unknown profile '{profile_id}'"}

        counts: dict[str, int] = {}
        for lane in effective_daily_lanes(profile):
            counts[lane.orb_label] = counts.get(lane.orb_label, 0) + 1
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            joined = ", ".join(duplicates)
            return {
                "status": "pass",
                "detail": (
                    "Shared-session lanes supported "
                    f"({joined}). Session hard gate will evaluate every lane in the selected session."
                ),
            }
        return {"status": "pass", "detail": "Session gates map cleanly to one lane per session"}
    except Exception as exc:
        return {"status": "warn", "detail": f"Could not evaluate session gate compatibility: {exc}"}


def _collect_data_status() -> dict[str, object]:
    if _refresh_snapshot()["running"]:
        return {
            "status": "busy",
            "instruments": {},
            "any_stale": True,
            "busy_reason": "Data refresh in progress",
        }
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        results: dict[str, dict[str, object]] = {}
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
        any_stale = any(info["stale"] for info in results.values())
        return {"status": "ok", "instruments": results, "any_stale": any_stale}
    except Exception as exc:
        return {"status": "error", "instruments": {}, "any_stale": True, "error": str(exc)}


def _collect_broker_status() -> dict[str, object]:
    try:
        from trading_app.live.broker_connections import connection_manager

        connections = connection_manager.list_connections()
        enabled = [conn for conn in connections if conn.get("enabled", True)]
        connected = [conn for conn in enabled if conn.get("status") == "connected"]
        errors = [conn for conn in enabled if conn.get("status") == "error"]
        return {
            "status": "ok",
            "connections": connections,
            "enabled_count": len(enabled),
            "connected_count": len(connected),
            "error_count": len(errors),
        }
    except Exception as exc:
        return {
            "status": "error",
            "connections": [],
            "enabled_count": 0,
            "connected_count": 0,
            "error_count": 1,
            "error": str(exc),
        }


def _connection_readiness(broker_summary: dict[str, object]) -> dict[str, object]:
    """Return operator-facing broker connection readiness.

    This is the single dashboard read model for the "am I connected?" question.
    It deliberately stays above broker-specific account/order details: those
    remain in /api/broker/list and /api/equity.
    """
    connections = _as_list(broker_summary.get("connections"))
    enabled_count = _as_int(broker_summary.get("enabled_count"))
    connected_count = _as_int(broker_summary.get("connected_count"))
    error_count = _as_int(broker_summary.get("error_count"))

    if broker_summary.get("status") == "error":
        message = str(broker_summary.get("error") or "Broker connection state is unavailable.")
        return {
            "status": "error",
            "message": message,
            "action": "open_connections",
            "connected_count": connected_count,
            "enabled_count": enabled_count,
        }

    if not connections:
        return {
            "status": "missing",
            "message": "No broker connection configured. Add a connection before starting.",
            "action": "open_connections",
            "connected_count": 0,
            "enabled_count": 0,
        }

    if enabled_count == 0:
        return {
            "status": "disabled",
            "message": "Broker connections exist, but all are disabled. Enable one before starting.",
            "action": "open_connections",
            "connected_count": 0,
            "enabled_count": 0,
        }

    if connected_count > 0:
        return {
            "status": "connected",
            "message": f"{connected_count}/{enabled_count} enabled broker connection(s) connected.",
            "action": "none",
            "connected_count": connected_count,
            "enabled_count": enabled_count,
        }

    if error_count > 0:
        errors = [
            str(conn_info.get("last_error") or "").strip()
            for conn in connections
            if (conn_info := _as_mapping(conn)).get("enabled", True) and conn_info.get("status") == "error"
        ]
        detail = next((msg for msg in errors if msg), "Enabled broker connection failed.")
        return {
            "status": "error",
            "message": detail,
            "action": "open_connections",
            "connected_count": 0,
            "enabled_count": enabled_count,
        }

    return {
        "status": "connecting",
        "message": "Broker connection is enabled but not connected yet.",
        "action": "wait",
        "connected_count": 0,
        "enabled_count": enabled_count,
    }


def _collect_alert_summary(limit: int = 25, profile: str | None = None, mode: str | None = None) -> dict[str, object]:
    try:
        alerts = read_operator_alerts(limit=limit, profile=profile, mode=mode)
        summary = summarize_operator_alerts(alerts)
        return {"status": "ok", "alerts": alerts, **summary}
    except Exception as exc:
        return {
            "status": "error",
            "alerts": [],
            "total": 0,
            "counts": {"critical": 0, "warning": 0, "info": 0},
            "recent_window_minutes": 30,
            "recent_counts": {"critical": 0, "warning": 0, "info": 0},
            "latest": None,
            "error": str(exc),
        }


def _choose_operator_profile(requested_profile: str | None, state: dict[str, object]) -> str | None:
    if requested_profile:
        return requested_profile

    account_name = str(state.get("account_name") or "")
    if account_name.startswith("profile_"):
        return account_name.removeprefix("profile_")

    try:
        from trading_app.prop_profiles import ACCOUNT_PROFILES, get_firm_spec

        for profile in ACCOUNT_PROFILES.values():
            if profile.active and (get_firm_spec(profile.firm).auto_trading in {"full", "semi"}):
                return profile.profile_id
    except Exception:
        pass
    return None


def _build_action_guard(
    *,
    data_summary: dict[str, object],
    preflight_summary: dict[str, object] | None,
) -> dict[str, object]:
    session = _session_snapshot()
    refresh = _refresh_snapshot()
    journal = _journal_lock_status()
    instance_locks = _instance_lock_status()
    handoff = _handoff_snapshot(data_summary=data_summary, preflight_summary=preflight_summary)

    blocked_action_ids: list[str] = []
    top_state = None
    reason = ""
    action = None

    if session["running"]:
        blocked_action_ids.extend(["start_signal", "start_demo", "start_live", "run_preflight", "refresh_data"])
    if refresh["running"]:
        blocked_action_ids.extend(["start_signal", "start_demo", "start_live", "run_preflight", "refresh_data"])
        top_state = "BLOCKED"
        reason = "Data refresh is in progress. Wait for it to finish."
        action = {"id": "wait_refresh", "label": "Refresh Running"}

    if handoff["active"]:
        top_state = "BLOCKED" if handoff["status"] != "ready_to_start" else "READY"
        reason = str(handoff["reason"])
        action = handoff["action"]
        blocked_action_ids.extend(["start_signal", "start_demo", "start_live"])
        if handoff["status"] in {"stopping", "waiting_cleanup", "waiting_refresh"}:
            blocked_action_ids.extend(["run_preflight", "refresh_data"])

    if not session["running"] and not refresh["running"] and not handoff["active"]:
        if data_summary.get("status") == "busy":
            top_state = "BLOCKED"
            reason = str(data_summary.get("busy_reason") or "Data refresh in progress.")
            action = {"id": "wait_refresh", "label": "Refresh Running"}
            blocked_action_ids.extend(["start_signal", "start_demo", "start_live", "run_preflight", "refresh_data"])
        elif bool(data_summary.get("any_stale", True)):
            blocked_action_ids.extend(["start_signal", "start_demo", "start_live"])

    return {
        "top_state": top_state,
        "reason": reason,
        "action": action,
        "blocked_action_ids": sorted({str(action_id) for action_id in blocked_action_ids}),
        "busy_reason": reason if top_state == "BLOCKED" else "",
        "resource_locks": {
            "journal": journal,
            "instance_lock": instance_locks,
            "refresh": refresh,
            "session": {
                "running": session["running"],
                "mode": session["raw_mode"],
                "profile": session["profile"],
            },
        },
        "handoff": handoff,
        "session": session,
        "refresh": refresh,
    }


def _derive_operator_state(
    *,
    raw_mode: str,
    heartbeat_age_s: float,
    broker_summary: dict[str, object],
    data_summary: dict[str, object],
    preflight_summary: dict[str, object] | None,
) -> tuple[str, str, dict[str, str]]:
    connected_count = _as_int(broker_summary.get("connected_count"))
    any_stale = bool(data_summary.get("any_stale", True))
    connection = _connection_readiness(broker_summary)

    if raw_mode in {"SIGNAL", "DEMO", "LIVE"}:
        if heartbeat_age_s >= HEARTBEAT_STALE_AFTER_S:
            return (
                "STALE",
                "Bot heartbeat is stale. Current runtime state cannot be trusted.",
                {"id": "stop_session", "label": "Stop Session"},
            )
        mapped = {
            "SIGNAL": "RUNNING_ALERTS",
            "DEMO": "RUNNING_PAPER",
            "LIVE": "RUNNING_LIVE",
        }[raw_mode]
        return (
            mapped,
            f"{mapped.replace('RUNNING_', '').replace('_', ' ').title()} session is active.",
            {"id": "stop_session", "label": "Stop Session"},
        )

    if connection["status"] in {"missing", "disabled"}:
        return (
            "BLOCKED",
            str(connection["message"]),
            {"id": "open_connections", "label": "Connect Broker"},
        )

    if connected_count == 0:
        return (
            "BLOCKED",
            str(connection["message"]),
            {"id": "open_connections", "label": "Fix Broker Connection"},
        )

    if any_stale:
        return (
            "BLOCKED",
            "Market data is stale. Refresh data before starting a session.",
            {"id": "refresh_data", "label": "Refresh Data"},
        )

    if preflight_summary is None:
        return (
            "READY",
            "System looks healthy. Start will auto-run readiness checks.",
            {"id": "start_signal", "label": "Start Alerts"},
        )

    preflight_status = str(preflight_summary.get("status") or "unknown")
    if preflight_status in {"fail", "error", "timeout"}:
        return (
            "DEGRADED",
            "Last readiness check failed. Start will rerun readiness checks automatically.",
            {"id": "start_signal", "label": "Start Alerts"},
        )

    if preflight_status == "warn":
        return (
            "DEGRADED",
            "Last readiness check had warnings. Start will rerun readiness checks automatically.",
            {"id": "start_signal", "label": "Start Alerts"},
        )

    return (
        "READY",
        "System is ready for a supervised session start.",
        {"id": "start_signal", "label": "Start Alerts"},
    )


def _build_operator_payload(profile: str | None = None) -> dict[str, object]:
    state = read_state()
    session_snapshot = _session_snapshot()
    raw_mode = str(session_snapshot.get("raw_mode") or "STOPPED")
    heartbeat_age_s = _as_float(session_snapshot.get("heartbeat_age_s"), 9999.0)

    operator_profile = _choose_operator_profile(profile, state)
    overlay_summary = None
    opportunity_summary = None
    if operator_profile:
        try:
            from trading_app.lifecycle_state import read_lifecycle_state

            lifecycle = read_lifecycle_state(operator_profile)
            overlay_summary = lifecycle.get("conditional_overlays")
            opportunity_summary = lifecycle.get("opportunity_awareness")
        except Exception as exc:
            overlay_summary = {"available": False, "error": str(exc)}
            opportunity_summary = {"available": False, "error": str(exc)}
    broker_summary = _collect_broker_status()
    connection_summary = _connection_readiness(broker_summary)
    data_summary = _collect_data_status()
    alert_summary = _collect_alert_summary(profile=operator_profile, mode=None if raw_mode == "STOPPED" else raw_mode)
    with _state_lock:
        preflight_summary = _preflight_cache.get(operator_profile or "")
    session_gate = _profile_session_ambiguity(operator_profile)
    top_state, reason, action = _derive_operator_state(
        raw_mode=raw_mode,
        heartbeat_age_s=heartbeat_age_s,
        broker_summary=broker_summary,
        data_summary=data_summary,
        preflight_summary=preflight_summary,
    )
    guard = _build_action_guard(
        data_summary=data_summary,
        preflight_summary=preflight_summary,
    )
    blocked_action_ids = _as_list(guard["blocked_action_ids"])
    if raw_mode == "STOPPED" and connection_summary["status"] != "connected":
        blocked_action_ids.extend(["start_signal", "start_demo", "start_live"])
    if guard["top_state"]:
        top_state = str(guard["top_state"])
        reason = str(guard["reason"])
        action = guard["action"]

    checks: list[dict[str, str]] = []

    connected_count = _as_int(broker_summary.get("connected_count"))
    broker_detail = str(connection_summary["message"])
    broker_status = "pass" if connected_count > 0 else "fail"
    checks.append({"name": "Broker", "status": broker_status, "detail": broker_detail})

    if data_summary.get("status") == "busy":
        checks.append(
            {
                "name": "Data",
                "status": "warn",
                "detail": str(data_summary.get("busy_reason") or "Data refresh in progress"),
            }
        )
    elif data_summary.get("status") == "error":
        checks.append(
            {
                "name": "Data",
                "status": "fail",
                "detail": str(data_summary.get("error") or "Data freshness check failed"),
            }
        )
    else:
        stale_instruments = [
            name
            for name, info in _as_mapping(data_summary.get("instruments")).items()
            if isinstance(info, dict) and info.get("stale")
        ]
        if stale_instruments:
            checks.append(
                {
                    "name": "Data",
                    "status": "fail",
                    "detail": f"Stale bars for {', '.join(stale_instruments)}",
                }
            )
        else:
            checks.append({"name": "Data", "status": "pass", "detail": "Bars are current for active instruments"})

    if preflight_summary is None:
        checks.append(
            {
                "name": "Preflight",
                "status": "info",
                "detail": "No cached preflight yet for this profile",
            }
        )
    else:
        preflight_detail = f"{preflight_summary.get('passed', '?')}/{preflight_summary.get('total', '?')} checks passed"
        ran_at = preflight_summary.get("ran_at")
        if ran_at:
            preflight_detail += f" · last run {ran_at}"
        checks.append(
            {
                "name": "Preflight",
                "status": str(preflight_summary.get("status") or "info"),
                "detail": preflight_detail,
            }
        )

    checks.append(
        {
            "name": "Runtime",
            "status": "warn" if top_state == "STALE" else "pass" if raw_mode != "STOPPED" else "info",
            "detail": f"Heartbeat {heartbeat_age_s:.0f}s old" if raw_mode != "STOPPED" else "No session running",
        }
    )
    checks.append(
        {
            "name": "Session Gates",
            "status": str(session_gate.get("status") or "info"),
            "detail": str(session_gate.get("detail") or ""),
        }
    )

    if overlay_summary and overlay_summary.get("available"):
        overlays = overlay_summary.get("overlays", [])
        invalid = [row for row in overlays if not row.get("valid") or row.get("status") == "invalid"]
        ready = [row for row in overlays if row.get("status") == "ready"]
        if invalid:
            detail = ", ".join(f"{row.get('overlay_id')}: {row.get('reason') or 'invalid'}" for row in invalid)
            checks.append({"name": "Conditional overlays", "status": "warn", "detail": detail})
        elif ready:
            detail = ", ".join(
                f"{row.get('overlay_id')} ready ({row.get('summary', {}).get('ready_count', 0)}/{row.get('summary', {}).get('row_count', 0)} rows)"
                for row in ready
            )
            checks.append({"name": "Conditional overlays", "status": "info", "detail": detail})
        else:
            detail = (
                ", ".join(f"{row.get('overlay_id')} {row.get('status')}" for row in overlays)
                or "No overlay rows available"
            )
            checks.append({"name": "Conditional overlays", "status": "info", "detail": detail})
    elif overlay_summary and overlay_summary.get("error"):
        checks.append(
            {
                "name": "Conditional overlays",
                "status": "warn",
                "detail": f"Overlay state unavailable: {overlay_summary['error']}",
            }
        )

    if opportunity_summary and opportunity_summary.get("available"):
        from trading_app.opportunity_awareness import describe_opportunity_awareness

        status, detail = describe_opportunity_awareness(opportunity_summary)
        checks.append({"name": "Opportunity awareness", "status": status, "detail": detail})
    elif opportunity_summary and opportunity_summary.get("error"):
        checks.append(
            {
                "name": "Opportunity awareness",
                "status": "warn",
                "detail": f"Opportunity state unavailable: {opportunity_summary['error']}",
            }
        )

    if alert_summary.get("status") == "error":
        checks.append(
            {
                "name": "Alerts",
                "status": "warn",
                "detail": str(alert_summary.get("error") or "Could not read runtime alerts"),
            }
        )
    else:
        recent_counts = _as_mapping(alert_summary.get("recent_counts"))
        latest = alert_summary.get("latest")
        latest_message = ""
        if isinstance(latest, dict):
            latest_message = str(latest.get("message") or "").strip()
        if raw_mode != "STOPPED" and _as_int(recent_counts.get("critical")) > 0:
            alert_status = "fail"
        elif raw_mode != "STOPPED" and _as_int(recent_counts.get("warning")) > 0:
            alert_status = "warn"
        elif _as_int(alert_summary.get("total")) > 0:
            alert_status = "info"
        else:
            alert_status = "pass"
        detail = f"Latest: {latest_message}" if latest_message else "No runtime alerts recorded yet"
        checks.append({"name": "Alerts", "status": alert_status, "detail": detail})

    # Read-only broker-edge health snapshot written by SessionOrchestrator
    # heartbeat. Reader is fail-closed: any failure surfaces broker_status="unknown".
    live_health = read_live_health()

    return {
        "profile": operator_profile,
        "raw_mode": raw_mode,
        "top_state": top_state,
        "reason": reason,
        "recommended_action": action,
        "checks": checks,
        "heartbeat_age_s": heartbeat_age_s,
        "blocked_action_ids": sorted({str(action_id) for action_id in blocked_action_ids}),
        "busy_reason": guard["busy_reason"],
        "resource_locks": guard["resource_locks"],
        "handoff": guard["handoff"],
        "preflight": preflight_summary,
        "broker_summary": broker_summary,
        "connection_readiness": connection_summary,
        "data_summary": data_summary,
        "alert_summary": alert_summary,
        "conditional_overlays": overlay_summary,
        "opportunity_awareness": opportunity_summary,
        "live_health": live_health,
    }


def _start_broker_connect_background(connection_manager: Any) -> None:
    def _worker() -> None:
        try:
            connection_manager.connect_all_enabled()
        except Exception as exc:
            log.warning("Startup: broker auto-connect failed: %s", exc)

    threading.Thread(target=_worker, name="dashboard-broker-connect", daemon=True).start()


# ── Signal-history helpers (Stage 1 of cockpit-v3) ────────────────────────────
# SIGNALS_DIR pins coupling to session_orchestrator.py:296 — both must derive
# the same path or the dashboard reads from a different log than the orchestrator
# writes. Project root = Path(__file__).parent.parent.parent.
SIGNALS_DIR: Path = PROJECT_ROOT


def _read_recent_signals(limit: int = 50, since_ts: str | None = None) -> list[dict]:
    """Read recent live-signal records from today's partition file.

    Returns at most ``limit`` records, newest first. ``since_ts`` (ISO-8601 UTC)
    filters to records strictly after that timestamp. Tolerates missing file
    by returning []. Malformed lines are skipped with WARNING — institutional
    -rigor.md § 6 forbids silent failures.
    """
    import json

    from trading_app.live.signal_log_rotator import signals_file_for_day

    today = datetime.now(UTC).date()
    path = signals_file_for_day(SIGNALS_DIR, today)
    if not path.exists():
        return []

    since_dt: datetime | None = None
    if since_ts:
        try:
            since_dt = datetime.fromisoformat(since_ts.replace("Z", "+00:00"))
        except ValueError:
            log.warning("_read_recent_signals: bad since_ts %r — ignoring filter", since_ts)
            since_dt = None

    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line_num, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning(
                        "_read_recent_signals: skipped malformed line %d in %s: %s",
                        line_num,
                        path.name,
                        exc,
                    )
                    continue
                if since_dt is not None:
                    rec_ts_raw = rec.get("ts")
                    if isinstance(rec_ts_raw, str):
                        try:
                            rec_ts = datetime.fromisoformat(rec_ts_raw.replace("Z", "+00:00"))
                            if rec_ts <= since_dt:
                                continue
                        except ValueError:
                            pass
                records.append(rec)
    except OSError as exc:
        log.error("_read_recent_signals: could not read %s: %s", path, exc)
        return []

    records.reverse()
    return records[:limit]


# ── API Endpoints ─────────────────────────────────────────────────────────────


@app.get("/api/signals-recent")
async def api_signals_recent(limit: int = 50, since: str | None = None):
    """Return recent live-signal records from today's partition file.

    Query params:
      limit (int, default 50, max 500) — newest first
      since (ISO-8601 UTC, optional) — only records strictly after this ts

    Returns {"signals": [...], "server_ts": "...", "trading_day": "YYYY-MM-DD"}.
    Tolerates missing file by returning empty list. Stage 2 builds SSE push
    on top of this same data source.
    """
    capped_limit = max(1, min(int(limit), 500))
    signals = _read_recent_signals(limit=capped_limit, since_ts=since)
    return {
        "signals": signals,
        "server_ts": datetime.now(UTC).isoformat(),
        "trading_day": datetime.now(UTC).date().isoformat(),
    }


@app.get("/api/planned-launch")
async def api_planned_launch():
    """Return the canonical planned-launch record (or status=unknown).

    Source of truth: ``data/bot_planned_launch.json`` written by
    ``trading_app.live.planned_launch`` from START_BOT.bat or the CLI codepath
    in ``scripts/run_live_session.py``. Pre-start, the dashboard renders this
    payload above the START CTA so the operator can see — unambiguously —
    which mode (SIGNAL/DEMO/LIVE), which profile, and how many broker accounts
    are about to receive orders.

    Fail-visible: missing/stale/malformed file returns ``status=unknown`` or
    ``status=stale`` rather than guessing. The UI must NOT show a green CTA
    in those cases.
    """
    from trading_app.live.planned_launch import read_planned_launch

    return read_planned_launch()


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


@app.get("/api/operator-state")
async def api_operator_state(profile: str | None = None):
    """Operator-grade summary for the dashboard shell."""
    return _build_operator_payload(profile)


@app.get("/api/trades")
async def api_trades():
    """Today's trades from live_journal.db.

    When a live session process holds the journal file exclusively (Windows
    DuckDB file-lock behaviour), return a graceful locked=True note instead
    of a raw IOException string. The UI renders a friendly "session writing
    — refresh after close" message in that case.
    """
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
    except duckdb.IOException as e:
        # Live session process holds live_journal.db exclusively (Windows
        # file-lock behaviour). Return a structured note the UI can render
        # as a friendly "session active" message rather than a raw error.
        return {
            "trades": [],
            "locked": True,
            "note": "Live session is writing the journal — blotter refreshes after session close.",
            "error_type": "file_locked",
            "error_detail": str(e),
        }
    except Exception as e:
        return {"trades": [], "error": str(e)}


@app.get("/api/trade-book")
async def api_trade_book():
    """All-time trade book — real broker fills + paper-trade simulations, separated.

    Live blotter (/api/trades) only shows the last 24h. This endpoint returns
    the full history so the operator can see lifetime fills without leaving
    the dashboard. Two arrays so the UI can render them as distinct tables —
    real money trades must never be visually conflated with paper-trade
    research output.

    Read-only on both DBs. Empty arrays if a DB is locked or absent.
    """
    live_trades: list[dict] = []
    live_note: str | None = None
    paper_trades: list[dict] = []
    paper_note: str | None = None
    paper_truncated: bool = False
    paper_total_count: int = 0
    # Rationale: Browser DOM rendering and JSON payload size grow linearly
    # with row count; at the current 580-row baseline (2026-05-07 smoke
    # check) latency is unobservable, but at 10k+ rows the operator-facing
    # Trade Book panel becomes laggy and the JSON response approaches MB
    # scale. 5000 is a comfortable headroom over today's count while
    # keeping the page snappy. Truncation is signalled to the UI via
    # paper_truncated + paper_total_count for an explicit "showing first
    # N of M" hint instead of silent row loss.
    PAPER_TRADES_LIMIT = 5000

    if JOURNAL_PATH.exists():
        try:
            with duckdb.connect(str(JOURNAL_PATH), read_only=True) as con:
                configure_connection(con)
                rows = con.execute(
                    """
                    SELECT trade_id, trading_day, instrument, strategy_id, direction,
                           entry_model, fill_entry, fill_exit, actual_r, expected_r,
                           pnl_dollars, exit_reason, contracts, session_mode, broker,
                           created_at, exited_at
                    FROM live_trades
                    ORDER BY created_at DESC
                    """
                ).fetchall()
                cols = [
                    "trade_id",
                    "trading_day",
                    "instrument",
                    "strategy_id",
                    "direction",
                    "entry_model",
                    "fill_entry",
                    "fill_exit",
                    "actual_r",
                    "expected_r",
                    "pnl_dollars",
                    "exit_reason",
                    "contracts",
                    "session_mode",
                    "broker",
                    "created_at",
                    "exited_at",
                ]
                live_trades = [dict(zip(cols, r, strict=False)) for r in rows]
        except duckdb.IOException as e:
            live_note = f"live_journal.db locked by active session: {e}"
        except Exception as e:
            live_note = f"live_journal.db read failed: {e}"
    else:
        live_note = "No live_journal.db found"

    if GOLD_DB_PATH.exists():
        try:
            with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
                configure_connection(con)
                # LIMIT N+1 lets us detect truncation in one query (rows
                # returned > N → truncated). Only when truncated do we run
                # the second COUNT(*) for the exact total. Untruncated path
                # stays single-query.
                rows = con.execute(
                    f"""
                    SELECT trading_day, instrument, strategy_id, lane_name, direction,
                           entry_model, orb_label, orb_minutes, rr_target, filter_type,
                           entry_time, exit_time, entry_price, exit_price,
                           pnl_r, pnl_dollar, slippage_ticks, exit_reason,
                           execution_source
                    FROM paper_trades
                    ORDER BY trading_day DESC, entry_time DESC
                    LIMIT {PAPER_TRADES_LIMIT + 1}
                    """
                ).fetchall()
                if len(rows) > PAPER_TRADES_LIMIT:
                    paper_truncated = True
                    rows = rows[:PAPER_TRADES_LIMIT]
                    count_row = con.execute("SELECT COUNT(*) FROM paper_trades").fetchone()
                    paper_total_count = int(count_row[0]) if count_row else len(rows)
                else:
                    paper_total_count = len(rows)
                cols = [
                    "trading_day",
                    "instrument",
                    "strategy_id",
                    "lane_name",
                    "direction",
                    "entry_model",
                    "orb_label",
                    "orb_minutes",
                    "rr_target",
                    "filter_type",
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "pnl_r",
                    "pnl_dollar",
                    "slippage_ticks",
                    "exit_reason",
                    "execution_source",
                ]
                paper_trades = [dict(zip(cols, r, strict=False)) for r in rows]
        except duckdb.IOException as e:
            paper_note = f"gold.db locked: {e}"
        except Exception as e:
            paper_note = f"gold.db read failed: {e}"
    else:
        paper_note = "No gold.db found"

    return {
        "live_trades": live_trades,
        "live_note": live_note,
        "paper_trades": paper_trades,
        "paper_note": paper_note,
        "paper_truncated": paper_truncated,
        "paper_total_count": paper_total_count,
        "counts": {
            "live": len(live_trades),
            "paper": len(paper_trades),
        },
    }


@app.get("/api/lane-status")
async def api_lane_status(profile: str = "topstep_50k_mnq_auto"):
    """Lane pause status from the canonical lane_ctl accessor.

    Returns the strategy_ids currently paused via lane_overrides JSON, with
    the human-readable reason and expiry. The dashboard uses this to render
    a "Paused — SR alarm" badge on lane cards so the operator understands
    why a deployed lane is silent.

    Reads through trading_app.lane_ctl (canonical accessor) — never touches
    the override JSON directly.
    """
    try:
        from trading_app.lane_ctl import get_lane_override, get_paused_strategy_ids

        paused_ids = sorted(get_paused_strategy_ids(profile))
        items = []
        for sid in paused_ids:
            override = get_lane_override(profile, sid)
            if override is None:
                items.append(
                    {
                        "strategy_id": sid,
                        "paused": True,
                        "reason": None,
                        "expires_on": None,
                        "warning": "paused per get_paused_strategy_ids but get_lane_override returned None",
                    }
                )
                continue
            items.append(
                {
                    "strategy_id": sid,
                    "paused": True,
                    "reason": override.get("reason") or None,
                    "expires_on": override.get("expires") or None,
                    "paused_at": override.get("paused_at") or None,
                }
            )
        return {
            "profile": profile,
            "paused_count": len(items),
            "paused": items,
        }
    except Exception as e:
        log.warning("api/lane-status failed for profile=%s: %s", profile, e)
        return {"profile": profile, "paused_count": 0, "paused": [], "error": str(e)}


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
        with _state_lock:
            if bool(_handoff_state.get("active")):
                _handoff_state["status"] = "stopping"
                _handoff_state["message"] = "Waiting for the current session to stop."
        return {"status": "ok", "message": "Stop sent — waiting for session shutdown"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/action/preflight")
async def action_preflight(profile: str | None = None):
    """Run preflight checks and return output."""
    try:
        profile = profile or _resolve_profile()
        session = _session_snapshot()
        refresh = _refresh_snapshot()
        if refresh["running"]:
            return {
                "status": "blocked",
                "profile": profile,
                "output": "Data refresh is running — wait for it to finish before preflight.",
            }
        if session["running"]:
            return {
                "status": "blocked",
                "profile": profile,
                "output": "A session is already running — stop it before preflight.",
            }
        prepared = await _prepare_profile_for_start(profile)
        preflight = prepared.get("preflight")
        if not isinstance(preflight, dict):
            return {
                "status": prepared.get("status", "error"),
                "profile": profile,
                "output": prepared.get("output", prepared.get("message", "No output")),
            }
        return {
            "status": preflight.get("status"),
            "output": prepared.get("output", preflight.get("output")),
            "returncode": preflight.get("returncode"),
            "profile": profile,
            "checks": preflight.get("checks", []),
            "passed": preflight.get("passed"),
            "total": preflight.get("total"),
            "overall": preflight.get("overall"),
        }
    except subprocess.TimeoutExpired:
        cache_entry = {
            "status": "timeout",
            "profile": profile or "",
            "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "checks": [],
            "passed": 0,
            "total": None,
            "overall": "fail",
            "output": "Preflight timed out after 60s",
        }
        if profile:
            with _state_lock:
                _preflight_cache[profile] = cache_entry
        return cache_entry
    except Exception as e:
        cache_entry = {
            "status": "error",
            "profile": profile or "",
            "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "checks": [],
            "passed": 0,
            "total": None,
            "overall": "fail",
            "output": str(e),
        }
        if profile:
            with _state_lock:
                _preflight_cache[profile] = cache_entry
        return cache_entry


@app.get("/api/accounts")
async def api_accounts():
    """All trading profiles with human-readable names, firm info, and lane summaries.

    Per-profile build is wrapped in try/except so one misconfigured profile
    (e.g. an account_size that has no matching tier in ACCOUNT_TIERS) does not
    nuke the entire endpoint. Broken profiles surface in the 'skipped' array
    of the response so the UI / operator can see what was excluded and why.
    """
    from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes, get_account_tier, get_firm_spec

    accounts = []
    skipped = []
    try:
        profile_items = list(ACCOUNT_PROFILES.items())
    except Exception as e:
        return {"accounts": [], "error": f"ACCOUNT_PROFILES unavailable: {e}"}

    for pid, p in profile_items:
        try:
            tier = get_account_tier(p.firm, p.account_size)
            firm = get_firm_spec(p.firm)
            p_lanes = effective_daily_lanes(p)
            lanes_summary = []
            for lane in p_lanes:
                meta = _strategy_meta(lane.strategy_id)
                session_time = _get_session_time_brisbane(lane.orb_label)
            for lane in p_lanes:
                meta = _strategy_meta(lane.strategy_id)
                session_time = _get_session_time_brisbane(lane.orb_label)
                rr_target = meta.get("rr_target")
                rr_label = f"RR{rr_target:g}" if isinstance(rr_target, float) else None
                setup_parts = [
                    str(part) for part in (meta.get("entry_model"), rr_label, meta.get("filter_type")) if part
                ]
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
        except Exception as e:
            log.warning("api_accounts: skipping profile %r — %s", pid, e)
            skipped.append(
                {
                    "profile_id": pid,
                    "firm": getattr(p, "firm", None),
                    "account_size": getattr(p, "account_size", None),
                    "error": str(e),
                }
            )

    return {"accounts": accounts, "skipped": skipped}


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

    Uses BrokerHTTPClient with READ_POLICY (bounded retry, deadline) so a
    transient read-timeout or TCP RST is recovered instead of surfacing as
    `Balance —` to the operator (2026-05-18 resilience baseline).
    """
    from trading_app.live.http_client import READ_POLICY, BrokerHTTPClient

    if broker_type == "projectx":
        from trading_app.live.projectx.auth import BASE_URL

        client = BrokerHTTPClient(
            base_url=BASE_URL,
            refresh_token=auth.refresh_if_needed,  # type: ignore[union-attr]
            name=f"dashboard-{broker_type}",
        )
        data = client.post_json(
            "/api/Account/search",
            headers=auth.headers(),  # type: ignore[union-attr]
            body={"onlyActiveAccounts": False},
            policy=READ_POLICY,
            timeout=15,
        )

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

            accounts.append(
                {
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
                }
            )

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
                brokers.append(
                    {
                        "name": broker_type,
                        "display": display_name,
                        "connection_id": conn_id,
                        "error": str(e),
                        "accounts": [],
                    }
                )
                continue

        if auth is None:
            brokers.append(
                {
                    "name": broker_type,
                    "display": display_name,
                    "connection_id": conn_id,
                    "error": "Auth not available",
                    "accounts": [],
                }
            )
            continue

        try:
            fetch_result = await asyncio.to_thread(_fetch_accounts_for_connection, conn_id, broker_type, auth)
            connection_manager.update_account_count(conn_id, fetch_result["count"])
            brokers.append(
                {
                    "name": broker_type,
                    "display": display_name,
                    "connection_id": conn_id,
                    "error": None,
                    "accounts": fetch_result["accounts"],
                }
            )
        except Exception as e:
            log.warning("Equity fetch failed for %s: %s", display_name, e)
            brokers.append(
                {
                    "name": broker_type,
                    "display": display_name,
                    "connection_id": conn_id,
                    "error": str(e),
                    "accounts": [],
                }
            )

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
    """Server-side DST-correct session schedule with next-session computation.

    Market-state truth comes from pipeline.market_calendar (the same source the
    orchestrator uses), NOT the Brisbane weekday: the machine is "Saturday"
    locally during the entire Friday US session when the CME market is open, so
    a Brisbane-date check would mislabel an open market as closed. ``market_open``
    is surfaced so the dashboard can render closed-on-weekend / holiday correctly.
    """
    try:
        from pipeline.dst import SESSION_CATALOG
        from pipeline.market_calendar import is_market_open_at

        now_bris = datetime.now(ZoneInfo("Australia/Brisbane"))
        today = now_bris.date()  # Use Brisbane date — not system local (matters at NYSE_OPEN midnight crossing)
        market_open = is_market_open_at(datetime.now(ZoneInfo("UTC")))
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
        return {"sessions": sessions, "next": next_session, "market_open": market_open}
    except Exception as e:
        return {"sessions": [], "next": None, "market_open": None, "error": str(e)}


@app.get("/api/alerts")
async def api_alerts(limit: int = 25, profile: str | None = None, mode: str | None = None):
    """Recent runtime alerts for operator-facing monitoring."""
    limit = max(1, min(limit, 100))
    alerts = read_operator_alerts(limit=limit, profile=profile, mode=mode)
    return {"alerts": alerts, "summary": summarize_operator_alerts(alerts)}


@app.get("/api/data-status")
async def api_data_status():
    """Data freshness for all active instruments."""
    data = _collect_data_status()
    if data.get("status") == "error":
        return {"instruments": {}, "any_stale": True, "error": data.get("error")}
    return {"instruments": data.get("instruments", {}), "any_stale": data.get("any_stale", True)}


@app.post("/api/action/refresh")
async def action_refresh():
    """Download fresh data from Databento + rebuild pipeline. Runs in background.

    Output goes to logs/refresh.log (not a pipe — prevents deadlock on Windows
    where the 64KB pipe buffer fills and blocks the child process forever).
    """
    session = _session_snapshot()
    if session["running"]:
        return {
            "status": "blocked",
            "message": "A session is active and owns runtime state. Stop it before refreshing data.",
        }
    handoff = _handoff_snapshot()
    if handoff["active"] and handoff["status"] in {"stopping", "waiting_cleanup", "waiting_refresh"}:
        return {
            "status": "blocked",
            "message": str(handoff["reason"]),
        }

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

    profile = profile or _resolve_profile()
    session = _session_snapshot()
    refresh = _refresh_snapshot()
    with _state_lock:
        preflight_summary = _preflight_cache.get(profile or "")
    handoff = _handoff_snapshot(preflight_summary=preflight_summary)
    data_summary = _collect_data_status()

    if refresh["running"]:
        return {"status": "blocked", "message": "Data refresh is running. Wait for it to finish."}

    if session["running"]:
        same_mode = str(session["raw_mode"]).lower() == mode
        same_profile = session["profile"] == profile
        if same_mode and same_profile:
            return {"status": "running", "message": "Requested session is already running"}
        _set_handoff(profile, mode, f"Stopping current session before switching to {mode.upper()}.")
        await action_kill()
        return {
            "status": "handoff_started",
            "message": f"Stopping current session before switching to {mode.upper()}.",
            "handoff": _handoff_snapshot(),
        }

    if handoff["active"]:
        target_profile = str(handoff.get("target_profile") or "")
        target_mode = str(handoff.get("target_mode") or "")
        if target_profile and target_mode and (profile != target_profile or mode != target_mode):
            return {
                "status": "blocked",
                "message": (
                    f"Handoff in progress to {target_mode.upper()} for {target_profile}. Finish or cancel that first."
                ),
            }
        if handoff["status"] != "ready_to_start":
            return {
                "status": "handoff_wait",
                "message": str(handoff["reason"]),
                "handoff": handoff,
            }

    if bool(data_summary.get("any_stale", True)):
        return {"status": "blocked", "message": "Market data is stale. Refresh data before starting."}

    broker_summary = _collect_broker_status()
    connection = _connection_readiness(broker_summary)
    if connection["status"] != "connected":
        return {
            "status": "blocked",
            "message": str(connection["message"]),
            "connection_readiness": connection,
        }

    prepared = await _prepare_profile_for_start(profile, mode)
    prep_status = str(prepared.get("status") or "error")
    if prep_status in {"fail", "error", "timeout"}:
        return {
            "status": "blocked",
            "message": str(prepared.get("message") or "Automatic readiness checks failed."),
            "profile": profile,
            "output": prepared.get("output", ""),
        }

    with _bg_lock:
        if "session" in _bg_processes:
            proc = _bg_processes["session"]
            if proc.poll() is None:
                return {"status": "running", "message": "Session already running"}

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
            cmd.extend(_live_pilot_cli_args(profile))
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
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "CANOMPX3_DASHBOARD_ORIGIN": "1"},
            )
            _bg_processes["session"] = proc
            _bg_processes["_session_logfile"] = log_file  # type: ignore[assignment]
            _clear_handoff()
            return {
                "status": "started",
                "message": f"{mode_label} session started: {profile}",
                "output": prepared.get("output", ""),
                "readiness_status": prep_status,
                "pid": proc.pid,
                "profile": profile,
                "mode": mode,
            }
        except Exception as e:
            if log_file is not None:
                log_file.close()
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ── Stage 2 of cockpit-v3: SSE infrastructure ─────────────────────────────────
# Dashboard runs as subprocess (line ~2270 launch_dashboard_background) —
# JSONL files + bot_state.json are the IPC bus. Stage 2 adds NO orchestrator
# code. Single-uvicorn-worker is asserted in run_dashboard().

import asyncio
import json as _json
from collections import deque

_SSE_QUEUE_MAXSIZE: int = 256
_SSE_MAX_SUBSCRIBERS: int = 4
_SSE_RING_SIZE: int = 100
_SSE_HEARTBEAT_INTERVAL_S: float = 1.0
_SSE_STATE_POLL_INTERVAL_S: float = 0.5


class _SSEBroker:
    """In-process pub/sub for dashboard SSE events.

    Ring buffer keeps last ``_SSE_RING_SIZE`` non-heartbeat events for
    Last-Event-ID replay. Slow-consumer queues are dropped (logged WARNING)
    rather than blocking publish — institutional-rigor.md § 6.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._ring: deque[dict[str, Any]] = deque(maxlen=_SSE_RING_SIZE)
        self._next_event_id: int = 0

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_SSE_QUEUE_MAXSIZE)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        envelope: dict[str, Any] = {"event": event_type, "data": data}
        if event_type != "heartbeat":
            self._next_event_id += 1
            envelope["id"] = self._next_event_id
            self._ring.append(envelope)
        for q in list(self._subscribers):
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                log.warning("SSE queue full — dropping %s event", event_type)

    def replay_since(self, last_event_id: int) -> list[dict[str, Any]]:
        return [e for e in self._ring if e.get("id", 0) > last_event_id]


_sse_broker = _SSEBroker()
_sse_tasks: list[asyncio.Task[None]] = []


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0
    except OSError as exc:
        log.warning("_file_mtime: stat failed on %s: %s", path, exc)
        return 0.0


async def _heartbeat_watcher() -> None:
    """Emit heartbeat every _SSE_HEARTBEAT_INTERVAL_S with feed-liveness."""
    from trading_app.live.bot_state import read_state

    while True:
        try:
            state = read_state() or {}
            hb_raw = state.get("heartbeat_utc")
            last_tick_age_s: float | None = None
            bot_alive = False
            if isinstance(hb_raw, str):
                try:
                    hb_dt = datetime.fromisoformat(hb_raw.replace("Z", "+00:00"))
                    last_tick_age_s = (datetime.now(UTC) - hb_dt).total_seconds()
                    bot_alive = last_tick_age_s < HEARTBEAT_STALE_AFTER_S
                except ValueError:
                    pass
            _sse_broker.publish(
                "heartbeat",
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "bot_alive": bot_alive,
                    "session_state": state.get("mode"),
                    "last_tick_age_s": last_tick_age_s,
                },
            )
        except Exception as exc:
            log.warning("_heartbeat_watcher tick failed: %s", exc)
        await asyncio.sleep(_SSE_HEARTBEAT_INTERVAL_S)


async def _state_watcher() -> None:
    """Publish state events on bot_state.json mtime change or presence flip."""
    from trading_app.live.bot_state import STATE_FILE, read_state

    last_mtime: float = 0.0
    last_present: bool | None = None
    while True:
        try:
            cur_mtime = _file_mtime(STATE_FILE)
            cur_present = STATE_FILE.exists()
            if cur_mtime != last_mtime or cur_present != last_present:
                if cur_present:
                    _sse_broker.publish("state", {"present": True, "state": read_state() or {}})
                else:
                    _sse_broker.publish("state", {"present": False})
                last_mtime = cur_mtime
                last_present = cur_present
        except Exception as exc:
            log.warning("_state_watcher tick failed: %s", exc)
        await asyncio.sleep(_SSE_STATE_POLL_INTERVAL_S)


async def _signals_watcher() -> None:
    """Tail today's live_signals JSONL; publish signal event per new line."""
    from trading_app.live.signal_log_rotator import signals_file_for_day

    cur_day: date | None = None
    offset: int = 0
    while True:
        try:
            today = datetime.now(UTC).date()
            if today != cur_day:
                cur_day = today
                # Start at EOF so dashboard startup doesn't re-emit history.
                # Stage 1's /api/signals-recent serves backfill on connect.
                p0 = signals_file_for_day(SIGNALS_DIR, today)
                offset = p0.stat().st_size if p0.exists() else 0

            path = signals_file_for_day(SIGNALS_DIR, today)
            if path.exists():
                size = path.stat().st_size
                if size < offset:
                    log.warning("_signals_watcher: %s shrank — resetting offset", path.name)
                    offset = 0
                if size > offset:
                    with open(path, "rb") as fh:
                        fh.seek(offset)
                        chunk = fh.read(size - offset)
                    text = chunk.decode("utf-8", errors="replace")
                    last_nl = text.rfind("\n")
                    if last_nl == -1:
                        await asyncio.sleep(_SSE_STATE_POLL_INTERVAL_S)
                        continue
                    complete = text[: last_nl + 1]
                    offset += len(complete.encode("utf-8"))
                    for line in complete.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = _json.loads(line)
                        except _json.JSONDecodeError as exc:
                            log.warning("_signals_watcher: malformed line: %s", exc)
                            continue
                        _sse_broker.publish("signal", rec)
        except Exception as exc:
            log.warning("_signals_watcher tick failed: %s", exc)
        await asyncio.sleep(_SSE_STATE_POLL_INTERVAL_S)


async def _alerts_watcher() -> None:
    """Tail operator_alerts.jsonl; publish alert event per new line.

    Canonical path: data/runtime/operator_alerts.jsonl
    (alert_engine.ALERTS_PATH:21). Same byte-offset pattern as the signals
    watcher — single file, no day-roll. Missing file at startup tolerated.
    """
    alerts_path = PROJECT_ROOT / "data" / "runtime" / "operator_alerts.jsonl"
    # Start at EOF so we don't re-emit historical alerts on dashboard boot.
    offset: int = alerts_path.stat().st_size if alerts_path.exists() else 0
    while True:
        try:
            if alerts_path.exists():
                size = alerts_path.stat().st_size
                if size < offset:
                    log.warning("_alerts_watcher: %s shrank — resetting offset", alerts_path.name)
                    offset = 0
                if size > offset:
                    with open(alerts_path, "rb") as fh:
                        fh.seek(offset)
                        chunk = fh.read(size - offset)
                    text = chunk.decode("utf-8", errors="replace")
                    last_nl = text.rfind("\n")
                    if last_nl == -1:
                        await asyncio.sleep(_SSE_STATE_POLL_INTERVAL_S)
                        continue
                    complete = text[: last_nl + 1]
                    offset += len(complete.encode("utf-8"))
                    for line in complete.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = _json.loads(line)
                        except _json.JSONDecodeError as exc:
                            log.warning("_alerts_watcher: malformed line: %s", exc)
                            continue
                        _sse_broker.publish("alert", rec)
        except Exception as exc:
            log.warning("_alerts_watcher tick failed: %s", exc)
        await asyncio.sleep(_SSE_STATE_POLL_INTERVAL_S)


async def _bars_watcher() -> None:
    """Poll the live-bars ring (and gold.db on bootstrap) and publish bar SSE events.

    During an active session BarPersister writes ``data/live_bars/<SYMBOL>.json``
    atomically on every closed bar. This watcher reads that ring file — no
    DuckDB write-lock contention possible. When the ring is empty (no session
    or post-shutdown) we fall back to gold.db so a stopped-bot dashboard still
    shows the latest historical chart.

    Stale-gating: if BOTH the ring and bot_state.heartbeat are stale we skip
    SSE pushes — prevents yesterday's ring (after a crash) showing as "live".

    Tracks last-seen ts_utc per instrument; only emits strictly-newer rows.
    Instrument set is taken from bot_state.json active lanes; defaults to MNQ.
    """
    from trading_app.live import bar_ring
    from trading_app.live.bot_state import read_state

    last_seen: dict[str, datetime] = {}
    # Per-instrument "is the chart currently fed by the live ring?" tracker.
    # On transition ring_is_live -> heartbeat_stale we publish a
    # bars_source_changed SSE event so the client repaints from gold.db via
    # /api/bars-recent without requiring a browser reload. Initial state is
    # None (unknown) — only emit on actual True -> False transition so
    # cold-start dashboards don't get a spurious fallback event.
    ring_source_live: dict[str, bool | None] = {}
    while True:
        try:
            state = read_state() or {}
            lanes = state.get("lanes") or {}
            instruments: set[str] = set()
            if isinstance(lanes, dict):
                for lane in lanes.values():
                    if isinstance(lane, dict):
                        inst = lane.get("instrument")
                        if isinstance(inst, str):
                            instruments.add(inst)
            if not instruments:
                instruments = {"MNQ"}

            # Cross-stale gate: if both the bot_state heartbeat AND the ring
            # are stale, don't push anything — the bot is dead.
            heartbeat_raw = state.get("heartbeat_utc")
            heartbeat_stale = True
            if isinstance(heartbeat_raw, str):
                try:
                    hb_ts = datetime.fromisoformat(heartbeat_raw)
                    if hb_ts.tzinfo is None:
                        hb_ts = hb_ts.replace(tzinfo=UTC)
                    heartbeat_stale = (datetime.now(UTC) - hb_ts).total_seconds() > HEARTBEAT_STALE_AFTER_S
                except ValueError:
                    pass

            for inst in instruments:
                snap = bar_ring.read_bar_ring(inst)
                # Crash-detection gate: a stale heartbeat means the bot is
                # dead; do NOT push ring bars to SSE even if the ring file
                # is still within its 90s freshness window. The post-crash
                # ring is a historical artifact, not live data. Falls
                # through to the cross-stale branch below which defers to
                # /api/bars-recent.
                ring_is_live = not snap.is_empty() and not bar_ring.is_stale(snap) and not heartbeat_stale
                if ring_is_live:
                    ring_source_live[inst] = True
                    # Prefer ring during a live session.
                    since = last_seen.get(inst)
                    if since is None:
                        # Bootstrap — record latest ts only; chart bulk-loads
                        # via /api/bars-recent on browser connect.
                        latest_ts: datetime | None = None
                        for entry in snap.bars:
                            ts_raw = entry.get("ts_utc")
                            if isinstance(ts_raw, str):
                                try:
                                    ts = datetime.fromisoformat(ts_raw)
                                    if ts.tzinfo is None:
                                        ts = ts.replace(tzinfo=UTC)
                                    if latest_ts is None or ts > latest_ts:
                                        latest_ts = ts
                                except ValueError:
                                    continue
                        if latest_ts is not None:
                            last_seen[inst] = latest_ts
                        continue
                    for entry in snap.bars:
                        ts_raw = entry.get("ts_utc")
                        if not isinstance(ts_raw, str):
                            continue
                        try:
                            ts_utc = datetime.fromisoformat(ts_raw)
                        except ValueError:
                            continue
                        if ts_utc.tzinfo is None:
                            ts_utc = ts_utc.replace(tzinfo=UTC)
                        if ts_utc <= since:
                            continue
                        last_seen[inst] = ts_utc
                        _sse_broker.publish(
                            "bar",
                            {
                                "instrument": inst,
                                "time": int(ts_utc.astimezone(UTC).timestamp()),
                                "open": float(entry.get("open", 0.0)),
                                "high": float(entry.get("high", 0.0)),
                                "low": float(entry.get("low", 0.0)),
                                "close": float(entry.get("close", 0.0)),
                                "volume": int(entry.get("volume", 0)),
                            },
                        )
                    continue

                # Bot dead (stale heartbeat) → suppress SSE pushes entirely
                # for this instrument and defer chart loads to
                # /api/bars-recent's gold.db historical view. Prevents
                # post-crash false-live where the ring stays fresh for up
                # to 90s after the heartbeat thread dies.
                if heartbeat_stale:
                    # Transition emit: if the ring was previously live for
                    # this instrument, tell the client to repaint from
                    # /api/bars-recent (gold.db historical). Idempotent —
                    # ring_source_live[inst] is reset to False so only one
                    # event per transition.
                    if ring_source_live.get(inst) is True:
                        log.info(
                            "_bars_watcher(%s): ring_is_live -> heartbeat_stale "
                            "transition; publishing bars_source_changed",
                            inst,
                        )
                        _sse_broker.publish(
                            "bars_source_changed",
                            {"instrument": inst, "source": "gold_db"},
                        )
                    ring_source_live[inst] = False
                    if inst not in last_seen:
                        log.info(
                            "_bars_watcher(%s): heartbeat stale — deferring to "
                            "/api/bars-recent for historical view "
                            "(ring_stale=%s, ring_empty=%s)",
                            inst,
                            bar_ring.is_stale(snap),
                            snap.is_empty(),
                        )
                        last_seen[inst] = datetime.now(UTC) - timedelta(days=365)
                    continue

                if not GOLD_DB_PATH.exists():
                    continue
                try:
                    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
                        configure_connection(con)
                        since = last_seen.get(inst)
                        if since is None:
                            row = con.execute(
                                "SELECT max(ts_utc) FROM bars_1m WHERE symbol = ?",
                                [inst],
                            ).fetchone()
                            if row and row[0] is not None:
                                ts = row[0]
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=UTC)
                                last_seen[inst] = ts
                            continue
                        rows = con.execute(
                            "SELECT ts_utc, open, high, low, close, volume "
                            "FROM bars_1m WHERE symbol = ? AND ts_utc > ? "
                            "ORDER BY ts_utc ASC",
                            [inst, since],
                        ).fetchall()
                        for ts_utc, o, h, lo, c, v in rows:
                            if ts_utc.tzinfo is None:
                                ts_utc = ts_utc.replace(tzinfo=UTC)
                            last_seen[inst] = ts_utc
                            _sse_broker.publish(
                                "bar",
                                {
                                    "instrument": inst,
                                    "time": int(ts_utc.astimezone(UTC).timestamp()),
                                    "open": float(o),
                                    "high": float(h),
                                    "low": float(lo),
                                    "close": float(c),
                                    "volume": int(v),
                                },
                            )
                except duckdb.IOException as exc:
                    log.warning("_bars_watcher: gold.db locked: %s", exc)
        except Exception as exc:
            log.warning("_bars_watcher tick failed: %s", exc)
        await asyncio.sleep(2.0)


async def _sse_start_watchers() -> None:
    """Lazy-start SSE watcher tasks on first subscriber connect.

    Idempotency note: if a prior set of tasks was cancelled (e.g. lifespan
    shutdown) but the module-level _sse_tasks list still references them,
    a stale done-task check prevents zombie state. Tests that re-enter the
    module across event loops exercise this path.
    """
    # Drop any cancelled/done tasks left over from a prior lifespan.
    _sse_tasks[:] = [t for t in _sse_tasks if not t.done()]
    if _sse_tasks:
        return
    _sse_tasks.append(asyncio.create_task(_heartbeat_watcher(), name="sse-heartbeat"))
    _sse_tasks.append(asyncio.create_task(_state_watcher(), name="sse-state"))
    _sse_tasks.append(asyncio.create_task(_signals_watcher(), name="sse-signals"))
    _sse_tasks.append(asyncio.create_task(_alerts_watcher(), name="sse-alerts"))
    _sse_tasks.append(asyncio.create_task(_bars_watcher(), name="sse-bars"))


async def _sse_cancel_watchers() -> None:
    """Cancel all SSE watcher tasks and await their finalisation.

    Called from the FastAPI lifespan shutdown so uvicorn graceful-stop does
    not leave orphan coroutines polling files during teardown. Idempotent:
    safe to call when _sse_tasks is empty or already cancelled.
    """
    for t in _sse_tasks:
        if not t.done():
            t.cancel()
    if _sse_tasks:
        # Swallow CancelledError; watchers are infinite loops by design.
        await asyncio.gather(*_sse_tasks, return_exceptions=True)
    _sse_tasks.clear()


async def _sse_lazy_stop_if_idle() -> None:
    """Cancel watcher tasks when the last subscriber disconnects.

    Called from the SSE endpoint's finally-block after a real subscriber
    drop. Mirrors `_sse_start_watchers` (lazy-start on first connect) so
    the five file-polling watchers do not run indefinitely once no browser
    tab is consuming events. Idempotent: a re-connecting subscriber will
    relaunch via `_sse_start_watchers` on the next `/api/events/stream`
    GET.

    Skips when `subscriber_count > 0` so this remains safe to call from any
    unsubscribe path (including the TOCTOU-rejection cleanup at the
    endpoint, which already has live subscribers under the cap).
    """
    if _sse_broker.subscriber_count() > 0:
        return
    if not _sse_tasks:
        return
    await _sse_cancel_watchers()


# ── HTML Frontend ─────────────────────────────────────────────────────────────


DASHBOARD_HTML = Path(__file__).parent / "bot_dashboard.html"


# ── /api/bars-recent (Stage 2 of cockpit-v3) ──────────────────────────────────


def _query_bars_recent(
    instrument: str, lookback_minutes: int, since_ts: str | None
) -> tuple[list[dict[str, Any]], str | None]:
    """Read recent 1m bars, preferring the live-bars ring over gold.db.

    Order: read the ring snapshot first (no DuckDB lock contention during a
    live session). If the lookback window extends earlier than the ring's
    oldest bar, also query gold.db for the historical tail and merge with
    dedup on epoch-second ``time``. Ring takes precedence on overlap so the
    operator sees current-session OHLCV even before flush_to_db lands.
    """
    from trading_app.live import bar_ring

    since_dt: datetime | None = None
    if since_ts:
        try:
            since_dt = datetime.fromisoformat(since_ts.replace("Z", "+00:00"))
        except ValueError:
            log.warning("_query_bars_recent: bad since_ts %r — ignoring", since_ts)

    # ── Ring read ───────────────────────────────────────────────────────────
    snap = bar_ring.read_bar_ring(instrument)
    ring_bars: list[dict[str, Any]] = []
    ring_oldest: datetime | None = None
    for entry in snap.bars:
        ts_raw = entry.get("ts_utc")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if since_dt is not None and ts <= since_dt:
            continue
        if ring_oldest is None or ts < ring_oldest:
            ring_oldest = ts
        ring_bars.append(
            {
                "time": int(ts.astimezone(UTC).timestamp()),
                "open": float(entry.get("open", 0.0)),
                "high": float(entry.get("high", 0.0)),
                "low": float(entry.get("low", 0.0)),
                "close": float(entry.get("close", 0.0)),
                "volume": int(entry.get("volume", 0)),
            }
        )

    # ── Decide whether to also query gold.db ────────────────────────────────
    # If the ring already covers the lookback window from oldest to "now",
    # skip the DB. Otherwise pull the missing historical tail.
    now = datetime.now(UTC)
    lookback_floor = since_dt if since_dt is not None else now - timedelta(minutes=max(0, lookback_minutes))
    need_db = ring_oldest is None or ring_oldest > lookback_floor

    db_bars: list[dict[str, Any]] = []
    if need_db:
        if not GOLD_DB_PATH.exists():
            if not ring_bars:
                return [], f"gold.db not found at {GOLD_DB_PATH}"
        else:
            db_upper = ring_oldest  # cap DB read at ring's oldest to avoid duplicate rows
            try:
                with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
                    configure_connection(con)
                    if since_dt is not None and db_upper is not None:
                        rows = con.execute(
                            "SELECT ts_utc, open, high, low, close, volume FROM bars_1m "
                            "WHERE symbol = ? AND ts_utc > ? AND ts_utc < ? "
                            "ORDER BY ts_utc ASC",
                            [instrument, since_dt, db_upper],
                        ).fetchall()
                    elif since_dt is not None:
                        rows = con.execute(
                            "SELECT ts_utc, open, high, low, close, volume FROM bars_1m "
                            "WHERE symbol = ? AND ts_utc > ? ORDER BY ts_utc ASC",
                            [instrument, since_dt],
                        ).fetchall()
                    elif db_upper is not None:
                        rows = con.execute(
                            "SELECT ts_utc, open, high, low, close, volume FROM bars_1m "
                            "WHERE symbol = ? AND ts_utc > now() - INTERVAL (? || ' minutes') "
                            "AND ts_utc < ? ORDER BY ts_utc ASC",
                            [instrument, str(lookback_minutes), db_upper],
                        ).fetchall()
                    else:
                        rows = con.execute(
                            "SELECT ts_utc, open, high, low, close, volume FROM bars_1m "
                            "WHERE symbol = ? AND ts_utc > now() - INTERVAL (? || ' minutes') "
                            "ORDER BY ts_utc ASC",
                            [instrument, str(lookback_minutes)],
                        ).fetchall()
            except duckdb.IOException as exc:
                if not ring_bars:
                    return [], f"gold.db locked: {exc}"
                log.warning("_query_bars_recent: gold.db locked (ring-only): %s", exc)
                rows = []
            except Exception as exc:
                if not ring_bars:
                    log.error("_query_bars_recent: query failed: %s", exc)
                    return [], f"query failed: {exc}"
                log.warning("_query_bars_recent: query failed (ring-only): %s", exc)
                rows = []

            for ts_utc, o, h, lo, c, v in rows:
                if ts_utc.tzinfo is None:
                    ts_utc = ts_utc.replace(tzinfo=UTC)
                db_bars.append(
                    {
                        "time": int(ts_utc.astimezone(UTC).timestamp()),
                        "open": float(o),
                        "high": float(h),
                        "low": float(lo),
                        "close": float(c),
                        "volume": int(v),
                    }
                )

    # ── Merge: DB tail + ring; dedup on epoch-second `time`; ring wins ──────
    by_time: dict[int, dict[str, Any]] = {b["time"]: b for b in db_bars}
    for b in ring_bars:
        by_time[b["time"]] = b
    merged = sorted(by_time.values(), key=lambda b: b["time"])
    return merged, None


def _count_open_positions_from_state(state: dict[str, Any] | None) -> int | None:
    """Count lanes currently IN_TRADE in a ``bot_state`` payload.

    Returns ``None`` when state is missing/stale (caller should treat as
    "unknown → show confirmation modal" — fail-CLOSED on operator
    confirmation per institutional-rigor.md § 6). Returns ``0`` when state
    is present but no lane has ``status == "IN_TRADE"``. Returns ``n``
    (positive int) when n lanes are IN_TRADE.

    Canonical lane-status taxonomy lives in
    ``trading_app.live.bot_state.build_state_snapshot`` (values:
    WAITING / ARMED / IN_TRADE / FLAT). The non-existent fields
    ``position_qty`` / ``open_position`` referenced by the pre-2026-05-14
    cockpit HTML are NOT in the canonical shape — relying on them caused
    the kill-switch confirmation modal to be skipped unconditionally.
    """
    if not isinstance(state, dict):
        return None
    if not state.get("present", False):
        return None
    inner = state.get("state")
    if not isinstance(inner, dict):
        return None
    lanes = inner.get("lanes")
    if not isinstance(lanes, dict):
        return 0
    count = 0
    for lane in lanes.values():
        if isinstance(lane, dict) and lane.get("status") == "IN_TRADE":
            count += 1
    return count


_ORB_PAYLOAD_NULL_KEYS = (
    "orb_high",
    "orb_low",
    "orb_complete",
    "orb_minutes",
    "session_name",
    "session_time_brisbane",
    "orb_break_direction",
    "orb_window_start_utc",
    "orb_window_end_utc",
)


def _empty_orb_payload() -> dict[str, Any]:
    """Stable-shape null payload — frontend never sees missing keys."""
    payload: dict[str, Any] = {k: None for k in _ORB_PAYLOAD_NULL_KEYS}
    payload["orb_complete"] = False
    return payload


def _orb_levels_for_instrument(instrument: str, days: int = 1) -> dict[str, Any]:
    """Read ORB high/low/window for ``instrument`` from bot_state.json.

    Reads through canonical ``read_state`` accessor — never re-derives ORB
    levels (institutional-rigor.md § 4). ORB UTC window is sourced from
    ``pipeline.dst.orb_utc_window`` (the canonical resolver per
    institutional-rigor.md § 10 + postmortem 2026-04-07) — never inlined.
    Returns a stable-shape dict with null fields if no lane has computed
    ORB yet, the trading_day is missing, or the canonical resolver
    rejects the session/aperture combination.

    Stage 2 (2026-05-23): adds ``daily_orb_windows`` — per-day ORB-aperture
    windows for the last ``days`` trading days, each computed via the same
    canonical ``orb_utc_window`` resolver. Width = literal ``orb_minutes``
    aperture (5/15/30m). No fabricated display constant. Each entry has
    schema ``{trading_day, session_name, orb_minutes, window_start_utc,
    window_end_utc}``. ``days`` is server-clamped to 1..5.
    """
    from pipeline.dst import orb_utc_window
    from trading_app.live.bot_state import read_state

    payload = _empty_orb_payload()
    payload["daily_orb_windows"] = []
    state = read_state() or {}
    lanes = state.get("lanes") or {}
    if not isinstance(lanes, dict):
        return payload

    trading_day_str = state.get("trading_day")
    trading_day_obj: date | None = None
    if isinstance(trading_day_str, str):
        try:
            trading_day_obj = date.fromisoformat(trading_day_str)
        except ValueError:
            log.warning(
                "_orb_levels_for_instrument: bad trading_day %r — window fields null",
                trading_day_str,
            )

    clamped_days = max(1, min(int(days), 5))

    for lane in lanes.values():
        if not isinstance(lane, dict):
            continue
        if lane.get("instrument") != instrument:
            continue
        oh, ol = lane.get("orb_high"), lane.get("orb_low")
        if oh is None or ol is None:
            continue

        session_name = lane.get("session_name")
        orb_minutes = lane.get("orb_minutes")
        payload.update(
            {
                "orb_high": oh,
                "orb_low": ol,
                "orb_complete": bool(lane.get("orb_complete")),
                "orb_minutes": orb_minutes,
                "session_name": session_name,
                "session_time_brisbane": lane.get("session_time_brisbane"),
                "orb_break_direction": lane.get("orb_break_direction"),
            }
        )

        if trading_day_obj is not None and isinstance(session_name, str) and isinstance(orb_minutes, int):
            try:
                start_dt, end_dt = orb_utc_window(trading_day_obj, session_name, orb_minutes)
                payload["orb_window_start_utc"] = int(start_dt.timestamp())
                payload["orb_window_end_utc"] = int(end_dt.timestamp())
            except (ValueError, KeyError) as exc:
                log.warning(
                    "_orb_levels_for_instrument: orb_utc_window(%s, %s, %s) rejected (%s) — window fields null",
                    trading_day_obj,
                    session_name,
                    orb_minutes,
                    exc,
                )

            # Stage 2: per-day canonical aperture windows for the last N trading days.
            # Each call delegates to the same canonical orb_utc_window — width is the
            # literal orb_minutes aperture, never a fabricated display constant.
            for i in range(clamped_days):
                td = trading_day_obj - timedelta(days=i)
                try:
                    s, e = orb_utc_window(td, session_name, orb_minutes)
                except (ValueError, KeyError) as exc:
                    log.warning(
                        "_orb_levels_for_instrument: per-day orb_utc_window(%s, %s, %s) rejected (%s) — entry skipped",
                        td,
                        session_name,
                        orb_minutes,
                        exc,
                    )
                    continue
                payload["daily_orb_windows"].append(
                    {
                        "trading_day": td.isoformat(),
                        "session_name": session_name,
                        "orb_minutes": orb_minutes,
                        "window_start_utc": int(s.timestamp()),
                        "window_end_utc": int(e.timestamp()),
                    }
                )
        return payload
    return payload


@app.get("/api/bars-recent")
async def api_bars_recent(
    instrument: str = "MNQ",
    lookback_minutes: int = 90,
    since: str | None = None,
    days: int = 1,
):
    """Return recent 1m bars + ORB levels shaped for Lightweight Charts.

    lookback_minutes capped at 7200 (5d). since= overrides lookback.
    days capped at 5 — controls how many per-day ORB-aperture windows are
    returned in `daily_orb_windows` (Stage 2 2026-05-23). ORB levels and
    windows delegated to canonical bot_state.json + pipeline.dst.orb_utc_window
    — never re-derived.
    """
    capped_lookback = max(1, min(int(lookback_minutes), 7200))
    bars, err = _query_bars_recent(instrument, capped_lookback, since)
    orb = _orb_levels_for_instrument(instrument, days=days)
    payload: dict[str, Any] = {
        "instrument": instrument,
        "bars": bars,
        "server_ts": datetime.now(UTC).isoformat(),
        **orb,
    }
    if err:
        payload["warning"] = err
    return payload


@app.get("/api/events/stream")
async def api_events_stream(request: Request):
    """Server-Sent Events stream feeding the cockpit dashboard.

    Events: heartbeat (1s), state (mtime-driven), signal (JSONL-tailed).
    Subscriber cap _SSE_MAX_SUBSCRIBERS; over-cap returns 429. Browser
    Last-Event-ID header triggers replay from the ring buffer.
    """
    from sse_starlette.sse import EventSourceResponse

    # Atomic subscribe-then-check pattern. The naive "check then subscribe"
    # leaves a window where two concurrent connects at cap-1 can both pass
    # the guard before either is registered — TOCTOU race surfaced by the
    # Stage 2 adversarial audit (commit cff1efcd verdict). Subscribing first
    # then unsubscribing on overflow closes the window.
    queue = _sse_broker.subscribe()
    if _sse_broker.subscriber_count() > _SSE_MAX_SUBSCRIBERS:
        _sse_broker.unsubscribe(queue)
        return JSONResponse(
            status_code=429,
            content={
                "error": "SSE subscriber limit reached",
                "limit": _SSE_MAX_SUBSCRIBERS,
                "retry_after_s": 5,
            },
            headers={"Retry-After": "5"},
        )

    await _sse_start_watchers()

    last_event_id_raw = request.headers.get("last-event-id")
    try:
        last_event_id = int(last_event_id_raw) if last_event_id_raw else 0
    except (ValueError, TypeError):
        last_event_id = 0

    async def _generator():
        try:
            # On-connect state snapshot — covers "operator opens dashboard
            # mid-night when nothing changes" case.
            try:
                from trading_app.live.bot_state import STATE_FILE, read_state

                if STATE_FILE.exists():
                    yield {
                        "event": "state",
                        "data": _json.dumps({"present": True, "state": read_state() or {}}),
                    }
                else:
                    yield {"event": "state", "data": _json.dumps({"present": False})}
            except Exception as exc:
                log.warning("SSE on-connect snapshot failed: %s", exc)

            if last_event_id > 0:
                for env in _sse_broker.replay_since(last_event_id):
                    yield {
                        "event": env["event"],
                        "id": str(env.get("id", "")),
                        "data": _json.dumps(env["data"]),
                    }

            while True:
                env = await queue.get()
                payload: dict[str, str] = {
                    "event": env["event"],
                    "data": _json.dumps(env["data"]),
                }
                if "id" in env:
                    payload["id"] = str(env["id"])
                yield payload
        finally:
            _sse_broker.unsubscribe(queue)
            await _sse_lazy_stop_if_idle()

    return EventSourceResponse(_generator())


@app.get("/", response_class=HTMLResponse)
async def index():
    if DASHBOARD_HTML.exists():
        return DASHBOARD_HTML.read_text(encoding="utf-8")
    return "<h1>Dashboard HTML not found</h1>"


# ── Server launch ─────────────────────────────────────────────────────────────


def run_dashboard(host: str = "127.0.0.1", port: int = PORT) -> None:
    """Run the dashboard server (blocking).

    Localhost-only — SSE leaks live position state, /api/action/kill mutates
    real-money state. Drift check ``check_dashboard_localhost_only_binding``
    is the first defense; this assertion is the second. workers=1 pinned
    because the SSE subscriber set is in-process.
    """
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            f"Refusing to start dashboard on non-localhost host {host!r}: "
            f"SSE + kill endpoint leak live position state over LAN."
        )
    uvicorn.run(app, host=host, port=port, log_level="warning", workers=1)


def launch_dashboard_background(port: int = PORT) -> subprocess.Popen | None:
    """Launch dashboard as a background subprocess (Windows-safe).

    Returns the Popen handle so the caller can terminate it on exit.
    Using subprocess instead of threading avoids asyncio event loop
    conflicts between uvicorn and the main bot's event loop on Windows.

    Browser open is the caller's responsibility — START_BOT.bat opens the
    browser explicitly (line 79) and the dashboard-button start path sets
    CANOMPX3_DASHBOARD_ORIGIN=1 which skips this function entirely
    (run_live_session.py:937). Auto-opening here was producing a redundant
    second browser tab when the launcher path also opened one. See
    docs/runtime/stages/2026-05-26-start-bot-double-dashboard-bug.md.
    """
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

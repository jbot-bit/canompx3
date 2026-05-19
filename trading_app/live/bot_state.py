"""Shared bot state file — written by SessionOrchestrator, read by dashboard.

Atomic JSON write (write to .tmp, os.replace) prevents partial reads.
Dashboard reads this file on each poll cycle (every 5s).
Bot writes on each bar and on each trade event.

Strict-type contamination guard (2026-05-17): write_state refuses to serialize
unittest.mock.Mock/MagicMock/AsyncMock or any non-JSON-native type. The prior
``json.dumps(default=str, ...)`` silently coerced mocks to literal
``<MagicMock name='...'>`` strings, polluting data/bot_state.json with fake
TEST_STRAT_001 lane state. See docs/runtime/diagnostics/2026-05-17-live-
throughput-triage.md and the companion test in test_bot_state_strict_types.py.
"""

import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pipeline.dst import SESSION_CATALOG

log = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent.parent.parent / "data" / "bot_state.json"
LIVE_HEALTH_FILE = Path(__file__).parent.parent.parent / "runtime" / "state" / "live_health.json"

# Rationale: heartbeat writes the snapshot once per HEARTBEAT_INTERVAL
# (1800 s / 30 min in production per session_orchestrator.HEARTBEAT_INTERVAL).
# A snapshot older than 600 s while the bot is supposed to be running means
# at least one heartbeat tick has been skipped — the operator dashboard
# treats that as broker_status="unknown" (fail-closed). The tolerance is
# deliberately below HEARTBEAT_INTERVAL so a stale snapshot is detected on
# the very next heartbeat, not two heartbeats late.
LIVE_HEALTH_STALE_AFTER_SECS = 600

_MOCK_CLASS_NAMES = frozenset({"Mock", "MagicMock", "AsyncMock", "NonCallableMock", "NonCallableMagicMock"})


def _is_mock_object(value: Any) -> bool:
    """True if value is an instance of unittest.mock.* — detected by class name.

    Avoids importing unittest.mock in production. Mock subclasses are caught
    because mro inspection includes their class names.
    """
    for cls in type(value).__mro__:
        if cls.__name__ in _MOCK_CLASS_NAMES:
            return True
    return False


def _sanitize_for_state(value: Any, path: str = "<root>") -> Any:
    """Recursively validate ``value`` is JSON-safe for bot_state serialization.

    Raises ``TypeError`` on Mock-class objects or any non-(JSON-native|datetime|
    date|Path) leaf. dict / list / tuple are walked. tuple is converted to list.

    Path argument is dotted-path context for the error message.
    """
    if _is_mock_object(value):
        raise TypeError(
            f"bot_state contamination: {path} = {type(value).__name__} (unittest.mock object); refuse to serialize"
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date, Path)):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"bot_state contamination: {path} dict key {k!r} is {type(k).__name__}, not str")
            out[k] = _sanitize_for_state(v, f"{path}.{k}")
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_state(v, f"{path}[{i}]") for i, v in enumerate(value)]
    raise TypeError(
        f"bot_state contamination: {path} = {type(value).__module__}.{type(value).__name__} "
        f"is not JSON-safe (allowed: dict, list, tuple, str, int, float, bool, None, datetime, date, Path)"
    )


def _json_default(value: Any) -> str:
    """Strict json.dumps default: serialize datetime/date/Path; raise otherwise."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"bot_state json encoder: refusing {type(value).__module__}.{type(value).__name__}")


def write_state(data: dict[str, Any]) -> None:
    """Atomically write bot state to JSON file.

    Strict-type guarded: refuses to serialize Mock objects or non-JSON-safe
    leaves (see ``_sanitize_for_state``). On contamination, logs CRITICAL and
    does NOT write. Fail-open on disk/encoder errors per the original
    contract — trade execution must not block on a dashboard write failure.
    """
    try:
        sanitized = _sanitize_for_state(data)
    except TypeError as exc:
        log.critical(
            "bot_state write REFUSED — contamination detected; not overwriting canonical state: %s",
            exc,
        )
        return
    sanitized["heartbeat_utc"] = datetime.now(UTC).isoformat()
    tmp = STATE_FILE.with_suffix(".json.tmp")
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(sanitized, default=_json_default, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(STATE_FILE))
    except Exception:
        log.warning("bot_state write failed — dashboard state may be stale", exc_info=True)
        tmp.unlink(missing_ok=True)


def read_state() -> dict[str, Any]:
    """Read bot state from JSON file. Returns empty dict if missing/corrupt."""
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning("bot_state read failed — returning empty state", exc_info=True)
    return {}


def clear_state() -> None:
    """Remove the state file (called on bot shutdown)."""
    STATE_FILE.unlink(missing_ok=True)


def write_live_health(data: dict[str, Any]) -> None:
    """Atomic snapshot write for broker-edge operator visibility.

    Mirrors write_state's atomic-replace idiom. The trading loop calls this
    best-effort from the heartbeat tick — failure is logged, never raised.
    """
    data["snapshot_ts_utc"] = datetime.now(UTC).isoformat()
    tmp = LIVE_HEALTH_FILE.with_suffix(".json.tmp")
    try:
        LIVE_HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(LIVE_HEALTH_FILE))
    except OSError:
        log.warning("live_health write failed — dashboard health card may be stale", exc_info=True)
        tmp.unlink(missing_ok=True)


def read_live_health() -> dict[str, Any]:
    """Read broker-edge snapshot. Returns fail-closed unknown payload on any issue.

    Surfaced verbatim into the dashboard operator payload; the dashboard
    never derives its own broker health. Fail-closed: missing file, corrupt
    JSON, or stale snapshot all return broker_status="unknown" with a reason.
    """
    if not LIVE_HEALTH_FILE.exists():
        return {"broker_status": "unknown", "reason": "snapshot_missing"}
    try:
        raw = LIVE_HEALTH_FILE.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("live_health read failed — returning unknown", exc_info=True)
        return {"broker_status": "unknown", "reason": f"read_error:{type(e).__name__}"}

    ts = payload.get("snapshot_ts_utc")
    if not isinstance(ts, str):
        return {"broker_status": "unknown", "reason": "snapshot_ts_missing"}
    try:
        snapshot_ts = datetime.fromisoformat(ts)
    except ValueError:
        return {"broker_status": "unknown", "reason": "snapshot_ts_invalid"}
    age = (datetime.now(UTC) - snapshot_ts).total_seconds()
    if age > LIVE_HEALTH_STALE_AFTER_SECS:
        payload["broker_status"] = "unknown"
        payload["reason"] = f"snapshot_stale:{int(age)}s"
    return payload


def _get_session_time_brisbane(orb_label: str, trading_day: date | None = None) -> str:
    """Resolve a session's Brisbane clock time for the given trading day."""
    entry = SESSION_CATALOG.get(orb_label)
    if entry is None:
        return "unknown"
    resolver = entry.get("resolver")
    if resolver is None:
        return "unknown"
    hour, minute = resolver(trading_day or date.today())
    return f"{hour:02d}:{minute:02d}"


def _sort_time_key(time_text: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = time_text.split(":", maxsplit=1)
        return int(hour_text), int(minute_text)
    except (ValueError, AttributeError):
        return (99, 99)


def _iso_utc(value: Any) -> str | None:
    """Best-effort ISO formatter for datetimes held on runtime objects.

    None passes through silently. datetime is normalized to UTC ISO8601.
    Any other non-None type returns None with a logger warning — operator-
    visible fields must never silently drop type-mismatched values per
    institutional-rigor.md sec 6 (no silent failures). Upstream callers
    that route non-datetime timestamps (e.g. pandas Timestamp from
    execution_engine.py:978/1099/1374 — see follow-up F6) should coerce
    at the assignment site, not here.

    STABLE CROSS-MODULE API: imported by trading_app.live.session_orchestrator
    (see fix/code-review-hotpatches-2026-05-05). Despite the leading
    underscore, this helper is the canonical operator-visible-timestamp
    formatter for the live package. Do not rename without updating every
    in-package caller; promotion to a public name (``iso_utc``) is queued
    as low-priority debt — track in the memory feedback file for the
    iso_utc silent-None class pattern.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    log.warning("_iso_utc: unsupported type %s — returning None", type(value).__name__)
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _orb_snapshot(orbs: dict[tuple[str, int], Any] | None, orb_label: str, orb_minutes: int) -> dict[str, Any]:
    """Project runtime ORB state into a JSON-safe dict for operator surfaces."""
    if not orbs:
        return {}
    orb = orbs.get((orb_label, orb_minutes))
    if orb is None:
        return {}
    high = _coerce_float(getattr(orb, "high", None))
    low = _coerce_float(getattr(orb, "low", None))
    size = None
    if high is not None and low is not None:
        size = high - low
    return {
        "orb_high": high,
        "orb_low": low,
        "orb_size": size,
        "orb_complete": bool(getattr(orb, "complete", False)),
        "orb_break_direction": getattr(orb, "break_dir", None),
        "orb_break_time_utc": _iso_utc(getattr(orb, "break_ts", None)),
        "orb_complete_time_utc": _iso_utc(getattr(orb, "complete_ts", None)),
    }


def build_state_snapshot(
    *,
    mode: str,
    instrument: str,
    contract: str,
    trading_day: date,
    account_id: int,
    account_name: str,
    daily_pnl_r: float,
    daily_loss_limit_r: float,
    max_equity_dd_r: float | None,
    bars_received: int,
    strategies: list,
    active_trades: list,
    completed_trades: list,
    orbs: dict[tuple[str, int], Any] | None = None,
    feed_status: dict[str, Any] | None = None,
    router_status: dict[str, Any] | None = None,
    broker_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a state dict from orchestrator internals."""
    lanes: dict[str, dict] = {}
    lane_cards: list[dict[str, Any]] = []
    for s in strategies:
        lane = {
            "lane_key": s.strategy_id,
            "strategy_id": s.strategy_id,
            "instrument": s.instrument,
            "session_name": s.orb_label,
            "session_time_brisbane": _get_session_time_brisbane(s.orb_label, trading_day),
            "filter_type": s.filter_type,
            "rr_target": s.rr_target,
            "orb_minutes": s.orb_minutes,
            "entry_model": s.entry_model,
            "confirm_bars": s.confirm_bars,
            "status": "WAITING",
            "status_detail": None,
            "direction": None,
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "risk_points": None,
            "signal_time_utc": None,
            "entry_time_utc": None,
            "exit_time_utc": None,
            "current_pnl_r": None,
        }
        lane.update(_orb_snapshot(orbs, s.orb_label, s.orb_minutes))
        # Check if this strategy has an active trade
        for t in active_trades:
            if t.strategy_id == s.strategy_id:
                lane["status_detail"] = getattr(getattr(t, "state", None), "value", None)
                lane["direction"] = getattr(t, "direction", None)
                lane["entry_price"] = _coerce_float(getattr(t, "entry_price", None))
                lane["stop_price"] = _coerce_float(getattr(t, "stop_price", None))
                lane["target_price"] = _coerce_float(getattr(t, "target_price", None))
                lane["entry_time_utc"] = _iso_utc(getattr(t, "entry_ts", None))
                lane["signal_time_utc"] = lane["entry_time_utc"] or lane.get("orb_break_time_utc")
                if t.state.value == "ENTERED":
                    lane["status"] = "IN_TRADE"
                    lane["current_pnl_r"] = _coerce_float(getattr(t, "pnl_r", None))
                elif t.state.value in ("ARMED", "CONFIRMING"):
                    lane["status"] = "ARMED"
                if lane["entry_price"] is not None and lane["stop_price"] is not None:
                    lane["risk_points"] = abs(lane["entry_price"] - lane["stop_price"])
                break
        # Check completed trades
        for t in completed_trades:
            if t.strategy_id == s.strategy_id:
                lane["status"] = "FLAT"
                lane["status_detail"] = getattr(getattr(t, "state", None), "value", lane["status_detail"])
                lane["direction"] = getattr(t, "direction", lane["direction"])
                lane["entry_price"] = _coerce_float(getattr(t, "entry_price", lane["entry_price"]))
                lane["stop_price"] = _coerce_float(getattr(t, "stop_price", lane["stop_price"]))
                lane["target_price"] = _coerce_float(getattr(t, "target_price", lane["target_price"]))
                lane["entry_time_utc"] = _iso_utc(getattr(t, "entry_ts", None)) or lane["entry_time_utc"]
                lane["exit_time_utc"] = _iso_utc(getattr(t, "exit_ts", None))
                lane["signal_time_utc"] = lane["entry_time_utc"] or lane.get("orb_break_time_utc")
                lane["current_pnl_r"] = _coerce_float(getattr(t, "pnl_r", None))
                if lane["entry_price"] is not None and lane["stop_price"] is not None:
                    lane["risk_points"] = abs(lane["entry_price"] - lane["stop_price"])
                break
        lanes[s.strategy_id] = lane
        lane_cards.append(lane)

    lane_cards.sort(
        key=lambda item: (_sort_time_key(item["session_time_brisbane"]), item["instrument"], item["strategy_id"])
    )

    return {
        "mode": mode,
        "instrument": instrument,
        "contract": contract,
        "trading_day": trading_day.isoformat(),
        "account_id": account_id,
        "account_name": account_name,
        "daily_pnl_r": round(daily_pnl_r, 4),
        "daily_loss_limit_r": daily_loss_limit_r,
        "max_equity_dd_r": max_equity_dd_r,
        "bars_received": bars_received,
        "strategies_loaded": len(strategies),
        "lanes": lanes,
        "lane_cards": lane_cards,
        "feed_status": feed_status or {},
        "router_status": router_status or {},
        "broker_status": broker_status or {},
    }

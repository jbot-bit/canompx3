"""Shared bot state file — written by SessionOrchestrator, read by dashboard.

Atomic JSON write (write to .tmp, os.replace) prevents partial reads.
Dashboard reads this file on each poll cycle (every 5s).
Bot writes on each bar and on each trade event.
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


def write_state(data: dict[str, Any]) -> None:
    """Atomically write bot state to JSON file."""
    data["heartbeat_utc"] = datetime.now(UTC).isoformat()
    tmp = STATE_FILE.with_suffix(".json.tmp")
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
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
            "direction": None,
            "entry_price": None,
            "current_pnl_r": None,
        }
        # Check if this strategy has an active trade
        for t in active_trades:
            if t.strategy_id == s.strategy_id:
                if t.state.value == "ENTERED":
                    lane["status"] = "IN_TRADE"
                    lane["direction"] = t.direction
                    lane["entry_price"] = t.entry_price
                    lane["current_pnl_r"] = t.pnl_r
                elif t.state.value in ("ARMED", "CONFIRMING"):
                    lane["status"] = "ARMED"
                break
        # Check completed trades
        for t in completed_trades:
            if t.strategy_id == s.strategy_id:
                lane["status"] = "FLAT"
                lane["direction"] = t.direction
                lane["entry_price"] = t.entry_price
                lane["current_pnl_r"] = t.pnl_r
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
    }

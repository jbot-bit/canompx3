"""Shared bot state file — written by SessionOrchestrator, read by dashboard.

Atomic JSON write (write to .tmp, os.replace) prevents partial reads.
Dashboard reads this file on each poll cycle (every 5s).
Bot writes on each bar and on each trade event.
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        log.debug("bot_state write failed", exc_info=True)
        tmp.unlink(missing_ok=True)


def read_state() -> dict[str, Any]:
    """Read bot state from JSON file. Returns empty dict if missing/corrupt."""
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.debug("bot_state read failed", exc_info=True)
    return {}


def clear_state() -> None:
    """Remove the state file (called on bot shutdown)."""
    STATE_FILE.unlink(missing_ok=True)


def build_state_snapshot(
    *,
    mode: str,
    instrument: str,
    contract: str,
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
    for s in strategies:
        lane = {
            "strategy_id": s.strategy_id,
            "filter_type": s.filter_type,
            "rr_target": s.rr_target,
            "orb_minutes": s.orb_minutes,
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
        lanes[s.orb_label] = lane

    return {
        "mode": mode,
        "instrument": instrument,
        "contract": contract,
        "account_id": account_id,
        "account_name": account_name,
        "daily_pnl_r": round(daily_pnl_r, 4),
        "daily_loss_limit_r": daily_loss_limit_r,
        "max_equity_dd_r": max_equity_dd_r,
        "bars_received": bars_received,
        "strategies_loaded": len(strategies),
        "lanes": lanes,
    }

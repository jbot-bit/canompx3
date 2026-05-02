"""Passive-sidecar runtime snapshot persistence.

This reuses the repo's runtime-file pattern without touching bot_state.json.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .policy_gate import policy_gate_status

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = PROJECT_ROOT / "data" / "runtime" / "passive_sidecar_state.json"


def build_dashboard_snapshot(
    projection: Mapping[str, Any],
    *,
    recent_alert_summary: Mapping[str, Any] | None = None,
    policy_gate_state: str | None = None,
) -> dict[str, Any]:
    return {
        "connection_status": projection.get("connection_status", "idle"),
        "last_error": projection.get("last_error"),
        "last_event_utc": projection.get("last_event_utc"),
        "accounts": projection.get("accounts_by_id", {}),
        "orders": projection.get("orders_by_id", {}),
        "positions": projection.get("positions_by_contract", {}),
        "trades": projection.get("trades_by_id", {}),
        "recent_alert_summary": dict(recent_alert_summary or {}),
        "policy_gate_status": policy_gate_state or policy_gate_status(),
    }


def write_passive_sidecar_state(snapshot: Mapping[str, Any]) -> None:
    payload = dict(snapshot)
    payload["heartbeat_utc"] = datetime.now(UTC).isoformat()
    tmp = STATE_PATH.with_suffix(".json.tmp")
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(STATE_PATH))
    except Exception:
        log.warning("passive_sidecar state write failed", exc_info=True)
        tmp.unlink(missing_ok=True)


def read_passive_sidecar_state() -> dict[str, Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning("passive_sidecar state read failed", exc_info=True)
    return {}

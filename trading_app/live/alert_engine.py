"""Thin runtime alert persistence for the operator dashboard.

This is intentionally not a second event bus. The trading runtime already
knows when important things happen; this module gives the dashboard a durable,
structured read model for those events.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
ALERTS_PATH = PROJECT_ROOT / "data" / "runtime" / "operator_alerts.jsonl"
RECENT_ALERT_WINDOW_MINUTES = 30


@dataclass(frozen=True)
class OperatorAlert:
    timestamp_utc: str
    level: Literal["critical", "warning", "info"]
    category: str
    message: str
    instrument: str | None = None
    profile: str | None = None
    mode: str | None = None
    source: str = "runtime"
    trading_day: str | None = None


_ALERT_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("feed_dead", "critical", ("FEED DEAD",)),
    ("kill_switch", "critical", ("KILL SWITCH",)),
    ("manual_close_required", "critical", ("MANUAL CLOSE REQUIRED",)),
    ("account_dd_limit", "critical", ("ACCOUNT DD LIMIT", "HWM DD HALT")),
    ("engine_circuit_breaker", "critical", ("ENGINE CIRCUIT BREAKER",)),
    ("bad_fill", "critical", ("BAD FILL",)),
    ("stuck_exit", "critical", ("STUCK EXIT",)),
    ("exit_failed", "critical", ("EXIT FAILED",)),
    ("feed_stale", "warning", ("FEED STALE",)),
    ("bar_heartbeat", "warning", ("BAR HEARTBEAT",)),
    ("stale_orders", "warning", ("STALE ORDERS",)),
    ("cusum_alarm", "warning", ("CUSUM ALARM",)),
    ("engine_error", "warning", ("ENGINE ERROR",)),
    ("rollover_error", "warning", ("ROLLOVER ERROR",)),
    ("close_time_flatten", "warning", ("CLOSE TIME FLATTEN",)),
    ("entry_blocked", "warning", ("ENTRY BLOCKED",)),
    ("orphan", "warning", ("ORPHAN",)),
    ("heartbeat", "info", ("HEARTBEAT:",)),
]


def _normalize_profile_id(profile: str | None) -> str | None:
    if profile is None:
        return None
    text = str(profile).strip()
    if text.startswith("profile_"):
        return text.removeprefix("profile_")
    return text or None


def classify_operator_alert(message: str) -> tuple[str, str]:
    """Infer alert level/category from existing runtime messages."""
    upper = message.strip().upper()
    for category, level, markers in _ALERT_RULES:
        if any(marker in upper for marker in markers):
            return level, category
    return "info", "runtime_event"


def record_operator_alert(
    *,
    message: str,
    instrument: str | None = None,
    profile: str | None = None,
    mode: str | None = None,
    source: str = "runtime",
    trading_day: str | None = None,
) -> dict[str, object] | None:
    """Persist a structured operator alert. Fail-open by design."""
    try:
        level, category = classify_operator_alert(message)
        alert = OperatorAlert(
            timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            level=level,
            category=category,
            message=message.strip(),
            instrument=instrument,
            profile=profile,
            mode=mode,
            source=source,
            trading_day=trading_day,
        )
        ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ALERTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(alert), ensure_ascii=True) + "\n")
        return asdict(alert)
    except Exception:
        log.warning("Operator alert persistence failed", exc_info=True)
        return None


def read_operator_alerts(
    limit: int = 25,
    *,
    profile: str | None = None,
    mode: str | None = None,
) -> list[dict[str, object]]:
    """Return the most recent operator alerts, newest first."""
    if limit <= 0 or not ALERTS_PATH.exists():
        return []

    normalized_profile = _normalize_profile_id(profile)
    normalized_mode = str(mode).strip().upper() if mode else None
    alerts: list[dict[str, object]] = []
    try:
        with ALERTS_PATH.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if normalized_profile is not None:
                    payload_profile = _normalize_profile_id(payload.get("profile"))
                    if payload_profile != normalized_profile:
                        continue
                if normalized_mode is not None:
                    payload_mode = str(payload.get("mode") or "").strip().upper()
                    if payload_mode != normalized_mode:
                        continue
                alerts.append(payload)
    except Exception:
        log.warning("Operator alert read failed", exc_info=True)
        return []

    if len(alerts) > limit:
        alerts = alerts[-limit:]
    alerts.reverse()
    return alerts


def summarize_operator_alerts(
    alerts: list[dict[str, object]],
    *,
    recent_window_minutes: int = RECENT_ALERT_WINDOW_MINUTES,
) -> dict[str, object]:
    counts = Counter({"critical": 0, "warning": 0, "info": 0})
    recent_counts = Counter({"critical": 0, "warning": 0, "info": 0})
    now = datetime.now(UTC)

    for alert in alerts:
        level = str(alert.get("level") or "info")
        counts[level] += 1
        timestamp_text = alert.get("timestamp_utc")
        if not isinstance(timestamp_text, str):
            continue
        try:
            timestamp = datetime.fromisoformat(timestamp_text)
        except ValueError:
            continue
        age_minutes = (now - timestamp).total_seconds() / 60.0
        if age_minutes <= recent_window_minutes:
            recent_counts[level] += 1

    latest = alerts[0] if alerts else None
    return {
        "total": len(alerts),
        "counts": dict(counts),
        "recent_window_minutes": recent_window_minutes,
        "recent_counts": dict(recent_counts),
        "latest": latest,
    }

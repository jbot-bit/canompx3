"""Unified operational lifecycle-state reader for live control surfaces.

This module consolidates three existing truths without changing their source:
- Criterion 11 deployment gate (`account_survival`)
- Criterion 12 SR monitor state (`sr_state.json`)
- persisted lane pauses (`lane_ctl`)

Consumers should read these surfaces through one shared interpretation layer
instead of re-implementing partial logic independently.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pipeline.paths import GOLD_DB_PATH
from trading_app.account_survival import check_survival_report_gate, read_survival_report_state
from trading_app.derived_state import (
    build_code_fingerprint,
    build_db_identity,
    build_profile_fingerprint,
    validate_state_envelope,
)
from trading_app.lane_ctl import get_lane_override, get_paused_strategy_ids
from trading_app.prop_profiles import get_profile, get_profile_lane_definitions, resolve_profile_id
from trading_app.sr_review_registry import get_sr_alarm_review

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SR_STATE_PATH = PROJECT_ROOT / "data" / "state" / "sr_state.json"


def _age_days_from_timestamp(timestamp: str | None, *, now: datetime | None = None) -> int | None:
    if not timestamp:
        return None
    try:
        dt = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    effective_now = now or datetime.now(UTC)
    return max(0, (effective_now - dt).days)


def _age_days_from_date(day: str | None, *, today: date | None = None) -> int | None:
    if not day:
        return None
    try:
        parsed = date.fromisoformat(str(day))
    except ValueError:
        return None
    effective_today = today or date.today()
    return max(0, (effective_today - parsed).days)


def read_criterion11_state(
    profile_id: str | None = None,
    *,
    db_path: Path = GOLD_DB_PATH,
    today: date | None = None,
) -> dict[str, Any]:
    """Read the current Criterion 11 gate/report state for one profile."""
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    state = read_survival_report_state(resolved_profile_id, db_path=db_path, today=today)
    gate_ok, gate_msg = check_survival_report_gate(profile_id=resolved_profile_id, db_path=db_path, today=today)
    summary = state.get("summary", {})
    generated_at = summary.get("generated_at_utc") if isinstance(summary, dict) else None
    return {
        "profile_id": resolved_profile_id,
        "available": state["available"],
        "valid": state["valid"],
        "reason": state["reason"],
        "gate_ok": gate_ok,
        "gate_msg": gate_msg,
        "as_of_date": summary.get("as_of_date") if isinstance(summary, dict) else None,
        "generated_at_utc": generated_at,
        "report_age_days": _age_days_from_timestamp(generated_at),
        "operational_pass_probability": summary.get("operational_pass_probability") if isinstance(summary, dict) else None,
        "n_paths": summary.get("n_paths") if isinstance(summary, dict) else None,
        "horizon_days": summary.get("horizon_days") if isinstance(summary, dict) else None,
        "gate_pass": summary.get("gate_pass") if isinstance(summary, dict) else None,
        "path_model": summary.get("path_model") if isinstance(summary, dict) else None,
    }


def read_criterion12_state(
    profile_id: str | None = None,
    *,
    db_path: Path = GOLD_DB_PATH,
    today: date | None = None,
) -> dict[str, Any]:
    """Read and validate the current Criterion 12 SR state for one profile."""
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    profile = get_profile(resolved_profile_id)
    lane_ids = [str(lane["strategy_id"]) for lane in get_profile_lane_definitions(resolved_profile_id)]

    base: dict[str, Any] = {
        "profile_id": resolved_profile_id,
        "available": SR_STATE_PATH.exists(),
        "valid": False,
        "reason": None,
        "state_date": None,
        "state_age_days": None,
        "counts": {},
        "stream_counts": {},
        "apply_pauses": False,
        "status_by_strategy": {},
        "alarm_strategy_ids": [],
        "no_data_strategy_ids": [],
    }
    if not SR_STATE_PATH.exists():
        base["reason"] = "missing"
        return base

    try:
        data = json.loads(SR_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        base["reason"] = f"unreadable: {exc}"
        return base

    valid, reason, envelope = validate_state_envelope(
        data,
        expected_schema_version=1,
        expected_state_type="sr_monitor",
        current_profile_id=resolved_profile_id,
        current_profile_fingerprint=build_profile_fingerprint(profile),
        current_lane_ids=lane_ids,
        current_db_identity=build_db_identity(db_path),
        current_code_fingerprint=build_code_fingerprint(
            [
                PROJECT_ROOT / "trading_app" / "sr_monitor.py",
                PROJECT_ROOT / "trading_app" / "live" / "sr_monitor.py",
                PROJECT_ROOT / "trading_app" / "derived_state.py",
            ]
        ),
        today=today or date.today(),
    )
    if not valid or envelope is None:
        base["reason"] = reason
        return base

    freshness = envelope["freshness"]
    payload = envelope["payload"]
    results = payload.get("results", [])
    counts: dict[str, int] = {}
    stream_counts: dict[str, int] = {}
    status_by_strategy: dict[str, str] = {}
    alarm_strategy_ids: list[str] = []
    no_data_strategy_ids: list[str] = []
    for row in results:
        strategy_id = str(row.get("strategy_id", ""))
        status = str(row.get("status", "UNKNOWN"))
        stream = str(row.get("stream_source", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        stream_counts[stream] = stream_counts.get(stream, 0) + 1
        if strategy_id:
            status_by_strategy[strategy_id] = status
            if status == "ALARM":
                alarm_strategy_ids.append(strategy_id)
            elif status == "NO_DATA":
                no_data_strategy_ids.append(strategy_id)

    state_date = freshness.get("as_of_date")
    base.update(
        {
            "valid": True,
            "reason": None,
            "state_date": state_date,
            "state_age_days": _age_days_from_date(state_date, today=today),
            "counts": counts,
            "stream_counts": stream_counts,
            "apply_pauses": bool(payload.get("apply_pauses")),
            "status_by_strategy": status_by_strategy,
            "alarm_strategy_ids": sorted(alarm_strategy_ids),
            "no_data_strategy_ids": sorted(no_data_strategy_ids),
        }
    )
    return base


def read_pause_state(
    profile_id: str | None = None,
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Read active persisted lane pauses for one profile."""
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    paused_ids = sorted(get_paused_strategy_ids(resolved_profile_id, as_of=as_of))
    paused_details: dict[str, dict[str, Any]] = {}
    for strategy_id in paused_ids:
        override = get_lane_override(resolved_profile_id, strategy_id) or {}
        paused_details[strategy_id] = dict(override)
    return {
        "profile_id": resolved_profile_id,
        "paused_count": len(paused_ids),
        "paused_strategy_ids": paused_ids,
        "paused_details": paused_details,
    }


def read_lifecycle_state(
    profile_id: str | None = None,
    *,
    db_path: Path = GOLD_DB_PATH,
    today: date | None = None,
) -> dict[str, Any]:
    """Read the unified operational lifecycle truth for one profile."""
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    criterion11 = read_criterion11_state(resolved_profile_id, db_path=db_path, today=today)
    criterion12 = read_criterion12_state(resolved_profile_id, db_path=db_path, today=today)
    pauses = read_pause_state(resolved_profile_id, as_of=today)
    lane_ids = [str(lane["strategy_id"]) for lane in get_profile_lane_definitions(resolved_profile_id)]

    strategy_ids = sorted(
        set(lane_ids)
        | set(pauses["paused_strategy_ids"])
        | set(criterion12["status_by_strategy"].keys())
    )

    strategy_states: dict[str, dict[str, Any]] = {}
    blocked_reason_by_strategy: dict[str, str] = {}
    blocked_strategy_ids: list[str] = []
    for strategy_id in strategy_ids:
        pause_info = pauses["paused_details"].get(strategy_id)
        sr_status = criterion12["status_by_strategy"].get(strategy_id)
        sr_review = get_sr_alarm_review(resolved_profile_id, strategy_id) if sr_status == "ALARM" else None
        if pause_info is not None:
            blocked = True
            block_source = "pause"
            block_reason = str(pause_info.get("reason") or "Paused pending manual review")
        elif sr_status == "ALARM" and sr_review is not None and sr_review.outcome == "watch":
            blocked = False
            block_source = None
            block_reason = None
        elif sr_status == "ALARM" and sr_review is not None and sr_review.outcome == "pause":
            blocked = True
            block_source = "sr_review_pause"
            block_reason = sr_review.summary
        elif sr_status == "ALARM":
            blocked = True
            block_source = "sr_alarm"
            block_reason = "Criterion 12 SR ALARM — manual review required"
        else:
            blocked = False
            block_source = None
            block_reason = None

        strategy_states[strategy_id] = {
            "paused": pause_info is not None,
            "pause_reason": pause_info.get("reason") if pause_info else None,
            "pause_source": pause_info.get("source") if pause_info else None,
            "sr_status": sr_status,
            "sr_review_outcome": sr_review.outcome if sr_review is not None else None,
            "sr_review_summary": sr_review.summary if sr_review is not None else None,
            "sr_reviewed_at": sr_review.reviewed_at if sr_review is not None else None,
            "sr_recheck_trigger": sr_review.recheck_trigger if sr_review is not None else None,
            "blocked": blocked,
            "block_source": block_source,
            "block_reason": block_reason,
        }
        if blocked and block_reason is not None:
            blocked_strategy_ids.append(strategy_id)
            blocked_reason_by_strategy[strategy_id] = block_reason

    return {
        "profile_id": resolved_profile_id,
        "lane_ids": lane_ids,
        "criterion11": criterion11,
        "criterion12": criterion12,
        "pauses": pauses,
        "strategy_states": strategy_states,
        "blocked_strategy_ids": sorted(blocked_strategy_ids),
        "blocked_reason_by_strategy": blocked_reason_by_strategy,
    }

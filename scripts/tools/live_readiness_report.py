#!/usr/bin/env python3
"""One-command live-readiness report for an operator profile.

Read-only aggregation over canonical live-control surfaces:
- deployment vs validated-active truth
- Criterion 11 account-survival state
- Criterion 12 SR monitor state
- allocator lane state and rebalance provenance
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _preferred_repo_python() -> Path | None:
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
    return candidate if candidate.exists() else None


def _preferred_repo_prefix(expected_python: Path) -> Path:
    return expected_python.parent.parent.resolve()


def _ensure_repo_python() -> None:
    if "pytest" in sys.modules:
        return
    expected_python = _preferred_repo_python()
    if expected_python is None:
        return
    current_prefix = Path(sys.prefix).resolve()
    expected_prefix = _preferred_repo_prefix(expected_python)
    if current_prefix == expected_prefix or os.environ.get("CANOMPX3_BOOTSTRAP_DONE") == "1":
        return

    env = os.environ.copy()
    env["CANOMPX3_BOOTSTRAP_DONE"] = "1"
    env.setdefault("CANOMPX3_BOOTSTRAPPED_FROM", str(Path(sys.executable).resolve()))
    raise SystemExit(
        subprocess.call(
            [str(expected_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    )


_ensure_repo_python()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.lifecycle_state import read_lifecycle_state  # noqa: E402
from trading_app.live.telemetry_maturity import (  # noqa: E402
    VERDICT_MATURE,
    TelemetryMaturityReport,
    evaluate_telemetry_maturity,
)
from trading_app.prop_profiles import (  # noqa: E402
    get_profile_lane_definitions,
    legacy_lane_allocation_path,
    resolve_profile_id,
)
from trading_app.validated_shelf import deployable_validated_relation  # noqa: E402

# Defined after the bootstrap-deferred trading_app import so the resolver
# helper is in scope (the canonical legacy path lives in prop_profiles per
# Stage 1b authority inversion).
DEFAULT_ALLOCATION_PATH = legacy_lane_allocation_path()
DEFAULT_TELEMETRY_INSTRUMENT = "MNQ"
DEFAULT_SIGNALS_DIR = PROJECT_ROOT
LIVE_STAGE_PATHS: tuple[str, ...] = (
    "docs/runtime/stages/2026-05-22-live-bar-ring-chart.md",
    "docs/runtime/stages/2026-05-26-ring-orphan-startup-sweep.md",
)


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_branch(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def _load_validated_strategy_ids(db_path: Path) -> list[str]:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        shelf_relation = deployable_validated_relation(con)
        rows = con.execute(f"SELECT strategy_id FROM {shelf_relation} ORDER BY strategy_id").fetchall()
        return [str(row[0]) for row in rows]
    finally:
        con.close()


def _normalize_lane_row(
    row: dict[str, Any],
    *,
    bucket: str,
    strategy_states: dict[str, dict[str, Any]],
    blocked_reason_by_strategy: dict[str, str],
) -> dict[str, Any]:
    strategy_id = str(row.get("strategy_id", ""))
    state = strategy_states.get(strategy_id, {})
    normalized = {
        "strategy_id": strategy_id,
        "instrument": row.get("instrument"),
        "orb_label": row.get("orb_label"),
        "orb_minutes": row.get("orb_minutes"),
        "rr_target": row.get("rr_target"),
        "filter_type": row.get("filter_type"),
        "allocator_bucket": bucket,
        "status": row.get("status"),
        "status_reason": row.get("status_reason"),
        "chordia_verdict": row.get("chordia_verdict"),
        "chordia_audit_age_days": row.get("chordia_audit_age_days"),
        "lifecycle_blocked": bool(state.get("blocked")),
        "lifecycle_block_source": state.get("block_source"),
        "lifecycle_block_reason": blocked_reason_by_strategy.get(strategy_id) or state.get("block_reason"),
        "sr_status": state.get("sr_status"),
        "sr_review_outcome": state.get("sr_review_outcome"),
        "sr_review_summary": state.get("sr_review_summary"),
        "sr_reviewed_at": state.get("sr_reviewed_at"),
        "sr_recheck_trigger": state.get("sr_recheck_trigger"),
        "paused": bool(state.get("paused")),
        "pause_reason": state.get("pause_reason"),
    }
    return normalized


def _normalize_profile_lane_row(
    row: dict[str, Any],
    *,
    strategy_states: dict[str, dict[str, Any]],
    blocked_reason_by_strategy: dict[str, str],
) -> dict[str, Any]:
    strategy_id = str(row.get("strategy_id", ""))
    state = strategy_states.get(strategy_id, {})
    return {
        "strategy_id": strategy_id,
        "instrument": row.get("instrument"),
        "orb_label": row.get("orb_label"),
        "orb_minutes": row.get("orb_minutes"),
        "rr_target": row.get("rr_target"),
        "filter_type": row.get("filter_type"),
        "allocator_bucket": "profile_config",
        "status": "configured",
        "status_reason": None,
        "chordia_verdict": None,
        "chordia_audit_age_days": None,
        "lifecycle_blocked": bool(state.get("blocked")),
        "lifecycle_block_source": state.get("block_source"),
        "lifecycle_block_reason": blocked_reason_by_strategy.get(strategy_id) or state.get("block_reason"),
        "sr_status": state.get("sr_status"),
        "sr_review_outcome": state.get("sr_review_outcome"),
        "sr_review_summary": state.get("sr_review_summary"),
        "sr_reviewed_at": state.get("sr_reviewed_at"),
        "sr_recheck_trigger": state.get("sr_recheck_trigger"),
        "paused": bool(state.get("paused")),
        "pause_reason": state.get("pause_reason"),
    }


def _load_allocator_summary(
    profile_id: str,
    allocation_path: Path,
    strategy_states: dict[str, dict[str, Any]],
    blocked_reason_by_strategy: dict[str, str],
    profile_lane_ids: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": allocation_path.exists(),
        "source_path": str(allocation_path),
        "profile_match": None,
        "allocation_profile_id": None,
        "rebalance_date": None,
        "trailing_window_months": None,
        "all_scores_count": None,
        "active_lanes": [],
        "paused_lanes": [],
        "stale_lanes": [],
        "profile_lanes_missing_from_allocator": [],
    }
    if not allocation_path.exists():
        return summary

    try:
        data = json.loads(allocation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        summary["error"] = f"unreadable: {exc}"
        return summary

    allocation_profile_id = data.get("profile_id")
    summary["allocation_profile_id"] = allocation_profile_id
    summary["profile_match"] = allocation_profile_id == profile_id
    summary["rebalance_date"] = data.get("rebalance_date")
    summary["trailing_window_months"] = data.get("trailing_window_months")
    summary["all_scores_count"] = data.get("all_scores_count")
    for row in data.get("lanes", []):
        normalized = _normalize_lane_row(
            row,
            bucket="lanes",
            strategy_states=strategy_states,
            blocked_reason_by_strategy=blocked_reason_by_strategy,
        )
        status = str(row.get("status") or "").upper()
        if status in {"DEPLOY", "PROVISIONAL"}:
            summary["active_lanes"].append(normalized)
        elif status in {"PAUSE", "PAUSED"}:
            summary["paused_lanes"].append(normalized)
        elif status == "STALE":
            summary["stale_lanes"].append(normalized)
        else:
            summary["stale_lanes"].append(normalized)
    summary["paused_lanes"].extend(
        _normalize_lane_row(
            row,
            bucket="paused",
            strategy_states=strategy_states,
            blocked_reason_by_strategy=blocked_reason_by_strategy,
        )
        for row in data.get("paused", [])
    )
    summary["stale_lanes"].extend(
        _normalize_lane_row(
            row,
            bucket="stale",
            strategy_states=strategy_states,
            blocked_reason_by_strategy=blocked_reason_by_strategy,
        )
        for row in data.get("stale", [])
    )

    allocator_ids = {
        lane["strategy_id"]
        for lane in summary["active_lanes"] + summary["paused_lanes"] + summary["stale_lanes"]
        if lane.get("strategy_id")
    }
    summary["profile_lanes_missing_from_allocator"] = sorted(set(profile_lane_ids) - allocator_ids)
    return summary


def _discover_active_instrument(
    active_lanes: list[dict[str, Any]],
    profile_lanes: list[dict[str, Any]],
) -> str:
    instruments = {
        str(lane.get("instrument")).strip().upper()
        for lane in [*active_lanes, *profile_lanes]
        if lane.get("instrument")
    }
    if len(instruments) == 1:
        return next(iter(instruments))
    return DEFAULT_TELEMETRY_INSTRUMENT


def _normalize_telemetry_maturity(
    telemetry: TelemetryMaturityReport | dict[str, Any],
    instrument: str,
    profile_id: str,
) -> dict[str, Any]:
    if isinstance(telemetry, TelemetryMaturityReport):
        trading_days = [day.isoformat() for day in telemetry.trading_days]
        return {
            "instrument": telemetry.instrument,
            "profile_id": telemetry.profile_id or profile_id,
            "scope": "profile" if telemetry.profile_scoped else "instrument_global",
            "profile_scoped": telemetry.profile_scoped,
            "verdict": telemetry.verdict,
            "n_unique_trading_days": telemetry.n_unique_trading_days,
            "min_required": telemetry.min_required,
            "trading_days": trading_days,
            "signal_files_scanned": telemetry.signal_files_scanned,
            "records_scanned": telemetry.records_scanned,
            "records_qualifying": telemetry.records_qualifying,
        }

    trading_days = telemetry.get("trading_days") or []
    normalized_days = [str(day) for day in trading_days]
    return {
        "instrument": str(telemetry.get("instrument") or instrument),
        "profile_id": str(telemetry.get("profile_id") or profile_id),
        "scope": str(telemetry.get("scope") or "instrument_global"),
        "profile_scoped": bool(telemetry.get("profile_scoped")),
        "verdict": str(telemetry.get("verdict") or ""),
        "n_unique_trading_days": int(telemetry.get("n_unique_trading_days") or 0),
        "min_required": int(telemetry.get("min_required") or 0),
        "trading_days": normalized_days,
        "signal_files_scanned": int(telemetry.get("signal_files_scanned") or 0),
        "records_scanned": int(telemetry.get("records_scanned") or 0),
        "records_qualifying": int(telemetry.get("records_qualifying") or 0),
    }


def _load_telemetry_maturity_summary(
    profile_id: str,
    active_lanes: list[dict[str, Any]],
    profile_lanes: list[dict[str, Any]],
) -> dict[str, Any]:
    instrument = _discover_active_instrument(active_lanes, profile_lanes)
    telemetry = evaluate_telemetry_maturity(DEFAULT_SIGNALS_DIR, instrument=instrument, profile_id=profile_id)
    return _normalize_telemetry_maturity(telemetry, instrument, profile_id)


def _extract_stage_status_fields(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    status_fields: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        match = re.match(r"^(mode|status|implementation_status)\s*:\s*(.+?)\s*$", stripped, re.IGNORECASE)
        if match is None:
            continue
        key = match.group(1).lower()
        value = match.group(2).strip().strip("'\"")
        status_fields[key] = value
    return status_fields


def _read_stage_acceptance(path_text: str) -> dict[str, Any]:
    path = PROJECT_ROOT / path_text
    summary: dict[str, Any] = {
        "path": path_text,
        "green": False,
        "status_fields": {},
        "status_text": None,
    }
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        summary["status_text"] = f"UNREADABLE: {exc}"
        return summary

    status_fields = _extract_stage_status_fields(text)
    summary["status_fields"] = status_fields
    if not status_fields:
        summary["status_text"] = "NO_STATUS_FIELDS"
        return summary

    normalized_values = [value.upper() for value in status_fields.values()]
    summary["status_text"] = ", ".join(f"{field}={status_fields[field]}" for field in sorted(status_fields))
    has_closed = any(value == "CLOSED" for value in normalized_values)
    has_pending_or_implementation = any(
        ("PENDING" in value) or ("IMPLEMENT" in value) for value in normalized_values if value != "CLOSED"
    )
    summary["green"] = has_closed and not has_pending_or_implementation
    return summary


def _evaluate_live_stage_acceptance() -> dict[str, Any]:
    stages = [_read_stage_acceptance(path) for path in LIVE_STAGE_PATHS]
    return {
        "green": all(stage.get("green") is True for stage in stages),
        "stages": stages,
    }


def _build_strict_zero_warn_summary(
    *,
    deployment_summary: dict[str, Any],
    criterion11: dict[str, Any] | None,
    criterion12: dict[str, Any] | None,
    allocator_summary: dict[str, Any],
    active_lanes: list[dict[str, Any]],
    telemetry_maturity: dict[str, Any],
    live_stage_acceptance: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    c11 = criterion11 or {}
    c12 = criterion12 or {}

    if c11.get("gate_ok") is not True:
        blockers.append("Criterion 11 gate not OK")

    if c12.get("valid") is not True:
        blockers.append("Criterion 12 invalid")

    alarm_count = int((c12.get("counts") or {}).get("ALARM", 0) or 0)
    if alarm_count > 0:
        blockers.append(f"Criterion 12 alarm count > 0 ({alarm_count})")

    if allocator_summary.get("available") is not True:
        blockers.append("Allocator unavailable")
    elif allocator_summary.get("profile_match") is not True:
        blockers.append("Allocator profile mismatch")

    deployed_not_validated = deployment_summary.get("deployed_not_validated") or []
    if deployed_not_validated:
        blockers.append(
            "Deployed not validated: " + ", ".join(str(strategy_id) for strategy_id in deployed_not_validated)
        )

    for lane in active_lanes:
        strategy_id = str(lane.get("strategy_id") or "unknown")
        if lane.get("lifecycle_blocked"):
            blockers.append(f"Active lane lifecycle blocked: {strategy_id}")
        if str(lane.get("sr_status") or "").upper() == "ALARM":
            review_outcome = lane.get("sr_review_outcome")
            review_note = "watch reviewed" if review_outcome == "watch" else "no watch review"
            blockers.append(f"Active lane SR alarm ({review_note}): {strategy_id}")

    if str(telemetry_maturity.get("verdict") or "") != VERDICT_MATURE:
        blockers.append(
            "Telemetry not mature: "
            f"{telemetry_maturity.get('verdict')} "
            f"({telemetry_maturity.get('n_unique_trading_days')}/{telemetry_maturity.get('min_required')} trading days)"
        )

    if telemetry_maturity.get("profile_scoped") is not True:
        blockers.append(
            "Telemetry not profile-scoped: "
            f"profile={telemetry_maturity.get('profile_id')} "
            f"scope={telemetry_maturity.get('scope')}"
        )

    for stage in live_stage_acceptance.get("stages", []):
        if stage.get("green") is not True:
            blockers.append(f"Live stage not green: {stage.get('path')}")

    return {
        "green": not blockers,
        "blockers": blockers,
    }


def build_live_readiness_report(
    profile_id: str | None = None,
    *,
    db_path: Path = GOLD_DB_PATH,
    allocation_path: Path = DEFAULT_ALLOCATION_PATH,
) -> dict[str, Any]:
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    profile_lanes = get_profile_lane_definitions(resolved_profile_id)
    profile_lane_ids = [str(lane["strategy_id"]) for lane in profile_lanes]
    validated_ids = _load_validated_strategy_ids(db_path)
    lifecycle = read_lifecycle_state(resolved_profile_id, db_path=db_path)

    blocked_reason_by_strategy = lifecycle.get("blocked_reason_by_strategy", {})
    strategy_states = lifecycle.get("strategy_states", {})
    allocator_summary = _load_allocator_summary(
        resolved_profile_id,
        allocation_path,
        strategy_states,
        blocked_reason_by_strategy,
        profile_lane_ids,
    )

    # Fail-closed on profile mismatch: if the allocator JSON belongs to a
    # different profile, do NOT surface its lanes as the active set — that
    # would silently render the wrong profile's strategies under the
    # requested profile's banner. Fall back to profile_config lanes; the
    # mismatch stays visible via allocator_summary["profile_match"] so
    # operators see the integrity problem in the same report.
    if allocator_summary.get("active_lanes") and allocator_summary.get("profile_match") is True:
        active_lanes = allocator_summary["active_lanes"]
    else:
        active_lanes = [
            _normalize_profile_lane_row(
                lane,
                strategy_states=strategy_states,
                blocked_reason_by_strategy=blocked_reason_by_strategy,
            )
            for lane in profile_lanes
        ]

    deployed_set = set(profile_lane_ids)
    validated_set = set(validated_ids)
    deployment_summary = {
        "profile_id": resolved_profile_id,
        "deployed_count": len(profile_lane_ids),
        "validated_active_count": len(validated_ids),
        "deployed_not_validated": sorted(deployed_set - validated_set),
        "validated_not_deployed": sorted(validated_set - deployed_set),
    }
    telemetry_maturity = _load_telemetry_maturity_summary(resolved_profile_id, active_lanes, profile_lanes)
    live_stage_acceptance = _evaluate_live_stage_acceptance()
    strict_zero_warn = _build_strict_zero_warn_summary(
        deployment_summary=deployment_summary,
        criterion11=lifecycle.get("criterion11"),
        criterion12=lifecycle.get("criterion12"),
        allocator_summary=allocator_summary,
        active_lanes=active_lanes,
        telemetry_maturity=telemetry_maturity,
        live_stage_acceptance=live_stage_acceptance,
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "profile_id": resolved_profile_id,
        "git_branch": _git_branch(PROJECT_ROOT),
        "git_head": _git_head(PROJECT_ROOT),
        "db_path": str(db_path),
        "allocation_path": str(allocation_path),
        "deployment_summary": deployment_summary,
        "criterion11": lifecycle.get("criterion11"),
        "criterion12": lifecycle.get("criterion12"),
        "pauses": lifecycle.get("pauses"),
        "blocked_strategy_ids": lifecycle.get("blocked_strategy_ids", []),
        "blocked_reason_by_strategy": blocked_reason_by_strategy,
        "active_lanes": active_lanes,
        "allocator_summary": allocator_summary,
        "telemetry_maturity": telemetry_maturity,
        "live_stage_acceptance": live_stage_acceptance,
        "strict_zero_warn": strict_zero_warn,
        "conditional_overlays": lifecycle.get("conditional_overlays"),
    }


def _render_text(report: dict[str, Any]) -> str:
    deployment = report["deployment_summary"]
    c11 = report["criterion11"] or {}
    c12 = report["criterion12"] or {}
    allocator = report["allocator_summary"] or {}
    telemetry = report.get("telemetry_maturity") or {}
    live_stage_acceptance = report.get("live_stage_acceptance") or {}
    strict_zero_warn = report.get("strict_zero_warn") or {}

    lines = [
        f"Live Readiness | profile={report['profile_id']} | git={report.get('git_head') or 'unknown'}",
        (
            "Deployment: "
            f"deployed={deployment['deployed_count']} "
            f"validated_active={deployment['validated_active_count']} "
            f"validated_only={len(deployment['validated_not_deployed'])} "
            f"deployed_not_validated={len(deployment['deployed_not_validated'])}"
        ),
        (
            "Criterion 11: "
            f"gate_ok={bool(c11.get('gate_ok'))} "
            f"age_days={c11.get('report_age_days')} "
            f"msg={c11.get('gate_msg')}"
        ),
        (
            "Criterion 12: "
            f"valid={bool(c12.get('valid'))} "
            f"alarms={c12.get('counts', {}).get('ALARM', 0)} "
            f"state_age_days={c12.get('state_age_days')}"
        ),
        (
            "Allocator: "
            f"available={bool(allocator.get('available'))} "
            f"profile_match={allocator.get('profile_match')} "
            f"rebalance_date={allocator.get('rebalance_date')} "
            f"lanes={len(allocator.get('active_lanes', []))} "
            f"paused={len(allocator.get('paused_lanes', []))} "
            f"stale={len(allocator.get('stale_lanes', []))}"
        ),
        (
            "Strict zero-warn: "
            f"green={bool(strict_zero_warn.get('green'))} "
            f"blockers={len(strict_zero_warn.get('blockers', []))}"
        ),
        (
            "Telemetry: "
            f"instrument={telemetry.get('instrument')} "
            f"profile={telemetry.get('profile_id')} "
            f"scope={telemetry.get('scope')} "
            f"verdict={telemetry.get('verdict')} "
            f"trading_days={telemetry.get('n_unique_trading_days')}/{telemetry.get('min_required')} "
            f"files={telemetry.get('signal_files_scanned')} "
            f"records={telemetry.get('records_scanned')} "
            f"qualifying={telemetry.get('records_qualifying')}"
        ),
        "Live stages:",
    ]

    for stage in live_stage_acceptance.get("stages", []):
        lines.append(f"  - {stage.get('path')} green={stage.get('green')} status={stage.get('status_text') or '-'}")

    if strict_zero_warn.get("blockers"):
        lines.extend(["Strict blockers:"])
        for blocker in strict_zero_warn["blockers"]:
            lines.append(f"  - {blocker}")

    lines.extend(
        [
            "Active lanes:",
        ]
    )

    for lane in report.get("active_lanes", []):
        lines.append(
            "  - "
            f"{lane['strategy_id']} [{lane.get('instrument')}/{lane.get('orb_label')}] "
            f"blocked={lane.get('lifecycle_blocked')} "
            f"sr={lane.get('sr_status')} "
            f"reason={lane.get('lifecycle_block_reason') or lane.get('status_reason') or '-'}"
        )

    paused_or_stale = allocator.get("paused_lanes", []) + allocator.get("stale_lanes", [])
    if paused_or_stale:
        lines.append("Paused/stale lanes:")
        for lane in paused_or_stale:
            lines.append(
                "  - "
                f"{lane['strategy_id']} [{lane.get('allocator_bucket')}] "
                f"reason={lane.get('status_reason') or lane.get('lifecycle_block_reason') or '-'}"
            )

    return "\n".join(lines)


def _render_markdown(report: dict[str, Any]) -> str:
    deployment = report["deployment_summary"]
    c11 = report["criterion11"] or {}
    c12 = report["criterion12"] or {}
    allocator = report["allocator_summary"] or {}
    telemetry = report.get("telemetry_maturity") or {}
    live_stage_acceptance = report.get("live_stage_acceptance") or {}
    strict_zero_warn = report.get("strict_zero_warn") or {}

    lines = [
        f"# Live Readiness Report — `{report['profile_id']}`",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Git: `{report.get('git_head') or 'unknown'}` on `{report.get('git_branch') or 'unknown'}`",
        f"- DB: `{report['db_path']}`",
        f"- Allocator source: `{report['allocation_path']}`",
        "",
        "## Deployment",
        "",
        f"- Deployed lanes: `{deployment['deployed_count']}`",
        f"- Validated-active lanes: `{deployment['validated_active_count']}`",
        f"- Validated not deployed: `{len(deployment['validated_not_deployed'])}`",
        f"- Deployed not validated: `{len(deployment['deployed_not_validated'])}`",
        "",
        "## Criterion 11",
        "",
        f"- Gate OK: `{bool(c11.get('gate_ok'))}`",
        f"- Report age days: `{c11.get('report_age_days')}`",
        f"- Gate message: `{c11.get('gate_msg')}`",
        "",
        "## Criterion 12",
        "",
        f"- Valid: `{bool(c12.get('valid'))}`",
        f"- Alarm count: `{c12.get('counts', {}).get('ALARM', 0)}`",
        f"- State age days: `{c12.get('state_age_days')}`",
        "",
        "## Allocator",
        "",
        f"- Available: `{bool(allocator.get('available'))}`",
        f"- Profile match: `{allocator.get('profile_match')}`",
        f"- Rebalance date: `{allocator.get('rebalance_date')}`",
        f"- Active lanes: `{len(allocator.get('active_lanes', []))}`",
        f"- Paused lanes: `{len(allocator.get('paused_lanes', []))}`",
        f"- Stale lanes: `{len(allocator.get('stale_lanes', []))}`",
        "",
        "## Strict zero-warn",
        "",
        f"- Green: `{bool(strict_zero_warn.get('green'))}`",
        f"- Blockers: `{len(strict_zero_warn.get('blockers', []))}`",
        "",
        "## Telemetry",
        "",
        f"- Instrument: `{telemetry.get('instrument')}`",
        f"- Profile: `{telemetry.get('profile_id')}`",
        f"- Scope: `{telemetry.get('scope')}`",
        f"- Profile scoped: `{bool(telemetry.get('profile_scoped'))}`",
        f"- Verdict: `{telemetry.get('verdict')}`",
        f"- Trading days: `{telemetry.get('n_unique_trading_days')}` / `{telemetry.get('min_required')}`",
        f"- Files scanned: `{telemetry.get('signal_files_scanned')}`",
        f"- Records scanned: `{telemetry.get('records_scanned')}`",
        f"- Records qualifying: `{telemetry.get('records_qualifying')}`",
        "",
        "## Live stage acceptance",
        "",
    ]

    for stage in live_stage_acceptance.get("stages", []):
        lines.append(f"- `{stage.get('path')}` green=`{stage.get('green')}` status=`{stage.get('status_text') or '-'}`")

    if strict_zero_warn.get("blockers"):
        lines.extend(["", "## Strict blockers", ""])
        for blocker in strict_zero_warn["blockers"]:
            lines.append(f"- `{blocker}`")

    lines.extend(
        [
            "## Active Lanes",
            "",
        ]
    )

    for lane in report.get("active_lanes", []):
        lines.append(
            "- "
            f"`{lane['strategy_id']}` "
            f"{lane.get('instrument')}/{lane.get('orb_label')} "
            f"blocked=`{lane.get('lifecycle_blocked')}` "
            f"sr=`{lane.get('sr_status')}` "
            f"reason=`{lane.get('lifecycle_block_reason') or lane.get('status_reason') or '-'}`"
        )

    paused_or_stale = allocator.get("paused_lanes", []) + allocator.get("stale_lanes", [])
    if paused_or_stale:
        lines.extend(["", "## Paused / Stale", ""])
        for lane in paused_or_stale:
            lines.append(
                "- "
                f"`{lane['strategy_id']}` "
                f"bucket=`{lane.get('allocator_bucket')}` "
                f"reason=`{lane.get('status_reason') or lane.get('lifecycle_block_reason') or '-'}`"
            )

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit a one-command live-readiness report for a profile.")
    parser.add_argument("--profile", default=None, help="Profile id. Defaults to the repo's active profile resolver.")
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format.",
    )
    parser.add_argument("--out", default=None, help="Optional output file path.")
    parser.add_argument(
        "--allocation-path",
        default=str(DEFAULT_ALLOCATION_PATH),
        help="Path to lane allocation JSON file.",
    )
    parser.add_argument(
        "--strict-zero-warn",
        action="store_true",
        help="Exit nonzero when the strict zero-warn gate is not green.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = build_live_readiness_report(
        profile_id=args.profile,
        db_path=GOLD_DB_PATH,
        allocation_path=Path(args.allocation_path),
    )

    if args.format == "json":
        rendered = json.dumps(report, indent=2, sort_keys=True)
    elif args.format == "markdown":
        rendered = _render_markdown(report)
    else:
        rendered = _render_text(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        print(rendered)

    if args.strict_zero_warn and not bool((report.get("strict_zero_warn") or {}).get("green")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

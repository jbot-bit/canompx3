"""Shadow-only opportunity awareness for profile lanes.

This module ranks currently configured profile lanes for operator visibility.
It does not authorize trades, change allocation, or modify execution sizing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pipeline.paths import GOLD_DB_PATH
from trading_app.chordia import chordia_verdict_allows_deploy
from trading_app.derived_state import (
    build_code_fingerprint,
    build_db_identity,
    build_profile_fingerprint,
    build_state_envelope,
    get_git_head,
    validate_state_envelope,
)
from trading_app.prop_profiles import (
    get_profile,
    get_profile_lane_definitions,
    resolve_allocation_json,
    resolve_profile_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "data" / "state"

OPPORTUNITY_STATE_TYPE = "opportunity_awareness_shadow"
OPPORTUNITY_STATE_SCHEMA_VERSION = 1
OPPORTUNITY_MAX_AGE_DAYS = 1


@dataclass(frozen=True)
class OpportunityLane:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int | None
    entry_model: str | None
    rr_target: float | None
    confirm_bars: int | None
    filter_type: str | None
    opportunity_tier: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    fitness_state: str
    filter_state: str
    risk_state: str
    portfolio_collision: str
    allocation_status: str | None
    allocation_reason: str | None
    trailing_expr: float | None
    session_regime: str | None
    chordia_verdict: str | None


@dataclass(frozen=True)
class OpportunitySnapshot:
    profile_id: str
    trading_day: str
    summary: dict[str, int]
    lanes: tuple[OpportunityLane, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_opportunity_state_path(profile_id: str) -> Path:
    return STATE_DIR / f"opportunity_awareness_{profile_id}.json"


def _lane_code_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        PROJECT_ROOT / "trading_app" / "derived_state.py",
        PROJECT_ROOT / "trading_app" / "lifecycle_state.py",
    ]


def _as_lane_dict(lane: Any) -> dict[str, Any]:
    if isinstance(lane, dict):
        return dict(lane)
    return {
        "strategy_id": getattr(lane, "strategy_id", None),
        "instrument": getattr(lane, "instrument", None),
        "orb_label": getattr(lane, "orb_label", None),
        "orb_minutes": getattr(lane, "orb_minutes", None),
        "entry_model": getattr(lane, "entry_model", None),
        "rr_target": getattr(lane, "rr_target", None),
        "confirm_bars": getattr(lane, "confirm_bars", None),
        "filter_type": getattr(lane, "filter_type", None),
    }


def _load_allocation_payload(profile_id: str) -> dict[str, Any]:
    """Load allocation payload via the canonical resolver.

    Stage 1b authority inversion: ``resolve_allocation_json`` is the single
    owner of path resolution (new-path-first, legacy fallback,
    profile-mismatch fail-closed, multi-file ambiguity hard-fail). The
    "missing" / "error" envelope shape is preserved for the existing
    ``allocation_problem`` surface in ``build_snapshot``.
    """
    result = resolve_allocation_json(profile_id)
    if result.data is None:
        return {"lanes": [], "paused": [], "missing": True}
    return result.data


def _index_by_strategy(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        strategy_id = str(row.get("strategy_id") or "")
        if strategy_id:
            indexed[strategy_id] = dict(row)
    return indexed


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _lane_display(row: dict[str, Any], *, reason_keys: tuple[str, ...] = ()) -> str:
    label = "/".join(
        part
        for part in (
            str(row.get("orb_label") or "").strip(),
            str(row.get("instrument") or "").strip(),
        )
        if part
    )
    if not label:
        label = str(row.get("strategy_id") or "unknown")

    reasons: list[str] = []
    for key in reason_keys:
        value = row.get(key)
        if isinstance(value, list | tuple) and value:
            reasons.append(str(value[0]))
        elif value:
            reasons.append(str(value))
    if reasons:
        return f"{label} ({reasons[0]})"

    trailing_expr = _coerce_float(row.get("trailing_expr"))
    if trailing_expr is not None:
        return f"{label} expr={trailing_expr:.3f}"
    return label


def _tier_detail(
    lanes: list[dict[str, Any]],
    *,
    tier: str,
    label: str,
    reason_keys: tuple[str, ...] = (),
    limit: int = 3,
) -> str | None:
    tier_rows = [row for row in lanes if row.get("opportunity_tier") == tier]
    if not tier_rows:
        return None
    shown = [_lane_display(row, reason_keys=reason_keys) for row in tier_rows[:limit]]
    if len(tier_rows) > limit:
        shown.append(f"+{len(tier_rows) - limit} more")
    return f"{label}: {', '.join(shown)}"


def describe_opportunity_awareness(state: dict[str, Any]) -> tuple[str, str]:
    """Return dashboard/pre-session status and compact lane-aware detail."""
    if not state.get("available"):
        return "info", "Opportunity awareness unavailable"

    if not state.get("valid"):
        return "warn", f"Opportunity awareness invalid ({state.get('reason') or 'invalid'})"

    summary = dict(state.get("summary") or {})
    prime = _coerce_int(summary.get("prime_shadow_count"))
    watch = _coerce_int(summary.get("watch_count"))
    blocked = _coerce_int(summary.get("blocked_count"))
    total = _coerce_int(summary.get("lane_count"))
    detail = f"Opportunity awareness: {prime} PRIME_SHADOW, {watch} WATCH, {blocked} BLOCKED, {total} total"

    lanes = [dict(row) for row in state.get("lanes", []) if isinstance(row, dict)]
    tier_details = [
        _tier_detail(lanes, tier="PRIME_SHADOW", label="prime"),
        _tier_detail(lanes, tier="WATCH", label="watch", reason_keys=("warnings",)),
        _tier_detail(lanes, tier="BLOCKED", label="blocked", reason_keys=("blockers",)),
    ]
    detail_parts = [part for part in tier_details if part]
    if detail_parts:
        detail = f"{detail}; {'; '.join(detail_parts)}"

    status = "warn" if blocked or watch else "info"
    return status, detail


def _build_lane(
    lane: dict[str, Any],
    *,
    allocation_row: dict[str, Any] | None,
    paused_row: dict[str, Any] | None,
    strategy_state: dict[str, Any],
    allocation_problem: str | None,
) -> OpportunityLane:
    strategy_id = str(lane.get("strategy_id") or "")
    source = allocation_row or lane
    blockers: list[str] = []
    warnings: list[str] = []

    if paused_row is not None:
        blockers.append(str(paused_row.get("reason") or "Paused by lane allocation"))

    if strategy_state.get("blocked"):
        blockers.append(str(strategy_state.get("block_reason") or "Blocked by lifecycle state"))

    if allocation_problem is not None:
        warnings.append(allocation_problem)

    sr_status = strategy_state.get("sr_status")
    if sr_status and sr_status not in {"CONTINUE", "NO_DATA"}:
        warnings.append(f"Criterion 12 SR status: {sr_status}")

    allocation_status = source.get("status")
    if allocation_status == "PROVISIONAL":
        warnings.append("Allocation status is PROVISIONAL")

    chordia_verdict = source.get("chordia_verdict")
    if chordia_verdict and not chordia_verdict_allows_deploy(chordia_verdict):
        warnings.append(f"Chordia gate: {chordia_verdict}")

    trailing_expr = _coerce_float(source.get("trailing_expr"))
    session_regime = source.get("session_regime")
    if session_regime == "COLD":
        warnings.append("Session regime COLD")

    if blockers:
        opportunity_tier = "BLOCKED"
    elif warnings:
        opportunity_tier = "WATCH"
    elif (
        allocation_status == "DEPLOY"
        and session_regime == "HOT"
        and chordia_verdict_allows_deploy(chordia_verdict)
        and trailing_expr is not None
        and trailing_expr >= 0.20
    ):
        opportunity_tier = "PRIME_SHADOW"
    else:
        opportunity_tier = "NORMAL"

    p90_orb = source.get("p90_orb_pts")
    risk_state = f"p90_orb_pts={p90_orb}" if p90_orb is not None else "risk envelope unknown"
    filter_type = source.get("filter_type") or lane.get("filter_type")

    return OpportunityLane(
        strategy_id=strategy_id,
        instrument=str(source.get("instrument") or lane.get("instrument") or ""),
        orb_label=str(source.get("orb_label") or lane.get("orb_label") or ""),
        orb_minutes=source.get("orb_minutes") or lane.get("orb_minutes"),
        entry_model=source.get("entry_model") or lane.get("entry_model"),
        rr_target=_coerce_float(
            source.get("rr_target") if source.get("rr_target") is not None else lane.get("rr_target")
        ),
        confirm_bars=source.get("confirm_bars") or lane.get("confirm_bars"),
        filter_type=str(filter_type) if filter_type else None,
        opportunity_tier=opportunity_tier,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        fitness_state=str(source.get("status_reason") or "No allocation fitness note"),
        filter_state=f"{filter_type} configured" if filter_type else "No filter metadata",
        risk_state=risk_state,
        portfolio_collision="profile_selected_lane",
        allocation_status=str(allocation_status) if allocation_status else None,
        allocation_reason=str(source.get("status_reason")) if source.get("status_reason") else None,
        trailing_expr=trailing_expr,
        session_regime=str(session_regime) if session_regime else None,
        chordia_verdict=str(chordia_verdict) if chordia_verdict else None,
    )


def _summarize_lanes(lanes: tuple[OpportunityLane, ...]) -> dict[str, int]:
    counts = {
        "lane_count": len(lanes),
        "blocked_count": 0,
        "watch_count": 0,
        "normal_count": 0,
        "prime_shadow_count": 0,
    }
    for lane in lanes:
        if lane.opportunity_tier == "BLOCKED":
            counts["blocked_count"] += 1
        elif lane.opportunity_tier == "WATCH":
            counts["watch_count"] += 1
        elif lane.opportunity_tier == "PRIME_SHADOW":
            counts["prime_shadow_count"] += 1
        else:
            counts["normal_count"] += 1
    return counts


def build_opportunity_snapshot(
    profile_id: str | None = None,
    *,
    trading_day: date | None = None,
    lifecycle: dict[str, Any] | None = None,
    allocation_payload: dict[str, Any] | None = None,
) -> OpportunitySnapshot:
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    effective_day = trading_day or date.today()
    lane_defs = [_as_lane_dict(lane) for lane in get_profile_lane_definitions(resolved_profile_id)]
    allocation = allocation_payload if allocation_payload is not None else _load_allocation_payload(resolved_profile_id)
    allocated_by_id = _index_by_strategy(allocation.get("lanes", []))
    paused_by_id = _index_by_strategy(allocation.get("paused", []))

    allocation_problem = None
    if allocation.get("missing"):
        allocation_problem = "lane allocation missing"
    elif allocation.get("error"):
        allocation_problem = f"lane allocation unreadable: {allocation['error']}"

    strategy_states = dict((lifecycle or {}).get("strategy_states", {}))
    lanes = tuple(
        _build_lane(
            lane,
            allocation_row=allocated_by_id.get(str(lane.get("strategy_id") or "")),
            paused_row=paused_by_id.get(str(lane.get("strategy_id") or "")),
            strategy_state=dict(strategy_states.get(str(lane.get("strategy_id") or ""), {})),
            allocation_problem=allocation_problem,
        )
        for lane in lane_defs
    )
    return OpportunitySnapshot(
        profile_id=resolved_profile_id,
        trading_day=effective_day.isoformat(),
        summary=_summarize_lanes(lanes),
        lanes=lanes,
    )


def _current_canonical_inputs(profile_id: str, *, db_path: Path) -> dict[str, Any]:
    profile = get_profile(profile_id)
    lane_ids = [str(lane["strategy_id"]) for lane in get_profile_lane_definitions(profile_id)]
    return {
        "profile_id": profile_id,
        "profile_fingerprint": build_profile_fingerprint(profile),
        "lane_ids": lane_ids,
        "db_path": str(db_path.resolve()),
        "db_identity": build_db_identity(db_path),
        "code_fingerprint": build_code_fingerprint(_lane_code_paths()),
    }


def refresh_opportunity_state(
    profile_id: str | None = None,
    *,
    db_path: Path = GOLD_DB_PATH,
    today: date | None = None,
    lifecycle: dict[str, Any] | None = None,
    allocation_payload: dict[str, Any] | None = None,
    write_state: bool = True,
) -> dict[str, Any]:
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    effective_day = today or date.today()
    snapshot = build_opportunity_snapshot(
        resolved_profile_id,
        trading_day=effective_day,
        lifecycle=lifecycle,
        allocation_payload=allocation_payload,
    )
    envelope = build_state_envelope(
        schema_version=OPPORTUNITY_STATE_SCHEMA_VERSION,
        state_type=OPPORTUNITY_STATE_TYPE,
        tool="opportunity_awareness",
        canonical_inputs=_current_canonical_inputs(resolved_profile_id, db_path=db_path),
        freshness={
            "as_of_date": effective_day.isoformat(),
            "max_age_days": OPPORTUNITY_MAX_AGE_DAYS,
        },
        payload=snapshot.to_dict(),
        git_head=get_git_head(PROJECT_ROOT),
    )
    if write_state:
        state_path = get_opportunity_state_path(resolved_profile_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return envelope


def _validated_opportunity_state(
    profile_id: str,
    *,
    db_path: Path,
    today: date,
) -> tuple[bool, str | None, dict | None]:
    path = get_opportunity_state_path(profile_id)
    if not path.exists():
        return False, "missing", None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"unreadable: {exc}", None

    profile = get_profile(profile_id)
    lane_ids = [str(lane["strategy_id"]) for lane in get_profile_lane_definitions(profile_id)]
    return validate_state_envelope(
        data,
        expected_state_type=OPPORTUNITY_STATE_TYPE,
        expected_schema_version=OPPORTUNITY_STATE_SCHEMA_VERSION,
        current_profile_id=profile_id,
        current_profile_fingerprint=build_profile_fingerprint(profile),
        current_lane_ids=lane_ids,
        current_db_identity=build_db_identity(db_path),
        current_code_fingerprint=build_code_fingerprint(_lane_code_paths()),
        today=today,
    )


def read_opportunity_state(
    profile_id: str | None = None,
    *,
    db_path: Path = GOLD_DB_PATH,
    today: date | None = None,
    lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    effective_day = today or date.today()
    if lifecycle is not None:
        try:
            refresh_opportunity_state(
                resolved_profile_id,
                db_path=db_path,
                today=effective_day,
                lifecycle=lifecycle,
                write_state=True,
            )
        except Exception as exc:
            return {
                "profile_id": resolved_profile_id,
                "available": False,
                "valid": False,
                "reason": str(exc),
                "summary": {},
                "lanes": [],
            }

    valid, reason, envelope = _validated_opportunity_state(resolved_profile_id, db_path=db_path, today=effective_day)
    if not valid or envelope is None:
        try:
            envelope = refresh_opportunity_state(
                resolved_profile_id,
                db_path=db_path,
                today=effective_day,
                lifecycle=lifecycle,
                write_state=True,
            )
            valid, reason, envelope = _validated_opportunity_state(
                resolved_profile_id,
                db_path=db_path,
                today=effective_day,
            )
        except Exception as exc:
            return {
                "profile_id": resolved_profile_id,
                "available": False,
                "valid": False,
                "reason": str(exc),
                "summary": {},
                "lanes": [],
            }

    if not valid or envelope is None:
        return {
            "profile_id": resolved_profile_id,
            "available": True,
            "valid": False,
            "reason": reason,
            "summary": {},
            "lanes": [],
        }
    payload = dict(envelope["payload"])
    return {
        "profile_id": resolved_profile_id,
        "available": True,
        "valid": True,
        "reason": None,
        "state_date": envelope["freshness"].get("as_of_date"),
        "summary": payload.get("summary", {}),
        "lanes": payload.get("lanes", []),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build shadow-only opportunity awareness state")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--date", dest="trading_day", default=None, help="Trading day YYYY-MM-DD")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    trading_day = date.fromisoformat(args.trading_day) if args.trading_day else date.today()
    envelope = refresh_opportunity_state(args.profile, today=trading_day, write_state=not args.no_write)
    print(json.dumps(envelope["payload"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

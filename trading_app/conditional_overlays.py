"""Shadow-only conditional overlay registry and derived-state helpers."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from pipeline.db_config import configure_connection
from pipeline.dst import BRISBANE_TZ, compute_trading_day_from_timestamp
from pipeline.paths import GOLD_DB_PATH
from trading_app.derived_state import (
    build_code_fingerprint,
    build_db_identity,
    build_profile_fingerprint,
    build_state_envelope,
    get_git_head,
    validate_state_envelope,
)
from trading_app.prop_profiles import get_profile, get_profile_lane_definitions, resolve_profile_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

OVERLAY_STATE_TYPE = "conditional_overlay_shadow"
OVERLAY_STATE_SCHEMA_VERSION = 1
OVERLAY_MAX_AGE_DAYS = 1


@dataclass(frozen=True)
class ConditionalOverlaySpec:
    overlay_id: str
    profile_id: str
    mode: str
    role: str
    instrument: str
    orb_minutes: int
    entry_model: str
    confirm_bars: int
    rr_target: float
    sessions: tuple[str, ...]
    directions: tuple[str, ...]
    feature_family: str
    breakpoint_artifact_path: Path
    size_map: dict[int, float]
    holdout_frozen_from: str
    notes: str = ""


PR48_MGC_CONT_EXEC_V1 = ConditionalOverlaySpec(
    overlay_id="pr48_mgc_cont_exec_v1",
    profile_id="topstep_50k",
    mode="shadow_only",
    role="allocator",
    instrument="MGC",
    orb_minutes=5,
    entry_model="E2",
    confirm_bars=1,
    rr_target=1.5,
    sessions=(
        "CME_REOPEN",
        "COMEX_SETTLE",
        "EUROPE_FLOW",
        "LONDON_METALS",
        "NYSE_OPEN",
        "SINGAPORE_OPEN",
        "TOKYO_OPEN",
        "US_DATA_1000",
        "US_DATA_830",
    ),
    directions=("long", "short"),
    feature_family="rel_vol_session",
    breakpoint_artifact_path=PROJECT_ROOT / "research" / "output" / "pr48_mes_mgc_sizer_rule_breakpoints_v1.csv",
    size_map={1: 0.0, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0},
    holdout_frozen_from="2026-01-01",
    notes="PR48 MGC continuous-exec frozen rel-vol sizer, shadow-only Phase 1 carrier.",
)


CONDITIONAL_OVERLAYS: dict[str, ConditionalOverlaySpec] = {
    PR48_MGC_CONT_EXEC_V1.overlay_id: PR48_MGC_CONT_EXEC_V1,
}


def get_overlay_specs_for_profile(profile_id: str) -> tuple[ConditionalOverlaySpec, ...]:
    return tuple(spec for spec in CONDITIONAL_OVERLAYS.values() if spec.profile_id == profile_id)


def get_overlay_state_path(profile_id: str, overlay_id: str) -> Path:
    return STATE_DIR / f"conditional_overlay_{profile_id}_{overlay_id}.json"


def _current_trading_day(now: datetime | None = None) -> date:
    effective_now = now or datetime.now(BRISBANE_TZ)
    return compute_trading_day_from_timestamp(effective_now)


def _overlay_code_paths(spec: ConditionalOverlaySpec) -> list[Path]:
    return [
        Path(__file__).resolve(),
        PROJECT_ROOT / "trading_app" / "derived_state.py",
        spec.breakpoint_artifact_path.resolve(),
    ]


def _load_breakpoints(spec: ConditionalOverlaySpec) -> dict[tuple[str, str], dict[str, float]]:
    if not spec.breakpoint_artifact_path.exists():
        raise FileNotFoundError(f"Missing breakpoint artifact: {spec.breakpoint_artifact_path}")

    rows: dict[tuple[str, str], dict[str, float]] = {}
    with spec.breakpoint_artifact_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("instrument") != spec.instrument:
                continue
            session = str(row.get("session") or "")
            direction = str(row.get("direction") or "")
            if session not in spec.sessions or direction not in spec.directions:
                continue
            rows[(session, direction)] = {
                "p20": float(row["p20"]),
                "p40": float(row["p40"]),
                "p60": float(row["p60"]),
                "p80": float(row["p80"]),
            }
    return rows


def _assign_bucket(value: float, thresholds: dict[str, float]) -> int:
    if value < thresholds["p20"]:
        return 1
    if value < thresholds["p40"]:
        return 2
    if value < thresholds["p60"]:
        return 3
    if value < thresholds["p80"]:
        return 4
    return 5


def _load_feature_row(
    con: duckdb.DuckDBPyConnection,
    *,
    instrument: str,
    orb_minutes: int,
    trading_day: date,
) -> dict[str, Any] | None:
    row = con.execute(
        """
        SELECT *
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = ? AND trading_day = ?
        LIMIT 1
        """,
        [instrument, orb_minutes, trading_day],
    ).fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in con.description]
    return dict(zip(cols, row, strict=False))


def _build_rows(
    spec: ConditionalOverlaySpec,
    *,
    feature_row: dict[str, Any] | None,
    breakpoints: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in spec.sessions:
        feature_key = f"rel_vol_{session}"
        feature_value = None if feature_row is None else feature_row.get(feature_key)
        numeric_feature_value = None
        if feature_value is not None:
            numeric_feature_value = float(feature_value)
        for direction in spec.directions:
            thresholds = breakpoints.get((session, direction))
            row: dict[str, Any] = {
                "session": session,
                "direction": direction,
                "lane": f"{session}_{direction}",
                "feature_key": feature_key,
                "feature_value": numeric_feature_value,
                "bucket": None,
                "size_multiplier": None,
                "status": "unscored",
                "reason": None,
            }
            if thresholds is None:
                row["status"] = "invalid"
                row["reason"] = "missing breakpoint row"
            elif feature_row is None:
                row["status"] = "unscored"
                row["reason"] = "no daily_features row for trading day"
            elif feature_value is None:
                row["status"] = "unscored"
                row["reason"] = f"missing {feature_key}"
            elif numeric_feature_value is None or not math.isfinite(numeric_feature_value):
                row["status"] = "unscored"
                row["reason"] = f"non-finite {feature_key}"
            else:
                bucket = _assign_bucket(numeric_feature_value, thresholds)
                row["bucket"] = bucket
                row["size_multiplier"] = spec.size_map[bucket]
                row["status"] = "ready"
                row["reason"] = f"Q{bucket}"
            rows.append(row)
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [row for row in rows if row["status"] == "ready"]
    unscored = [row for row in rows if row["status"] == "unscored"]
    invalid = [row for row in rows if row["status"] == "invalid"]
    if invalid:
        overall = "invalid"
    elif unscored:
        overall = "unscored"
    else:
        overall = "ready"
    return {
        "status": overall,
        "row_count": len(rows),
        "ready_count": len(ready),
        "unscored_count": len(unscored),
        "invalid_count": len(invalid),
        "active_sessions": sorted({row["session"] for row in ready}),
    }


def _semantic_invalid_reason(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str | None:
    if summary.get("status") != "invalid":
        return None
    invalid_reasons = sorted({str(row.get("reason") or "invalid") for row in rows if row.get("status") == "invalid"})
    if not invalid_reasons:
        return "invalid overlay status"
    if len(invalid_reasons) == 1:
        return invalid_reasons[0]
    return ", ".join(invalid_reasons)


def _current_overlay_canonical_inputs(
    profile_id: str,
    *,
    db_path: Path,
    spec: ConditionalOverlaySpec,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    lane_ids = [str(lane["strategy_id"]) for lane in get_profile_lane_definitions(profile_id)]
    return {
        "profile_id": profile_id,
        "profile_fingerprint": build_profile_fingerprint(profile),
        "lane_ids": lane_ids,
        "db_path": str(db_path.resolve()),
        "db_identity": build_db_identity(db_path, con=con),
        "code_fingerprint": build_code_fingerprint(_overlay_code_paths(spec)),
        "overlay_id": spec.overlay_id,
    }


def refresh_overlay_state(
    overlay_id: str,
    *,
    profile_id: str | None = None,
    db_path: Path = GOLD_DB_PATH,
    today: date | None = None,
    write_state: bool = True,
) -> dict[str, Any]:
    spec = CONDITIONAL_OVERLAYS[overlay_id]
    resolved_profile_id = profile_id or spec.profile_id
    trading_day = today or _current_trading_day()

    with duckdb.connect(str(db_path), read_only=True) as con:
        configure_connection(con, writing=False)
        breakpoints = _load_breakpoints(spec)
        feature_row = _load_feature_row(
            con,
            instrument=spec.instrument,
            orb_minutes=spec.orb_minutes,
            trading_day=trading_day,
        )
        rows = _build_rows(spec, feature_row=feature_row, breakpoints=breakpoints)
        summary = _summarize_rows(rows)
        envelope = build_state_envelope(
            schema_version=OVERLAY_STATE_SCHEMA_VERSION,
            state_type=OVERLAY_STATE_TYPE,
            tool="conditional_overlays",
            canonical_inputs=_current_overlay_canonical_inputs(
                resolved_profile_id,
                db_path=db_path,
                spec=spec,
                con=con,
            ),
            freshness={
                "as_of_date": trading_day.isoformat(),
                "max_age_days": OVERLAY_MAX_AGE_DAYS,
            },
            payload={
                "overlay_id": spec.overlay_id,
                "profile_id": resolved_profile_id,
                "mode": spec.mode,
                "role": spec.role,
                "instrument": spec.instrument,
                "orb_minutes": spec.orb_minutes,
                "entry_model": spec.entry_model,
                "confirm_bars": spec.confirm_bars,
                "rr_target": spec.rr_target,
                "feature_family": spec.feature_family,
                "holdout_frozen_from": spec.holdout_frozen_from,
                "summary": summary,
                "rows": rows,
            },
            git_head=get_git_head(PROJECT_ROOT),
        )

    if write_state:
        get_overlay_state_path(resolved_profile_id, spec.overlay_id).write_text(
            json.dumps(envelope, indent=2),
            encoding="utf-8",
        )
    return envelope


def _validated_overlay_state(
    spec: ConditionalOverlaySpec,
    *,
    profile_id: str,
    db_path: Path,
    today: date,
) -> tuple[bool, str | None, dict | None]:
    path = get_overlay_state_path(profile_id, spec.overlay_id)
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
        expected_state_type=OVERLAY_STATE_TYPE,
        expected_schema_version=OVERLAY_STATE_SCHEMA_VERSION,
        current_profile_id=profile_id,
        current_profile_fingerprint=build_profile_fingerprint(profile),
        current_lane_ids=lane_ids,
        current_db_identity=build_db_identity(db_path),
        current_code_fingerprint=build_code_fingerprint(_overlay_code_paths(spec)),
        today=today,
    )


def read_overlay_states(
    profile_id: str | None = None,
    *,
    db_path: Path = GOLD_DB_PATH,
    today: date | None = None,
) -> dict[str, Any]:
    effective_today = today or _current_trading_day()
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    specs = get_overlay_specs_for_profile(resolved_profile_id)
    if not specs:
        return {"profile_id": resolved_profile_id, "available": False, "valid": True, "overlays": []}

    overlays: list[dict[str, Any]] = []
    for spec in specs:
        try:
            valid, reason, envelope = _validated_overlay_state(
                spec,
                profile_id=resolved_profile_id,
                db_path=db_path,
                today=effective_today,
            )
            if not valid:
                refresh_overlay_state(
                    spec.overlay_id,
                    profile_id=resolved_profile_id,
                    db_path=db_path,
                    today=effective_today,
                    write_state=True,
                )
                valid, reason, envelope = _validated_overlay_state(
                    spec,
                    profile_id=resolved_profile_id,
                    db_path=db_path,
                    today=effective_today,
                )
        except Exception as exc:
            valid, reason, envelope = False, str(exc), None

        if valid and envelope is not None:
            freshness = envelope["freshness"]
            payload = envelope["payload"]
            summary = dict(payload.get("summary") or {})
            rows = list(payload.get("rows") or [])
            invalid_reason = _semantic_invalid_reason(summary, rows)
            overlays.append(
                {
                    "overlay_id": spec.overlay_id,
                    "available": True,
                    "valid": invalid_reason is None,
                    "reason": invalid_reason,
                    "state_date": freshness.get("as_of_date"),
                    "mode": payload.get("mode"),
                    "role": payload.get("role"),
                    "status": summary.get("status"),
                    "summary": summary,
                    "rows": rows,
                }
            )
        else:
            overlays.append(
                {
                    "overlay_id": spec.overlay_id,
                    "available": True,
                    "valid": False,
                    "reason": reason,
                    "state_date": None,
                    "mode": spec.mode,
                    "role": spec.role,
                    "status": "invalid",
                    "summary": {},
                    "rows": [],
                }
            )

    return {
        "profile_id": resolved_profile_id,
        "available": True,
        "valid": all(overlay["valid"] for overlay in overlays),
        "overlays": overlays,
    }


class RoleResolver:
    """Native engine interface for resolving conditional-role decisions."""

    def __init__(self, profile_id: str, today: date | None = None, db_path: Path = GOLD_DB_PATH):
        self.profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
        self.today = today or date.today()
        self.db_path = db_path
        # Pre-load all overlay states for this profile and day
        self.state = read_overlay_states(self.profile_id, today=self.today, db_path=self.db_path)

    def get_overlay_context(self, strategy_id: str, session: str, direction: str) -> dict[str, dict[str, Any]]:
        """Return matching overlay decisions for a specific trade candidate."""
        if not self.state.get("available") or not self.state.get("valid"):
            return {}

        context: dict[str, dict[str, Any]] = {}
        for overlay in self.state.get("overlays", []):
            if not overlay.get("valid") or overlay.get("status") != "ready":
                continue

            # Check if this strategy_id matches the overlay spec.
            # Use the canonical parser instead of substring matching:
            # `spec.entry_model in strategy_id` would collide with hypothetical
            # "E20", `f"O{spec.orb_minutes}"` collides with O50/O150 etc.
            # See feedback_aperture_overlay_canonical_parser.md.
            spec = CONDITIONAL_OVERLAYS.get(overlay["overlay_id"])
            if not spec:
                continue
            try:
                from trading_app.eligibility.builder import parse_strategy_id

                parsed = parse_strategy_id(strategy_id)
            except (ValueError, ImportError):
                continue
            if parsed.get("instrument") != spec.instrument:
                continue
            if parsed.get("orb_minutes", 5) != spec.orb_minutes:
                continue
            if parsed.get("entry_model") != spec.entry_model:
                continue

            # Match on session and direction within the overlay rows
            matching_row = next(
                (
                    row
                    for row in overlay.get("rows", [])
                    if row["session"] == session and row["direction"] == direction and row["status"] == "ready"
                ),
                None,
            )

            if matching_row:
                context[spec.overlay_id] = {
                    "mode": overlay["mode"],
                    "role": overlay["role"],
                    "bucket": matching_row["bucket"],
                    "size_multiplier": matching_row["size_multiplier"],
                    "feature_key": matching_row["feature_key"],
                    "feature_value": matching_row["feature_value"],
                }

        return context

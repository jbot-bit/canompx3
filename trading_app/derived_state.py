"""Helpers for self-invalidating derived state files.

These helpers keep runtime/operator state tied to canonical repo + DB truth so
stale artifacts fail closed or degrade explicitly instead of being trusted by
convention.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Sequence

import duckdb

from pipeline.db_config import configure_connection
from trading_app.prop_profiles import AccountProfile, effective_daily_lanes, get_firm_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_profile_fingerprint(profile: AccountProfile) -> str:
    """Fingerprint the safety-relevant profile definition."""
    payload = {
        "profile_id": profile.profile_id,
        "firm": profile.firm,
        "account_size": profile.account_size,
        "dd_type": get_firm_spec(profile.firm).dd_type,
        "stop_multiplier": profile.stop_multiplier,
        "payout_policy_id": profile.payout_policy_id,
        "is_express_funded": profile.is_express_funded,
        "max_risk_per_trade": profile.max_risk_per_trade,
        "daily_lanes": [
            {
                "strategy_id": lane.strategy_id,
                "instrument": lane.instrument,
                "orb_label": lane.orb_label,
                "planned_stop_multiplier": lane.planned_stop_multiplier,
                "required_fitness": list(lane.required_fitness),
                "max_orb_size_pts": lane.max_orb_size_pts,
            }
            for lane in effective_daily_lanes(profile)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_code_fingerprint(paths: Sequence[Path]) -> str:
    """Hash normalized source content for a set of code files."""
    payload: list[tuple[str, str]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            text = resolved.read_text(encoding="utf-8")
        except OSError:
            text = ""
        payload.append((str(resolved.relative_to(PROJECT_ROOT)), text.replace("\r\n", "\n")))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fetch_one(con: duckdb.DuckDBPyConnection, query: str, default: str) -> str:
    try:
        row = con.execute(query).fetchone()
    except duckdb.Error:
        return default
    if not row:
        return default
    value = row[0]
    if value is None:
        return default
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def build_db_identity(db_path: Path, con: duckdb.DuckDBPyConnection | None = None) -> str:
    """Hash a compact tuple set describing the canonical DB control surface."""
    own_con = False
    db_path = db_path.resolve()
    if con is None:
        con = duckdb.connect(str(db_path), read_only=True)
        configure_connection(con)
        own_con = True

    try:
        tuples = {
            "db_path": str(db_path),
            "validated_setups_count": _fetch_one(con, "SELECT COUNT(*) FROM validated_setups", "missing"),
            "validated_active_count": _fetch_one(
                con,
                "SELECT COUNT(*) FROM validated_setups WHERE LOWER(status) = 'active'",
                "missing",
            ),
            "paper_trades_count": _fetch_one(con, "SELECT COUNT(*) FROM paper_trades", "missing"),
            "orb_outcomes_max_day": _fetch_one(con, "SELECT MAX(trading_day) FROM orb_outcomes", "missing"),
            "daily_features_max_day": _fetch_one(con, "SELECT MAX(trading_day) FROM daily_features", "missing"),
        }
        encoded = json.dumps(tuples, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    finally:
        if own_con:
            con.close()


def get_git_head(root: Path | None = None) -> str:
    """Return short HEAD SHA or 'unknown' if unavailable."""
    repo_root = root or PROJECT_ROOT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    head = result.stdout.strip()
    return head or "unknown"


def build_state_envelope(
    *,
    state_type: str,
    canonical_inputs: dict,
    freshness: dict,
    payload: dict,
    schema_version: int = 1,
    tool: str,
    generated_at_utc: str | None = None,
    git_head: str | None = None,
) -> dict:
    """Wrap derived state in a versioned identity envelope."""
    return {
        "schema_version": schema_version,
        "state_type": state_type,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "git_head": git_head or get_git_head(),
        "tool": tool,
        "canonical_inputs": canonical_inputs,
        "freshness": freshness,
        "payload": payload,
    }


def validate_state_envelope(
    envelope: dict,
    *,
    expected_state_type: str,
    expected_schema_version: int,
    today: date | None = None,
    current_profile_id: str | None = None,
    current_profile_fingerprint: str | None = None,
    current_lane_ids: Sequence[str] | None = None,
    current_db_identity: str | None = None,
    current_code_fingerprint: str | None = None,
) -> tuple[bool, str | None, dict | None]:
    """Validate a derived-state envelope against current canonical truth."""
    required_top = {
        "schema_version",
        "state_type",
        "generated_at_utc",
        "git_head",
        "tool",
        "canonical_inputs",
        "freshness",
        "payload",
    }
    missing_top = sorted(required_top - set(envelope))
    if missing_top:
        return False, f"legacy state: missing {', '.join(missing_top)}", None

    if envelope["state_type"] != expected_state_type:
        return False, f"wrong state_type: {envelope['state_type']}", None
    if int(envelope["schema_version"]) != expected_schema_version:
        return False, f"wrong schema_version: {envelope['schema_version']}", None

    canonical_inputs = envelope["canonical_inputs"]
    freshness = envelope["freshness"]
    if not isinstance(canonical_inputs, dict):
        return False, "invalid canonical_inputs", None
    if not isinstance(freshness, dict):
        return False, "invalid freshness", None

    required_inputs = {
        "profile_id",
        "profile_fingerprint",
        "lane_ids",
        "db_path",
        "db_identity",
        "code_fingerprint",
    }
    missing_inputs = sorted(required_inputs - set(canonical_inputs))
    if missing_inputs:
        return False, f"legacy state: missing canonical_inputs.{', canonical_inputs.'.join(missing_inputs)}", None

    required_freshness = {"as_of_date", "max_age_days"}
    missing_freshness = sorted(required_freshness - set(freshness))
    if missing_freshness:
        return False, f"legacy state: missing freshness.{', freshness.'.join(missing_freshness)}", None

    if current_profile_id is not None and canonical_inputs["profile_id"] != current_profile_id:
        return False, "profile mismatch", None
    if current_profile_fingerprint is not None and canonical_inputs["profile_fingerprint"] != current_profile_fingerprint:
        return False, "profile fingerprint mismatch", None
    if current_lane_ids is not None and list(canonical_inputs["lane_ids"]) != list(current_lane_ids):
        return False, "lane_ids mismatch", None
    if current_db_identity is not None and canonical_inputs["db_identity"] != current_db_identity:
        return False, "db identity mismatch", None
    if current_code_fingerprint is not None and canonical_inputs["code_fingerprint"] != current_code_fingerprint:
        return False, "code fingerprint mismatch", None

    try:
        as_of_date = date.fromisoformat(str(freshness["as_of_date"]))
        max_age_days = int(freshness["max_age_days"])
    except (TypeError, ValueError):
        return False, "invalid freshness metadata", None

    effective_today = today or date.today()
    age_days = (effective_today - as_of_date).days
    if age_days > max_age_days:
        return False, f"stale state: {age_days}d old > {max_age_days}d", None

    return True, None, envelope

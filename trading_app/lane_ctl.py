"""Lane control — pause/resume/list daily lane overrides.

Overrides are profile-scoped state files in data/state/.
They affect:
- the manual daily sheet (`resolve_daily_lanes`)
- `pre_session_check`
- `session_orchestrator` startup and entry gating

Usage:
    python -m trading_app.lane_ctl pause SINGAPORE_OPEN --reason "cold streak"
    python -m trading_app.lane_ctl pause SINGAPORE_OPEN --expires 2026-04-01
    python -m trading_app.lane_ctl resume SINGAPORE_OPEN
    python -m trading_app.lane_ctl list
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def _override_path(profile_id: str) -> Path:
    return STATE_DIR / f"lane_overrides_{profile_id}.json"


def _load_overrides(profile_id: str) -> dict:
    path = _override_path(profile_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_overrides(profile_id: str, overrides: dict) -> None:
    path = _override_path(profile_id)
    path.write_text(json.dumps(overrides, indent=2))


def _find_strategy_id(profile_id: str, session_name: str) -> str | None:
    """Find the strategy_id for a session in a profile's daily_lanes."""
    from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes

    profile = ACCOUNT_PROFILES.get(profile_id)
    if not profile:
        return None
    lanes = effective_daily_lanes(profile)
    if not lanes:
        return None
    for lane in lanes:
        if lane.orb_label == session_name:
            return lane.strategy_id
    return None


def pause_lane(
    profile_id: str,
    session_name: str,
    reason: str = "",
    expires: str | None = None,
) -> None:
    """Pause a lane by session name."""
    sid = _find_strategy_id(profile_id, session_name)
    if sid is None:
        print(f"ERROR: No lane for session '{session_name}' in profile '{profile_id}'")
        sys.exit(1)

    overrides = _load_overrides(profile_id)
    overrides[sid] = {
        "active": False,
        "reason": reason,
        "since": date.today().isoformat(),
        "paused_at": datetime.now(UTC).isoformat(),
    }
    if expires:
        overrides[sid]["expires"] = expires
    _save_overrides(profile_id, overrides)
    exp_str = f" (expires {expires})" if expires else " (no expiry — resume manually)"
    print(f"PAUSED: {session_name} ({sid}){exp_str}")
    if reason:
        print(f"  Reason: {reason}")


def pause_strategy_id(
    profile_id: str,
    strategy_id: str,
    reason: str = "",
    expires: str | None = None,
    source: str = "manual",
) -> bool:
    """Pause a lane directly by strategy_id.

    Returns True if a new pause was written, False if the strategy was already paused.
    """
    overrides = _load_overrides(profile_id)
    existing = overrides.get(strategy_id)
    if existing is not None and not existing.get("active", True):
        return False

    overrides[strategy_id] = {
        "active": False,
        "reason": reason,
        "since": date.today().isoformat(),
        "paused_at": datetime.now(UTC).isoformat(),
        "source": source,
    }
    if expires:
        overrides[strategy_id]["expires"] = expires
    _save_overrides(profile_id, overrides)
    return True


def resume_lane(profile_id: str, session_name: str) -> None:
    """Resume a paused lane."""
    sid = _find_strategy_id(profile_id, session_name)
    if sid is None:
        print(f"ERROR: No lane for session '{session_name}' in profile '{profile_id}'")
        sys.exit(1)

    overrides = _load_overrides(profile_id)
    if sid in overrides:
        del overrides[sid]
        _save_overrides(profile_id, overrides)
        print(f"RESUMED: {session_name} ({sid})")
    else:
        print(f"Lane {session_name} was not paused.")


def list_overrides(profile_id: str) -> None:
    """Show all active overrides for a profile."""
    overrides = _load_overrides(profile_id)
    if not overrides:
        print(f"No lane overrides for profile '{profile_id}'.")
        return

    # Check for orphans (strategy_id no longer in profile)
    from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes

    profile = ACCOUNT_PROFILES.get(profile_id)
    valid_sids = {la.strategy_id for la in effective_daily_lanes(profile)} if profile else set()

    today = date.today()
    print(f"\nLane overrides — {profile_id}:")
    for sid, info in overrides.items():
        status = "PAUSED" if not info.get("active", True) else "active"
        reason = info.get("reason", "")
        since = info.get("since", "?")
        expires = info.get("expires")
        orphan = " [ORPHANED — not in current lanes]" if sid not in valid_sids else ""

        # Age warning
        age_str = ""
        if since != "?":
            age = (today - date.fromisoformat(since)).days
            if age > 14:
                age_str = f" ({age} days — still intended?)"
            else:
                age_str = f" ({age} days)"

        exp_str = f", expires {expires}" if expires else ", no expiry"
        print(f"  {status}: {sid}{orphan}")
        print(f"    since {since}{age_str}{exp_str}")
        if reason:
            print(f"    reason: {reason}")


def get_lane_override(profile_id: str, strategy_id: str) -> dict | None:
    """Get override for a specific lane. Returns None if no override or expired."""
    overrides = _load_overrides(profile_id)
    info = overrides.get(strategy_id)
    if info is None:
        return None
    # Check expiry
    expires = info.get("expires")
    if expires and date.fromisoformat(expires) < date.today():
        return None  # Expired — lane auto-resumes
    if info.get("active", True):
        return None  # Not paused
    return info


def get_paused_strategy_ids(profile_id: str, as_of: date | None = None) -> set[str]:
    """Return currently active paused strategy_ids for a profile."""
    today = as_of or date.today()
    overrides = _load_overrides(profile_id)
    paused: set[str] = set()
    for strategy_id, info in overrides.items():
        if info.get("active", True):
            continue
        expires = info.get("expires")
        if expires and date.fromisoformat(expires) < today:
            continue
        paused.add(strategy_id)
    return paused


def main():
    parser = argparse.ArgumentParser(description="Lane control — pause/resume daily lanes")
    parser.add_argument("action", choices=["pause", "resume", "list"], help="Action to perform")
    parser.add_argument("session", nargs="?", help="Session name (e.g. SINGAPORE_OPEN)")
    parser.add_argument("--profile", default="topstep_50k_mnq_auto", help="Profile ID (default: topstep_50k_mnq_auto)")
    parser.add_argument("--reason", default="", help="Reason for pausing")
    parser.add_argument("--expires", default=None, help="Expiry date YYYY-MM-DD")
    args = parser.parse_args()

    if args.action == "list":
        list_overrides(args.profile)
    elif args.action == "pause":
        if not args.session:
            parser.error("pause requires a session name")
        pause_lane(args.profile, args.session, args.reason, args.expires)
    elif args.action == "resume":
        if not args.session:
            parser.error("resume requires a session name")
        resume_lane(args.profile, args.session)


if __name__ == "__main__":
    main()

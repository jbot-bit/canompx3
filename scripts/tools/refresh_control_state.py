#!/usr/bin/env python3
"""Refresh stale or mismatched control-state surfaces.

This reconciles the repo's two machine-derived control artifacts:
- Criterion 11 account-survival state
- Criterion 12 SR monitor state

It is intentionally narrow: refresh the state surfaces, then re-read the
unified lifecycle snapshot and fail closed if either surface is still invalid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pipeline.paths import GOLD_DB_PATH
from trading_app.account_survival import evaluate_profile_survival
from trading_app.lifecycle_state import read_lifecycle_state
from trading_app.prop_profiles import resolve_profile_id
from trading_app.sr_monitor import run_monitor


@dataclass(frozen=True)
class RefreshResult:
    refreshed: bool
    reason: str


def _needs_refresh(summary: dict | None) -> bool:
    if not isinstance(summary, dict):
        return True
    if not summary.get("available", True):
        return True
    if not summary.get("valid", True):
        return True
    return False


def refresh_control_state(
    profile_id: str | None = None,
    *,
    db_path: Path = GOLD_DB_PATH,
    force: bool = False,
    refresh_c11: bool = True,
    refresh_c12: bool = True,
) -> dict[str, object]:
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    before = read_lifecycle_state(resolved_profile_id, db_path=db_path)

    c11_before = before["criterion11"]
    c12_before = before["criterion12"]

    c11_result = RefreshResult(refreshed=False, reason="already valid")
    c12_result = RefreshResult(refreshed=False, reason="already valid")

    if refresh_c11 and (force or _needs_refresh(c11_before)):
        evaluate_profile_survival(profile_id=resolved_profile_id, db_path=db_path, write_state=True)
        c11_result = RefreshResult(
            refreshed=True, reason=str(c11_before.get("reason") or c11_before.get("gate_msg") or "refreshed")
        )

    if refresh_c12 and (force or _needs_refresh(c12_before)):
        run_monitor(apply_pauses=False)
        c12_result = RefreshResult(refreshed=True, reason=str(c12_before.get("reason") or "refreshed"))

    after = read_lifecycle_state(resolved_profile_id, db_path=db_path)
    return {
        "profile_id": resolved_profile_id,
        "before": before,
        "after": after,
        "criterion11": c11_result,
        "criterion12": c12_result,
    }


def _print_state(label: str, lifecycle: dict[str, object]) -> None:
    c11 = lifecycle["criterion11"]
    c12 = lifecycle["criterion12"]
    print(f"[{label}] {lifecycle['profile_id']}")
    print(
        "  C11: "
        f"valid={bool(c11.get('valid'))} gate_ok={bool(c11.get('gate_ok'))} "
        f"reason={c11.get('reason') or c11.get('gate_msg')}"
    )
    print(f"  C12: valid={bool(c12.get('valid'))} reason={c12.get('reason')} counts={c12.get('counts')}")
    print(f"  blocked={lifecycle.get('blocked_strategy_ids', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh stale or mismatched control-state surfaces")
    parser.add_argument("--profile", default=None, help="Profile id to refresh")
    parser.add_argument("--force", action="store_true", help="Refresh both C11 and C12 even if they appear valid")
    parser.add_argument("--skip-c11", action="store_true", help="Do not refresh Criterion 11")
    parser.add_argument("--skip-c12", action="store_true", help="Do not refresh Criterion 12")
    args = parser.parse_args()

    result = refresh_control_state(
        profile_id=args.profile,
        force=args.force,
        refresh_c11=not args.skip_c11,
        refresh_c12=not args.skip_c12,
    )

    _print_state("before", result["before"])
    print(
        "  refresh actions: "
        f"C11={'yes' if result['criterion11'].refreshed else 'no'} "
        f"({result['criterion11'].reason}); "
        f"C12={'yes' if result['criterion12'].refreshed else 'no'} "
        f"({result['criterion12'].reason})"
    )
    _print_state("after", result["after"])

    after = result["after"]
    c11 = after["criterion11"]
    c12 = after["criterion12"]
    if not bool(c11.get("valid")) or not bool(c11.get("gate_ok")):
        raise SystemExit(1)
    if not bool(c12.get("valid")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

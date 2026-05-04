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
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOCATION_PATH = PROJECT_ROOT / "docs" / "runtime" / "lane_allocation.json"


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
from trading_app.prop_profiles import (  # noqa: E402
    get_profile_lane_definitions,
    resolve_profile_id,
)
from trading_app.validated_shelf import deployable_validated_relation  # noqa: E402


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
    summary["active_lanes"] = [
        _normalize_lane_row(
            row,
            bucket="lanes",
            strategy_states=strategy_states,
            blocked_reason_by_strategy=blocked_reason_by_strategy,
        )
        for row in data.get("lanes", [])
    ]
    summary["paused_lanes"] = [
        _normalize_lane_row(
            row,
            bucket="paused",
            strategy_states=strategy_states,
            blocked_reason_by_strategy=blocked_reason_by_strategy,
        )
        for row in data.get("paused", [])
    ]
    summary["stale_lanes"] = [
        _normalize_lane_row(
            row,
            bucket="stale",
            strategy_states=strategy_states,
            blocked_reason_by_strategy=blocked_reason_by_strategy,
        )
        for row in data.get("stale", [])
    ]

    allocator_ids = {
        lane["strategy_id"]
        for lane in summary["active_lanes"] + summary["paused_lanes"] + summary["stale_lanes"]
        if lane.get("strategy_id")
    }
    summary["profile_lanes_missing_from_allocator"] = sorted(set(profile_lane_ids) - allocator_ids)
    return summary


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
        "conditional_overlays": lifecycle.get("conditional_overlays"),
    }


def _render_text(report: dict[str, Any]) -> str:
    deployment = report["deployment_summary"]
    c11 = report["criterion11"] or {}
    c12 = report["criterion12"] or {}
    allocator = report["allocator_summary"] or {}

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
            f"rebalance_date={allocator.get('rebalance_date')} "
            f"lanes={len(allocator.get('active_lanes', []))} "
            f"paused={len(allocator.get('paused_lanes', []))} "
            f"stale={len(allocator.get('stale_lanes', []))}"
        ),
        "Active lanes:",
    ]

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
        "## Active Lanes",
        "",
    ]

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
        help="Path to lane_allocation.json.",
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


if __name__ == "__main__":
    main()

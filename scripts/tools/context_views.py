#!/usr/bin/env python3
"""Task-scoped generated context views for cold-start agents.

Each view is structured into strict truth classes:
- canonical_state
- live_operational_state
- non_authoritative_context

The goal is to prevent mixed-truth summaries from becoming soft authority.
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
        return  # imported as library by test runner — do not re-exec
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

from context.registry import VERIFICATION_PROFILES, VERIFICATION_STEPS  # noqa: E402
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, DEPLOYABLE_ORB_INSTRUMENTS  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from scripts.tools.project_pulse import (  # noqa: E402
    _canonical_repo_root,
    _git_branch,
    _git_head,
    collect_deployment_state,
    collect_fitness_fast,
    collect_handoff,
    collect_lifecycle_control,
    collect_session_claims,
    collect_system_identity,
    collect_upcoming_sessions,
)
from trading_app.holdout_policy import (  # noqa: E402
    HOLDOUT_GRANDFATHER_CUTOFF,
    HOLDOUT_SACRED_FROM,
)

REQUIRED_SECTIONS: tuple[str, ...] = (
    "canonical_state",
    "live_operational_state",
    "non_authoritative_context",
)
RECENT_PERFORMANCE_ROLLING_MONTHS = 3
RECENT_PERFORMANCE_RANK_LIMIT = 5
RECENT_PERFORMANCE_NON_FIT_LIMIT = 10
DISALLOWED_DENSE_KEYS: tuple[str, ...] = (
    "recommendation",
    "opinion",
    "essay",
    "freeform_summary",
    "advice",
    "signals",
    "system_identity",
)


def _build_payload(
    *,
    view: str,
    root: Path,
    db_path: Path,
    required_canonical_reads: list[str],
    canonical_state: dict[str, Any],
    live_operational_state: dict[str, Any],
    non_authoritative_context: dict[str, Any],
    section_sources: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "view": view,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_branch": _git_branch(root),
        "git_head": _git_head(root),
        "canonical_db_path": str(db_path),
        "required_canonical_reads": required_canonical_reads,
        "sections": {
            "canonical_state": canonical_state,
            "live_operational_state": live_operational_state,
            "non_authoritative_context": non_authoritative_context,
        },
        "section_sources": section_sources,
    }


def validate_view_payload(payload: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    sections = payload.get("sections")
    if not isinstance(sections, dict):
        return ["payload missing sections dict"]
    for name in REQUIRED_SECTIONS:
        if name not in sections:
            violations.append(f"missing section {name}")
    section_sources = payload.get("section_sources")
    if not isinstance(section_sources, dict):
        violations.append("payload missing section_sources dict")
    else:
        for name in REQUIRED_SECTIONS:
            if name not in section_sources:
                violations.append(f"missing source map for section {name}")
    canonical_state = sections.get("canonical_state", {})
    if "handoff" in canonical_state or "next_steps" in canonical_state:
        violations.append("handoff context leaked into canonical_state")
    live_state = sections.get("live_operational_state", {})
    if "handoff" in live_state:
        violations.append("handoff context leaked into live_operational_state")
    for section_name in ("canonical_state", "live_operational_state"):
        section_payload = sections.get(section_name, {})
        if isinstance(section_payload, dict):
            for key in DISALLOWED_DENSE_KEYS:
                if key in section_payload:
                    violations.append(f"{key} is not allowed in {section_name}")
    return violations


def _compact_runtime_context(system_identity: dict[str, Any] | None, items: list[Any]) -> dict[str, Any]:
    if not isinstance(system_identity, dict):
        return {"available": False}
    interpreter = system_identity.get("interpreter", {})
    git = system_identity.get("git", {})
    policy = system_identity.get("policy", {})
    return {
        "available": True,
        "canonical_repo_root": system_identity.get("canonical_repo_root"),
        "selected_db_path": system_identity.get("selected_db_path"),
        "db_override_active": system_identity.get("db_override_active"),
        "interpreter": {
            "context": interpreter.get("context"),
            "matches_expected": interpreter.get("matches_expected"),
            "expected_prefix": interpreter.get("expected_prefix"),
        },
        "git": {
            "branch": git.get("branch"),
            "head_sha": git.get("head_sha"),
            "in_linked_worktree": git.get("in_linked_worktree"),
        },
        "work_queue": {
            "open_count": system_identity.get("work_queue", {}).get("open_count"),
            "close_first_open_count": system_identity.get("work_queue", {}).get("close_first_open_count"),
            "stale_count": system_identity.get("work_queue", {}).get("stale_count"),
            "top_item_ids": [
                item.get("id")
                for item in system_identity.get("work_queue", {}).get("top_items", [])
                if isinstance(item, dict) and item.get("id")
            ],
            "handoff_matches_rendered": system_identity.get("work_queue", {}).get("handoff_matches_rendered"),
        },
        "active_stage_count": len(system_identity.get("active_stages", [])),
        "fresh_claim_count": len(system_identity.get("fresh_claims", [])),
        "policy_warning_count": len(policy.get("warnings", [])),
        "health_item_count": len(items),
    }


def _project_fitness_score(score: Any) -> dict[str, Any]:
    return {
        "strategy_id": score.strategy_id,
        "fitness_status": score.fitness_status,
        "fitness_notes": score.fitness_notes,
        "rolling_window_months": score.rolling_window_months,
        "rolling_exp_r": score.rolling_exp_r,
        "rolling_sharpe": score.rolling_sharpe,
        "rolling_win_rate": score.rolling_win_rate,
        "rolling_sample": score.rolling_sample,
        "recent_sharpe_30": score.recent_sharpe_30,
        "recent_sharpe_60": score.recent_sharpe_60,
        "sharpe_delta_30": score.sharpe_delta_30,
        "sharpe_delta_60": score.sharpe_delta_60,
    }


def _sort_metric_low(value: float | None) -> tuple[int, float]:
    return (1, 0.0) if value is None else (0, float(value))


def _collect_recent_performance(instruments: list[str], db_path: Path) -> dict[str, Any]:
    from trading_app.strategy_fitness import compute_portfolio_fitness

    instrument_reports: dict[str, Any] = {}
    instrument_errors: dict[str, Any] = {}

    for instrument in instruments:
        try:
            report = compute_portfolio_fitness(
                db_path=db_path,
                instrument=instrument,
                rolling_months=RECENT_PERFORMANCE_ROLLING_MONTHS,
            )
        except Exception as exc:
            instrument_errors[instrument] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            continue

        projected_scores = [_project_fitness_score(score) for score in report.scores]
        non_fit_scores = [score for score in projected_scores if score["fitness_status"] != "FIT"]
        lowest_recent_sharpe_30 = sorted(
            projected_scores,
            key=lambda score: (_sort_metric_low(score["recent_sharpe_30"]), score["strategy_id"]),
        )[:RECENT_PERFORMANCE_RANK_LIMIT]
        lowest_rolling_exp_r = sorted(
            projected_scores,
            key=lambda score: (_sort_metric_low(score["rolling_exp_r"]), score["strategy_id"]),
        )[:RECENT_PERFORMANCE_RANK_LIMIT]

        instrument_reports[instrument] = {
            "as_of_date": report.as_of_date.isoformat(),
            "strategy_count": len(projected_scores),
            "summary": report.summary,
            "non_fit_count": len(non_fit_scores),
            "non_fit_strategies": non_fit_scores[:RECENT_PERFORMANCE_NON_FIT_LIMIT],
            "lowest_recent_sharpe_30": lowest_recent_sharpe_30,
            "lowest_rolling_exp_r": lowest_rolling_exp_r,
        }

    return {
        "available": bool(instrument_reports),
        "rolling_window_months": RECENT_PERFORMANCE_ROLLING_MONTHS,
        "instrument_reports": instrument_reports,
        "instrument_errors": instrument_errors,
    }


def build_research_context(root: Path, db_path: Path) -> dict[str, Any]:
    canonical = _canonical_repo_root(root)
    system_identity, identity_items = collect_system_identity(root, canonical, db_path)
    fitness_summary, _fitness_items = collect_fitness_fast(db_path)
    handoff_context, _handoff_items = collect_handoff(root)

    return _build_payload(
        view="research",
        root=root,
        db_path=db_path,
        required_canonical_reads=[
            "RESEARCH_RULES.md",
            "TRADING_RULES.md",
            "trading_app/holdout_policy.py",
            "pipeline/asset_configs.py",
            "pipeline/db_contracts.py",
        ],
        canonical_state={
            "holdout_policy": {
                "sacred_from": HOLDOUT_SACRED_FROM.isoformat(),
                "grandfather_cutoff": HOLDOUT_GRANDFATHER_CUTOFF.isoformat(),
                "canonical_source": "trading_app/holdout_policy.py",
            },
            "instrument_scope": {
                "active_orb_instruments": sorted(ACTIVE_ORB_INSTRUMENTS),
                "deployable_orb_instruments": sorted(DEPLOYABLE_ORB_INSTRUMENTS),
                "canonical_source": "pipeline/asset_configs.py",
            },
        },
        live_operational_state={
            "fitness_summary": fitness_summary,
            "repo_runtime_context": _compact_runtime_context(system_identity, identity_items),
        },
        non_authoritative_context={
            "handoff": {
                "date": handoff_context.get("date"),
                "summary": handoff_context.get("summary"),
                "next_steps": handoff_context.get("next_steps", []),
            }
        },
        section_sources={
            "canonical_state": ["trading_app/holdout_policy.py", "pipeline/asset_configs.py"],
            "live_operational_state": ["scripts/tools/project_pulse.py", "pipeline/system_context.py"],
            "non_authoritative_context": ["HANDOFF.md"],
        },
    )


def build_recent_performance_context(root: Path, db_path: Path) -> dict[str, Any]:
    canonical = _canonical_repo_root(root)
    system_identity, identity_items = collect_system_identity(root, canonical, db_path)
    instruments = sorted(ACTIVE_ORB_INSTRUMENTS)
    recent_performance = _collect_recent_performance(instruments, db_path)

    return _build_payload(
        view="recent_performance",
        root=root,
        db_path=db_path,
        required_canonical_reads=[
            "RESEARCH_RULES.md",
            "TRADING_RULES.md",
            "trading_app/strategy_fitness.py",
            "trading_app/holdout_policy.py",
            "pipeline/asset_configs.py",
        ],
        canonical_state={
            "recent_performance_contract": {
                "active_instruments": instruments,
                "rolling_window_months": RECENT_PERFORMANCE_ROLLING_MONTHS,
                "recent_trade_windows": [30, 60],
                "canonical_source": "trading_app/strategy_fitness.py",
            },
        },
        live_operational_state={
            "recent_performance": recent_performance,
            "repo_runtime_context": _compact_runtime_context(system_identity, identity_items),
        },
        non_authoritative_context={},
        section_sources={
            "canonical_state": ["trading_app/strategy_fitness.py", "pipeline/asset_configs.py"],
            "live_operational_state": ["trading_app/strategy_fitness.py", "pipeline/system_context.py"],
            "non_authoritative_context": [],
        },
    )


def build_trading_context(root: Path, db_path: Path) -> dict[str, Any]:
    canonical = _canonical_repo_root(root)
    system_identity, identity_items = collect_system_identity(root, canonical, db_path)
    deployment_summary, _deployment_items = collect_deployment_state(db_path)
    survival_summary, sr_summary, pause_summary, _lifecycle_items = collect_lifecycle_control(db_path)
    upcoming_sessions = collect_upcoming_sessions(db_path)

    return _build_payload(
        view="trading",
        root=root,
        db_path=db_path,
        required_canonical_reads=[
            "TRADING_RULES.md",
            "trading_app/prop_profiles.py",
            "trading_app/lifecycle_state.py",
            "pipeline/db_contracts.py",
        ],
        canonical_state={
            "deployable_shelf_contract": {
                "relation": "deployable_validated_setups",
                "canonical_source": "pipeline/db_contracts.py + trading_app/validated_shelf.py",
            },
            "lane_owner": {"canonical_source": "trading_app/prop_profiles.py"},
        },
        live_operational_state={
            "deployment_summary": deployment_summary,
            "survival_summary": survival_summary,
            "sr_summary": sr_summary,
            "pause_summary": pause_summary,
            "upcoming_sessions": upcoming_sessions,
            "repo_runtime_context": _compact_runtime_context(system_identity, identity_items),
        },
        non_authoritative_context={},
        section_sources={
            "canonical_state": ["pipeline/db_contracts.py", "trading_app/prop_profiles.py"],
            "live_operational_state": ["trading_app/lifecycle_state.py", "scripts/tools/project_pulse.py"],
            "non_authoritative_context": [],
        },
    )


def build_verification_context(root: Path, db_path: Path) -> dict[str, Any]:
    canonical = _canonical_repo_root(root)
    system_identity, identity_items = collect_system_identity(root, canonical, db_path)
    handoff_context, _handoff_items = collect_handoff(root)
    claim_items = collect_session_claims(root)

    done_profile = VERIFICATION_PROFILES["done"]
    verification_commands = [VERIFICATION_STEPS[step_id].command for step_id in done_profile.steps]

    return _build_payload(
        view="verification",
        root=root,
        db_path=db_path,
        required_canonical_reads=[
            "CLAUDE.md",
            "pipeline/check_drift.py",
            "context/registry.py",
            "pipeline/system_context.py",
        ],
        canonical_state={
            "verification_profile": {
                "id": done_profile.id,
                "commands": verification_commands,
                "canonical_source": "context/registry.py",
            }
        },
        live_operational_state={
            "repo_runtime_context": _compact_runtime_context(system_identity, identity_items),
            "session_claim_item_count": len(claim_items),
        },
        non_authoritative_context={
            "handoff": {
                "date": handoff_context.get("date"),
                "summary": handoff_context.get("summary"),
                "next_steps": handoff_context.get("next_steps", []),
            }
        },
        section_sources={
            "canonical_state": ["context/registry.py"],
            "live_operational_state": ["pipeline/system_context.py", "scripts/tools/project_pulse.py"],
            "non_authoritative_context": ["HANDOFF.md"],
        },
    )


VIEW_BUILDERS = {
    "research": build_research_context,
    "recent_performance": build_recent_performance_context,
    "trading": build_trading_context,
    "verification": build_verification_context,
}


def build_view(view: str, root: Path, db_path: Path) -> dict[str, Any]:
    builder = VIEW_BUILDERS.get(view)
    if builder is None:
        raise ValueError(f"Unknown view: {view}")
    payload = builder(root, db_path)
    violations = validate_view_payload(payload)
    if violations:
        raise ValueError(f"Invalid context view payload for {view}: {'; '.join(violations)}")
    return payload


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['view'].title()} Context",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Branch: `{payload['git_branch']}`",
        f"- HEAD: `{payload['git_head']}`",
        f"- DB: `{payload['canonical_db_path']}`",
        "",
        "## Required Canonical Reads",
        "",
    ]
    lines.extend(f"- `{path}`" for path in payload.get("required_canonical_reads", []))
    lines.append("")
    for section_name in REQUIRED_SECTIONS:
        lines.append(f"## {section_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("Sources:")
        for source in payload.get("section_sources", {}).get(section_name, []):
            lines.append(f"- `{source}`")
        if not payload.get("section_sources", {}).get(section_name):
            lines.append("- none")
        lines.append("")
        section_payload = payload.get("sections", {}).get(section_name, {})
        lines.append("```json")
        lines.append(json.dumps(section_payload, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render task-scoped generated context views.")
    parser.add_argument("--view", required=True, choices=sorted(VIEW_BUILDERS))
    parser.add_argument("--format", default="json", choices=["json", "markdown"])
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    db_path = Path(args.db_path).resolve()
    payload = build_view(args.view, root, db_path)
    if args.format == "markdown":
        print(format_markdown(payload))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

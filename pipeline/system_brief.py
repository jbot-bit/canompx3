"""Derived startup read-model for minimal complete task understanding."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from context.registry import TaskRoute, resolve_from_text, resolve_task
from pipeline.system_context import build_system_context, evaluate_system_policy

BriefingLevel = Literal["trivial", "read_only", "non_trivial", "mutating"]

DECISION_LEDGER_PATH = Path("docs/runtime/decision-ledger.md")
DEBT_LEDGER_PATH = Path("docs/runtime/debt-ledger.md")
DEFAULT_ORIENTATION_BUDGET_MS = {
    "trivial": 150,
    "read_only": 250,
    "non_trivial": 500,
    "mutating": 800,
}


@dataclass(frozen=True)
class BriefIssue:
    level: str
    code: str
    message: str
    detail: str | None = None


def _extract_bullet_refs(path: Path, limit: int = 5) -> tuple[list[str], list[str]]:
    if not path.exists():
        return [], [f"{path.as_posix()} missing"]
    refs: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("- "):
            refs.append(stripped[2:].strip())
            if len(refs) >= limit:
                break
    return refs, []


def _load_capsule_summary(root: Path) -> tuple[dict | None, list[BriefIssue]]:
    try:
        from pipeline.work_capsule import evaluate_current_capsule
    except Exception as exc:
        return None, [BriefIssue("warning", "work_capsule_unavailable", "Work capsule module unavailable.", str(exc))]
    summary, issues = evaluate_current_capsule(root)
    mapped = [BriefIssue(issue.level, issue.code, issue.message, issue.detail) for issue in issues]
    return summary, mapped


def _resolve_route(task_text: str | None, task_id: str | None) -> tuple[TaskRoute, list[BriefIssue]]:
    issues: list[BriefIssue] = []
    if task_id:
        return resolve_task(task_id), issues
    if task_text:
        route, candidates = resolve_from_text(task_text)
        if route is None:
            if candidates:
                issues.append(
                    BriefIssue(
                        "blocker",
                        "ambiguous_route",
                        "Mutating/system startup blocked: task routing is ambiguous.",
                        ", ".join(candidate.task_id for candidate in candidates),
                    )
                )
            else:
                issues.append(
                    BriefIssue(
                        "blocker",
                        "missing_route",
                        "Mutating/system startup blocked: no deterministic task route matched.",
                    )
                )
            return resolve_task("system_orientation"), issues
        return route, issues
    return resolve_task("system_orientation"), issues


def build_system_brief(
    root: Path,
    *,
    task_text: str | None = None,
    task_id: str | None = None,
    briefing_level: BriefingLevel = "read_only",
    context_name: str = "generic",
    active_tool: str | None = None,
    active_mode: str = "read-only",
    db_path: Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    route, route_issues = _resolve_route(task_text, task_id)
    snapshot = build_system_context(
        root,
        context_name=context_name,  # type: ignore[arg-type]
        active_tool=active_tool,
        active_mode=active_mode,  # type: ignore[arg-type]
        db_path=db_path,
    )
    decision = evaluate_system_policy(
        snapshot, "session_start_mutating" if briefing_level == "mutating" else "orientation"
    )
    capsule_summary, capsule_issues = _load_capsule_summary(root)
    decision_refs, decision_errors = _extract_bullet_refs(root / DECISION_LEDGER_PATH)
    debt_refs, debt_errors = _extract_bullet_refs(root / DEBT_LEDGER_PATH)

    blockers: list[dict[str, object]] = [issue.__dict__ for issue in route_issues if issue.level == "blocker"]
    warnings: list[dict[str, object]] = [issue.__dict__ for issue in route_issues if issue.level != "blocker"]
    blockers.extend(issue.model_dump(mode="json") for issue in decision.blockers)
    warnings.extend(issue.model_dump(mode="json") for issue in decision.warnings)

    for issue in capsule_issues:
        target = blockers if issue.level == "blocker" and briefing_level == "mutating" else warnings
        target.append(issue.__dict__)

    for path in (*route.doctrine_files, *route.canonical_files):
        if not (root / path).exists():
            issue = {
                "level": "blocker",
                "code": "missing_owner_path",
                "message": "Required route path missing.",
                "detail": path,
            }
            if briefing_level == "mutating":
                blockers.append(issue)
            else:
                warnings.append(issue)

    warnings.extend(
        {
            "level": "warning",
            "code": "missing_decision_ledger",
            "message": "Decision ledger missing or empty.",
            "detail": item,
        }
        for item in decision_errors
    )
    warnings.extend(
        {"level": "warning", "code": "missing_debt_ledger", "message": "Debt ledger missing or empty.", "detail": item}
        for item in debt_errors
    )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "task_id": route.manifest.id,
        "route_id": route.manifest.id,
        "task_kind": route.manifest.title,
        "briefing_level": briefing_level,
        "briefing_contract": route.briefing_contract.id,
        "doctrine_chain": list(route.doctrine_files),
        "canonical_owners": list(route.canonical_files),
        "required_live_views": [view.id for view in route.live_views],
        "verification_profile": route.verification.id,
        "verification_steps": [step.id for step in route.verification_steps],
        "work_capsule_ref": capsule_summary.get("path") if capsule_summary else None,
        "decision_refs": decision_refs,
        "debt_refs": debt_refs,
        "blocking_issues": blockers,
        "warning_issues": warnings,
        "expansion_triggers": list(route.expansion_triggers or route.briefing_contract.expansion_triggers),
        "generated_at": snapshot.generated_at,
        "startup_latency_ms": elapsed_ms,
        "orientation_cost_budget": {
            "budget_ms": DEFAULT_ORIENTATION_BUDGET_MS[briefing_level],
            "within_budget": elapsed_ms <= DEFAULT_ORIENTATION_BUDGET_MS[briefing_level],
        },
    }

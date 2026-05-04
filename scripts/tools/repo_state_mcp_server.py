"""MCP server for repo-local task routing, startup briefing, and project state.

Exposes read-only tools via stdio (fastmcp):
  - list_task_routes: list deterministic task routes available in the repo
  - list_context_views: list generated context views
  - resolve_task_route: resolve natural-language work into the canonical route
  - get_project_pulse: return the bounded operator pulse report
  - get_system_context: inspect canonical runtime / claim / stage state
  - get_context_view: load a strict-truth generated context view
  - get_startup_packet: derive the compact startup brief for a task
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ContextName = Literal["generic", "codex-wsl", "claude-windows", "claude-shell", "unknown"]
PolicyAction = Literal["orientation", "session_start_read_only", "session_start_mutating"]
BriefingLevel = Literal["trivial", "read_only", "non_trivial", "mutating"]

VALID_CONTEXT_NAMES: tuple[ContextName, ...] = ("generic", "codex-wsl", "claude-windows", "claude-shell", "unknown")
VALID_POLICY_ACTIONS: tuple[PolicyAction, ...] = (
    "orientation",
    "session_start_read_only",
    "session_start_mutating",
)
VALID_BRIEFING_LEVELS: tuple[BriefingLevel, ...] = ("trivial", "read_only", "non_trivial", "mutating")
CONTEXT_VIEW_INDEX: dict[str, dict[str, str]] = {
    "research": {"builder": "build_research_context", "owner": "scripts/tools/context_views.py"},
    "recent_performance": {"builder": "build_recent_performance_context", "owner": "scripts/tools/context_views.py"},
    "trading": {"builder": "build_trading_context", "owner": "scripts/tools/context_views.py"},
    "verification": {"builder": "build_verification_context", "owner": "scripts/tools/context_views.py"},
}


def _registry_bindings():
    from context.registry import TASKS, render_route_json, resolve_from_text, resolve_task

    return TASKS, render_route_json, resolve_from_text, resolve_task


def _candidate_payload(candidates: tuple[object, ...]) -> list[dict[str, object]]:
    return [
        {"task_id": candidate.task_id, "score": candidate.score, "matched_terms": list(candidate.matched_terms)}
        for candidate in candidates
    ]


def _list_task_routes() -> list[dict[str, object]]:
    TASKS, _render_route_json, _resolve_from_text, resolve_task = _registry_bindings()
    routes: list[dict[str, object]] = []
    for task_id in sorted(TASKS):
        route = resolve_task(task_id)
        routes.append(
            {
                "task_id": route.manifest.id,
                "title": route.manifest.title,
                "purpose": route.manifest.purpose,
                "verification_profile": route.verification.id,
                "briefing_contract": route.briefing_contract.id,
                "live_views": [view.id for view in route.live_views],
                "doctrine_files": list(route.doctrine_files),
                "canonical_files": list(route.canonical_files),
            }
        )
    return routes


def _list_context_views() -> list[dict[str, str]]:
    return [
        {
            "view": view_name,
            "builder": metadata["builder"],
            "owner": metadata["owner"],
        }
        for view_name, metadata in sorted(CONTEXT_VIEW_INDEX.items())
    ]


def _resolve_task_route(task_text: str | None = None, task_id: str | None = None) -> dict[str, object]:
    TASKS, render_route_json, resolve_from_text, resolve_task = _registry_bindings()
    if task_id:
        route = resolve_task(task_id)
        return json.loads(render_route_json(route))

    if not task_text or not task_text.strip():
        return {
            "matched": False,
            "reason": "Provide task_text or task_id.",
            "available_task_ids": sorted(TASKS),
            "candidates": [],
        }

    route, candidates = resolve_from_text(task_text.strip())
    if route is None:
        ambiguous = len(candidates) > 1 and candidates[1].score == candidates[0].score if candidates else False
        reason = "Ambiguous task match." if ambiguous else "No deterministic task match found."
        return {
            "matched": False,
            "task_text": task_text.strip(),
            "reason": reason,
            "available_task_ids": sorted(TASKS),
            "candidates": _candidate_payload(candidates),
        }

    return json.loads(render_route_json(route, candidates))


def _build_pulse_report(**kwargs):
    from scripts.tools.project_pulse import build_pulse

    return build_pulse(**kwargs)


def _format_pulse_report_json(report) -> str:
    from scripts.tools.project_pulse import format_json

    return format_json(report)


def _get_project_pulse(
    *,
    fast: bool = True,
    deep: bool = False,
    no_cache: bool = False,
    skip_drift: bool = False,
    skip_tests: bool = False,
    tool_name: str = "repo-state-mcp",
) -> dict[str, object]:
    report = _build_pulse_report(
        PROJECT_ROOT,
        fast=fast,
        deep=deep,
        no_cache=no_cache,
        skip_drift=skip_drift,
        skip_tests=skip_tests,
        tool_name=tool_name,
    )
    return json.loads(_format_pulse_report_json(report))


def _build_system_context_snapshot(**kwargs):
    from pipeline.system_context import build_system_context

    return build_system_context(**kwargs)


def _evaluate_policy(snapshot, action):
    from pipeline.system_context import evaluate_system_policy

    return evaluate_system_policy(snapshot, action)


def _get_system_context(
    *,
    context_name: ContextName = "generic",
    action: PolicyAction | None = None,
    active_tool: str | None = "repo-state-mcp",
    active_mode: str = "read-only",
) -> dict[str, object]:
    snapshot = _build_system_context_snapshot(
        PROJECT_ROOT,
        context_name=context_name,
        active_tool=active_tool,
        active_mode=active_mode,
    )
    payload: dict[str, object] = {"snapshot": snapshot.model_dump(mode="json")}
    if action is not None:
        payload["decision"] = _evaluate_policy(snapshot, action).model_dump(mode="json")
    return payload


def _build_context_view_payload(view_name: str) -> dict[str, object]:
    from pipeline.paths import GOLD_DB_PATH
    from scripts.tools.context_views import build_view

    return build_view(view_name, PROJECT_ROOT, GOLD_DB_PATH)


def _get_context_view(view_name: str) -> dict[str, object]:
    return _build_context_view_payload(view_name)


def _build_startup_brief_payload(**kwargs):
    from pipeline.system_brief import build_system_brief

    return build_system_brief(**kwargs)


def _get_startup_packet(
    *,
    task_text: str | None = None,
    task_id: str | None = None,
    briefing_level: BriefingLevel = "read_only",
    context_name: ContextName = "generic",
    active_tool: str = "repo-state-mcp",
    active_mode: str = "read-only",
) -> dict[str, object]:
    payload = _build_startup_brief_payload(
        PROJECT_ROOT,
        task_text=task_text,
        task_id=task_id,
        briefing_level=briefing_level,
        context_name=context_name,
        active_tool=active_tool,
        active_mode=active_mode,
    )
    return payload


def _build_server():
    from fastmcp import FastMCP

    mcp = FastMCP(
        "repo-state",
        instructions=(
            "Repo-local read-only control-plane and startup MCP for canompx3. "
            "Use it to resolve task routes, read the operator pulse, inspect system context, "
            "load strict-truth context views, and derive startup packets. "
            "Prefer fast pulse unless deeper verification is actually needed. "
            "This server explains repo state; it does not replace canonical code or gold-db truth."
        ),
    )

    @mcp.tool()
    def list_task_routes() -> list[dict[str, object]]:
        """List deterministic repo task routes with their live views and owner files."""

        return _list_task_routes()

    @mcp.tool()
    def list_context_views() -> list[dict[str, str]]:
        """List generated strict-truth context views available from this repo."""

        return _list_context_views()

    @mcp.tool()
    def resolve_task_route(task_text: str | None = None, task_id: str | None = None) -> dict[str, object]:
        """Resolve a natural-language task or explicit task_id into the canonical repo route.

        Args:
            task_text: Natural-language task description. Use this for ordinary work.
            task_id: Explicit route id when the exact route is already known.
        """

        return _resolve_task_route(task_text=task_text, task_id=task_id)

    @mcp.tool()
    def get_project_pulse(
        fast: bool = True,
        deep: bool = False,
        no_cache: bool = False,
        skip_drift: bool = False,
        skip_tests: bool = False,
        tool_name: str = "repo-state-mcp",
    ) -> dict[str, object]:
        """Return the bounded project pulse read-model.

        Args:
            fast: Prefer cached / cheap collectors only. Recommended default.
            deep: Run deeper fitness collection when available.
            no_cache: Ignore the expensive-results cache.
            skip_drift: Hard-skip drift checks in the pulse.
            skip_tests: Hard-skip test collection in the pulse.
            tool_name: Label used for session-delta / brief context.
        """

        return _get_project_pulse(
            fast=fast,
            deep=deep,
            no_cache=no_cache,
            skip_drift=skip_drift,
            skip_tests=skip_tests,
            tool_name=tool_name,
        )

    @mcp.tool()
    def get_system_context(
        context_name: ContextName = "generic",
        action: PolicyAction | None = None,
        active_tool: str | None = "repo-state-mcp",
        active_mode: str = "read-only",
    ) -> dict[str, object]:
        """Return canonical system context and optional policy decision.

        Args:
            context_name: Runtime context being evaluated.
            action: Optional policy action to evaluate against the snapshot.
            active_tool: Tool label for the snapshot.
            active_mode: Active session mode, usually `read-only` or `mutating`.
        """

        return _get_system_context(
            context_name=context_name,
            action=action,
            active_tool=active_tool,
            active_mode=active_mode,
        )

    @mcp.tool()
    def get_context_view(view_name: str) -> dict[str, object]:
        """Return one generated strict-truth context view.

        Args:
            view_name: One of `research`, `recent_performance`, `trading`, `verification`.
        """

        return _get_context_view(view_name)

    @mcp.tool()
    def get_startup_packet(
        task_text: str | None = None,
        task_id: str | None = None,
        briefing_level: BriefingLevel = "read_only",
        context_name: ContextName = "generic",
        active_tool: str = "repo-state-mcp",
        active_mode: str = "read-only",
    ) -> dict[str, object]:
        """Derive the compact startup packet for a task without writing `.session/task-route.md`.

        Args:
            task_text: Natural-language task description.
            task_id: Explicit route id when already known.
            briefing_level: Startup scope / latency budget to target.
            context_name: Runtime context being evaluated.
            active_tool: Tool label for the brief.
            active_mode: Active session mode, usually `read-only` or `mutating`.
        """

        return _get_startup_packet(
            task_text=task_text,
            task_id=task_id,
            briefing_level=briefing_level,
            context_name=context_name,
            active_tool=active_tool,
            active_mode=active_mode,
        )

    return mcp


if __name__ == "__main__":
    _build_server().run()

"""Canonical task routing registry for context resolution and startup briefing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from context.institutional import (
    ANSWER_CONTRACTS,
    BRIEFING_CONTRACTS,
    CONCEPTS,
    DECISION_PROTOCOLS,
    DRILLDOWN_PLAYBOOKS,
    UNDERSTANDING_PACKS,
    VARIABLES,
    AnswerContract,
    BriefingContract,
    Concept,
    DecisionProtocol,
    DrilldownPlaybook,
    UnderstandingPack,
    VariableOwner,
    model_to_dict,
    validate_institutional_contracts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DOCS_DIR = PROJECT_ROOT / "docs" / "context"
CONTEXT_GENERATED_NOTICE = "Generated from `context/registry.py` and `context/institutional.py`. Do not edit by hand."

FALLBACK_READ_SET: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "docs/governance/document_authority.md",
    "docs/governance/system_authority_map.md",
    "docs/runtime/action-queue.yaml",
    "HANDOFF.md",
)


@dataclass(frozen=True)
class VerificationStep:
    id: str
    title: str
    command: str
    summary: str


@dataclass(frozen=True)
class VerificationProfile:
    id: str
    title: str
    summary: str
    steps: tuple[str, ...]


@dataclass(frozen=True)
class LiveView:
    id: str
    title: str
    summary: str
    owner: str


@dataclass(frozen=True)
class TaskManifest:
    id: str
    title: str
    purpose: str
    intent_terms: tuple[str, ...]
    domains: tuple[str, ...]
    verification_profile: str
    concepts: tuple[str, ...]
    decision_protocol: str
    answer_contract: str
    understanding_packs: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    doctrine_files: tuple[str, ...] = ()
    canonical_files: tuple[str, ...] = ()
    live_views: tuple[str, ...] = ()
    drilldown_playbook: str | None = None
    briefing_contract: str = "orientation_briefing"
    expansion_triggers: tuple[str, ...] = ()
    priority: int = 1


@dataclass(frozen=True)
class TaskRoute:
    manifest: TaskManifest
    doctrine_files: tuple[str, ...]
    canonical_files: tuple[str, ...]
    live_views: tuple[LiveView, ...]
    verification: VerificationProfile
    verification_steps: tuple[VerificationStep, ...]
    concepts: tuple[Concept, ...]
    decision_protocol: DecisionProtocol
    answer_contract: AnswerContract
    understanding_packs: tuple[UnderstandingPack, ...]
    variables: tuple[VariableOwner, ...]
    drilldown_playbook: DrilldownPlaybook | None
    briefing_contract: BriefingContract
    expansion_triggers: tuple[str, ...]


@dataclass(frozen=True)
class TaskCandidate:
    task_id: str
    score: int
    matched_terms: tuple[str, ...]


VERIFICATION_STEPS: dict[str, VerificationStep] = {
    "project_pulse_fast": VerificationStep(
        id="project_pulse_fast",
        title="Project Pulse Fast",
        command="./.venv-wsl/bin/python scripts/tools/project_pulse.py --fast --format json",
        summary="Read a bounded operator summary without full expensive checks.",
    ),
    "system_context_text": VerificationStep(
        id="system_context_text",
        title="System Context",
        command="./.venv-wsl/bin/python scripts/tools/system_context.py --context codex-wsl --action orientation",
        summary="Read canonical startup control state.",
    ),
    "system_brief_json": VerificationStep(
        id="system_brief_json",
        title="System Brief",
        command="./.venv-wsl/bin/python scripts/tools/system_brief.py --format json",
        summary="Read the derived startup brief for the current route/task.",
    ),
    "pytest_full": VerificationStep(
        id="pytest_full",
        title="Full Targeted Pytest",
        command="./.venv-wsl/bin/python -m pytest -q",
        summary="Run the declared pytest surface before claiming done.",
    ),
    "drift_check": VerificationStep(
        id="drift_check",
        title="Drift Check",
        command="./.venv-wsl/bin/python pipeline/check_drift.py",
        summary="Verify canonical docs, contracts, and linked truth are in sync.",
    ),
    "render_context_catalog": VerificationStep(
        id="render_context_catalog",
        title="Render Context Catalog",
        command="./.venv-wsl/bin/python scripts/tools/render_context_catalog.py",
        summary="Re-render generated context docs from the registry.",
    ),
}


VERIFICATION_PROFILES: dict[str, VerificationProfile] = {
    "orientation": VerificationProfile(
        id="orientation",
        title="Orientation",
        summary="Bounded startup checks for read-only orientation.",
        steps=("system_brief_json", "project_pulse_fast", "system_context_text"),
    ),
    "investigation": VerificationProfile(
        id="investigation",
        title="Investigation",
        summary="Compact live-state verification for research or diagnosis work.",
        steps=("project_pulse_fast", "system_context_text"),
    ),
    "runtime_status": VerificationProfile(
        id="runtime_status",
        title="Runtime Status",
        summary="Operator-grade runtime status without broad research expansion.",
        steps=("system_brief_json", "project_pulse_fast", "system_context_text"),
    ),
    "done": VerificationProfile(
        id="done",
        title="Done",
        summary="High-confidence completion profile.",
        steps=("system_brief_json", "project_pulse_fast", "pytest_full", "drift_check", "render_context_catalog"),
    ),
}


LIVE_VIEWS: dict[str, LiveView] = {
    "gold_db_mcp": LiveView(
        id="gold_db_mcp",
        title="Gold DB MCP",
        summary="Canonical live trading/research query surface.",
        owner="trading_app/mcp_server.py",
    ),
    "research_context": LiveView(
        id="research_context",
        title="Research Context View",
        summary="Generated research view with strict truth boundaries.",
        owner="scripts/tools/context_views.py",
    ),
    "recent_performance_context": LiveView(
        id="recent_performance_context",
        title="Recent Performance Context View",
        summary="Generated recent-performance view for fit weakness questions.",
        owner="scripts/tools/context_views.py",
    ),
    "trading_context": LiveView(
        id="trading_context",
        title="Trading Context View",
        summary="Generated trading/runtime context view.",
        owner="scripts/tools/context_views.py",
    ),
    "verification_context": LiveView(
        id="verification_context",
        title="Verification Context View",
        summary="Generated verification-focused context view.",
        owner="scripts/tools/context_views.py",
    ),
    "system_brief": LiveView(
        id="system_brief",
        title="System Brief",
        summary="Derived minimal startup read-model for the current task.",
        owner="pipeline/system_brief.py",
    ),
}


TASKS: dict[str, TaskManifest] = {
    "repo_workflow_audit": TaskManifest(
        id="repo_workflow_audit",
        title="Repo Workflow Audit",
        purpose="Audit startup, routing, launcher, or hook workflow surfaces for rigor, token efficiency, and operator clarity.",
        intent_terms=(
            "context resolver",
            "context_resolver",
            "token waste",
            "tokens being wasted",
            "token burn",
            "startup",
            "hook",
            "hooks",
            "launcher",
            "launchers",
            "routing",
            "workflow",
            "operator surface",
        ),
        domains=("repo_governance",),
        verification_profile="investigation",
        concepts=("runtime_control_plane", "task_packet_contract"),
        decision_protocol="system_orientation_protocol",
        answer_contract="orientation_answer",
        understanding_packs=("coding_runtime_pack", "project_orientation_pack"),
        doctrine_files=(
            "AGENTS.md",
            "CLAUDE.md",
            "CODEX.md",
            "docs/governance/document_authority.md",
            "docs/governance/system_authority_map.md",
        ),
        canonical_files=(
            "context/registry.py",
            "pipeline/system_authority.py",
            "pipeline/system_context.py",
            "pipeline/system_brief.py",
            "pipeline/work_queue.py",
            "scripts/tools/context_resolver.py",
            "scripts/tools/work_queue.py",
            "scripts/tools/session_preflight.py",
            "scripts/tools/task_route_packet.py",
            "scripts/infra/windows_agent_launch.py",
            "scripts/infra/claude-worktree.sh",
            "scripts/infra/codex-project.sh",
            "scripts/infra/codex-worktree.sh",
        ),
        live_views=("verification_context", "system_brief"),
        briefing_contract="orientation_briefing",
        expansion_triggers=("unmatched route", "startup noise", "launcher drift", "hook churn"),
        priority=3,
    ),
    "research_discovery": TaskManifest(
        id="research_discovery",
        title="Research Discovery",
        purpose="Triage new edge ideas, discovery questions, and hypothesis-shaping work using the institutional discovery front door.",
        intent_terms=(
            "discover",
            "discovery",
            "find edge",
            "new edge",
            "edge idea",
            "hypothesis",
            "triage",
            "scan for edges",
            "edge discovery",
            "explore idea",
        ),
        domains=("research_methodology", "repo_governance"),
        verification_profile="investigation",
        concepts=("sacred_holdout_policy", "runtime_control_plane"),
        decision_protocol="research_investigation_protocol",
        answer_contract="research_investigation_answer",
        understanding_packs=("coding_runtime_pack", "trading_runtime_pack", "research_methodology_pack"),
        variables=("orb_utc_window", "holdout_policy_var"),
        doctrine_files=(
            "RESEARCH_RULES.md",
            "TRADING_RULES.md",
            "docs/STRATEGY_BLUEPRINT.md",
            "docs/institutional/research_pipeline_contract.md",
            "docs/institutional/pre_registered_criteria.md",
            "docs/institutional/mechanism_priors.md",
            "docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md",
        ),
        canonical_files=(
            "trading_app/holdout_policy.py",
            "pipeline/build_daily_features.py",
            "pipeline/asset_configs.py",
            "pipeline/cost_model.py",
            "pipeline/dst.py",
            "scripts/tools/prereg_front_door.py",
            "scripts/infra/prereg-loop.sh",
            "trading_app/strategy_discovery.py",
            "pipeline/db_contracts.py",
        ),
        live_views=("gold_db_mcp", "research_context", "system_brief"),
        briefing_contract="investigation_briefing",
        expansion_triggers=(
            "current-stack vs architecture ambiguity appears",
            "holdout ambiguity appears",
            "runtime translation becomes the real question",
        ),
        priority=6,
    ),
    "research_investigation": TaskManifest(
        id="research_investigation",
        title="Research Investigation",
        purpose="Investigate performance changes, edge behavior, or discovery questions using canonical research truth.",
        intent_terms=(
            "investigate",
            "why",
            "drop",
            "dropped",
            "win rate",
            "research",
            "performance",
            "mes",
            "mnq",
            "mgc",
            "orb",
        ),
        domains=("research_methodology", "repo_governance"),
        verification_profile="investigation",
        concepts=("sacred_holdout_policy", "runtime_control_plane"),
        decision_protocol="research_investigation_protocol",
        answer_contract="research_investigation_answer",
        understanding_packs=("coding_runtime_pack", "trading_runtime_pack", "research_methodology_pack"),
        variables=("orb_utc_window", "holdout_policy_var"),
        doctrine_files=(
            "RESEARCH_RULES.md",
            "TRADING_RULES.md",
            "docs/STRATEGY_BLUEPRINT.md",
            "docs/institutional/pre_registered_criteria.md",
        ),
        canonical_files=(
            "trading_app/holdout_policy.py",
            "pipeline/asset_configs.py",
            "pipeline/cost_model.py",
            "pipeline/dst.py",
            "trading_app/strategy_fitness.py",
            "pipeline/db_contracts.py",
        ),
        live_views=("gold_db_mcp", "research_context", "recent_performance_context", "system_brief"),
        drilldown_playbook="research_recent_performance_drilldown",
        briefing_contract="investigation_briefing",
        expansion_triggers=(
            "live evidence conflicts",
            "holdout ambiguity appears",
            "lane-level trade evidence is needed",
        ),
        priority=5,
    ),
    "live_trading_status": TaskManifest(
        id="live_trading_status",
        title="Live Trading Status",
        purpose="Understand the current runtime/deployment state without loading research doctrine.",
        intent_terms=(
            "what's live",
            "live tonight",
            "runtime",
            "status",
            "operator",
            "deployment",
            "readiness",
            "preflight",
            "session",
        ),
        domains=("trading_runtime",),
        verification_profile="runtime_status",
        concepts=("runtime_control_plane",),
        decision_protocol="live_status_protocol",
        answer_contract="live_status_answer",
        understanding_packs=("coding_runtime_pack", "trading_runtime_pack"),
        variables=("deployable_validated_relation",),
        doctrine_files=("TRADING_RULES.md", "CLAUDE.md"),
        canonical_files=("trading_app/prop_profiles.py", "trading_app/lifecycle_state.py", "pipeline/db_contracts.py"),
        live_views=("trading_context", "system_brief"),
        briefing_contract="orientation_briefing",
        expansion_triggers=("operator asks for deeper investigation", "runtime control reports a blocker"),
        priority=4,
    ),
    "completion_claim": TaskManifest(
        id="completion_claim",
        title="Completion Claim",
        purpose="Verify whether a change is actually done before closing it.",
        intent_terms=("done", "complete", "verify", "close", "ship", "finished"),
        domains=("repo_governance",),
        verification_profile="done",
        concepts=("runtime_control_plane", "task_packet_contract"),
        decision_protocol="completion_protocol",
        answer_contract="completion_answer",
        understanding_packs=("coding_runtime_pack",),
        doctrine_files=("CLAUDE.md", "docs/governance/system_authority_map.md"),
        canonical_files=(
            "pipeline/check_drift.py",
            "scripts/tools/session_preflight.py",
            "scripts/tools/project_pulse.py",
        ),
        live_views=("verification_context", "system_brief"),
        briefing_contract="mutating_briefing",
        expansion_triggers=("verification is failing", "scope is unclear"),
        priority=4,
    ),
    "docs_drift_audit": TaskManifest(
        id="docs_drift_audit",
        title="Docs Drift Audit",
        purpose="Audit whether docs and generated context contracts are stale against code-backed truth.",
        intent_terms=("doc drift", "documentation drift", "stale docs", "authority map", "context docs"),
        domains=("repo_governance",),
        verification_profile="investigation",
        concepts=("runtime_control_plane",),
        decision_protocol="implementation_protocol",
        answer_contract="implementation_answer",
        understanding_packs=("coding_runtime_pack", "project_orientation_pack"),
        doctrine_files=(
            "CLAUDE.md",
            "docs/governance/document_authority.md",
            "docs/governance/system_authority_map.md",
        ),
        canonical_files=(
            "pipeline/system_authority.py",
            "scripts/tools/render_context_catalog.py",
            "pipeline/check_drift.py",
        ),
        live_views=("verification_context", "system_brief"),
        briefing_contract="orientation_briefing",
        expansion_triggers=("generated docs differ", "missing authority path"),
        priority=2,
    ),
    "system_orientation": TaskManifest(
        id="system_orientation",
        title="System Orientation",
        purpose="Return the smallest complete startup model for the repo or current workstream.",
        intent_terms=(
            "minimal complete model",
            "orientation",
            "startup",
            "understand the repo",
            "mental model",
            "project brain",
        ),
        domains=("repo_governance",),
        verification_profile="orientation",
        concepts=("runtime_control_plane", "task_packet_contract"),
        decision_protocol="system_orientation_protocol",
        answer_contract="orientation_answer",
        understanding_packs=("coding_runtime_pack", "trading_runtime_pack", "project_orientation_pack"),
        variables=("deployable_validated_relation", "orb_utc_window"),
        doctrine_files=(
            "AGENTS.md",
            "CLAUDE.md",
            "CODEX.md",
            "docs/governance/document_authority.md",
            "docs/governance/system_authority_map.md",
            "docs/runtime/action-queue.yaml",
            "HANDOFF.md",
        ),
        canonical_files=(
            "pipeline/system_authority.py",
            "pipeline/system_context.py",
            "pipeline/work_capsule.py",
            "pipeline/system_brief.py",
            "pipeline/work_queue.py",
            "context/registry.py",
            "context/institutional.py",
        ),
        live_views=("system_brief",),
        briefing_contract="orientation_briefing",
        expansion_triggers=(
            "active work capsule exists",
            "user asks for depth",
            "operator/runtime state must be queried",
        ),
        priority=1,
    ),
}


def _unique(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def resolve_task(task_id: str) -> TaskRoute:
    manifest = TASKS[task_id]
    verification = VERIFICATION_PROFILES[manifest.verification_profile]
    return TaskRoute(
        manifest=manifest,
        doctrine_files=_unique(manifest.doctrine_files),
        canonical_files=_unique(manifest.canonical_files),
        live_views=tuple(LIVE_VIEWS[view_id] for view_id in manifest.live_views),
        verification=verification,
        verification_steps=tuple(VERIFICATION_STEPS[step_id] for step_id in verification.steps),
        concepts=tuple(CONCEPTS[item_id] for item_id in manifest.concepts),
        decision_protocol=DECISION_PROTOCOLS[manifest.decision_protocol],
        answer_contract=ANSWER_CONTRACTS[manifest.answer_contract],
        understanding_packs=tuple(UNDERSTANDING_PACKS[item_id] for item_id in manifest.understanding_packs),
        variables=tuple(VARIABLES[item_id] for item_id in manifest.variables),
        drilldown_playbook=(
            DRILLDOWN_PLAYBOOKS[manifest.drilldown_playbook] if manifest.drilldown_playbook is not None else None
        ),
        briefing_contract=BRIEFING_CONTRACTS[manifest.briefing_contract],
        expansion_triggers=manifest.expansion_triggers,
    )


def candidate_tasks(task_text: str) -> list[TaskCandidate]:
    lowered = task_text.lower()
    candidates: list[TaskCandidate] = []
    for manifest in TASKS.values():
        matched_terms = tuple(term for term in manifest.intent_terms if term.lower() in lowered)
        if not matched_terms:
            continue
        score = len(matched_terms) * 10 + manifest.priority
        candidates.append(TaskCandidate(task_id=manifest.id, score=score, matched_terms=matched_terms))
    candidates.sort(key=lambda item: (-item.score, item.task_id))
    return candidates


def resolve_from_text(task_text: str) -> tuple[TaskRoute | None, tuple[TaskCandidate, ...]]:
    candidates = tuple(candidate_tasks(task_text))
    if not candidates:
        return None, candidates
    top = candidates[0]
    tied = tuple(candidate for candidate in candidates if candidate.score == top.score)
    if len(tied) > 1:
        return None, tied
    return resolve_task(top.task_id), candidates


def validate_registry() -> list[str]:
    violations = list(validate_institutional_contracts())

    for task in TASKS.values():
        if task.verification_profile not in VERIFICATION_PROFILES:
            violations.append(f"task {task.id} references unknown verification profile {task.verification_profile}")
        if task.decision_protocol not in DECISION_PROTOCOLS:
            violations.append(f"task {task.id} references unknown decision protocol {task.decision_protocol}")
        if task.answer_contract not in ANSWER_CONTRACTS:
            violations.append(f"task {task.id} references unknown answer contract {task.answer_contract}")
        if task.briefing_contract not in BRIEFING_CONTRACTS:
            violations.append(f"task {task.id} references unknown briefing contract {task.briefing_contract}")
        for field_name, ids, registry in (
            ("concept", task.concepts, CONCEPTS),
            ("understanding pack", task.understanding_packs, UNDERSTANDING_PACKS),
            ("variable", task.variables, VARIABLES),
            ("live view", task.live_views, LIVE_VIEWS),
        ):
            for item_id in ids:
                if item_id not in registry:
                    violations.append(f"task {task.id} references unknown {field_name} {item_id}")
        if task.drilldown_playbook is not None and task.drilldown_playbook not in DRILLDOWN_PLAYBOOKS:
            violations.append(f"task {task.id} references unknown drilldown playbook {task.drilldown_playbook}")
        for path in (*task.doctrine_files, *task.canonical_files):
            if not (PROJECT_ROOT / path).exists():
                violations.append(f"task {task.id} references missing path {path}")

    for step in VERIFICATION_STEPS.values():
        if not step.command.strip():
            violations.append(f"verification step {step.id} has an empty command")

    return violations


def _route_payload(route: TaskRoute, candidates: tuple[TaskCandidate, ...] = ()) -> dict[str, Any]:
    return {
        "matched": True,
        "task": model_to_dict(route.manifest),
        "route_id": route.manifest.id,
        "briefing_contract": model_to_dict(route.briefing_contract),
        "decision_protocol": model_to_dict(route.decision_protocol),
        "answer_contract": model_to_dict(route.answer_contract),
        "drilldown_playbook": model_to_dict(route.drilldown_playbook) if route.drilldown_playbook else None,
        "concepts": [model_to_dict(item) for item in route.concepts],
        "understanding_packs": [model_to_dict(item) for item in route.understanding_packs],
        "variables": [model_to_dict(item) for item in route.variables],
        "doctrine_files": list(route.doctrine_files),
        "canonical_files": list(route.canonical_files),
        "live_views": [model_to_dict(item) for item in route.live_views],
        "verification": model_to_dict(route.verification),
        "verification_steps": [model_to_dict(item) for item in route.verification_steps],
        "expansion_triggers": list(route.expansion_triggers),
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def render_route_json(route: TaskRoute, candidates: tuple[TaskCandidate, ...] = ()) -> str:
    return json.dumps(_route_payload(route, candidates), indent=2, sort_keys=True)


def render_route_text(route: TaskRoute, candidates: tuple[TaskCandidate, ...] = ()) -> str:
    lines = [
        f"Task route: {route.manifest.id}",
        f"Title: {route.manifest.title}",
        f"Purpose: {route.manifest.purpose}",
        f"Decision protocol: {route.decision_protocol.id}",
        f"Answer contract: {route.answer_contract.id}",
        f"Briefing contract: {route.briefing_contract.id}",
        "Doctrine files:",
        *[f"  - {path}" for path in route.doctrine_files],
        "Canonical files:",
        *[f"  - {path}" for path in route.canonical_files],
        "Live views:",
        *[f"  - {view.id}" for view in route.live_views],
        "Verification steps:",
        *[f"  - {step.id}" for step in route.verification_steps],
    ]
    if route.drilldown_playbook is not None:
        lines.append(f"Drilldown playbook: {route.drilldown_playbook.id}")
    if candidates:
        lines.append("Candidates:")
        lines.extend(f"  - {candidate.task_id} ({candidate.score})" for candidate in candidates)
    return "\n".join(lines) + "\n"


def render_route_markdown(route: TaskRoute, candidates: tuple[TaskCandidate, ...] = ()) -> str:
    lines = [
        "# Context Route",
        "",
        f"- **Task:** `{route.manifest.id}`",
        f"- **Decision protocol:** `{route.decision_protocol.id}`",
        f"- **Answer contract:** `{route.answer_contract.id}`",
        f"- **Briefing contract:** `{route.briefing_contract.id}`",
        "",
        "## Doctrine Files",
        "",
        *[f"- `{path}`" for path in route.doctrine_files],
        "",
        "## Canonical Files",
        "",
        *[f"- `{path}`" for path in route.canonical_files],
        "",
        "## Live Views",
        "",
        *[f"- `{view.id}` — {view.summary}" for view in route.live_views],
        "",
        "## Verification Steps",
        "",
        *[f"- `{step.id}` — `{step.command}`" for step in route.verification_steps],
    ]
    if route.drilldown_playbook is not None:
        lines.extend(
            [
                "",
                "## Drilldown Playbook",
                "",
                f"- `{route.drilldown_playbook.id}` — {route.drilldown_playbook.summary}",
            ]
        )
    if candidates:
        lines.extend(["", "## Candidates", ""])
        lines.extend(f"- `{candidate.task_id}` (score={candidate.score})" for candidate in candidates)
    return "\n".join(lines) + "\n"


def render_source_catalog_markdown() -> str:
    lines = [
        "# Context Source Catalog",
        "",
        CONTEXT_GENERATED_NOTICE,
        "",
        "Generated catalog of canonical routing sources and published read models.",
        "",
        "## Domains",
        "",
    ]
    for domain in sorted({domain for task in TASKS.values() for domain in task.domains}):
        lines.append(f"- `{domain}`")
    lines.extend(["", "## Concepts", ""])
    for concept in CONCEPTS.values():
        lines.append(f"- `{concept.id}` — `{concept.owner_path}`")
    lines.extend(["", "## Control-Plane Truth", ""])
    lines.extend(
        [
            "- `docs/runtime/action-queue.yaml` — canonical active-work truth",
            "- `.session/work_queue_leases.json` — local runtime ownership only",
            "- `HANDOFF.md` — rendered baton view, not canonical active-work truth",
        ]
    )
    lines.extend(["", "## Live Views", ""])
    for view in LIVE_VIEWS.values():
        lines.append(f"- `{view.id}` — {view.summary} (`{view.owner}`)")
    lines.extend(["", "## Verification Steps", ""])
    for step in VERIFICATION_STEPS.values():
        lines.append(f"- `{step.id}` — `{step.command}`")
    lines.extend(["", "## Understanding Packs", ""])
    for pack in UNDERSTANDING_PACKS.values():
        lines.append(f"- `{pack.id}` — {pack.summary}")
    lines.extend(["", "## Variables", ""])
    for variable in VARIABLES.values():
        lines.append(f"- `{variable.id}` — `{variable.owner_path}`")
    return "\n".join(lines) + "\n"


def render_task_routes_markdown() -> str:
    lines = [
        "# Task Routes",
        "",
        CONTEXT_GENERATED_NOTICE,
        "",
        "Generated canonical task routes for cold-start orientation.",
        "",
    ]
    for task_id in sorted(TASKS):
        route = resolve_task(task_id)
        lines.extend(
            [
                f"## `{task_id}`",
                "",
                f"- Purpose: {route.manifest.purpose}",
                f"- Verification profile: `{route.verification.id}`",
                f"- Briefing contract: `{route.briefing_contract.id}`",
                f"- Packs: {', '.join(f'`{item.id}`' for item in route.understanding_packs) or 'none'}",
                f"- Doctrine: {', '.join(f'`{item}`' for item in route.doctrine_files)}",
                f"- Canonical owners: {', '.join(f'`{item}`' for item in route.canonical_files)}",
                f"- Live views: {', '.join(f'`{item.id}`' for item in route.live_views)}",
                "",
            ]
        )
    lines.extend(["## Fallback Read Set", ""])
    lines.extend(f"- `{path}`" for path in FALLBACK_READ_SET)
    lines.extend(
        [
            "",
            "## Control-Plane Notes",
            "",
            "- `docs/runtime/action-queue.yaml` is the canonical active-work registry when present.",
            "- `HANDOFF.md` is baton context only and should be rendered from the queue, not maintained as a parallel backlog.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_institutional_markdown() -> str:
    lines = [
        "# Institutional Routing Contracts",
        "",
        CONTEXT_GENERATED_NOTICE,
        "",
        "Generated registry of concepts, protocols, answer contracts, and briefing rules.",
        "",
        "## Concepts",
        "",
    ]
    for concept in CONCEPTS.values():
        lines.append(f"- `{concept.id}` — {concept.summary} (`{concept.owner_path}`)")
    lines.extend(["", "## Decision Protocols", ""])
    for item in DECISION_PROTOCOLS.values():
        lines.append(f"- `{item.id}` — {item.summary}")
    lines.extend(["", "## Answer Contracts", ""])
    for item in ANSWER_CONTRACTS.values():
        lines.append(f"- `{item.id}` — {item.summary}")
    lines.extend(["", "## Understanding Packs", ""])
    for item in UNDERSTANDING_PACKS.values():
        lines.append(f"- `{item.id}` — {item.summary}")
    lines.extend(["", "## Drilldown Playbooks", ""])
    for item in DRILLDOWN_PLAYBOOKS.values():
        lines.append(f"- `{item.id}` — {item.summary}")
    lines.extend(["", "## Briefing Contracts", ""])
    for item in BRIEFING_CONTRACTS.values():
        lines.append(f"- `{item.id}` — {item.summary}")
    return "\n".join(lines) + "\n"


def render_readme_markdown() -> str:
    lines = [
        "# Context Routing",
        "",
        CONTEXT_GENERATED_NOTICE,
        "",
        "This directory contains generated docs for the canonical task routing registry.",
        "",
        "## Files",
        "",
        "- `source-catalog.md` — published live views, packs, variables, and verification steps",
        "- `task-routes.md` — deterministic task routes",
        "- `institutional-contracts.md` — concepts, protocols, and briefing contracts",
        "",
        "## Control Plane",
        "",
        "- `docs/runtime/action-queue.yaml` is the canonical active-work registry.",
        "- `HANDOFF.md` is a generated baton view for session startup only.",
    ]
    return "\n".join(lines) + "\n"

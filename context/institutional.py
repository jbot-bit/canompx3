"""Institutional routing contracts for task context and startup briefing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from trading_app.ai.sql_adapter import QueryTemplate


@dataclass(frozen=True)
class Concept:
    id: str
    title: str
    summary: str
    owner_path: str


@dataclass(frozen=True)
class DecisionProtocol:
    id: str
    title: str
    summary: str
    required_moves: tuple[str, ...]


@dataclass(frozen=True)
class AnswerContract:
    id: str
    title: str
    summary: str
    required_sections: tuple[str, ...]


@dataclass(frozen=True)
class UnderstandingPack:
    id: str
    title: str
    summary: str
    owner_paths: tuple[str, ...]


@dataclass(frozen=True)
class VariableOwner:
    id: str
    title: str
    summary: str
    owner_path: str


@dataclass(frozen=True)
class DrilldownStep:
    id: str
    title: str
    template: str
    goal: str


@dataclass(frozen=True)
class DrilldownPlaybook:
    id: str
    title: str
    summary: str
    steps: tuple[DrilldownStep, ...]


@dataclass(frozen=True)
class BriefingContract:
    id: str
    title: str
    summary: str
    required_fields: tuple[str, ...]
    expansion_triggers: tuple[str, ...]


CONCEPTS: dict[str, Concept] = {
    "sacred_holdout_policy": Concept(
        id="sacred_holdout_policy",
        title="Sacred Holdout Policy",
        summary="Research and validation work must honor the holdout boundary and grandfather rules.",
        owner_path="trading_app/holdout_policy.py",
    ),
    "runtime_control_plane": Concept(
        id="runtime_control_plane",
        title="Runtime Control Plane",
        summary="Repo startup, interpreter, git, and claim state must be read from canonical control surfaces.",
        owner_path="pipeline/system_context.py",
    ),
    "shared_root_mutation_override": Concept(
        id="shared_root_mutation_override",
        title="Shared Root Mutation Override",
        summary="Mutating work in the shared root is exceptional and must require an explicit override.",
        owner_path="pipeline/system_context.py",
    ),
    "task_packet_contract": Concept(
        id="task_packet_contract",
        title="Task Packet Contract",
        summary="Active scoped work should carry one explicit capsule with scope, authorities, and verification.",
        owner_path="pipeline/work_capsule.py",
    ),
}


DECISION_PROTOCOLS: dict[str, DecisionProtocol] = {
    "research_investigation_protocol": DecisionProtocol(
        id="research_investigation_protocol",
        title="Research Investigation Protocol",
        summary="Resolve the live question, confirm the canonical research constraints, and gather evidence before explaining.",
        required_moves=(
            "state the question precisely",
            "name the holdout and canonical owners",
            "query live evidence instead of citing stale docs",
            "separate explanation from recommendation",
        ),
    ),
    "completion_protocol": DecisionProtocol(
        id="completion_protocol",
        title="Completion Protocol",
        summary="Do not claim done until the declared verification profile is green or the exact blocker is named.",
        required_moves=(
            "state the target outcome",
            "run the declared verification profile",
            "name residual risk or blocking failures",
        ),
    ),
    "implementation_protocol": DecisionProtocol(
        id="implementation_protocol",
        title="Implementation Protocol",
        summary="Read the owning rules first, stay inside the scoped surface, and verify behavior after edits.",
        required_moves=(
            "identify canonical owners",
            "keep scope bounded",
            "verify the changed behavior",
        ),
    ),
    "live_status_protocol": DecisionProtocol(
        id="live_status_protocol",
        title="Live Status Protocol",
        summary="For runtime questions, prefer control state and deployment summaries over research doctrine.",
        required_moves=(
            "read live control state",
            "separate current runtime truth from historical notes",
            "state operator-relevant blockers first",
        ),
    ),
    "system_orientation_protocol": DecisionProtocol(
        id="system_orientation_protocol",
        title="System Orientation Protocol",
        summary="Return the smallest complete startup model for the current task or repo surface.",
        required_moves=(
            "identify the route",
            "load only required doctrine and owners",
            "expand only on explicit triggers",
        ),
    ),
}


ANSWER_CONTRACTS: dict[str, AnswerContract] = {
    "research_investigation_answer": AnswerContract(
        id="research_investigation_answer",
        title="Research Investigation Answer",
        summary="A research answer should report the question, evidence surface, and constrained conclusion.",
        required_sections=("question", "canonical_constraints", "live_evidence", "conclusion"),
    ),
    "completion_answer": AnswerContract(
        id="completion_answer",
        title="Completion Answer",
        summary="A completion answer should say whether the target is done, verified, or blocked.",
        required_sections=("status", "verification", "blockers_or_risks"),
    ),
    "implementation_answer": AnswerContract(
        id="implementation_answer",
        title="Implementation Answer",
        summary="An implementation answer should explain the intended change and bounded verification.",
        required_sections=("change", "scope", "verification"),
    ),
    "live_status_answer": AnswerContract(
        id="live_status_answer",
        title="Live Status Answer",
        summary="A runtime answer should foreground readiness, blockers, and near-term operator state.",
        required_sections=("status", "blockers", "next_runtime_move"),
    ),
    "orientation_answer": AnswerContract(
        id="orientation_answer",
        title="Orientation Answer",
        summary="An orientation answer should compress the startup model to the minimum sufficient surface.",
        required_sections=("route", "owners", "live_views", "expansion_triggers"),
    ),
}


UNDERSTANDING_PACKS: dict[str, UnderstandingPack] = {
    "coding_runtime_pack": UnderstandingPack(
        id="coding_runtime_pack",
        title="Coding Runtime Pack",
        summary="Repo shell, git, interpreter, and verification control surfaces.",
        owner_paths=("CLAUDE.md", "CODEX.md", "pipeline/system_context.py", "scripts/tools/session_preflight.py"),
    ),
    "trading_runtime_pack": UnderstandingPack(
        id="trading_runtime_pack",
        title="Trading Runtime Pack",
        summary="Live/trading runtime owners, deployment state, and operator truth surfaces.",
        owner_paths=("TRADING_RULES.md", "trading_app/prop_profiles.py", "trading_app/lifecycle_state.py"),
    ),
    "research_methodology_pack": UnderstandingPack(
        id="research_methodology_pack",
        title="Research Methodology Pack",
        summary="Research doctrine, holdout policy, and institutional thresholds.",
        owner_paths=(
            "RESEARCH_RULES.md",
            "docs/STRATEGY_BLUEPRINT.md",
            "docs/institutional/pre_registered_criteria.md",
            "trading_app/holdout_policy.py",
        ),
    ),
    "project_orientation_pack": UnderstandingPack(
        id="project_orientation_pack",
        title="Project Orientation Pack",
        summary="Minimal project-wide startup truth for cold-start repo work.",
        owner_paths=(
            "AGENTS.md",
            "HANDOFF.md",
            "docs/governance/document_authority.md",
            "docs/governance/system_authority_map.md",
        ),
    ),
}


VARIABLES: dict[str, VariableOwner] = {
    "orb_utc_window": VariableOwner(
        id="orb_utc_window",
        title="ORB UTC Window",
        summary="The canonical ORB timing resolver; never derive from break timestamps.",
        owner_path="pipeline/dst.py",
    ),
    "holdout_policy_var": VariableOwner(
        id="holdout_policy_var",
        title="Holdout Policy",
        summary="Sacred-from and grandfather cutoffs governing research and validation.",
        owner_path="trading_app/holdout_policy.py",
    ),
    "deployable_validated_relation": VariableOwner(
        id="deployable_validated_relation",
        title="Deployable Validated Relation",
        summary="Published read-model for deployable validated setups.",
        owner_path="pipeline/db_contracts.py",
    ),
}


DRILLDOWN_PLAYBOOKS: dict[str, DrilldownPlaybook] = {
    "research_recent_performance_drilldown": DrilldownPlaybook(
        id="research_recent_performance_drilldown",
        title="Recent Performance Drilldown",
        summary="Move from broad fit weakness to lane-level recent-performance evidence using approved query templates.",
        steps=(
            DrilldownStep(
                id="validated_summary",
                title="Validate the affected shelf",
                template="validated_summary",
                goal="Confirm the active/deployable surface before deeper diagnosis.",
            ),
            DrilldownStep(
                id="performance_stats",
                title="Inspect aggregate performance",
                template="performance_stats",
                goal="Check whether the question is about real performance deterioration or a thin-sample artifact.",
            ),
            DrilldownStep(
                id="rolling_stability",
                title="Check recent stability",
                template="rolling_stability",
                goal="Find the recent time window where the edge weakened.",
            ),
            DrilldownStep(
                id="trade_history",
                title="Inspect lane-level trades",
                template="trade_history",
                goal="Gather lane-level evidence once the weak window is identified.",
            ),
        ),
    )
}


BRIEFING_CONTRACTS: dict[str, BriefingContract] = {
    "orientation_briefing": BriefingContract(
        id="orientation_briefing",
        title="Orientation Briefing",
        summary="Minimal complete repo orientation for cold-start work.",
        required_fields=(
            "task_id",
            "route_id",
            "briefing_level",
            "doctrine_chain",
            "canonical_owners",
            "required_live_views",
            "verification_profile",
            "blocking_issues",
            "warning_issues",
        ),
        expansion_triggers=(
            "route ambiguity remains",
            "user asks for deeper explanation",
            "exact institutional wording is required",
            "live state must be queried before answering",
        ),
    ),
    "investigation_briefing": BriefingContract(
        id="investigation_briefing",
        title="Investigation Briefing",
        summary="Compact but complete research/truth-finding startup brief.",
        required_fields=(
            "task_id",
            "route_id",
            "doctrine_chain",
            "canonical_owners",
            "required_live_views",
            "verification_profile",
            "decision_refs",
            "debt_refs",
        ),
        expansion_triggers=(
            "evidence conflicts",
            "holdout ambiguity appears",
            "the answer needs exact numbers from live state",
        ),
    ),
    "mutating_briefing": BriefingContract(
        id="mutating_briefing",
        title="Mutating Briefing",
        summary="Fail-closed mutating startup brief with scoped ownership and verification.",
        required_fields=(
            "task_id",
            "route_id",
            "briefing_level",
            "canonical_owners",
            "verification_profile",
            "work_capsule_ref",
            "blocking_issues",
            "warning_issues",
        ),
        expansion_triggers=(
            "work capsule is missing or ambiguous",
            "canonical owner file is missing",
            "policy blocker is present",
        ),
    ),
}


def validate_institutional_contracts() -> list[str]:
    """Validate internal referential integrity for the institutional contracts."""
    violations: list[str] = []
    valid_templates = {item.value for item in QueryTemplate}

    for concept in CONCEPTS.values():
        if "/" not in concept.owner_path and not concept.owner_path.endswith(".md"):
            violations.append(f"concept {concept.id} has a non-path owner: {concept.owner_path}")

    for variable in VARIABLES.values():
        if "/" not in variable.owner_path and not variable.owner_path.endswith(".md"):
            violations.append(f"variable {variable.id} has a non-path owner: {variable.owner_path}")

    for pack in UNDERSTANDING_PACKS.values():
        if not pack.owner_paths:
            violations.append(f"understanding pack {pack.id} has no owner paths")

    for playbook in DRILLDOWN_PLAYBOOKS.values():
        if not playbook.steps:
            violations.append(f"drilldown playbook {playbook.id} has no steps")
        for step in playbook.steps:
            if step.template not in valid_templates:
                violations.append(
                    f"drilldown playbook {playbook.id} step {step.id} uses unknown query template {step.template}"
                )

    for contract in BRIEFING_CONTRACTS.values():
        if not contract.required_fields:
            violations.append(f"briefing contract {contract.id} has no required fields")

    return violations


def model_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)

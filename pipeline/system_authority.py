"""Canonical whole-project authority registry.

This module is the code-backed source for the system authority map and related
orientation surfaces. The goal is to keep project identity, truth surfaces,
and ownership linked to one canonical registry instead of duplicated in prose,
drift checks, and entrypoint tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SYSTEM_AUTHORITY_MAP_RELATIVE_PATH = Path("docs/governance/system_authority_map.md")
SYSTEM_AUTHORITY_RENDER_SCRIPT_RELATIVE_PATH = Path("scripts/tools/render_system_authority_map.py")
SYSTEM_AUTHORITY_GENERATED_MARKER = (
    "<!-- Auto-generated from pipeline/system_authority.py via "
    "scripts/tools/render_system_authority_map.py -->"
)

DOCTRINE_DOCS: tuple[str, ...] = (
    "CLAUDE.md",
    "TRADING_RULES.md",
    "RESEARCH_RULES.md",
    "docs/institutional/pre_registered_criteria.md",
    "docs/governance/document_authority.md",
)

SYSTEM_AUTHORITY_BACKBONE_MODULES: tuple[str, ...] = (
    "pipeline/system_authority.py",
    "pipeline/system_context.py",
    "pipeline/system_brief.py",
    "pipeline/work_capsule.py",
    "context/institutional.py",
    "context/registry.py",
    "pipeline/db_contracts.py",
    "trading_app/holdout_policy.py",
    "trading_app/validated_shelf.py",
    "trading_app/prop_profiles.py",
    "trading_app/lifecycle_state.py",
)


@dataclass(frozen=True)
class SurfaceCategory:
    title: str
    purpose: str
    examples: tuple[str, ...]
    mutation_rule: str


@dataclass(frozen=True)
class CanonicalTruthEntry:
    question: str
    source: str


SURFACE_TAXONOMY: tuple[SurfaceCategory, ...] = (
    SurfaceCategory(
        title="Doctrine",
        purpose="Human-facing binding rules",
        examples=DOCTRINE_DOCS,
        mutation_rule="Update only when policy or workflow changes",
    ),
    SurfaceCategory(
        title="Canonical registries",
        purpose="Stable code truth for changing facts and rules",
        examples=(
            "pipeline/system_authority.py",
            "pipeline/system_context.py",
            "pipeline/system_brief.py",
            "pipeline/work_capsule.py",
            "context/institutional.py",
            "context/registry.py",
            "pipeline/asset_configs.py",
            "pipeline/cost_model.py",
            "pipeline/dst.py",
            "trading_app/config.py",
            "trading_app/holdout_policy.py",
            "trading_app/prop_profiles.py",
        ),
        mutation_rule="One owned source per concept; no duplicate literals downstream",
    ),
    SurfaceCategory(
        title="Command writers",
        purpose="The only places allowed to mutate durable state",
        examples=(
            "pipeline/init_db.py",
            "trading_app/db_manager.py",
            "trading_app/strategy_validator.py",
            "trading_app/edge_families.py",
            "scripts/migrations/",
        ),
        mutation_rule="Mutations must go through owned command paths",
    ),
    SurfaceCategory(
        title="Published read models / contracts",
        purpose="Stable query surfaces for operational consumers",
        examples=(
            "pipeline/db_contracts.py",
            "trading_app/validated_shelf.py",
            "DB views active_validated_setups and deployable_validated_setups",
            "scripts/tools/system_brief.py",
            "scripts/tools/work_capsule.py",
            "scripts/tools/project_pulse.py",
            "scripts/tools/context_views.py",
            "scripts/tools/context_resolver.py",
        ),
        mutation_rule="Readers consume these instead of rebuilding semantics ad hoc",
    ),
    SurfaceCategory(
        title="Derived operational state",
        purpose="Runtime snapshots and envelopes derived from canonical truth",
        examples=(
            "trading_app/lifecycle_state.py",
            "trading_app/derived_state.py",
            "data/state/sr_state.json",
            "Criterion 11 survival reports",
        ),
        mutation_rule="Must validate envelope/fingerprint before trust",
    ),
    SurfaceCategory(
        title="Audit / verification",
        purpose="Checks that linked truth and downstream consumers stay aligned",
        examples=(
            "pipeline/check_drift.py",
            "scripts/audits/",
            "scripts/tools/audit_integrity.py",
            "scripts/tools/audit_behavioral.py",
        ),
        mutation_rule="Audits must import canonical truth where possible",
    ),
    SurfaceCategory(
        title="Plans / history / baton",
        purpose="Decision history and in-flight context",
        examples=(
            "docs/runtime/decision-ledger.md",
            "docs/runtime/debt-ledger.md",
            "docs/plans/",
            "HANDOFF.md",
            "ROADMAP.md",
            "docs/postmortems/",
        ),
        mutation_rule="Never cited as live runtime truth",
    ),
    SurfaceCategory(
        title="Reference / generated docs",
        purpose="Orientation aids and generated inventory",
        examples=(
            "docs/ARCHITECTURE.md",
            "docs/MONOREPO_ARCHITECTURE.md",
            "REPO_MAP.md",
            "docs/governance/system_authority_map.md",
            "docs/context/task-routes.md",
            "docs/context/source-catalog.md",
            "docs/context/institutional-contracts.md",
        ),
        mutation_rule="Must be marked non-authoritative and kept linked/generated",
    ),
)


CANONICAL_TRUTH_MAP: tuple[CanonicalTruthEntry, ...] = (
    CanonicalTruthEntry("Which instruments are active or dead?", "pipeline/asset_configs.py"),
    CanonicalTruthEntry("What are the live session definitions and DOW alignment rules?", "pipeline/dst.py"),
    CanonicalTruthEntry("What are the cost specs?", "pipeline/cost_model.py"),
    CanonicalTruthEntry("What is the sacred holdout policy?", "trading_app/holdout_policy.py"),
    CanonicalTruthEntry("What filters, entry models, and routing rules exist?", "trading_app/config.py"),
    CanonicalTruthEntry(
        "What is deployable on the validated shelf?",
        "pipeline/db_contracts.py + deployable_validated_setups",
    ),
    CanonicalTruthEntry("What are the active execution lanes?", "trading_app/prop_profiles.py"),
    CanonicalTruthEntry("What is the unified operational block/allow state?", "trading_app/lifecycle_state.py"),
    CanonicalTruthEntry(
        "What is the canonical repo/dev control-plane context?",
        "pipeline/system_context.py + scripts/tools/system_context.py",
    ),
    CanonicalTruthEntry(
        "What is the minimal complete startup understanding for the current task?",
        "pipeline/system_brief.py + scripts/tools/system_brief.py",
    ),
    CanonicalTruthEntry(
        "What is the active task packet for this workstream?",
        "pipeline/work_capsule.py + scripts/tools/work_capsule.py",
    ),
    CanonicalTruthEntry(
        "What are the project's institutional concepts, decision protocols, and answer contracts?",
        "context/institutional.py",
    ),
    CanonicalTruthEntry(
        "How should a cold-start agent route task context?",
        "context/registry.py + scripts/tools/context_resolver.py",
    ),
    CanonicalTruthEntry(
        "What is the current task-scoped live context for research, trading, or verification work?",
        "scripts/tools/context_views.py",
    ),
    CanonicalTruthEntry(
        "What is planning vs current implementation?",
        "ROADMAP.md is planning only; code/DB decide current implementation",
    ),
)


ENFORCEMENT_RULES: tuple[str, ...] = (
    "New mutable workflow surfaces must declare their category here or in docs/governance/document_authority.md in the same change.",
    "If a consumer needs deployable shelf semantics, it should read deployable_validated_setups or deployable_validated_relation(...), not validated_setups WHERE status='active'.",
    "If a rule changes frequently with data, profiles, or runtime state, do not hardcode it in prose. Link the source or expose a published contract.",
    "Audits should fail when they read deprecated truth surfaces after a newer canonical surface exists.",
    "Reference docs must say what they are not authoritative for.",
)


def authority_reference_paths() -> tuple[str, ...]:
    """Return the unique linked references that must stay represented in the authority map."""
    refs: list[str] = []
    seen: set[str] = set()
    for category in SURFACE_TAXONOMY:
        for ref in category.examples:
            if "/" not in ref and ".md" not in ref and ".py" not in ref:
                continue
            if ref in seen:
                continue
            seen.add(ref)
            refs.append(ref)
    for entry in CANONICAL_TRUTH_MAP:
        source = entry.source
        for token in source.split(" + "):
            if "/" not in token and ".md" not in token and ".py" not in token:
                continue
            if token in seen:
                continue
            seen.add(token)
            refs.append(token)
    return tuple(refs)


def _fmt_ref(ref: str) -> str:
    return f"`{ref}`"


def _render_examples(examples: tuple[str, ...]) -> str:
    return ", ".join(_fmt_ref(ref) for ref in examples)


def render_system_authority_map() -> str:
    """Render the authoritative markdown doc for docs/governance/system_authority_map.md."""
    lines: list[str] = [
        "# System Authority Map",
        "",
        SYSTEM_AUTHORITY_GENERATED_MARKER,
        "",
        "## Purpose",
        "",
        "This file is the project-level map of where truth lives.",
        "",
        "The point is simple: the repo should explain itself without relying on stale",
        'folklore or someone remembering how "Josh\'s weird machine" works.',
        "",
        "If a value or rule changes often, it should be linked to a canonical code or",
        "DB surface, not copied into scattered docs and audits.",
        "",
        f"Generated from `{SYSTEM_AUTHORITY_RENDER_SCRIPT_RELATIVE_PATH.as_posix()}` and",
        "`pipeline/system_authority.py`.",
        "",
        "## Design Rule",
        "",
        "**Linked truth, not copied truth.**",
        "",
        "- Doctrine may live in prose.",
        "- Frequently changing truth must live in code, data, or published read models.",
        "- Audits must verify the linked source, not restate their own local version of the rule.",
        "- Plans and handoffs may explain decisions, but they do not become runtime truth.",
        "",
        "## Surface Taxonomy",
        "",
        "| Category | Purpose | Canonical examples | Mutation rule |",
        "|---|---|---|---|",
    ]
    for category in SURFACE_TAXONOMY:
        lines.append(
            f"| {category.title} | {category.purpose} | {_render_examples(category.examples)} | {category.mutation_rule} |"
        )

    lines.extend(
        [
            "",
            "## Canonical Truth Map",
            "",
            "| Question | Canonical source |",
            "|---|---|",
        ]
    )
    for entry in CANONICAL_TRUTH_MAP:
        lines.append(f"| {entry.question} | {_fmt_ref(entry.source)} |")

    lines.extend(
        [
            "",
            "## Enforcement Rules",
            "",
        ]
    )
    for idx, rule in enumerate(ENFORCEMENT_RULES, 1):
        lines.append(f"{idx}. {rule}")
    lines.append("")
    return "\n".join(lines)

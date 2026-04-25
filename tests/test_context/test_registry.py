from __future__ import annotations

import context.registry as registry
from context.registry import (
    FALLBACK_READ_SET,
    TASKS,
    TaskManifest,
    candidate_tasks,
    render_institutional_markdown,
    render_source_catalog_markdown,
    render_task_routes_markdown,
    resolve_from_text,
    resolve_task,
    validate_registry,
)


def test_registry_is_structurally_valid() -> None:
    assert validate_registry() == []


def test_research_investigation_route_expands_domains_and_profiles() -> None:
    route = resolve_task("research_investigation")

    assert route.manifest.id == "research_investigation"
    assert "RESEARCH_RULES.md" in route.doctrine_files
    assert "trading_app/holdout_policy.py" in route.canonical_files
    assert route.verification.id == "investigation"
    assert route.decision_protocol.id == "research_investigation_protocol"
    assert route.answer_contract.id == "research_investigation_answer"
    assert route.drilldown_playbook is not None
    assert route.drilldown_playbook.id == "research_recent_performance_drilldown"
    assert any(pack.id == "coding_runtime_pack" for pack in route.understanding_packs)
    assert any(pack.id == "trading_runtime_pack" for pack in route.understanding_packs)
    assert any(pack.id == "research_methodology_pack" for pack in route.understanding_packs)
    assert any(variable.id == "orb_utc_window" for variable in route.variables)
    assert any(variable.id == "holdout_policy_var" for variable in route.variables)
    assert any(concept.id == "sacred_holdout_policy" for concept in route.concepts)
    assert route.verification.steps == ("project_pulse_fast", "system_context_text")
    assert any(step.id == "project_pulse_fast" for step in route.verification_steps)
    assert any(view.id == "gold_db_mcp" for view in route.live_views)
    assert any(view.id == "research_context" for view in route.live_views)
    assert any(view.id == "recent_performance_context" for view in route.live_views)
    assert all(view.id != "verification_context" for view in route.live_views)
    assert "pipeline/check_drift.py" not in route.canonical_files
    assert "scripts/tools/audit_integrity.py" not in route.canonical_files


def test_live_trading_route_does_not_pull_research_context() -> None:
    route = resolve_task("live_trading_status")

    assert any(view.id == "trading_context" for view in route.live_views)
    assert all(view.id != "research_context" for view in route.live_views)
    assert route.drilldown_playbook is None
    assert any(pack.id == "trading_runtime_pack" for pack in route.understanding_packs)


def test_candidate_tasks_matches_mes_win_rate_investigation() -> None:
    candidates = candidate_tasks("Investigate why MES 5m ORB win rate dropped last week.")

    assert candidates
    assert candidates[0].task_id == "research_investigation"


def test_candidate_tasks_matches_docs_drift_audit() -> None:
    candidates = candidate_tasks("Run a documentation drift audit on stale docs.")

    assert candidates
    assert candidates[0].task_id == "docs_drift_audit"


def test_candidate_tasks_matches_repo_workflow_audit() -> None:
    candidates = candidate_tasks("Audit token waste in startup hooks and context resolver routing.")

    assert candidates
    assert candidates[0].task_id == "repo_workflow_audit"


def test_rendered_catalogs_include_expected_entries() -> None:
    source_catalog = render_source_catalog_markdown()
    task_routes = render_task_routes_markdown()
    institutional = render_institutional_markdown()

    assert "Context Source Catalog" in source_catalog
    assert "`research_methodology`" in source_catalog
    assert "`project_pulse_fast`" in source_catalog
    assert "`research_context`" in source_catalog
    assert "`recent_performance_context`" in source_catalog
    assert "`sacred_holdout_policy`" in source_catalog
    assert "`trading_runtime_pack`" in source_catalog
    assert "`orb_utc_window`" in source_catalog
    assert "Task Routes" in task_routes
    assert "`research_investigation`" in task_routes
    assert "`repo_workflow_audit`" in task_routes
    assert "`research_methodology_pack`" in task_routes
    assert all(path in task_routes for path in FALLBACK_READ_SET)
    assert "Institutional Routing Contracts" in institutional
    assert "`research_investigation_protocol`" in institutional
    assert "`research_recent_performance_drilldown`" in institutional
    assert "`coding_runtime_pack`" in institutional
    assert "`shared_root_mutation_override`" in institutional


def test_all_tasks_reference_existing_ids() -> None:
    for task in TASKS.values():
        assert task.verification_profile
        assert task.domains


def test_validate_registry_examples_resolve_to_declared_task() -> None:
    assert validate_registry() == []


def test_resolve_from_text_fails_closed_on_ambiguous_tie(monkeypatch) -> None:
    custom_tasks = {
        "alpha": TaskManifest(
            id="alpha",
            title="Alpha",
            purpose="alpha route",
            intent_terms=("same phrase",),
            domains=("repo_governance",),
            verification_profile="investigation",
            concepts=("runtime_control_plane",),
            decision_protocol="implementation_protocol",
            answer_contract="implementation_answer",
            priority=1,
        ),
        "beta": TaskManifest(
            id="beta",
            title="Beta",
            purpose="beta route",
            intent_terms=("same phrase",),
            domains=("repo_governance",),
            verification_profile="investigation",
            concepts=("runtime_control_plane",),
            decision_protocol="implementation_protocol",
            answer_contract="implementation_answer",
            priority=1,
        ),
    }
    monkeypatch.setattr(registry, "TASKS", custom_tasks)

    route, candidates = resolve_from_text("same phrase")

    assert route is None
    assert [candidate.task_id for candidate in candidates] == ["alpha", "beta"]

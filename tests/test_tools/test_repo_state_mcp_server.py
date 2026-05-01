"""Tests for scripts.tools.repo_state_mcp_server."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from scripts.tools import repo_state_mcp_server


def test_list_task_routes_includes_known_route() -> None:
    routes = repo_state_mcp_server._list_task_routes()
    ids = {route["task_id"] for route in routes}
    assert "repo_workflow_audit" in ids
    assert "completion_claim" in ids


def test_list_context_views_includes_verification() -> None:
    views = repo_state_mcp_server._list_context_views()
    names = {view["view"] for view in views}
    assert {"research", "recent_performance", "trading", "verification"} <= names


def test_resolve_task_route_matches_known_task() -> None:
    payload = repo_state_mcp_server._resolve_task_route(
        task_text="Does my project use context resolver properly and where are tokens being wasted in hooks and launchers?"
    )

    assert payload["matched"] is True
    assert payload["route_id"] == "repo_workflow_audit"


def test_resolve_task_route_reports_no_match() -> None:
    payload = repo_state_mcp_server._resolve_task_route(task_text="Compose a limerick about penguins.")

    assert payload["matched"] is False
    assert payload["reason"] == "No deterministic task match found."


def test_get_project_pulse_uses_json_formatter() -> None:
    fake_report = SimpleNamespace()
    with (
        patch.object(repo_state_mcp_server, "_build_pulse_report", return_value=fake_report) as build_pulse_mock,
        patch.object(
            repo_state_mcp_server,
            "_format_pulse_report_json",
            return_value='{"git_branch":"main","counts":{"broken":0},"items":[]}',
        ) as format_json_mock,
    ):
        payload = repo_state_mcp_server._get_project_pulse()

    build_pulse_mock.assert_called_once()
    format_json_mock.assert_called_once_with(fake_report)
    assert payload["git_branch"] == "main"
    assert payload["counts"]["broken"] == 0


def test_get_system_context_with_action_returns_snapshot_and_decision() -> None:
    snapshot = SimpleNamespace(model_dump=lambda mode="json": {"branch": "main"})
    decision = SimpleNamespace(model_dump=lambda mode="json": {"allowed": True})
    with (
        patch.object(repo_state_mcp_server, "_build_system_context_snapshot", return_value=snapshot) as snapshot_mock,
        patch.object(repo_state_mcp_server, "_evaluate_policy", return_value=decision) as decision_mock,
    ):
        payload = repo_state_mcp_server._get_system_context(action="orientation")

    snapshot_mock.assert_called_once()
    decision_mock.assert_called_once_with(snapshot, "orientation")
    assert payload == {"snapshot": {"branch": "main"}, "decision": {"allowed": True}}


def test_get_context_view_delegates_to_registered_builder() -> None:
    with patch.object(
        repo_state_mcp_server, "_build_context_view_payload", return_value={"view": "verification"}
    ) as build_view_mock:
        payload = repo_state_mcp_server._get_context_view("verification")

    build_view_mock.assert_called_once()
    assert payload["view"] == "verification"


def test_get_startup_packet_uses_system_brief() -> None:
    with patch.object(
        repo_state_mcp_server,
        "_build_startup_brief_payload",
        return_value={"task_id": "repo_workflow_audit", "briefing_level": "read_only"},
    ) as brief_mock:
        payload = repo_state_mcp_server._get_startup_packet(task_text="audit hooks", briefing_level="read_only")

    brief_mock.assert_called_once()
    assert payload["task_id"] == "repo_workflow_audit"

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.tools.task_route_packet import (
    clear_task_route_packet,
    read_task_route_packet,
    read_task_route_packet_metadata,
    write_task_route_packet,
)


def test_write_task_route_packet_renders_compact_packet(tmp_path: Path) -> None:
    payload = {
        "route_id": "repo_workflow_audit",
        "task_kind": "Repo Workflow Audit",
        "briefing_contract": "orientation_briefing",
        "verification_profile": "investigation",
        "verification_steps": ["project_pulse_fast", "system_context_text"],
        "required_live_views": ["verification_context", "system_brief"],
        "doctrine_chain": ["AGENTS.md", "CLAUDE.md"],
        "canonical_owners": ["scripts/tools/context_resolver.py", "scripts/tools/task_route_packet.py"],
        "blocking_issues": [],
        "warning_issues": [],
    }

    with patch("scripts.tools.task_route_packet.build_system_brief", return_value=payload):
        path, returned_payload = write_task_route_packet(
            tmp_path,
            task_text="Audit token waste and context routing",
            tool="claude",
            queue_item="prior_day_bridge_execution_triage",
            override_note="Urgent runtime coordination before the next session handoff.",
        )

    assert returned_payload == payload
    content = path.read_text(encoding="utf-8")
    assert "# Startup Task Route" in content
    assert "- Tool: `claude`" in content
    assert "- Route id: `repo_workflow_audit`" in content
    assert "- Queue item: `prior_day_bridge_execution_triage`" in content
    assert "- Override note: Urgent runtime coordination before the next session handoff." in content
    assert "scripts/tools/context_resolver.py" in content
    assert read_task_route_packet_metadata(tmp_path)["queue_item"] == "prior_day_bridge_execution_triage"


def test_read_and_clear_task_route_packet(tmp_path: Path) -> None:
    packet = tmp_path / ".session" / "task-route.md"
    packet.parent.mkdir(parents=True)
    packet.write_text("# Startup Task Route\n- Tool: `codex`\n", encoding="utf-8")

    assert read_task_route_packet(tmp_path) == ["# Startup Task Route", "- Tool: `codex`"]

    cleared_path = clear_task_route_packet(tmp_path)

    assert cleared_path == packet
    assert not packet.exists()


def test_read_task_route_packet_metadata_returns_empty_for_legacy_packet(tmp_path: Path) -> None:
    packet = tmp_path / ".session" / "task-route.md"
    packet.parent.mkdir(parents=True)
    packet.write_text("# Startup Task Route\n- Tool: `codex`\n", encoding="utf-8")

    assert read_task_route_packet_metadata(tmp_path) == {}

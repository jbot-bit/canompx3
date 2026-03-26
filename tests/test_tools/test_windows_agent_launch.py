"""Tests for scripts.infra.windows_agent_launch."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("readchar", reason="readchar not installed")

from scripts.infra import windows_agent_launch


class TestEnsureManagedWorktree:
    def test_uses_saved_purpose_when_present(self) -> None:
        payload = '{"path":"C:\\\\repo\\\\.worktrees\\\\tasks\\\\task-a"}'
        with (
            patch.object(windows_agent_launch, "get_existing_purpose", return_value="Review / verify"),
            patch.object(windows_agent_launch, "invoke_manager", return_value=(True, payload)) as invoke_manager,
        ):
            path, purpose = windows_agent_launch.ensure_managed_worktree("claude", "task-a", "Build / edit")

        assert path == Path(r"C:\repo\.worktrees\tasks\task-a")
        assert purpose == "Review / verify"
        invoke_manager.assert_called_once_with(
            [
                "create",
                "--tool",
                "claude",
                "--name",
                "task-a",
                "--json",
                "--purpose",
                "Review / verify",
            ]
        )

    def test_raises_on_invalid_manager_payload(self) -> None:
        with (
            patch.object(windows_agent_launch, "get_existing_purpose", return_value=None),
            patch.object(windows_agent_launch, "invoke_manager", return_value=(True, "not-json")),
        ):
            with pytest.raises(RuntimeError, match="Invalid worktree manager output"):
                windows_agent_launch.ensure_managed_worktree("claude", "task-a", "Build / edit")


class TestRunPreflight:
    def test_runs_preflight_with_windows_python(self, tmp_path: Path) -> None:
        preflight = tmp_path / "scripts" / "tools" / "session_preflight.py"
        preflight.parent.mkdir(parents=True)
        preflight.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with (
            patch.object(windows_agent_launch, "pick_python", return_value=["python"]),
            patch.object(windows_agent_launch.subprocess, "run") as run_mock,
        ):
            windows_agent_launch.run_preflight(tmp_path, claim_tool="claude")

        run_mock.assert_called_once_with(
            ["python", str(preflight), "--quiet", "--context", "generic", "--claim", "claude"],
            cwd=tmp_path,
            check=False,
        )

    def test_skips_missing_preflight(self, tmp_path: Path) -> None:
        with patch.object(windows_agent_launch.subprocess, "run") as run_mock:
            windows_agent_launch.run_preflight(tmp_path, claim_tool="claude")
        run_mock.assert_not_called()


class TestFindClaudeCli:
    def test_prefers_first_available_candidate(self) -> None:
        with patch.object(windows_agent_launch.shutil, "which", side_effect=[r"C:\Users\joshd\.local\bin\claude.exe"]):
            assert windows_agent_launch.find_claude_cli() == r"C:\Users\joshd\.local\bin\claude.exe"

    def test_raises_when_not_found(self) -> None:
        with patch.object(windows_agent_launch.shutil, "which", return_value=None):
            with pytest.raises(RuntimeError, match="Claude CLI not found"):
                windows_agent_launch.find_claude_cli()


class TestOpenClaudeWorkstream:
    def test_uses_native_claude_with_worktree_path(self, tmp_path: Path) -> None:
        with (
            patch.object(
                windows_agent_launch,
                "ensure_managed_worktree",
                return_value=(tmp_path, "Build / edit"),
            ) as ensure_mock,
            patch.object(windows_agent_launch, "run_preflight") as preflight_mock,
            patch.object(windows_agent_launch, "find_claude_cli", return_value=r"C:\Users\joshd\.local\bin\claude.exe"),
            patch.object(windows_agent_launch.subprocess, "call", return_value=0) as call_mock,
        ):
            exit_code = windows_agent_launch.open_claude_workstream("task-a", "Build / edit")

        assert exit_code == 0
        ensure_mock.assert_called_once_with("claude", "task-a", "Build / edit")
        preflight_mock.assert_called_once_with(tmp_path, claim_tool="claude", context="generic")
        call_mock.assert_called_once_with([r"C:\Users\joshd\.local\bin\claude.exe", "-C", str(tmp_path)])


class TestWorkflowCommands:
    def test_handoff_workstream_invokes_manager(self) -> None:
        with patch.object(windows_agent_launch, "invoke_manager", return_value=(True, "ok")) as invoke_mock:
            success, output = windows_agent_launch.handoff_workstream(
                "task-a",
                current_tool="claude",
                target_tool="codex",
                purpose="Build / edit",
                note="pick this up",
            )

        assert success is True
        assert output == "ok"
        invoke_mock.assert_called_once_with(
            [
                "handoff",
                "--name",
                "task-a",
                "--tool",
                "claude",
                "--target-tool",
                "codex",
                "--purpose",
                "Build / edit",
                "--note",
                "pick this up",
            ]
        )

    def test_ship_workstream_invokes_manager(self) -> None:
        with patch.object(windows_agent_launch, "invoke_manager", return_value=(True, "ok")) as invoke_mock:
            success, output = windows_agent_launch.ship_workstream("task-a", tool="codex", commit_message="workstream: task-a")

        assert success is True
        assert output == "ok"
        invoke_mock.assert_called_once_with(
            [
                "ship",
                "--name",
                "task-a",
                "--tool",
                "codex",
                "--commit-message",
                "workstream: task-a",
            ]
        )

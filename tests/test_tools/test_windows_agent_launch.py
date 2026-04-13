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
            patch.object(windows_agent_launch.subprocess, "run", return_value=type("R", (), {"returncode": 0})()) as run_mock,
        ):
            windows_agent_launch.run_preflight(tmp_path, claim_tool="claude")

        args, kwargs = run_mock.call_args
        assert args == (
            ["python", str(preflight), "--quiet", "--context", "generic", "--claim", "claude", "--mode", "mutating"],
        )
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is False
        assert kwargs["env"]["CANOMPX3_SESSION_OWNER"].startswith("launcher-")

    def test_skips_missing_preflight(self, tmp_path: Path) -> None:
        with patch.object(windows_agent_launch.subprocess, "run") as run_mock:
            windows_agent_launch.run_preflight(tmp_path, claim_tool="claude")
        run_mock.assert_not_called()

    def test_raises_when_preflight_blocks(self, tmp_path: Path) -> None:
        preflight = tmp_path / "scripts" / "tools" / "session_preflight.py"
        preflight.parent.mkdir(parents=True)
        preflight.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with (
            patch.object(windows_agent_launch, "pick_python", return_value=["python"]),
            patch.object(windows_agent_launch.subprocess, "run", return_value=type("R", (), {"returncode": 2})()),
        ):
            with pytest.raises(RuntimeError, match="Session preflight blocked launch"):
                windows_agent_launch.run_preflight(tmp_path, claim_tool="claude")


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
        preflight_mock.assert_called_once_with(tmp_path, claim_tool="claude", context="generic", mode="mutating")
        call_mock.assert_called_once_with([r"C:\Users\joshd\.local\bin\claude.exe", "-C", str(tmp_path)])


class TestCodexWslCommand:
    def test_builds_bootstrap_then_open_command(self) -> None:
        command = windows_agent_launch.build_codex_wsl_command(
            "/mnt/c/repo",
            "task-a",
            "Build / edit",
            False,
        )

        assert "set -euo pipefail" in command
        assert "cd /mnt/c/repo" in command
        assert "export UV_PROJECT_ENVIRONMENT=.venv-wsl" in command
        assert "export UV_CACHE_DIR=/tmp/uv-cache" in command
        assert "export UV_PYTHON_INSTALL_DIR=/tmp/uv-python" in command
        assert "export UV_LINK_MODE=copy" in command
        assert 'mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"' in command
        assert "uv sync --frozen --python 3.13 --group dev" in command
        assert "export CANOMPX3_WORKSTREAM_PURPOSE='Build / edit'" in command
        assert "exec ./scripts/infra/codex-worktree.sh open task-a -- --no-alt-screen" in command

    def test_builds_search_command_without_purpose(self) -> None:
        command = windows_agent_launch.build_codex_wsl_command(
            "/mnt/c/repo",
            "task-a",
            None,
            True,
        )

        assert "exec ./scripts/infra/codex-worktree.sh search task-a -- --no-alt-screen" in command
        assert "CANOMPX3_WORKSTREAM_PURPOSE" not in command

    def test_builds_plain_project_command(self) -> None:
        command = windows_agent_launch.build_codex_project_wsl_command("/mnt/c/repo")

        assert "set -euo pipefail" in command
        assert "cd /mnt/c/repo" in command
        assert "exec ./scripts/infra/codex-project.sh --no-alt-screen" in command

    def test_builds_gold_db_project_command(self) -> None:
        command = windows_agent_launch.build_codex_project_wsl_command(
            "/mnt/c/repo", enable_gold_db=True
        )

        assert "exec ./scripts/infra/codex-project-gold-db.sh --no-alt-screen" in command

    def test_builds_gold_db_search_project_command(self) -> None:
        command = windows_agent_launch.build_codex_project_wsl_command(
            "/mnt/c/repo", search_mode=True, enable_gold_db=True
        )

        assert "exec ./scripts/infra/codex-project-search-gold-db.sh --no-alt-screen" in command


class TestOpenCodexWorkstream:
    def test_uses_saved_search_purpose_for_wsl_launch(self) -> None:
        with (
            patch.object(windows_agent_launch, "repo_root", return_value=Path(r"C:\repo")),
            patch.object(windows_agent_launch, "windows_to_wsl", return_value="/mnt/c/repo"),
            patch.object(windows_agent_launch, "get_existing_purpose", return_value="Investigate / search"),
            patch.object(windows_agent_launch, "run_wsl", return_value=0) as run_wsl_mock,
        ):
            exit_code = windows_agent_launch.open_codex_workstream("task-a", "Build / edit", False)

        assert exit_code == 0
        command = run_wsl_mock.call_args.args[0]
        assert "exec ./scripts/infra/codex-worktree.sh search task-a -- --no-alt-screen" in command
        assert "export CANOMPX3_WORKSTREAM_PURPOSE='Investigate / search'" in command


class TestOpenCodexProject:
    def test_opens_repo_project_launcher_in_wsl(self) -> None:
        with (
            patch.object(windows_agent_launch, "repo_root", return_value=Path(r"C:\repo")),
            patch.object(windows_agent_launch, "windows_to_wsl", return_value="/mnt/c/repo"),
            patch.object(windows_agent_launch, "run_wsl", return_value=0) as run_wsl_mock,
        ):
            exit_code = windows_agent_launch.open_codex_project()

        assert exit_code == 0
        command = run_wsl_mock.call_args.args[0]
        assert "exec ./scripts/infra/codex-project.sh --no-alt-screen" in command

    def test_opens_repo_gold_db_project_launcher_in_wsl(self) -> None:
        with (
            patch.object(windows_agent_launch, "repo_root", return_value=Path(r"C:\repo")),
            patch.object(windows_agent_launch, "windows_to_wsl", return_value="/mnt/c/repo"),
            patch.object(windows_agent_launch, "run_wsl", return_value=0) as run_wsl_mock,
        ):
            exit_code = windows_agent_launch.open_codex_project(enable_gold_db=True)

        assert exit_code == 0
        command = run_wsl_mock.call_args.args[0]
        assert "exec ./scripts/infra/codex-project-gold-db.sh --no-alt-screen" in command


class TestWindowsBatchWrappers:
    def test_claude_workstream_batch_uses_windows_launcher_claude_mode(self) -> None:
        batch_path = windows_agent_launch.repo_root() / "claude-workstream.bat"
        content = batch_path.read_text(encoding="utf-8")

        assert '-Mode claude -Task "%TASK%"' in content
        assert "Claude Isolated Workstream" in content

    def test_codex_workstream_batch_uses_windows_launcher_codex_mode(self) -> None:
        batch_path = windows_agent_launch.repo_root() / "codex-workstream.bat"
        content = batch_path.read_text(encoding="utf-8")

        assert '-Mode codex -Task "%TASK%"' in content
        assert "Codex Isolated Workstream" in content

    def test_claude_green_baseline_batch_targets_clean_worktree(self) -> None:
        batch_path = windows_agent_launch.repo_root() / "claude-green-baseline.bat"
        content = batch_path.read_text(encoding="utf-8")

        assert ".worktrees\\tasks\\green-baseline" in content
        assert "Claude Green Baseline" in content
        assert "Get-Command claude,claude.exe" in content

    def test_codex_green_baseline_batch_targets_clean_worktree(self) -> None:
        batch_path = windows_agent_launch.repo_root() / "codex-green-baseline.bat"
        content = batch_path.read_text(encoding="utf-8")

        assert ".worktrees\\tasks\\green-baseline" in content
        assert "exec ./scripts/infra/codex-project.sh --no-alt-screen" in content
        assert "Codex Green Baseline" in content

    def test_root_workstream_helper_batches_use_launcher_modes(self) -> None:
        root = windows_agent_launch.repo_root()
        list_content = (root / "workstream-list.bat").read_text(encoding="utf-8")
        finish_content = (root / "workstream-finish.bat").read_text(encoding="utf-8")
        clean_content = (root / "workstream-clean.bat").read_text(encoding="utf-8")

        assert "-Mode list" in list_content
        assert "-Mode close-pick" in finish_content
        assert "-Mode prune" in clean_content


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
            success, output = windows_agent_launch.ship_workstream(
                "task-a", tool="codex", commit_message="workstream: task-a"
            )

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

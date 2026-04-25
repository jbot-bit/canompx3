from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch


def _load_module():
    stub = types.SimpleNamespace(
        key=types.SimpleNamespace(UP="UP", DOWN="DOWN", ENTER="ENTER"),
        readkey=lambda: "ENTER",
    )
    sys.modules.setdefault("readchar", stub)
    return importlib.import_module("scripts.infra.windows_agent_launch")


def test_build_codex_wsl_command_includes_mount_guard() -> None:
    module = _load_module()

    command = module.build_codex_wsl_command("/mnt/c/repo", "task-a", "Build / edit", False)

    assert "python3 scripts/tools/wsl_mount_guard.py --root /mnt/c/repo" in command
    assert "uv sync --frozen --python 3.13 --group dev" in command


def test_build_linux_home_project_command_uses_linux_root() -> None:
    module = _load_module()

    command = module.build_codex_project_wsl_command("/mnt/c/repo", use_linux_home=True)

    assert 'ROOT="${CANOMPX3_CODEX_WSL_ROOT:-$HOME/canompx3}"' in command
    assert 'bash /mnt/c/repo/scripts/infra/codex-wsl-sync.sh --source /mnt/c/repo --target "$ROOT"' in command
    assert 'cd "$ROOT"' in command
    assert "exec ./scripts/infra/codex-project.sh --no-alt-screen" in command


def test_build_power_project_command_exports_power_profile() -> None:
    module = _load_module()

    command = module.build_codex_project_wsl_command("/mnt/c/repo", profile="canompx3_power")

    assert "export CANOMPX3_CODEX_PROFILE=canompx3_power" in command
    assert "exec ./scripts/infra/codex-project.sh --no-alt-screen" in command


def test_open_codex_project_linux_home_uses_linux_home_builder() -> None:
    module = _load_module()

    with (
        patch.object(module, "repo_root", return_value=Path(r"C:\repo")),
        patch.object(module, "windows_to_wsl", return_value="/mnt/c/repo"),
        patch.object(module, "run_wsl", return_value=0) as run_wsl_mock,
    ):
        exit_code = module.open_codex_project_linux_home()

    assert exit_code == 0
    command = run_wsl_mock.call_args.args[0]
    assert 'ROOT="${CANOMPX3_CODEX_WSL_ROOT:-$HOME/canompx3}"' in command


def test_linux_project_batches_target_linux_modes() -> None:
    root = Path(__file__).resolve().parents[2]
    codex_bat = (root / "codex.bat").read_text(encoding="utf-8")

    assert 'set "MODE=codex-project-linux"' in codex_bat
    assert 'if /I "%ACTION%"=="power" (' in codex_bat
    assert 'set "MODE=codex-project-linux-power"' in codex_bat
    assert 'if /I "%ACTION%"=="windows" (' in codex_bat
    assert 'set "MODE=codex-project"' in codex_bat
    assert 'if /I "%ACTION%"=="linux" (' in codex_bat
    assert 'set "MODE=codex-project-linux"' in codex_bat
    assert 'if /I "%ACTION%"=="linux-power" (' in codex_bat
    assert 'set "MODE=codex-project-linux-power"' in codex_bat
    assert 'if /I "%ACTION%"=="linux-gold-db" (' in codex_bat
    assert 'set "MODE=codex-project-linux-gold-db"' in codex_bat


def test_codex_bat_routes_task_shortcuts_to_ai_workstreams() -> None:
    root = Path(__file__).resolve().parents[2]
    codex_bat = (root / "codex.bat").read_text(encoding="utf-8")

    assert 'call "ai-workstreams.bat" codex %*' in codex_bat
    assert 'call "ai-workstreams.bat" search %*' in codex_bat
    assert "Unknown codex mode: %ACTION%" in codex_bat
    assert "codex.bat help" in codex_bat


def test_claude_bat_is_simple_front_door() -> None:
    root = Path(__file__).resolve().parents[2]
    claude_bat = (root / "claude.bat").read_text(encoding="utf-8")

    assert 'call "ai-workstreams.bat" claude %*' in claude_bat
    assert 'call "ai-workstreams.bat" green claude' in claude_bat
    assert "Unknown claude mode: %ACTION%" in claude_bat
    assert "claude.bat help" in claude_bat


def test_open_codex_green_baseline_routes_to_green_worktree() -> None:
    module = _load_module()

    with (
        patch.object(module, "_green_baseline_worktree", return_value=Path(r"C:\repo\.worktrees\tasks\green-baseline")),
        patch.object(module, "windows_to_wsl", return_value="/mnt/c/repo/.worktrees/tasks/green-baseline"),
        patch.object(module, "run_wsl", return_value=0) as run_wsl_mock,
    ):
        exit_code = module.open_codex_green_baseline()

    assert exit_code == 0
    command = run_wsl_mock.call_args.args[0]
    assert "cd /mnt/c/repo/.worktrees/tasks/green-baseline" in command
    assert "exec ./scripts/infra/codex-project.sh --no-alt-screen" in command


def test_open_claude_green_baseline_uses_claude_cli() -> None:
    module = _load_module()

    with (
        patch.object(module, "_green_baseline_worktree", return_value=Path(r"C:\repo\.worktrees\tasks\green-baseline")),
        patch.object(module, "prepare_task_route_packet") as packet_mock,
        patch.object(module, "find_claude_cli", return_value=r"C:\tools\claude.exe"),
        patch.object(module.subprocess, "call", return_value=0) as call_mock,
    ):
        exit_code = module.open_claude_green_baseline()

    assert exit_code == 0
    packet_mock.assert_called_once_with(Path(r"C:\repo\.worktrees\tasks\green-baseline"), tool="claude", clear=True)
    call_mock.assert_called_once_with([r"C:\tools\claude.exe", "-C", r"C:\repo\.worktrees\tasks\green-baseline"])

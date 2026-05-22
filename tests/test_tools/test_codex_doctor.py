from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.infra import codex_doctor


def _tool_probe_result(*args, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout="present\n", stderr="")


def _doctor_git(win_head: str) -> callable:
    def fake_git(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "main"
        if args == ["status", "--short"]:
            return 0, ""
        if args == ["rev-parse", "HEAD"]:
            return 0, win_head
        raise AssertionError(f"unexpected git args: {args}")

    return fake_git


def _doctor_wsl(wsl_head: str, *, dirty: bool = False) -> callable:
    def fake_wsl(script: str) -> tuple[int, str]:
        if script == "true":
            return 0, ""
        if 'echo "$ROOT"' in script:
            return 0, "/home/joshd/canompx3"
        if "rev-parse --abbrev-ref HEAD" in script:
            return 0, "main"
        if "rev-parse HEAD" in script:
            return 0, wsl_head
        if "status --short" in script:
            return (0, " M repo_state") if dirty else (0, "")
        raise AssertionError(f"unexpected wsl script: {script}")

    return fake_wsl


def test_doctor_reports_aligned_smart_path(capsys) -> None:
    with (
        patch.object(codex_doctor, "repo_root", return_value=Path(r"C:\repo")),
        patch.object(codex_doctor, "_git", side_effect=_doctor_git("abc123abc123")),
        patch.object(codex_doctor, "_wsl", side_effect=_doctor_wsl("abc123abc123")),
        patch.object(codex_doctor, "wsl_home_clone_available", return_value=True),
        patch.object(
            codex_doctor, "_resolve_sync_relation", return_value=("aligned", "comparison repo: /home/joshd/canompx3")
        ),
        patch.object(codex_doctor.subprocess, "run", side_effect=_tool_probe_result),
    ):
        assert codex_doctor.run() == 0

    output = capsys.readouterr().out
    assert "SMART_PATH=READY: Windows checkout and WSL clone are aligned" in output


def test_doctor_reports_fast_forwardable_wsl_clone(capsys) -> None:
    with (
        patch.object(codex_doctor, "repo_root", return_value=Path(r"C:\repo")),
        patch.object(codex_doctor, "_git", side_effect=_doctor_git("def456def456")),
        patch.object(codex_doctor, "_wsl", side_effect=_doctor_wsl("abc123abc123")),
        patch.object(codex_doctor, "wsl_home_clone_available", return_value=True),
        patch.object(
            codex_doctor,
            "_resolve_sync_relation",
            return_value=("target_behind_source", "comparison repo: /home/joshd/canompx3"),
        ),
        patch.object(codex_doctor.subprocess, "run", side_effect=_tool_probe_result),
    ):
        assert codex_doctor.run() == 0

    output = capsys.readouterr().out
    assert "SMART_PATH=READY: WSL clone is behind but can be fast-forwarded" in output


def test_doctor_blocks_when_windows_checkout_is_behind(capsys) -> None:
    with (
        patch.object(codex_doctor, "repo_root", return_value=Path(r"C:\repo")),
        patch.object(codex_doctor, "_git", side_effect=_doctor_git("abc123abc123")),
        patch.object(codex_doctor, "_wsl", side_effect=_doctor_wsl("def456def456")),
        patch.object(codex_doctor, "wsl_home_clone_available", return_value=True),
        patch.object(
            codex_doctor,
            "_resolve_sync_relation",
            return_value=("source_behind_target", "comparison repo: /home/joshd/canompx3"),
        ),
        patch.object(codex_doctor.subprocess, "run", side_effect=_tool_probe_result),
    ):
        assert codex_doctor.run() == 0

    output = capsys.readouterr().out
    assert "SMART_PATH=BLOCKED: Windows checkout is behind the WSL clone on the same branch" in output
    assert "Manual remedy: update the Windows checkout" in output


def test_doctor_blocks_when_repos_diverge(capsys) -> None:
    with (
        patch.object(codex_doctor, "repo_root", return_value=Path(r"C:\repo")),
        patch.object(codex_doctor, "_git", side_effect=_doctor_git("abc123abc123")),
        patch.object(codex_doctor, "_wsl", side_effect=_doctor_wsl("def456def456")),
        patch.object(codex_doctor, "wsl_home_clone_available", return_value=True),
        patch.object(
            codex_doctor,
            "_resolve_sync_relation",
            return_value=("diverged", "comparison repo: /home/joshd/canompx3"),
        ),
        patch.object(codex_doctor.subprocess, "run", side_effect=_tool_probe_result),
    ):
        assert codex_doctor.run() == 0

    output = capsys.readouterr().out
    assert "SMART_PATH=BLOCKED: Windows checkout and WSL clone diverged on the same branch" in output
    assert "Manual remedy: reconcile the two repos manually" in output


def test_doctor_blocks_dirty_wsl_clone_before_relation_check(capsys) -> None:
    with (
        patch.object(codex_doctor, "repo_root", return_value=Path(r"C:\repo")),
        patch.object(codex_doctor, "_git", side_effect=_doctor_git("abc123abc123")),
        patch.object(codex_doctor, "_wsl", side_effect=_doctor_wsl("abc123abc123", dirty=True)),
        patch.object(codex_doctor, "wsl_home_clone_available", return_value=True),
        patch.object(codex_doctor.subprocess, "run", side_effect=_tool_probe_result),
        patch.object(codex_doctor, "_resolve_sync_relation") as relation_mock,
    ):
        assert codex_doctor.run() == 0

    output = capsys.readouterr().out
    assert "SMART_PATH=BLOCKED: WSL clone dirty" in output
    assert "Manual remedy: inspect the WSL-home clone before relaunching" in output
    assert "cd ~/canompx3" in output
    assert "git status --short --branch" in output
    relation_mock.assert_not_called()

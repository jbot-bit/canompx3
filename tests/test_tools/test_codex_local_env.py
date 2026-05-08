from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.infra import codex_local_env


def test_cleanup_paths_removes_caches_and_skips_envs(tmp_path: Path) -> None:
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "state").write_text("x", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "__pycache__" / "mod.pyc").write_text("x", encoding="utf-8")
    (tmp_path / ".venv" / "__pycache__").mkdir(parents=True)
    (tmp_path / ".venv" / "__pycache__" / "keep.pyc").write_text("x", encoding="utf-8")
    (tmp_path / ".coverage").write_text("x", encoding="utf-8")

    removed = codex_local_env.cleanup_paths(tmp_path)

    removed_names = {path.name for path in removed}
    assert ".pytest_cache" in removed_names
    assert "__pycache__" in removed_names
    assert ".coverage" in removed_names
    assert (tmp_path / ".venv" / "__pycache__" / "keep.pyc").exists()


def test_env_for_platform_sets_expected_uv_project_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shared_codex_home = tmp_path / ".codex"
    shared_codex_home.mkdir()
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_local_env, "default_shared_codex_home", lambda: shared_codex_home)

    wsl_env = codex_local_env.env_for_platform("wsl")
    windows_env = codex_local_env.env_for_platform("windows")

    assert wsl_env["UV_PROJECT_ENVIRONMENT"] == ".venv-wsl"
    assert wsl_env["JOBLIB_MULTIPROCESSING"] == "0"
    assert wsl_env["CODEX_HOME"] == str(shared_codex_home)
    assert windows_env["UV_PROJECT_ENVIRONMENT"] == ".venv"
    assert windows_env["JOBLIB_MULTIPROCESSING"] == "0"


def test_parse_args_accepts_doctor_command() -> None:
    args = codex_local_env.parse_args(["doctor", "--platform", "wsl"])

    assert args.command == "doctor"
    assert args.platform == "wsl"


def test_is_wsl_native_root_detects_wsl_home_paths() -> None:
    assert codex_local_env.is_wsl_native_root(Path("/home/joshd/canompx3"))
    assert not codex_local_env.is_wsl_native_root(Path("/mnt/c/Users/joshd/canompx3"))


def test_run_setup_blocks_wsl_mount_repo_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_local_env, "ROOT", Path("/mnt/c/Users/joshd/canompx3"))

    with pytest.raises(SystemExit, match="Codex WSL setup is blocked for non-native repo root"):
        codex_local_env.run_setup("wsl")


def test_safe_path_exists_handles_inaccessible_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    target = Path("C:/fake/.venv-wsl/bin/python")

    def _raise_oserror(self: Path) -> bool:
        raise OSError("[WinError 1920] The file cannot be accessed by the system")

    monkeypatch.setattr(Path, "exists", _raise_oserror)

    exists, detail = codex_local_env.safe_path_exists(target)

    assert not exists
    assert "WinError 1920" in detail


def test_parse_mount_guard_detail_reports_warn_for_sandbox_protected_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_local_env, "ROOT", Path("/home/joshd/canompx3"))
    monkeypatch.setattr(codex_local_env, "platform_python", lambda platform: ["python3"])
    monkeypatch.setattr(
        codex_local_env,
        "capture_command",
        lambda command, env=None: codex_local_env.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "state": "sandbox-protected",
                    "fatal_issues": [],
                    "warnings": ["sandbox-protected path .git is read-only"],
                }
            ),
            stderr="",
        ),
    )

    status, detail = codex_local_env._parse_mount_guard_detail("wsl", {})

    assert status == "WARN"
    assert ".git" in detail


def test_run_doctor_prints_warn_without_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(codex_local_env, "ROOT", Path("/home/joshd/canompx3"))
    monkeypatch.setattr(codex_local_env, "safe_path_exists", lambda path: (True, str(path)))
    monkeypatch.setattr(codex_local_env, "is_wsl_native_root", lambda root: True)
    monkeypatch.setattr(codex_local_env.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(
        codex_local_env, "_parse_mount_guard_detail", lambda platform, env: ("WARN", "sandbox-protected")
    )
    monkeypatch.setattr(codex_local_env, "_detect_recent_wsl_reset", lambda: None)
    monkeypatch.setattr(codex_local_env, "_detect_model_mismatch_warning", lambda: None)
    monkeypatch.setattr(codex_local_env, "_detect_primary_model_drift", lambda: None)
    monkeypatch.setattr(codex_local_env, "_shared_codex_home_status", lambda: ("PASS", "managed launchers default"))

    def fake_capture(command: list[str], env=None):
        if command[:3] == ["git", "worktree", "list"]:
            return codex_local_env.subprocess.CompletedProcess(
                command, 0, stdout="/home/joshd/canompx3 main\n", stderr=""
            )
        return codex_local_env.subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(codex_local_env, "capture_command", fake_capture)

    codex_local_env.run_doctor("wsl")

    output = capsys.readouterr().out
    assert "[WARN] WSL mount guard: sandbox-protected" in output
    assert "[PASS] Shared CODEX_HOME: managed launchers default" in output
    assert "[PASS] Session preflight: ok" in output


def test_shared_codex_home_status_prefers_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_HOME", "/tmp/shared-codex")

    status, detail = codex_local_env._shared_codex_home_status()

    assert status == "PASS"
    assert "/tmp/shared-codex" in detail


def test_shared_codex_home_status_reports_managed_launcher_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shared_codex_home = tmp_path / ".codex"
    shared_codex_home.mkdir()
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_local_env, "default_shared_codex_home", lambda: shared_codex_home)

    status, detail = codex_local_env._shared_codex_home_status()

    assert status == "PASS"
    assert str(shared_codex_home) in detail

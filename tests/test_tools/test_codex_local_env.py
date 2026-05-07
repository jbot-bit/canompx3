from __future__ import annotations

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


def test_env_for_platform_sets_expected_uv_project_environment() -> None:
    wsl_env = codex_local_env.env_for_platform("wsl")
    windows_env = codex_local_env.env_for_platform("windows")

    assert wsl_env["UV_PROJECT_ENVIRONMENT"] == ".venv-wsl"
    assert wsl_env["JOBLIB_MULTIPROCESSING"] == "0"
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

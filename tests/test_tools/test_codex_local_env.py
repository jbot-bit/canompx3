from __future__ import annotations

from pathlib import Path

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

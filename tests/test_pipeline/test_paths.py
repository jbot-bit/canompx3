"""
Tests for pipeline.paths path constants.

Verifies paths resolve correctly relative to project root.
"""

import subprocess
from pathlib import Path

import pytest

import pipeline.paths as path_mod
from pipeline.paths import DBN_DIR, GOLD_DB_PATH, OHLCV_DIR, PROJECT_ROOT


class TestPaths:
    """Tests for canonical path constants."""

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_pipeline(self):
        assert (PROJECT_ROOT / "pipeline").exists()
        assert (PROJECT_ROOT / "pipeline").is_dir()

    def test_gold_db_path_is_absolute_and_named_correctly(self):
        # DB may be in the local checkout, a shared git-common-root checkout,
        # or an explicit DUCKDB_PATH override.
        import os

        assert GOLD_DB_PATH.is_absolute()
        assert GOLD_DB_PATH.name == "gold.db"
        if "DUCKDB_PATH" not in os.environ:
            assert (
                str(GOLD_DB_PATH).startswith(str(PROJECT_ROOT))
                or path_mod._discover_git_common_root(PROJECT_ROOT) is not None
            )

    def test_dbn_dir_is_in_project_root(self):
        assert DBN_DIR.parent == PROJECT_ROOT

    def test_ohlcv_dir_is_in_project_root(self):
        assert OHLCV_DIR.parent == PROJECT_ROOT

    def test_paths_are_absolute(self):
        assert PROJECT_ROOT.is_absolute()
        assert GOLD_DB_PATH.is_absolute()

    def test_default_canonical_db_uses_shared_git_root_when_worktree_has_no_local_db(self, tmp_path, monkeypatch):
        worktree_root = tmp_path / "worktree"
        common_root = tmp_path / "canonical"
        worktree_root.mkdir()
        common_root.mkdir()
        (common_root / "pipeline").mkdir()
        canonical_db = common_root / "gold.db"
        canonical_db.write_bytes(b"db")

        completed = subprocess.CompletedProcess(
            args=["git", "rev-parse", "--git-common-dir"],
            returncode=0,
            stdout=str(common_root / ".git"),
            stderr="",
        )
        monkeypatch.setattr(path_mod._subprocess, "run", lambda *args, **kwargs: completed)

        assert path_mod._default_canonical_db(worktree_root) == canonical_db

    def test_default_canonical_db_prefers_local_db_when_present(self, tmp_path, monkeypatch):
        project_root = tmp_path / "repo"
        project_root.mkdir()
        local_db = project_root / "gold.db"
        local_db.write_bytes(b"db")

        def _unexpected(*args, **kwargs):
            raise AssertionError("git common-dir lookup should not run when local gold.db exists")

        monkeypatch.setattr(path_mod._subprocess, "run", _unexpected)
        assert path_mod._default_canonical_db(project_root) == local_db

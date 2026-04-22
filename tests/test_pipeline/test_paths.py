"""
Tests for pipeline.paths path constants.

Verifies paths resolve correctly relative to project root.
"""

from pathlib import Path

import pytest

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
        # DB may be in the shared repo root (worktree-safe), project root, or
        # a DUCKDB_PATH override.
        import os

        assert GOLD_DB_PATH.is_absolute()
        assert GOLD_DB_PATH.name == "gold.db"
        if "DUCKDB_PATH" not in os.environ:
            assert GOLD_DB_PATH.exists()

    def test_dbn_dir_is_in_project_root(self):
        assert DBN_DIR.parent == PROJECT_ROOT

    def test_ohlcv_dir_is_in_project_root(self):
        assert OHLCV_DIR.parent == PROJECT_ROOT

    def test_paths_are_absolute(self):
        assert PROJECT_ROOT.is_absolute()
        assert GOLD_DB_PATH.is_absolute()

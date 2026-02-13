"""
Tests for pipeline.paths path constants.

Verifies paths resolve correctly relative to project root.
"""

import pytest
from pathlib import Path

from pipeline.paths import PROJECT_ROOT, GOLD_DB_PATH, DBN_DIR, OHLCV_DIR


class TestPaths:
    """Tests for canonical path constants."""

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_pipeline(self):
        assert (PROJECT_ROOT / "pipeline").exists()
        assert (PROJECT_ROOT / "pipeline").is_dir()

    def test_gold_db_path_is_under_project_root(self):
        # DB may be in project root or local_db/ junction
        assert str(GOLD_DB_PATH).startswith(str(PROJECT_ROOT))
        assert GOLD_DB_PATH.name == "gold.db"

    def test_dbn_dir_is_in_project_root(self):
        assert DBN_DIR.parent == PROJECT_ROOT

    def test_ohlcv_dir_is_in_project_root(self):
        assert OHLCV_DIR.parent == PROJECT_ROOT

    def test_paths_are_absolute(self):
        assert PROJECT_ROOT.is_absolute()
        assert GOLD_DB_PATH.is_absolute()

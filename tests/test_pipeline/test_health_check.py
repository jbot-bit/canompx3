"""Tests for pipeline.health_check — pipeline health check CLI."""

import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from pipeline.health_check import (
    check_python_deps,
    check_database,
    check_dbn_files,
    check_drift,
    check_tests,
    check_git_hooks,
)


class TestCheckPythonDeps:
    def test_all_present(self):
        """With all deps installed, should pass."""
        ok, msg = check_python_deps()
        assert ok is True
        assert "all deps installed" in msg
        assert "Python" in msg

    def test_missing_dep(self):
        """Simulate a missing dependency."""
        import builtins
        original = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "zstandard":
                raise ImportError("mocked")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            ok, msg = check_python_deps()
            assert ok is False
            assert "zstandard" in msg


class TestCheckDatabase:
    def test_missing_db(self, tmp_path):
        """Non-existent DB returns failure."""
        with patch("pipeline.health_check.GOLD_DB_PATH", tmp_path / "missing.db"):
            ok, msg = check_database()
            assert ok is False
            assert "does not exist" in msg

    def test_existing_db(self, tmp_path):
        """Valid DB with tables returns success."""
        import duckdb
        db_path = tmp_path / "gold.db"
        con = duckdb.connect(str(db_path))
        con.execute("CREATE TABLE bars_1m (x INT)")
        con.execute("CREATE TABLE bars_5m (x INT)")
        con.execute("CREATE TABLE daily_features (x INT)")
        con.close()

        with patch("pipeline.health_check.GOLD_DB_PATH", db_path):
            ok, msg = check_database()
            assert ok is True
            assert "bars_1m" in msg


class TestCheckDbnFiles:
    def test_missing_dir(self, tmp_path):
        """Missing data directory returns failure."""
        with patch("pipeline.health_check.DAILY_DBN_DIR", tmp_path / "nope"):
            ok, msg = check_dbn_files()
            assert ok is False
            assert "missing" in msg.lower()

    def test_no_files(self, tmp_path):
        """Empty directory returns failure."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("pipeline.health_check.DAILY_DBN_DIR", empty_dir):
            ok, msg = check_dbn_files()
            assert ok is False
            assert "No .dbn.zst files" in msg

    def test_files_present(self, tmp_path):
        """Directory with DBN files returns success."""
        dbn_dir = tmp_path / "dbn"
        dbn_dir.mkdir()
        (dbn_dir / "glbx-mdp3-20240101.ohlcv-1m.dbn.zst").touch()
        (dbn_dir / "glbx-mdp3-20240102.ohlcv-1m.dbn.zst").touch()
        with patch("pipeline.health_check.DAILY_DBN_DIR", dbn_dir):
            ok, msg = check_dbn_files()
            assert ok is True
            assert "2" in msg


class TestCheckDrift:
    def test_drift_passes(self):
        """Successful drift check."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Check 1: PASSED [OK]\nCheck 2: PASSED [OK]\n"
        with patch("subprocess.run", return_value=mock_result):
            ok, msg = check_drift()
            assert ok is True
            assert "2/2" in msg

    def test_drift_fails(self):
        """Failed drift check."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "FAILED"
        with patch("subprocess.run", return_value=mock_result):
            ok, msg = check_drift()
            assert ok is False
            assert "FAILED" in msg

    def test_drift_timeout(self):
        """Timeout during drift check."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="", timeout=30)):
            ok, msg = check_drift()
            assert ok is False
            assert "error" in msg.lower()


class TestCheckTests:
    def test_tests_pass(self):
        """Successful test run."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "42 passed in 10s\n"
        with patch("subprocess.run", return_value=mock_result):
            ok, msg = check_tests()
            assert ok is True
            assert "42 passed" in msg

    def test_tests_fail(self):
        """Failed test run."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "1 failed, 41 passed\n"
        with patch("subprocess.run", return_value=mock_result):
            ok, msg = check_tests()
            assert ok is False


class TestCheckGitHooks:
    def test_hooks_configured(self, tmp_path):
        """Correctly configured hooks."""
        hooks_dir = tmp_path / ".githooks"
        hooks_dir.mkdir()
        (hooks_dir / "pre-commit").touch()

        mock_result = MagicMock()
        mock_result.stdout = ".githooks\n"

        with patch("pipeline.health_check.PROJECT_ROOT", tmp_path), \
             patch("subprocess.run", return_value=mock_result):
            ok, msg = check_git_hooks()
            assert ok is True
            assert "configured" in msg

    def test_missing_pre_commit(self, tmp_path):
        """Missing pre-commit hook file."""
        hooks_dir = tmp_path / ".githooks"
        hooks_dir.mkdir()
        # No pre-commit file

        with patch("pipeline.health_check.PROJECT_ROOT", tmp_path):
            ok, msg = check_git_hooks()
            assert ok is False
            assert "missing" in msg.lower()

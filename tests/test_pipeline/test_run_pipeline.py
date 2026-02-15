"""Tests for pipeline/run_pipeline.py — subprocess orchestration.

All tests mock subprocess.run. No real subprocesses, no database access.
"""

import argparse
import sys
from unittest.mock import patch, MagicMock

import pytest

from pipeline.run_pipeline import (
    PIPELINE_STEPS,
    main,
    step_ingest,
    step_build_5m,
    step_build_features,
    step_audit,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_args(**overrides):
    """Build a fake argparse.Namespace with sensible defaults."""
    defaults = dict(
        start="2024-01-01",
        end="2024-12-31",
        resume=False,
        retry_failed=False,
        dry_run=False,
        chunk_days=7,
        batch_size=50000,
        orb_minutes=5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def mock_subprocess_ok():
    """Return a MagicMock that simulates subprocess.run returning 0."""
    m = MagicMock()
    m.returncode = 0
    return m


# =============================================================================
# 1. STEP REGISTRY STRUCTURE
# =============================================================================

class TestStepRegistry:
    def test_pipeline_steps_count(self):
        assert len(PIPELINE_STEPS) == 4

    def test_pipeline_steps_are_3_tuples(self):
        for entry in PIPELINE_STEPS:
            assert len(entry) == 3
            name, desc, func = entry
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert callable(func)

    def test_pipeline_steps_order(self):
        names = [name for name, _, _ in PIPELINE_STEPS]
        assert names == ["ingest", "build_5m", "build_features", "audit"]

    def test_pipeline_steps_functions(self):
        funcs = [func for _, _, func in PIPELINE_STEPS]
        assert funcs == [step_ingest, step_build_5m, step_build_features, step_audit]


# =============================================================================
# 2. COMMAND CONSTRUCTION
# =============================================================================

class TestStepIngest:
    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_ingest_basic_cmd(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        args = make_args()
        step_ingest("MGC", args)

        cmd = mock_run.call_args[0][0]
        assert f"--instrument=MGC" in cmd
        assert f"--start=2024-01-01" in cmd
        assert f"--end=2024-12-31" in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_ingest_resume_flag(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()

        # resume=True -> flag present
        step_ingest("MGC", make_args(resume=True))
        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd

        # resume=False -> flag absent
        step_ingest("MGC", make_args(resume=False))
        cmd = mock_run.call_args[0][0]
        assert "--resume" not in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_ingest_retry_failed_flag(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()

        step_ingest("MGC", make_args(retry_failed=True))
        cmd = mock_run.call_args[0][0]
        assert "--retry-failed" in cmd

        step_ingest("MGC", make_args(retry_failed=False))
        cmd = mock_run.call_args[0][0]
        assert "--retry-failed" not in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_ingest_chunk_days_nondefault(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_ingest("MGC", make_args(chunk_days=14))
        cmd = mock_run.call_args[0][0]
        assert "--chunk-days=14" in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_ingest_chunk_days_default_omitted(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_ingest("MGC", make_args(chunk_days=7))
        cmd = mock_run.call_args[0][0]
        assert all("--chunk-days" not in c for c in cmd)

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_ingest_batch_size_nondefault(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_ingest("MGC", make_args(batch_size=100000))
        cmd = mock_run.call_args[0][0]
        assert "--batch-size=100000" in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_ingest_batch_size_default_omitted(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_ingest("MGC", make_args(batch_size=50000))
        cmd = mock_run.call_args[0][0]
        assert all("--batch-size" not in c for c in cmd)

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_ingest_dry_run_flag(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_ingest("MGC", make_args(dry_run=True))
        cmd = mock_run.call_args[0][0]
        assert "--dry-run" in cmd


class TestStepBuild5m:
    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_build_5m_missing_dates_returns_1(self, mock_run):
        rc = step_build_5m("MGC", make_args(start=None, end=None))
        assert rc == 1
        mock_run.assert_not_called()

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_build_5m_basic_cmd(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_build_5m("MGC", make_args())
        cmd = mock_run.call_args[0][0]
        assert "--instrument=MGC" in cmd
        assert "--start=2024-01-01" in cmd
        assert "--end=2024-12-31" in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_build_5m_dry_run_flag(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_build_5m("MGC", make_args(dry_run=True))
        cmd = mock_run.call_args[0][0]
        assert "--dry-run" in cmd


class TestStepBuildFeatures:
    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_build_features_missing_dates_returns_1(self, mock_run):
        rc = step_build_features("MGC", make_args(start=None, end=None))
        assert rc == 1
        mock_run.assert_not_called()

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_build_features_basic_cmd(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_build_features("MGC", make_args())
        cmd = mock_run.call_args[0][0]
        assert "--instrument=MGC" in cmd
        assert "--start=2024-01-01" in cmd
        assert "--end=2024-12-31" in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_build_features_orb_minutes(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_build_features("MGC", make_args(orb_minutes=15))
        cmd = mock_run.call_args[0][0]
        assert "--orb-minutes=15" in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_build_features_orb_minutes_default_always_appended(self, mock_run):
        """orb_minutes is always in cmd, even at default (unlike chunk_days/batch_size)."""
        mock_run.return_value = mock_subprocess_ok()
        step_build_features("MGC", make_args(orb_minutes=5))
        cmd = mock_run.call_args[0][0]
        assert "--orb-minutes=5" in cmd

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_build_features_dry_run_flag(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_build_features("MGC", make_args(dry_run=True))
        cmd = mock_run.call_args[0][0]
        assert "--dry-run" in cmd


class TestStepAudit:
    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_audit_cmd(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        step_audit("MGC", make_args())
        cmd = mock_run.call_args[0][0]
        # Should run check_db.py with no instrument/date args
        assert any("check_db.py" in c for c in cmd)
        assert all("--instrument" not in c for c in cmd)
        assert all("--start" not in c for c in cmd)


# =============================================================================
# 3. FAIL-CLOSED ORCHESTRATION
# =============================================================================

class TestMainOrchestration:
    """Test main() via sys.argv + mocked subprocess.run."""

    BASE_ARGV = [
        "run_pipeline.py",
        "--instrument=MGC",
        "--start=2024-01-01",
        "--end=2024-12-31",
    ]

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_main_all_steps_pass(self, mock_run):
        mock_run.return_value = mock_subprocess_ok()
        with patch.object(sys, "argv", self.BASE_ARGV):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        assert mock_run.call_count == 4

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_main_step1_fails_skips_rest(self, mock_run):
        fail = MagicMock()
        fail.returncode = 1
        mock_run.return_value = fail
        with patch.object(sys, "argv", self.BASE_ARGV):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        assert mock_run.call_count == 1

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_main_step2_fails_skips_rest(self, mock_run):
        ok = mock_subprocess_ok()
        fail = MagicMock()
        fail.returncode = 1
        mock_run.side_effect = [ok, fail]
        with patch.object(sys, "argv", self.BASE_ARGV):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        assert mock_run.call_count == 2

    @patch("pipeline.run_pipeline.subprocess.run")
    def test_main_step3_fails_skips_step4(self, mock_run):
        ok = mock_subprocess_ok()
        fail = MagicMock()
        fail.returncode = 1
        mock_run.side_effect = [ok, ok, fail]
        with patch.object(sys, "argv", self.BASE_ARGV):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        assert mock_run.call_count == 3


# =============================================================================
# 4. DRY RUN
# =============================================================================

class TestDryRun:
    DRY_ARGV = [
        "run_pipeline.py",
        "--instrument=MGC",
        "--start=2024-01-01",
        "--end=2024-12-31",
        "--dry-run",
    ]

    @patch("pipeline.asset_configs.get_asset_config")
    @patch("pipeline.run_pipeline.subprocess.run")
    def test_dry_run_exits_before_subprocess(self, mock_run, mock_config):
        mock_config.return_value = {
            "dbn_path": "/fake/path",
            "outright_pattern": MagicMock(pattern="GC.*"),
            "minimum_start_date": "2016-01-01",
        }
        with patch.object(sys, "argv", self.DRY_ARGV):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        mock_run.assert_not_called()

    @patch("pipeline.asset_configs.get_asset_config")
    @patch("pipeline.run_pipeline.subprocess.run")
    def test_dry_run_prints_all_step_names(self, mock_run, mock_config, capsys):
        mock_config.return_value = {
            "dbn_path": "/fake/path",
            "outright_pattern": MagicMock(pattern="GC.*"),
            "minimum_start_date": "2016-01-01",
        }
        with patch.object(sys, "argv", self.DRY_ARGV):
            with pytest.raises(SystemExit):
                main()
        output = capsys.readouterr().out
        for name, _, _ in PIPELINE_STEPS:
            assert name in output


# =============================================================================
# 5. RETURN CODE PROPAGATION
# =============================================================================

class TestReturnCodePropagation:
    @patch("pipeline.run_pipeline.subprocess.run")
    def test_step_returns_subprocess_returncode(self, mock_run):
        ret = MagicMock()
        ret.returncode = 42
        mock_run.return_value = ret
        assert step_ingest("MGC", make_args()) == 42

    def test_step_build_5m_returns_1_without_dates(self):
        assert step_build_5m("MGC", make_args(start=None, end=None)) == 1

"""
Tests for pipeline.ingest_dbn_daily module.

Tests file discovery, MGC filtering, and checkpoint round-trip logic.
"""

import json
import re
import pytest
from pathlib import Path
from datetime import date

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.ingest_dbn_daily import (
    discover_daily_files,
    DAILY_FILE_PATTERN,
)
from pipeline.ingest_dbn_mgc import (
    CheckpointManager,
    MGC_OUTRIGHT_PATTERN,
)


class TestDailyFilePattern:
    """Tests for the daily file name regex."""

    def test_matches_valid_file(self):
        assert DAILY_FILE_PATTERN.match("glbx-mdp3-20210205.ohlcv-1m.dbn.zst")

    def test_extracts_date(self):
        m = DAILY_FILE_PATTERN.match("glbx-mdp3-20240315.ohlcv-1m.dbn.zst")
        assert m.group(1) == "20240315"

    def test_rejects_bad_name(self):
        assert DAILY_FILE_PATTERN.match("random-file.csv") is None

    def test_rejects_wrong_schema(self):
        assert DAILY_FILE_PATTERN.match("glbx-mdp3-20210205.trades.dbn.zst") is None


class TestDiscoverDailyFiles:
    """Tests for discover_daily_files()."""

    def test_discovers_files_in_range(self, tmp_path):
        # Create fake daily files
        for d in ["20240101", "20240102", "20240103", "20240104", "20240105"]:
            (tmp_path / f"glbx-mdp3-{d}.ohlcv-1m.dbn.zst").write_bytes(b"fake")

        files = discover_daily_files(tmp_path, date(2024, 1, 2), date(2024, 1, 4))
        assert len(files) == 3
        assert files[0][0] == date(2024, 1, 2)
        assert files[-1][0] == date(2024, 1, 4)

    def test_sorted_by_date(self, tmp_path):
        for d in ["20240105", "20240101", "20240103"]:
            (tmp_path / f"glbx-mdp3-{d}.ohlcv-1m.dbn.zst").write_bytes(b"fake")

        files = discover_daily_files(tmp_path, date(2024, 1, 1), date(2024, 12, 31))
        dates = [f[0] for f in files]
        assert dates == sorted(dates)

    def test_empty_dir(self, tmp_path):
        files = discover_daily_files(tmp_path, date(2024, 1, 1), date(2024, 12, 31))
        assert files == []

    def test_ignores_non_matching_files(self, tmp_path):
        (tmp_path / "glbx-mdp3-20240101.ohlcv-1m.dbn.zst").write_bytes(b"fake")
        (tmp_path / "symbology.json").write_text("{}")
        (tmp_path / "readme.txt").write_text("hello")

        files = discover_daily_files(tmp_path, date(2024, 1, 1), date(2024, 12, 31))
        assert len(files) == 1


class TestMgcOutrightFilter:
    """Tests for the MGC outright contract pattern."""

    def test_matches_mgc_outrights(self):
        assert MGC_OUTRIGHT_PATTERN.match("MGCG4")
        assert MGC_OUTRIGHT_PATTERN.match("MGCZ25")
        assert MGC_OUTRIGHT_PATTERN.match("MGCM1")
        assert MGC_OUTRIGHT_PATTERN.match("MGCQ24")

    def test_rejects_gc_contracts(self):
        assert MGC_OUTRIGHT_PATTERN.match("GCG4") is None
        assert MGC_OUTRIGHT_PATTERN.match("GCZ25") is None

    def test_rejects_spreads(self):
        assert MGC_OUTRIGHT_PATTERN.match("MGCG4-MGCM4") is None

    def test_rejects_non_futures(self):
        assert MGC_OUTRIGHT_PATTERN.match("MGC") is None
        assert MGC_OUTRIGHT_PATTERN.match("MGCX") is None


class TestCheckpointRoundtrip:
    """Tests for checkpoint write + read + resume logic."""

    def test_write_and_read(self, tmp_path):
        # Create a fake source file
        src = tmp_path / "source.dbn.zst"
        src.write_bytes(b"fake data")

        cp = CheckpointManager(tmp_path / "checkpoints", src)

        # Write a done checkpoint
        cp.write_checkpoint("2024-01-01", "2024-01-07", "done", rows_written=1000)

        # Reload and verify
        cp2 = CheckpointManager(tmp_path / "checkpoints", src)
        assert cp2.get_chunk_status("2024-01-01", "2024-01-07") == "done"

    def test_resume_skips_done(self, tmp_path):
        src = tmp_path / "source.dbn.zst"
        src.write_bytes(b"fake data")

        cp = CheckpointManager(tmp_path / "checkpoints", src)
        cp.write_checkpoint("2024-01-01", "2024-01-07", "done", rows_written=1000)

        # should_process_chunk returns False for done chunks
        assert cp.should_process_chunk("2024-01-01", "2024-01-07", retry_failed=False) is False

    def test_resume_processes_new(self, tmp_path):
        src = tmp_path / "source.dbn.zst"
        src.write_bytes(b"fake data")

        cp = CheckpointManager(tmp_path / "checkpoints", src)
        # Never-seen chunk should be processed
        assert cp.should_process_chunk("2024-02-01", "2024-02-07", retry_failed=False) is True

    def test_retry_failed(self, tmp_path):
        src = tmp_path / "source.dbn.zst"
        src.write_bytes(b"fake data")

        cp = CheckpointManager(tmp_path / "checkpoints", src)
        cp.write_checkpoint("2024-01-01", "2024-01-07", "failed", error="test error")

        # Without retry_failed, skip
        assert cp.should_process_chunk("2024-01-01", "2024-01-07", retry_failed=False) is False
        # With retry_failed, process
        assert cp.should_process_chunk("2024-01-01", "2024-01-07", retry_failed=True) is True

    def test_checkpoint_jsonl_format(self, tmp_path):
        src = tmp_path / "source.dbn.zst"
        src.write_bytes(b"fake data")

        cp = CheckpointManager(tmp_path / "checkpoints", src)
        cp.write_checkpoint("2024-01-01", "2024-01-07", "done", rows_written=500)

        # Verify JSONL file is valid
        cp_file = list((tmp_path / "checkpoints").glob("checkpoint_*.jsonl"))[0]
        with open(cp_file) as f:
            for line in f:
                record = json.loads(line)
                assert "chunk_start" in record
                assert "status" in record

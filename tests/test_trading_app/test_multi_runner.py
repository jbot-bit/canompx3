"""Tests for multi-instrument live session runner."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from trading_app.live.multi_runner import _STOP_FILE, MultiInstrumentRunner


class TestMultiInstrumentRunner:
    """Test MultiInstrumentRunner creation and lifecycle."""

    @patch("trading_app.live.multi_runner.SessionOrchestrator")
    def test_creates_orchestrators_for_all_instruments(self, mock_orch_cls):
        """Each instrument gets its own orchestrator."""
        mock_orch = MagicMock()
        mock_orch.portfolio.strategies = [MagicMock()] * 3
        mock_orch_cls.return_value = mock_orch

        runner = MultiInstrumentRunner(
            instruments=["MGC", "MNQ"],
            broker="projectx",
            demo=True,
            signal_only=True,
        )

        assert len(runner.orchestrators) == 2
        assert "MGC" in runner.orchestrators
        assert "MNQ" in runner.orchestrators
        assert mock_orch_cls.call_count == 2

    @patch("trading_app.live.multi_runner.SessionOrchestrator")
    def test_partial_failure_continues_with_remaining(self, mock_orch_cls):
        """If one instrument fails to init, others still run."""
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["instrument"] == "MES":
                raise RuntimeError("MES has no strategies")
            mock = MagicMock()
            mock.portfolio.strategies = [MagicMock()]
            return mock

        mock_orch_cls.side_effect = side_effect

        runner = MultiInstrumentRunner(
            instruments=["MGC", "MES", "MNQ"],
            broker="projectx",
            demo=True,
            signal_only=True,
        )

        assert len(runner.orchestrators) == 2
        assert "MES" not in runner.orchestrators

    @patch("trading_app.live.multi_runner.SessionOrchestrator")
    def test_total_failure_raises(self, mock_orch_cls):
        """If ALL instruments fail, raise RuntimeError."""
        mock_orch_cls.side_effect = RuntimeError("no strategies")

        with pytest.raises(RuntimeError, match="No orchestrators created"):
            MultiInstrumentRunner(
                instruments=["MGC"],
                broker="projectx",
                demo=True,
                signal_only=True,
            )

    @patch("trading_app.live.multi_runner.SessionOrchestrator")
    def test_run_calls_all_orchestrators(self, mock_orch_cls):
        """run() launches all orchestrators concurrently."""

        def make_orch(**kwargs):
            m = MagicMock()
            m.portfolio.strategies = [MagicMock()]
            m.run = AsyncMock()
            return m

        mock_orch_cls.side_effect = make_orch

        runner = MultiInstrumentRunner(
            instruments=["MGC", "MNQ"],
            broker="projectx",
            demo=True,
            signal_only=True,
        )

        asyncio.run(runner.run())

        # Both orchestrators' run() should have been called
        for orch in runner.orchestrators.values():
            orch.run.assert_awaited_once()

    @patch("trading_app.live.multi_runner.SessionOrchestrator")
    def test_post_session_calls_all(self, mock_orch_cls):
        """post_session() runs for every instrument, even if one fails."""
        mock_orch1 = MagicMock()
        mock_orch1.portfolio.strategies = [MagicMock()]
        mock_orch1.post_session.side_effect = RuntimeError("cleanup failed")

        mock_orch2 = MagicMock()
        mock_orch2.portfolio.strategies = [MagicMock()]

        mock_orch_cls.side_effect = [mock_orch1, mock_orch2]

        runner = MultiInstrumentRunner(
            instruments=["MGC", "MNQ"],
            broker="projectx",
            demo=True,
            signal_only=True,
        )

        # Should not raise even though MGC's post_session fails
        runner.post_session()

        mock_orch1.post_session.assert_called_once()
        mock_orch2.post_session.assert_called_once()

    @patch("trading_app.live.multi_runner.SessionOrchestrator")
    def test_stop_file_cleaned_after_run(self, mock_orch_cls):
        """Stop file is deleted after all feeds exit, not by individual feeds."""
        mock_orch = MagicMock()
        mock_orch.portfolio.strategies = [MagicMock()]
        mock_orch.run = AsyncMock()
        mock_orch_cls.return_value = mock_orch

        runner = MultiInstrumentRunner(
            instruments=["MGC"],
            broker="projectx",
            demo=True,
            signal_only=True,
        )

        # Create stop file
        _STOP_FILE.touch()
        assert _STOP_FILE.exists()

        asyncio.run(runner.run())

        # Runner should have cleaned it up
        assert not _STOP_FILE.exists()

    @patch("trading_app.live.multi_runner.SessionOrchestrator")
    def test_default_instruments_from_active_config(self, mock_orch_cls):
        """instruments=None uses ACTIVE_ORB_INSTRUMENTS."""
        mock_orch = MagicMock()
        mock_orch.portfolio.strategies = [MagicMock()]
        mock_orch_cls.return_value = mock_orch

        runner = MultiInstrumentRunner(
            instruments=None,
            broker="projectx",
            demo=True,
            signal_only=True,
        )

        # Should have created orchestrators for all active instruments
        assert len(runner.orchestrators) >= len(ACTIVE_ORB_INSTRUMENTS)

    @patch("trading_app.live.multi_runner.SessionOrchestrator")
    def test_one_crash_others_survive(self, mock_orch_cls):
        """If one orchestrator crashes mid-run, others still complete."""

        async def crash():
            raise RuntimeError("MGC feed disconnected")

        async def succeed():
            pass

        mock_mgc = MagicMock()
        mock_mgc.portfolio.strategies = [MagicMock()]
        mock_mgc.run = AsyncMock(side_effect=crash)

        mock_mnq = MagicMock()
        mock_mnq.portfolio.strategies = [MagicMock()]
        mock_mnq.run = AsyncMock(side_effect=succeed)

        mock_orch_cls.side_effect = [mock_mgc, mock_mnq]

        runner = MultiInstrumentRunner(
            instruments=["MGC", "MNQ"],
            broker="projectx",
            demo=True,
            signal_only=True,
        )

        # Should NOT raise — gather(return_exceptions=True) isolates failures
        asyncio.run(runner.run())

        # Both should have been called
        mock_mgc.run.assert_awaited_once()
        mock_mnq.run.assert_awaited_once()


class TestStopFileRaceCondition:
    """Verify stop-file is NOT deleted by individual feeds."""

    def test_projectx_feed_does_not_delete_stop_file(self):
        """ProjectX feed should not unlink stop file (multi-instrument safe)."""
        import ast

        feed_path = Path("trading_app/live/projectx/data_feed.py")
        source = feed_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "unlink":
                    line = source.splitlines()[node.lineno - 1]
                    assert "_STOP_FILE" not in line, (
                        f"Line {node.lineno}: feed still deletes stop file — breaks multi-instrument shutdown"
                    )

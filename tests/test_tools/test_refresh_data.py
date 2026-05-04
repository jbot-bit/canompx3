"""Regression tests for scripts/tools/refresh_data.py.

Covers the "bars up to date but features stale" case that caused
2026-04-14 drift (MES/MNQ missing all daily_features rows, MGC orphan).
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

from scripts.tools import refresh_data


class TestRefreshInstrumentAlwaysBuilds:
    """refresh_instrument must invoke build steps even when bars are current.

    Prior bug: early return on gap_days <= 0 skipped daily_features and outcome
    builds entirely. Out-of-band bar ingestion (e.g. live bot's BarPersister
    writes bars_1m at session end) would leave features stale indefinitely.
    """

    def _stub_today(self, today: date):
        """Patch date.today() inside refresh_data to a fixed value."""

        class _FixedDate(date):
            @classmethod
            def today(cls):
                return today

        return patch.object(refresh_data, "date", _FixedDate)

    def test_bars_current_still_invokes_build_steps_for_yesterday(self):
        """Regression for 2026-04-14: gap_days=0 must still trigger build."""
        today = date(2026, 4, 14)
        yesterday = today - timedelta(days=1)

        calls: list[tuple] = []

        def fake_build(instrument, start, end, full_rebuild=False):
            calls.append(("build", instrument, start, end, full_rebuild))
            return True

        with (
            self._stub_today(today),
            patch.object(refresh_data, "get_last_bar_date", return_value=yesterday),
            patch.object(refresh_data, "run_build_steps", side_effect=fake_build),
            patch.object(refresh_data, "download_dbn") as mock_download,
            patch.object(refresh_data, "run_ingest") as mock_ingest,
            patch("pipeline.daily_backfill._patch_atr_percentiles", return_value=None),
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=False, full_rebuild=False)

        assert ok is True
        # Critical regression assertion: build runs even with no download
        assert len(calls) == 1, f"Expected 1 build call, got {len(calls)}"
        _, inst, build_start, build_end, _ = calls[0]
        assert inst == "MNQ"
        assert build_start == yesterday
        assert build_end == yesterday
        # No download or ingest should have been triggered
        mock_download.assert_not_called()
        mock_ingest.assert_not_called()

    def test_bars_stale_triggers_download_and_builds_gap(self):
        """Happy-path regression: gap_days > 0 still downloads + builds full gap."""
        today = date(2026, 4, 14)
        last = date(2026, 4, 10)  # 3-day gap

        calls: list[tuple] = []

        def fake_build(instrument, start, end, full_rebuild=False):
            calls.append((instrument, start, end))
            return True

        with (
            self._stub_today(today),
            patch.object(refresh_data, "get_last_bar_date", return_value=last),
            patch.object(refresh_data, "run_build_steps", side_effect=fake_build),
            patch.object(refresh_data, "download_dbn", return_value=object()),  # non-None
            patch.object(refresh_data, "run_ingest", return_value=True),
            patch("pipeline.daily_backfill._patch_atr_percentiles", return_value=None),
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=False, full_rebuild=False)

        assert ok is True
        assert len(calls) == 1
        inst, build_start, build_end = calls[0]
        assert inst == "MNQ"
        assert build_start == last + timedelta(days=1)  # fetch_start
        assert build_end == today - timedelta(days=1)  # yesterday

    def test_dry_run_no_builds_when_current(self):
        """Dry run with current bars should return True without building."""
        today = date(2026, 4, 14)
        yesterday = today - timedelta(days=1)

        with (
            self._stub_today(today),
            patch.object(refresh_data, "get_last_bar_date", return_value=yesterday),
            patch.object(refresh_data, "run_build_steps") as mock_build,
            patch.object(refresh_data, "download_dbn") as mock_download,
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=True, full_rebuild=False)

        assert ok is True
        mock_build.assert_not_called()
        mock_download.assert_not_called()

    def test_no_bars_at_all_skips(self):
        """get_last_bar_date=None means no initial load — refresh must skip."""
        today = date(2026, 4, 14)

        with (
            self._stub_today(today),
            patch.object(refresh_data, "get_last_bar_date", return_value=None),
            patch.object(refresh_data, "run_build_steps") as mock_build,
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=False, full_rebuild=False)

        assert ok is False
        mock_build.assert_not_called()

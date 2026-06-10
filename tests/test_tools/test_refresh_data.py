"""Regression tests for scripts/tools/refresh_data.py.

Covers two regression classes:
  1. "bars up to date but features stale" (2026-04-14): gap_days<=0 must still
     trigger a build for the last complete trading day.
  2. "calendar boundary silently skips a complete trading day" (2026-06-10):
     the build END must come from the canonical trading-day window
     (pipeline.dst), NOT calendar `today - 1`. When Databento availability
     clamped api_end back a day, the old `yesterday = api_end - 1` dropped the
     complete 06-09 trading day and blocked an unrelated commit on drift Check 79.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import patch

from pipeline.dst import compute_trading_day_utc_range
from scripts.tools import refresh_data


class TestLastCompleteTradingDay:
    """`_last_complete_trading_day` delegates to canonical pipeline.dst — a day
    is COMPLETE only when its 09:00->09:00 Brisbane window has fully elapsed."""

    def test_just_after_window_close_returns_that_day(self):
        """now == window END (exclusive) => that day is complete. EXACT 06-09 case."""
        td = date(2026, 6, 9)
        _start, end = compute_trading_day_utc_range(td)
        assert refresh_data._last_complete_trading_day(end) == td

    def test_inside_in_progress_window_returns_prior_day(self):
        """now 1s before close => window not elapsed => prior day is last complete."""
        td = date(2026, 6, 9)
        _start, end = compute_trading_day_utc_range(td)
        now = end - timedelta(seconds=1)
        assert refresh_data._last_complete_trading_day(now) == td - timedelta(days=1)

    def test_well_into_next_window_returns_just_closed_day(self):
        """5h into the 06-10 window => 06-09 is the last complete trading day."""
        td = date(2026, 6, 9)
        _start, end = compute_trading_day_utc_range(td)
        now = end + timedelta(hours=5)
        assert refresh_data._last_complete_trading_day(now) == td

    def test_exact_0900_brisbane_2300_utc_edge(self):
        """At exactly 23:00 UTC (09:00 Brisbane) the new window opens; the day
        whose window JUST closed (== that 23:00 UTC instant) is complete."""
        td = date(2026, 6, 9)
        _start, end = compute_trading_day_utc_range(td)
        assert end == datetime(2026, 6, 9, 23, 0, 0, tzinfo=UTC)
        # At the boundary instant, td's window end <= now => td complete.
        assert refresh_data._last_complete_trading_day(end) == td


class TestBuildBoundaryFromCanonicalTradingDay:
    """refresh_instrument's build END must be the canonical last-complete trading
    day, clamped to bars that exist — never calendar `today - 1`."""

    def _stub_today(self, today: date):
        """Patch date.today() inside refresh_data to a fixed value."""

        class _FixedDate(date):
            @classmethod
            def today(cls):
                return today

        return patch.object(refresh_data, "date", _FixedDate)

    def test_bars_current_builds_through_last_complete_trading_day(self):
        """2026-06-10 regression: bars current through the just-closed trading day
        (06-09) => build END must be 06-09, not calendar today-1 (06-09 here, but
        the boundary is driven by the canonical day, proven by the stub)."""
        today = date(2026, 6, 10)
        last_bar = date(2026, 6, 9)  # BarPersister wrote 06-09 out-of-band
        complete_td = date(2026, 6, 9)

        calls: list[tuple] = []

        def fake_build(instrument, start, end, full_rebuild=False):
            calls.append(("build", instrument, start, end, full_rebuild))
            return True

        with (
            self._stub_today(today),
            patch.object(refresh_data, "get_last_bar_date", return_value=last_bar),
            patch.object(refresh_data, "_last_complete_trading_day", return_value=complete_td),
            patch.object(refresh_data, "run_build_steps", side_effect=fake_build),
            patch.object(refresh_data, "download_dbn") as mock_download,
            patch.object(refresh_data, "run_ingest") as mock_ingest,
            patch("pipeline.daily_backfill._patch_atr_percentiles", return_value=None),
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=False, full_rebuild=False)

        assert ok is True
        assert len(calls) == 1, f"Expected 1 build call, got {len(calls)}"
        _, inst, build_start, build_end, _ = calls[0]
        assert inst == "MNQ"
        # Boundary is the canonical complete trading day, clamped to last bar.
        assert build_end == complete_td
        assert build_start == complete_td
        mock_download.assert_not_called()
        mock_ingest.assert_not_called()

    def test_bars_current_build_end_never_exceeds_last_bar(self):
        """If the canonical complete day is AHEAD of the bars we have (data
        outage), build END is clamped to the last bar — never build past bars."""
        today = date(2026, 6, 12)
        last_bar = date(2026, 6, 9)  # bars stop at 06-09
        complete_td = date(2026, 6, 11)  # but 06-11 has fully elapsed

        calls: list[tuple] = []

        def fake_build(instrument, start, end, full_rebuild=False):
            calls.append((instrument, start, end))
            return True

        with (
            self._stub_today(today),
            patch.object(refresh_data, "get_last_bar_date", return_value=last_bar),
            patch.object(refresh_data, "_last_complete_trading_day", return_value=complete_td),
            patch.object(refresh_data, "run_build_steps", side_effect=fake_build),
            patch.object(refresh_data, "download_dbn", return_value=object()),
            patch.object(refresh_data, "run_ingest", return_value=True),
            patch("pipeline.daily_backfill._patch_atr_percentiles", return_value=None),
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=False, full_rebuild=False)

        assert ok is True
        assert len(calls) == 1
        _inst, _build_start, build_end = calls[0]
        # min(complete_td=06-11, last_bar=06-09) == 06-09
        assert build_end == last_bar

    def test_bars_stale_downloads_then_builds_through_complete_day(self):
        """gap_days>0: download+ingest, then build END = min(canonical complete
        day, POST-INGEST last bar). After ingest brings bars current to 06-09,
        the build must extend to 06-09 (not the pre-fetch last bar)."""
        today = date(2026, 6, 10)
        pre_fetch_last = date(2026, 6, 6)  # 4-day gap pre-fetch
        post_ingest_last = date(2026, 6, 9)  # ingest landed bars through 06-09
        complete_td = date(2026, 6, 9)

        calls: list[tuple] = []

        def fake_build(instrument, start, end, full_rebuild=False):
            calls.append((instrument, start, end))
            return True

        # get_last_bar_date is called twice: pre-fetch (returns pre_fetch_last)
        # then post-ingest (returns post_ingest_last).
        with (
            self._stub_today(today),
            patch.object(
                refresh_data,
                "get_last_bar_date",
                side_effect=[pre_fetch_last, post_ingest_last],
            ),
            patch.object(refresh_data, "_last_complete_trading_day", return_value=complete_td),
            patch.object(refresh_data, "run_build_steps", side_effect=fake_build),
            patch.object(refresh_data, "download_dbn", return_value=object()),
            patch.object(refresh_data, "run_ingest", return_value=True),
            patch("pipeline.daily_backfill._patch_atr_percentiles", return_value=None),
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=False, full_rebuild=False)

        assert ok is True
        assert len(calls) == 1
        inst, build_start, build_end = calls[0]
        assert inst == "MNQ"
        assert build_start == pre_fetch_last + timedelta(days=1)  # fetch_start
        # Build END extends to the post-ingest complete trading day, NOT the
        # stale pre-fetch last bar (the calendar-boundary bug).
        assert build_end == post_ingest_last

    def test_dry_run_no_builds_when_current(self):
        """Dry run with current bars should return True without building."""
        today = date(2026, 6, 10)
        last_bar = date(2026, 6, 9)

        with (
            self._stub_today(today),
            patch.object(refresh_data, "get_last_bar_date", return_value=last_bar),
            patch.object(refresh_data, "_last_complete_trading_day", return_value=last_bar),
            patch.object(refresh_data, "run_build_steps") as mock_build,
            patch.object(refresh_data, "download_dbn") as mock_download,
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=True, full_rebuild=False)

        assert ok is True
        mock_build.assert_not_called()
        mock_download.assert_not_called()

    def test_no_bars_at_all_skips(self):
        """get_last_bar_date=None means no initial load — refresh must skip."""
        today = date(2026, 6, 10)

        with (
            self._stub_today(today),
            patch.object(refresh_data, "get_last_bar_date", return_value=None),
            patch.object(refresh_data, "run_build_steps") as mock_build,
        ):
            ok = refresh_data.refresh_instrument("MNQ", dry_run=False, full_rebuild=False)

        assert ok is False
        mock_build.assert_not_called()

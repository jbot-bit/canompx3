"""
Tests for calendar skip filters: NFP, OPEX, Friday detection + filter classes.
"""

from datetime import date

import pytest

from pipeline.calendar_filters import is_nfp_day, is_opex_day, is_friday, is_monday, is_tuesday, day_of_week
from trading_app.config import (
    CalendarSkipFilter, CompositeFilter, OrbSizeFilter, NoFilter,
)


# =============================================================================
# NFP: First Friday of month
# =============================================================================

class TestIsNfpDay:
    """Known NFP dates (first Friday of each month)."""

    @pytest.mark.parametrize("d", [
        date(2024, 1, 5),   # Jan 2024
        date(2024, 2, 2),   # Feb 2024
        date(2024, 3, 1),   # Mar 2024
        date(2024, 4, 5),   # Apr 2024
        date(2024, 5, 3),   # May 2024
        date(2024, 6, 7),   # Jun 2024
        date(2024, 7, 5),   # Jul 2024
        date(2024, 8, 2),   # Aug 2024
        date(2024, 9, 6),   # Sep 2024
        date(2024, 10, 4),  # Oct 2024
        date(2024, 11, 1),  # Nov 2024
        date(2024, 12, 6),  # Dec 2024
        date(2025, 1, 3),   # Jan 2025
        date(2025, 8, 1),   # Aug 2025 — first day of month IS a Friday
        date(2026, 2, 6),   # Feb 2026
    ])
    def test_known_nfp_dates(self, d):
        assert is_nfp_day(d), f"{d} should be NFP day"

    @pytest.mark.parametrize("d", [
        date(2024, 1, 12),  # second Friday
        date(2024, 1, 19),  # third Friday (OPEX)
        date(2024, 1, 4),   # Thursday before first Friday
        date(2024, 1, 8),   # Monday after first Friday
        date(2024, 3, 15),  # OPEX day, not NFP
    ])
    def test_non_nfp_dates(self, d):
        assert not is_nfp_day(d), f"{d} should NOT be NFP day"


# =============================================================================
# OPEX: Third Friday of month
# =============================================================================

class TestIsOpexDay:
    """Known OPEX dates (third Friday of each month)."""

    @pytest.mark.parametrize("d", [
        date(2024, 1, 19),  # Jan 2024
        date(2024, 2, 16),  # Feb 2024
        date(2024, 3, 15),  # Mar 2024
        date(2024, 4, 19),  # Apr 2024
        date(2024, 5, 17),  # May 2024
        date(2024, 6, 21),  # Jun 2024
        date(2024, 7, 19),  # Jul 2024
        date(2024, 8, 16),  # Aug 2024
        date(2024, 9, 20),  # Sep 2024
        date(2024, 10, 18), # Oct 2024
        date(2024, 11, 15), # Nov 2024
        date(2024, 12, 20), # Dec 2024
        date(2025, 1, 17),  # Jan 2025
    ])
    def test_known_opex_dates(self, d):
        assert is_opex_day(d), f"{d} should be OPEX day"

    @pytest.mark.parametrize("d", [
        date(2024, 1, 5),   # first Friday (NFP)
        date(2024, 1, 12),  # second Friday
        date(2024, 1, 26),  # fourth Friday
        date(2024, 1, 18),  # Thursday before OPEX
        date(2024, 3, 1),   # NFP day, not OPEX
    ])
    def test_non_opex_dates(self, d):
        assert not is_opex_day(d), f"{d} should NOT be OPEX day"

    def test_nfp_and_opex_never_overlap(self):
        """NFP (day 1-7) and OPEX (day 15-21) can never be the same date."""
        for year in range(2020, 2030):
            for month in range(1, 13):
                for day in range(1, 29):
                    try:
                        d = date(year, month, day)
                    except ValueError:
                        continue
                    assert not (is_nfp_day(d) and is_opex_day(d)), \
                        f"{d} flagged as both NFP and OPEX"


# =============================================================================
# Friday
# =============================================================================

class TestIsFriday:

    def test_friday(self):
        assert is_friday(date(2024, 1, 5))   # Friday

    def test_not_friday(self):
        assert not is_friday(date(2024, 1, 4))   # Thursday
        assert not is_friday(date(2024, 1, 6))   # Saturday
        assert not is_friday(date(2024, 1, 7))   # Sunday
        assert not is_friday(date(2024, 1, 8))   # Monday


# =============================================================================
# Day-of-week functions (Feb 2026 DOW research)
# =============================================================================

class TestIsMonday:

    def test_monday(self):
        assert is_monday(date(2024, 1, 8))   # Monday

    def test_not_monday(self):
        assert not is_monday(date(2024, 1, 9))   # Tuesday
        assert not is_monday(date(2024, 1, 5))   # Friday
        assert not is_monday(date(2024, 1, 7))   # Sunday


class TestIsTuesday:

    def test_tuesday(self):
        assert is_tuesday(date(2024, 1, 9))   # Tuesday

    def test_not_tuesday(self):
        assert not is_tuesday(date(2024, 1, 8))   # Monday
        assert not is_tuesday(date(2024, 1, 10))  # Wednesday
        assert not is_tuesday(date(2024, 1, 5))   # Friday


class TestDayOfWeek:

    def test_full_week_cycle(self):
        """Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6."""
        # 2024-01-08 is Monday
        assert day_of_week(date(2024, 1, 8)) == 0   # Mon
        assert day_of_week(date(2024, 1, 9)) == 1   # Tue
        assert day_of_week(date(2024, 1, 10)) == 2  # Wed
        assert day_of_week(date(2024, 1, 11)) == 3  # Thu
        assert day_of_week(date(2024, 1, 12)) == 4  # Fri
        assert day_of_week(date(2024, 1, 13)) == 5  # Sat
        assert day_of_week(date(2024, 1, 14)) == 6  # Sun

    def test_consistent_with_is_functions(self):
        """day_of_week must agree with is_monday/is_tuesday/is_friday."""
        for day_offset in range(7):
            d = date(2024, 1, 8 + day_offset)  # Mon Jan 8 through Sun Jan 14
            dow = day_of_week(d)
            assert is_monday(d) == (dow == 0)
            assert is_tuesday(d) == (dow == 1)
            assert is_friday(d) == (dow == 4)


# =============================================================================
# CalendarSkipFilter.matches_row
# =============================================================================

class TestCalendarSkipFilter:

    def _row(self, *, nfp=False, opex=False, friday=False):
        return {"is_nfp_day": nfp, "is_opex_day": opex, "is_friday": friday}

    def test_nfp_skip(self):
        f = CalendarSkipFilter(
            filter_type="test", description="test",
            skip_nfp=True, skip_opex=False,
        )
        assert not f.matches_row(self._row(nfp=True), "CME_REOPEN")
        assert f.matches_row(self._row(nfp=False), "CME_REOPEN")

    def test_opex_skip(self):
        f = CalendarSkipFilter(
            filter_type="test", description="test",
            skip_nfp=False, skip_opex=True,
        )
        assert not f.matches_row(self._row(opex=True), "CME_REOPEN")
        assert f.matches_row(self._row(opex=False), "CME_REOPEN")

    def test_friday_session_specific(self):
        """Friday skip only blocks the specified session."""
        f = CalendarSkipFilter(
            filter_type="test", description="test",
            skip_nfp=False, skip_opex=False, skip_friday_session="CME_REOPEN",
        )
        row = self._row(friday=True)
        # CME_REOPEN blocked on Friday
        assert not f.matches_row(row, "CME_REOPEN")
        # TOKYO_OPEN NOT blocked on Friday
        assert f.matches_row(row, "TOKYO_OPEN")
        # LONDON_METALS NOT blocked on Friday
        assert f.matches_row(row, "LONDON_METALS")

    def test_friday_skip_no_effect_on_non_friday(self):
        f = CalendarSkipFilter(
            filter_type="test", description="test",
            skip_nfp=False, skip_opex=False, skip_friday_session="CME_REOPEN",
        )
        row = self._row(friday=False)
        assert f.matches_row(row, "CME_REOPEN")

    def test_all_skips_combined(self):
        f = CalendarSkipFilter(
            filter_type="test", description="test",
            skip_nfp=True, skip_opex=True, skip_friday_session="CME_REOPEN",
        )
        # Normal day passes
        assert f.matches_row(self._row(), "CME_REOPEN")
        # NFP blocked
        assert not f.matches_row(self._row(nfp=True), "CME_REOPEN")
        # OPEX blocked
        assert not f.matches_row(self._row(opex=True), "LONDON_METALS")
        # Friday at CME_REOPEN blocked
        assert not f.matches_row(self._row(friday=True), "CME_REOPEN")
        # Friday at LONDON_METALS passes (only CME_REOPEN skipped)
        assert f.matches_row(self._row(friday=True), "LONDON_METALS")

    def test_no_skips_passes_everything(self):
        f = CalendarSkipFilter(
            filter_type="test", description="test",
            skip_nfp=False, skip_opex=False, skip_friday_session=None,
        )
        assert f.matches_row(self._row(nfp=True, opex=True, friday=True), "CME_REOPEN")

    def test_missing_flags_treated_as_false(self):
        """Rows without calendar flags should pass (fail-open for backwards compat)."""
        f = CalendarSkipFilter(
            filter_type="test", description="test",
            skip_nfp=True, skip_opex=True, skip_friday_session="CME_REOPEN",
        )
        assert f.matches_row({}, "CME_REOPEN")


# =============================================================================
# CompositeFilter
# =============================================================================

class TestCompositeFilter:

    def test_both_pass(self):
        base = OrbSizeFilter(
            filter_type="ORB_G4", description="G4", min_size=4.0,
        )
        overlay = CalendarSkipFilter(
            filter_type="CAL", description="cal",
            skip_nfp=True, skip_opex=True,
        )
        composite = CompositeFilter(
            filter_type="COMP", description="comp",
            base=base, overlay=overlay,
        )
        row = {"orb_CME_REOPEN_size": 5.0, "is_nfp_day": False, "is_opex_day": False}
        assert composite.matches_row(row, "CME_REOPEN")

    def test_base_fails(self):
        base = OrbSizeFilter(
            filter_type="ORB_G4", description="G4", min_size=4.0,
        )
        overlay = CalendarSkipFilter(
            filter_type="CAL", description="cal",
            skip_nfp=True, skip_opex=True,
        )
        composite = CompositeFilter(
            filter_type="COMP", description="comp",
            base=base, overlay=overlay,
        )
        row = {"orb_CME_REOPEN_size": 2.0, "is_nfp_day": False, "is_opex_day": False}
        assert not composite.matches_row(row, "CME_REOPEN")

    def test_overlay_fails(self):
        base = NoFilter()
        overlay = CalendarSkipFilter(
            filter_type="CAL", description="cal",
            skip_nfp=True, skip_opex=False,
        )
        composite = CompositeFilter(
            filter_type="COMP", description="comp",
            base=base, overlay=overlay,
        )
        row = {"is_nfp_day": True, "is_opex_day": False}
        assert not composite.matches_row(row, "CME_REOPEN")

    def test_both_fail(self):
        base = OrbSizeFilter(
            filter_type="ORB_G4", description="G4", min_size=4.0,
        )
        overlay = CalendarSkipFilter(
            filter_type="CAL", description="cal",
            skip_nfp=True, skip_opex=True,
        )
        composite = CompositeFilter(
            filter_type="COMP", description="comp",
            base=base, overlay=overlay,
        )
        row = {"orb_CME_REOPEN_size": 2.0, "is_nfp_day": True, "is_opex_day": False}
        assert not composite.matches_row(row, "CME_REOPEN")

"""Tests for pipeline/dst.py -- DST detection and dynamic session resolvers."""

from datetime import date

import pytest

from pipeline.dst import (
    is_us_dst,
    is_uk_dst,
    us_equity_open_brisbane,
    us_data_open_brisbane,
    london_open_brisbane,
    DYNAMIC_ORB_RESOLVERS,
)


# =========================================================================
# US DST transitions (second Sunday Mar, first Sunday Nov)
# =========================================================================

class TestUsDst:
    """US Eastern DST detection."""

    # 2024: DST starts Mar 10, ends Nov 3
    def test_2024_winter_before_transition(self):
        assert is_us_dst(date(2024, 3, 9)) is False

    def test_2024_summer_after_spring_forward(self):
        assert is_us_dst(date(2024, 3, 10)) is True

    def test_2024_summer_midyear(self):
        assert is_us_dst(date(2024, 7, 15)) is True

    def test_2024_summer_before_fall_back(self):
        assert is_us_dst(date(2024, 11, 2)) is True

    def test_2024_winter_after_fall_back(self):
        assert is_us_dst(date(2024, 11, 3)) is False

    def test_2024_deep_winter(self):
        assert is_us_dst(date(2024, 1, 15)) is False

    # 2025: DST starts Mar 9, ends Nov 2
    def test_2025_winter_before_transition(self):
        assert is_us_dst(date(2025, 3, 8)) is False

    def test_2025_summer_after_spring_forward(self):
        assert is_us_dst(date(2025, 3, 9)) is True

    def test_2025_winter_after_fall_back(self):
        assert is_us_dst(date(2025, 11, 2)) is False

    # 2026
    def test_2026_summer(self):
        assert is_us_dst(date(2026, 6, 1)) is True

    def test_2026_winter(self):
        assert is_us_dst(date(2026, 12, 1)) is False


# =========================================================================
# UK DST transitions (last Sunday Mar, last Sunday Oct)
# =========================================================================

class TestUkDst:
    """UK British Summer Time detection."""

    # 2024: BST starts Mar 31, ends Oct 27
    def test_2024_winter_before_transition(self):
        assert is_uk_dst(date(2024, 3, 30)) is False

    def test_2024_summer_after_spring_forward(self):
        assert is_uk_dst(date(2024, 3, 31)) is True

    def test_2024_summer_midyear(self):
        assert is_uk_dst(date(2024, 7, 15)) is True

    def test_2024_summer_before_fall_back(self):
        assert is_uk_dst(date(2024, 10, 26)) is True

    def test_2024_winter_after_fall_back(self):
        assert is_uk_dst(date(2024, 10, 27)) is False

    def test_2024_deep_winter(self):
        assert is_uk_dst(date(2024, 1, 15)) is False

    # 2025: BST starts Mar 30, ends Oct 26
    def test_2025_winter_before_transition(self):
        assert is_uk_dst(date(2025, 3, 29)) is False

    def test_2025_summer_after_spring_forward(self):
        assert is_uk_dst(date(2025, 3, 30)) is True

    def test_2025_winter_after_fall_back(self):
        assert is_uk_dst(date(2025, 10, 26)) is False


# =========================================================================
# US_EQUITY_OPEN resolver (NYSE 09:30 ET)
# =========================================================================

class TestUsEquityOpenBrisbane:
    """NYSE 09:30 ET -> Brisbane local time."""

    def test_summer_edt(self):
        # EDT: 09:30 ET = 13:30 UTC = 23:30 AEST
        h, m = us_equity_open_brisbane(date(2024, 7, 15))
        assert (h, m) == (23, 30)

    def test_winter_est(self):
        # EST: 09:30 ET = 14:30 UTC = 00:30 AEST (next day)
        h, m = us_equity_open_brisbane(date(2024, 1, 15))
        assert (h, m) == (0, 30)

    def test_transition_day_spring_2024(self):
        # Mar 10 2024 is DST start -- 09:30 EDT
        h, m = us_equity_open_brisbane(date(2024, 3, 10))
        assert (h, m) == (23, 30)

    def test_transition_day_fall_2024(self):
        # Nov 3 2024 is DST end -- 09:30 EST
        h, m = us_equity_open_brisbane(date(2024, 11, 3))
        assert (h, m) == (0, 30)


# =========================================================================
# US_DATA_OPEN resolver (Econ data 08:30 ET)
# =========================================================================

class TestUsDataOpenBrisbane:
    """US econ data 08:30 ET -> Brisbane local time."""

    def test_summer_edt(self):
        # EDT: 08:30 ET = 12:30 UTC = 22:30 AEST
        h, m = us_data_open_brisbane(date(2024, 7, 15))
        assert (h, m) == (22, 30)

    def test_winter_est(self):
        # EST: 08:30 ET = 13:30 UTC = 23:30 AEST
        h, m = us_data_open_brisbane(date(2024, 1, 15))
        assert (h, m) == (23, 30)

    def test_transition_day_spring_2025(self):
        # Mar 9 2025 is DST start -- 08:30 EDT
        h, m = us_data_open_brisbane(date(2025, 3, 9))
        assert (h, m) == (22, 30)

    def test_transition_day_fall_2025(self):
        # Nov 2 2025 is DST end -- 08:30 EST
        h, m = us_data_open_brisbane(date(2025, 11, 2))
        assert (h, m) == (23, 30)


# =========================================================================
# LONDON_OPEN resolver (08:00 London)
# =========================================================================

class TestLondonOpenBrisbane:
    """London metals 08:00 LT -> Brisbane local time."""

    def test_summer_bst(self):
        # BST: 08:00 London = 07:00 UTC = 17:00 AEST
        h, m = london_open_brisbane(date(2024, 7, 15))
        assert (h, m) == (17, 0)

    def test_winter_gmt(self):
        # GMT: 08:00 London = 08:00 UTC = 18:00 AEST
        h, m = london_open_brisbane(date(2024, 1, 15))
        assert (h, m) == (18, 0)

    def test_transition_day_spring_2024(self):
        # Mar 31 2024 is BST start -- 08:00 BST
        h, m = london_open_brisbane(date(2024, 3, 31))
        assert (h, m) == (17, 0)

    def test_transition_day_fall_2024(self):
        # Oct 27 2024 is BST end -- 08:00 GMT
        h, m = london_open_brisbane(date(2024, 10, 27))
        assert (h, m) == (18, 0)

    def test_transition_day_spring_2025(self):
        # Mar 30 2025 is BST start -- 08:00 BST
        h, m = london_open_brisbane(date(2025, 3, 30))
        assert (h, m) == (17, 0)


# =========================================================================
# Registry
# =========================================================================

class TestDynamicOrbResolvers:
    """DYNAMIC_ORB_RESOLVERS registry completeness."""

    def test_has_all_three_sessions(self):
        assert set(DYNAMIC_ORB_RESOLVERS.keys()) == {
            "US_EQUITY_OPEN", "US_DATA_OPEN", "LONDON_OPEN",
        }

    def test_resolvers_return_tuples(self):
        td = date(2024, 6, 15)
        for name, resolver in DYNAMIC_ORB_RESOLVERS.items():
            result = resolver(td)
            assert isinstance(result, tuple), f"{name} should return tuple"
            assert len(result) == 2, f"{name} should return (hour, minute)"
            assert 0 <= result[0] <= 23, f"{name} hour out of range"
            assert 0 <= result[1] <= 59, f"{name} minute out of range"


# =========================================================================
# Consistency: same resolver gives different results summer vs winter
# =========================================================================

class TestSeasonalShift:
    """Verify that dynamic sessions actually shift between summer and winter."""

    def test_us_equity_shifts_by_1h(self):
        summer = us_equity_open_brisbane(date(2024, 7, 15))
        winter = us_equity_open_brisbane(date(2024, 1, 15))
        # Summer = 23:30, winter = 00:30 -- 1 hour difference
        assert summer != winter

    def test_us_data_shifts_by_1h(self):
        summer = us_data_open_brisbane(date(2024, 7, 15))
        winter = us_data_open_brisbane(date(2024, 1, 15))
        # Summer = 22:30, winter = 23:30 -- 1 hour difference
        assert summer != winter

    def test_london_shifts_by_1h(self):
        summer = london_open_brisbane(date(2024, 7, 15))
        winter = london_open_brisbane(date(2024, 1, 15))
        # Summer = 17:00, winter = 18:00 -- 1 hour difference
        assert summer != winter

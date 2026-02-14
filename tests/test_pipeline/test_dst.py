"""Tests for pipeline/dst.py -- DST detection and dynamic session resolvers."""

from datetime import date

import pytest

from pipeline.dst import (
    is_us_dst,
    is_uk_dst,
    cme_open_brisbane,
    us_equity_open_brisbane,
    us_data_open_brisbane,
    london_open_brisbane,
    DYNAMIC_ORB_RESOLVERS,
    SESSION_CATALOG,
    validate_catalog,
    get_break_group,
)
from pipeline.init_db import ORB_LABELS


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
# CME_OPEN resolver (CME Globex 5:00 PM CT)
# =========================================================================

class TestCmeOpenBrisbane:
    """CME Globex 5:00 PM CT -> Brisbane local time."""

    def test_winter_cst(self):
        # CST: 5PM CT = 23:00 UTC = 09:00 AEST
        h, m = cme_open_brisbane(date(2025, 1, 15))
        assert (h, m) == (9, 0)

    def test_summer_cdt(self):
        # CDT: 5PM CT = 22:00 UTC = 08:00 AEST
        h, m = cme_open_brisbane(date(2025, 7, 15))
        assert (h, m) == (8, 0)

    def test_transition_day_spring_2025(self):
        # Mar 9 2025 is US DST start -- CDT kicks in
        h, m = cme_open_brisbane(date(2025, 3, 9))
        assert (h, m) == (8, 0)

    def test_transition_day_fall_2025(self):
        # Nov 2 2025 is US DST end -- CST kicks in
        h, m = cme_open_brisbane(date(2025, 11, 2))
        assert (h, m) == (9, 0)

    def test_shifts_by_1h(self):
        """CME_OPEN shifts 1 hour between summer and winter."""
        summer = cme_open_brisbane(date(2025, 7, 15))
        winter = cme_open_brisbane(date(2025, 1, 15))
        assert summer != winter


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

    def test_has_all_four_sessions(self):
        assert set(DYNAMIC_ORB_RESOLVERS.keys()) == {
            "CME_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN", "LONDON_OPEN",
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
# SESSION_CATALOG
# =========================================================================

class TestSessionCatalog:
    """SESSION_CATALOG master registry validation."""

    def test_all_non_alias_have_required_keys(self):
        for label, entry in SESSION_CATALOG.items():
            assert "type" in entry, f"{label} missing 'type'"
            assert "event" in entry, f"{label} missing 'event'"
            if entry["type"] == "dynamic":
                assert "resolver" in entry, f"{label} missing 'resolver'"
                assert callable(entry["resolver"]), f"{label} resolver not callable"
            elif entry["type"] == "fixed":
                assert "brisbane" in entry, f"{label} missing 'brisbane'"
                h, m = entry["brisbane"]
                assert 0 <= h <= 23 and 0 <= m <= 59, f"{label} invalid brisbane time"
            elif entry["type"] == "alias":
                assert "maps_to" in entry, f"{label} missing 'maps_to'"

    def test_aliases_map_to_valid_orb_labels(self):
        for label, entry in SESSION_CATALOG.items():
            if entry["type"] == "alias":
                assert entry["maps_to"] in ORB_LABELS, (
                    f"Alias {label} maps to {entry['maps_to']} which is not in ORB_LABELS"
                )

    def test_no_alias_in_orb_labels(self):
        aliases = {
            label for label, entry in SESSION_CATALOG.items()
            if entry["type"] == "alias"
        }
        for alias in aliases:
            assert alias not in ORB_LABELS, (
                f"Alias {alias} should NOT appear in ORB_LABELS"
            )

    def test_validate_catalog_no_collisions(self):
        validate_catalog()

    def test_dynamic_resolvers_built_from_catalog(self):
        dynamic_in_catalog = {
            label for label, entry in SESSION_CATALOG.items()
            if entry["type"] == "dynamic"
        }
        assert set(DYNAMIC_ORB_RESOLVERS.keys()) == dynamic_in_catalog

    def test_all_non_alias_non_dynamic_are_fixed(self):
        for label, entry in SESSION_CATALOG.items():
            assert entry["type"] in ("dynamic", "fixed", "alias"), (
                f"{label} has unknown type {entry['type']}"
            )

    def test_all_non_alias_have_break_group(self):
        for label, entry in SESSION_CATALOG.items():
            if entry["type"] == "alias":
                continue
            assert "break_group" in entry, f"{label} missing 'break_group'"
            assert isinstance(entry["break_group"], str), f"{label} break_group not str"

    def test_break_groups_same_group_shares_boundary(self):
        """1000, 1100, 1130 are all 'asia' -- same group."""
        assert get_break_group("1000") == "asia"
        assert get_break_group("1100") == "asia"
        assert get_break_group("1130") == "asia"

    def test_break_groups_different_groups(self):
        """0900/CME_OPEN are 'cme', 1800 is 'london', 2300/0030 are 'us'."""
        assert get_break_group("0900") == "cme"
        assert get_break_group("CME_OPEN") == "cme"
        assert get_break_group("1800") == "london"
        assert get_break_group("LONDON_OPEN") == "london"
        assert get_break_group("2300") == "us"
        assert get_break_group("0030") == "us"

    def test_get_break_group_alias_returns_none(self):
        assert get_break_group("TOKYO_OPEN") is None

    def test_get_break_group_unknown_returns_none(self):
        assert get_break_group("NONEXISTENT") is None


# =========================================================================
# Break window grouping: verify 1100 is NOT truncated by 1130
# =========================================================================

class TestBreakWindowGrouping:
    """Break detection windows use group boundaries, not next label."""

    def test_1100_break_window_extends_past_1130(self):
        """1100 and 1130 are in the same 'asia' group.
        1100's break window should extend to the next DIFFERENT group,
        NOT stop at 1130.

        On June 15 (summer), LONDON_OPEN (london group) resolves to
        17:00 Brisbane = 07:00 UTC, which is BEFORE the fixed 1800
        (08:00 UTC). So the boundary is 07:00 UTC.
        """
        from pipeline.build_daily_features import _break_detection_window
        td = date(2024, 6, 15)  # summer weekday

        _, window_end_1100 = _break_detection_window(td, "1100", 5)
        _, window_end_1130 = _break_detection_window(td, "1130", 5)

        # Both should end at the same boundary (next group = london)
        # In summer: LONDON_OPEN = 17:00 Brisbane = 07:00 UTC (earliest london group)
        assert window_end_1100.hour == 7 and window_end_1100.minute == 0
        assert window_end_1130.hour == 7 and window_end_1130.minute == 0

        # 1100's window should be much longer than 25 minutes
        orb_end_1100_start, _ = _break_detection_window(td, "1100", 5)
        duration_minutes = (window_end_1100 - orb_end_1100_start).total_seconds() / 60
        assert duration_minutes > 60, (
            f"1100 break window is only {duration_minutes:.0f} min, should be ~6 hours"
        )

    def test_2300_and_0030_share_us_boundary(self):
        """2300 and 0030 are both 'us' group.
        Both should extend to next day's 0900 (cme group).
        """
        from pipeline.build_daily_features import _break_detection_window
        td = date(2024, 6, 15)

        _, window_end_2300 = _break_detection_window(td, "2300", 5)
        _, window_end_0030 = _break_detection_window(td, "0030", 5)

        # Both should end at next day 0900 Brisbane = 23:00 UTC
        # which is the end of the trading day
        assert window_end_2300 == window_end_0030


# =========================================================================
# Consistency: same resolver gives different results summer vs winter
# =========================================================================

class TestSeasonalShift:
    """Verify that dynamic sessions actually shift between summer and winter."""

    def test_cme_open_shifts_by_1h(self):
        summer = cme_open_brisbane(date(2025, 7, 15))
        winter = cme_open_brisbane(date(2025, 1, 15))
        assert summer != winter

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

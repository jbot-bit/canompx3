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
    us_post_equity_brisbane,
    cme_close_brisbane,
    comex_settle_brisbane,
    nyse_close_brisbane,
    tokyo_open_brisbane,
    singapore_open_brisbane,
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
# CME_REOPEN resolver (CME Globex 5:00 PM CT)
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
        """CME_REOPEN shifts 1 hour between summer and winter."""
        summer = cme_open_brisbane(date(2025, 7, 15))
        winter = cme_open_brisbane(date(2025, 1, 15))
        assert summer != winter


# =========================================================================
# TOKYO_OPEN resolver (TSE 9:00 AM JST — fixed)
# =========================================================================

class TestTokyoOpenBrisbane:
    """Tokyo Stock Exchange 9:00 AM JST -> Brisbane local time (always 10:00)."""

    def test_winter(self):
        # JST = UTC+9, Brisbane = UTC+10. 9:00 JST = 10:00 Brisbane.
        h, m = tokyo_open_brisbane(date(2025, 1, 15))
        assert (h, m) == (10, 0)

    def test_summer(self):
        # No DST in Japan or Brisbane. Always 10:00.
        h, m = tokyo_open_brisbane(date(2025, 7, 15))
        assert (h, m) == (10, 0)

    def test_no_seasonal_shift(self):
        """TOKYO_OPEN does NOT shift between summer and winter."""
        summer = tokyo_open_brisbane(date(2025, 7, 15))
        winter = tokyo_open_brisbane(date(2025, 1, 15))
        assert summer == winter


# =========================================================================
# SINGAPORE_OPEN resolver (SGX 9:00 AM SGT — fixed)
# =========================================================================

class TestSingaporeOpenBrisbane:
    """SGX/HKEX 9:00 AM SGT -> Brisbane local time (always 11:00)."""

    def test_winter(self):
        # SGT = UTC+8, Brisbane = UTC+10. 9:00 SGT = 11:00 Brisbane.
        h, m = singapore_open_brisbane(date(2025, 1, 15))
        assert (h, m) == (11, 0)

    def test_summer(self):
        # No DST in Singapore or Brisbane. Always 11:00.
        h, m = singapore_open_brisbane(date(2025, 7, 15))
        assert (h, m) == (11, 0)

    def test_no_seasonal_shift(self):
        """SINGAPORE_OPEN does NOT shift between summer and winter."""
        summer = singapore_open_brisbane(date(2025, 7, 15))
        winter = singapore_open_brisbane(date(2025, 1, 15))
        assert summer == winter


# =========================================================================
# NYSE_OPEN resolver (NYSE 09:30 ET)
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
# US_DATA_830 resolver (Econ data 08:30 ET)
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
# LONDON_METALS resolver (08:00 London)
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

    def test_has_all_dynamic_sessions(self):
        assert set(DYNAMIC_ORB_RESOLVERS.keys()) == {
            "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
            "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
            "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
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

    def test_all_entries_have_required_keys(self):
        for label, entry in SESSION_CATALOG.items():
            assert "type" in entry, f"{label} missing 'type'"
            assert "event" in entry, f"{label} missing 'event'"
            if entry["type"] == "dynamic":
                assert "resolver" in entry, f"{label} missing 'resolver'"
                assert callable(entry["resolver"]), f"{label} resolver not callable"

    def test_validate_catalog_no_collisions(self):
        validate_catalog()

    def test_dynamic_resolvers_built_from_catalog(self):
        dynamic_in_catalog = {
            label for label, entry in SESSION_CATALOG.items()
            if entry["type"] == "dynamic"
        }
        assert set(DYNAMIC_ORB_RESOLVERS.keys()) == dynamic_in_catalog

    def test_all_entries_are_dynamic(self):
        """After event-based rename, all sessions are dynamic (no fixed/alias)."""
        for label, entry in SESSION_CATALOG.items():
            assert entry["type"] == "dynamic", (
                f"{label} has type {entry['type']}, expected 'dynamic'"
            )

    def test_all_have_break_group(self):
        for label, entry in SESSION_CATALOG.items():
            assert "break_group" in entry, f"{label} missing 'break_group'"
            assert isinstance(entry["break_group"], str), f"{label} break_group not str"

    def test_break_groups_asia(self):
        """TOKYO_OPEN and SINGAPORE_OPEN are both 'asia' group."""
        assert get_break_group("TOKYO_OPEN") == "asia"
        assert get_break_group("SINGAPORE_OPEN") == "asia"

    def test_break_groups_cme(self):
        """CME_REOPEN is 'cme' group."""
        assert get_break_group("CME_REOPEN") == "cme"

    def test_break_groups_london(self):
        """LONDON_METALS is 'london' group."""
        assert get_break_group("LONDON_METALS") == "london"

    def test_break_groups_us(self):
        """US sessions are all 'us' group."""
        assert get_break_group("US_DATA_830") == "us"
        assert get_break_group("NYSE_OPEN") == "us"
        assert get_break_group("US_DATA_1000") == "us"
        assert get_break_group("COMEX_SETTLE") == "us"
        assert get_break_group("CME_PRECLOSE") == "us"
        assert get_break_group("NYSE_CLOSE") == "us"

    def test_get_break_group_unknown_returns_none(self):
        assert get_break_group("NONEXISTENT") is None

    def test_catalog_has_exactly_10_sessions(self):
        assert len(SESSION_CATALOG) == 10

    def test_dst_sets_cover_all_sessions(self):
        """Every session must be in either DST_AFFECTED or DST_CLEAN."""
        from pipeline.dst import DST_AFFECTED_SESSIONS, DST_CLEAN_SESSIONS
        all_sessions = set(SESSION_CATALOG.keys())
        covered = set(DST_AFFECTED_SESSIONS.keys()) | DST_CLEAN_SESSIONS
        assert covered == all_sessions, f"Missing from DST sets: {all_sessions - covered}, Extra: {covered - all_sessions}"


# =========================================================================
# Break window grouping: verify asia sessions share boundary
# =========================================================================

class TestBreakWindowGrouping:
    """Break detection windows use group boundaries, not next label."""

    def test_asia_sessions_share_break_boundary(self):
        """TOKYO_OPEN and SINGAPORE_OPEN are both 'asia' group.
        Both should extend to the next DIFFERENT group (london).

        On June 15 (summer), LONDON_METALS (london group) resolves to
        17:00 Brisbane = 07:00 UTC.
        """
        from pipeline.build_daily_features import _break_detection_window
        td = date(2024, 6, 15)  # summer weekday

        _, window_end_tokyo = _break_detection_window(td, "TOKYO_OPEN", 5)
        _, window_end_singapore = _break_detection_window(td, "SINGAPORE_OPEN", 5)

        # Both should end at the same boundary (next group = london)
        # In summer: LONDON_METALS = 17:00 Brisbane = 07:00 UTC
        assert window_end_tokyo.hour == 7 and window_end_tokyo.minute == 0
        assert window_end_singapore.hour == 7 and window_end_singapore.minute == 0

        # TOKYO_OPEN's window should be much longer than 25 minutes
        orb_end_tokyo_start, _ = _break_detection_window(td, "TOKYO_OPEN", 5)
        duration_minutes = (window_end_tokyo - orb_end_tokyo_start).total_seconds() / 60
        assert duration_minutes > 60, (
            f"TOKYO_OPEN break window is only {duration_minutes:.0f} min, should be ~6 hours"
        )

    def test_us_sessions_share_boundary(self):
        """US_DATA_830 and NYSE_OPEN are both 'us' group.
        Both should extend to end of trading day (next cme group).
        """
        from pipeline.build_daily_features import _break_detection_window
        td = date(2024, 6, 15)

        _, window_end_data = _break_detection_window(td, "US_DATA_830", 5)
        _, window_end_nyse = _break_detection_window(td, "NYSE_OPEN", 5)

        # Both should end at end of trading day
        assert window_end_data == window_end_nyse


# =========================================================================
# US_DATA_1000 resolver (10:00 AM ET, ~30min after NYSE cash open)
# =========================================================================

class TestUsPostEquityBrisbane:
    """US post-equity-open 10:00 AM ET -> Brisbane local time."""

    def test_winter_est(self):
        # EST: 10:00 ET = 15:00 UTC = 01:00 AEST (next cal day)
        h, m = us_post_equity_brisbane(date(2025, 1, 15))
        assert (h, m) == (1, 0)

    def test_summer_edt(self):
        # EDT: 10:00 ET = 14:00 UTC = 00:00 AEST (next cal day)
        h, m = us_post_equity_brisbane(date(2025, 7, 15))
        assert (h, m) == (0, 0)

    def test_transition_day_spring_2025(self):
        # Mar 9 2025 is US DST start -- 10:00 EDT
        h, m = us_post_equity_brisbane(date(2025, 3, 9))
        assert (h, m) == (0, 0)

    def test_transition_day_fall_2025(self):
        # Nov 2 2025 is US DST end -- 10:00 EST
        h, m = us_post_equity_brisbane(date(2025, 11, 2))
        assert (h, m) == (1, 0)

    def test_shifts_by_1h(self):
        """US_DATA_1000 shifts 1 hour between summer and winter."""
        summer = us_post_equity_brisbane(date(2025, 7, 15))
        winter = us_post_equity_brisbane(date(2025, 1, 15))
        assert summer != winter

    def test_break_group_is_us(self):
        assert get_break_group("US_DATA_1000") == "us"


# =========================================================================
# CME_PRECLOSE resolver (2:45 PM CT, CME equity futures pre-close)
# =========================================================================

class TestCmeCloseBrisbane:
    """CME equity futures pre-close 2:45 PM CT -> Brisbane local time."""

    def test_winter_cst(self):
        # CST: 2:45 PM CT = 20:45 UTC = 06:45 AEST
        h, m = cme_close_brisbane(date(2025, 1, 15))
        assert (h, m) == (6, 45)

    def test_summer_cdt(self):
        # CDT: 2:45 PM CT = 19:45 UTC = 05:45 AEST
        h, m = cme_close_brisbane(date(2025, 7, 15))
        assert (h, m) == (5, 45)

    def test_transition_day_spring_2025(self):
        # Mar 9 2025 is US DST start -- 2:45 PM CDT
        h, m = cme_close_brisbane(date(2025, 3, 9))
        assert (h, m) == (5, 45)

    def test_transition_day_fall_2025(self):
        # Nov 2 2025 is US DST end -- 2:45 PM CST
        h, m = cme_close_brisbane(date(2025, 11, 2))
        assert (h, m) == (6, 45)

    def test_shifts_by_1h(self):
        """CME_PRECLOSE shifts 1 hour between summer and winter."""
        summer = cme_close_brisbane(date(2025, 7, 15))
        winter = cme_close_brisbane(date(2025, 1, 15))
        assert summer != winter

    def test_break_group_is_us(self):
        assert get_break_group("CME_PRECLOSE") == "us"


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

    def test_tokyo_no_shift(self):
        """Tokyo does NOT shift (no DST in Japan or Brisbane)."""
        summer = tokyo_open_brisbane(date(2025, 7, 15))
        winter = tokyo_open_brisbane(date(2025, 1, 15))
        assert summer == winter

    def test_singapore_no_shift(self):
        """Singapore does NOT shift (no DST in Singapore or Brisbane)."""
        summer = singapore_open_brisbane(date(2025, 7, 15))
        winter = singapore_open_brisbane(date(2025, 1, 15))
        assert summer == winter


# =========================================================================
# COMEX_SETTLE resolver (COMEX gold settlement 1:30 PM ET)
# =========================================================================

class TestComexSettleBrisbane:
    """COMEX gold settlement at 1:30 PM ET."""

    def test_winter_est(self):
        # 1:30 PM EST = 18:30 UTC = 04:30 Brisbane (next day)
        h, m = comex_settle_brisbane(date(2026, 1, 15))
        assert (h, m) == (4, 30)

    def test_summer_edt(self):
        # 1:30 PM EDT = 17:30 UTC = 03:30 Brisbane (next day)
        h, m = comex_settle_brisbane(date(2025, 7, 15))
        assert (h, m) == (3, 30)

    def test_spring_transition(self):
        # Mar 8 2026 is US spring-forward
        h, m = comex_settle_brisbane(date(2026, 3, 7))  # still EST
        assert (h, m) == (4, 30)
        h, m = comex_settle_brisbane(date(2026, 3, 9))  # now EDT
        assert (h, m) == (3, 30)

    def test_shifts_by_1h(self):
        """COMEX_SETTLE shifts 1 hour between summer and winter."""
        summer = comex_settle_brisbane(date(2025, 7, 15))
        winter = comex_settle_brisbane(date(2025, 1, 15))
        assert summer != winter


# =========================================================================
# NYSE_CLOSE resolver (NYSE closing bell 4:00 PM ET)
# =========================================================================

class TestNyseCloseBrisbane:
    """NYSE closing bell at 4:00 PM ET."""

    def test_winter_est(self):
        # 4:00 PM EST = 21:00 UTC = 07:00 Brisbane (next day)
        h, m = nyse_close_brisbane(date(2026, 1, 15))
        assert (h, m) == (7, 0)

    def test_summer_edt(self):
        # 4:00 PM EDT = 20:00 UTC = 06:00 Brisbane (next day)
        h, m = nyse_close_brisbane(date(2025, 7, 15))
        assert (h, m) == (6, 0)

    def test_spring_transition(self):
        h, m = nyse_close_brisbane(date(2026, 3, 7))  # EST
        assert (h, m) == (7, 0)
        h, m = nyse_close_brisbane(date(2026, 3, 9))  # EDT
        assert (h, m) == (6, 0)

    def test_shifts_by_1h(self):
        """NYSE_CLOSE shifts 1 hour between summer and winter."""
        summer = nyse_close_brisbane(date(2025, 7, 15))
        winter = nyse_close_brisbane(date(2025, 1, 15))
        assert summer != winter

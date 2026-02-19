"""
Tests for trading_app.config filter definitions.
"""

import json
import pytest

from trading_app.config import (
    StrategyFilter,
    NoFilter,
    OrbSizeFilter,
    VolumeFilter,
    DayOfWeekSkipFilter,
    CompositeFilter,
    ALL_FILTERS,
    MGC_ORB_SIZE_FILTERS,
    MGC_VOLUME_FILTERS,
    get_filters_for_grid,
)


class TestNoFilter:
    """NoFilter always matches."""

    def test_matches_any_row(self):
        f = NoFilter()
        assert f.matches_row({}, "0900") is True

    def test_matches_with_data(self):
        f = NoFilter()
        row = {"orb_0900_size": 3.5, "rsi_14_at_0900": 45.0}
        assert f.matches_row(row, "0900") is True

    def test_to_json_roundtrip(self):
        f = NoFilter()
        data = json.loads(f.to_json())
        assert data["filter_type"] == "NO_FILTER"

    def test_frozen(self):
        f = NoFilter()
        with pytest.raises(AttributeError):
            f.filter_type = "CHANGED"

    def test_hashable(self):
        f1 = NoFilter()
        f2 = NoFilter()
        assert hash(f1) == hash(f2)
        assert {f1, f2} == {f1}


class TestOrbSizeFilter:
    """OrbSizeFilter matches by ORB size ranges."""

    def test_max_size_under(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        assert f.matches_row({"orb_0900_size": 3.5}, "0900") is True

    def test_max_size_at_boundary(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        # max_size uses < (exclusive), so 4.0 should NOT match
        assert f.matches_row({"orb_0900_size": 4.0}, "0900") is False

    def test_min_size_above(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", min_size=4.0)
        assert f.matches_row({"orb_0900_size": 5.0}, "0900") is True

    def test_min_size_at_boundary(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", min_size=4.0)
        # min_size uses >= (inclusive at boundary check is <), so 4.0 matches
        assert f.matches_row({"orb_0900_size": 4.0}, "0900") is True

    def test_null_size_no_match(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        assert f.matches_row({"orb_0900_size": None}, "0900") is False

    def test_missing_size_no_match(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        assert f.matches_row({}, "0900") is False

    def test_uses_correct_orb_label(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        row = {"orb_0900_size": 5.0, "orb_1100_size": 3.0}
        assert f.matches_row(row, "0900") is False
        assert f.matches_row(row, "1100") is True

    def test_to_json_roundtrip(self):
        f = MGC_ORB_SIZE_FILTERS["L4"]
        data = json.loads(f.to_json())
        assert data["max_size"] == 4.0

    def test_g_filters_have_min_size(self):
        """All G filters have min_size set."""
        for key in ["G2", "G3", "G4", "G5", "G6", "G8"]:
            f = MGC_ORB_SIZE_FILTERS[key]
            assert f.min_size is not None
            assert f.max_size is None

    def test_l_filters_have_max_size(self):
        """All L filters have max_size set."""
        for key in ["L2", "L3", "L4", "L6", "L8"]:
            f = MGC_ORB_SIZE_FILTERS[key]
            assert f.max_size is not None
            assert f.min_size is None

    def test_g_thresholds_ascending(self):
        """G filter thresholds are strictly ascending."""
        thresholds = [MGC_ORB_SIZE_FILTERS[k].min_size for k in ["G2", "G3", "G4", "G5", "G6", "G8"]]
        assert thresholds == sorted(thresholds)
        assert len(set(thresholds)) == len(thresholds)

    def test_l_thresholds_ascending(self):
        """L filter thresholds are strictly ascending."""
        thresholds = [MGC_ORB_SIZE_FILTERS[k].max_size for k in ["L2", "L3", "L4", "L6", "L8"]]
        assert thresholds == sorted(thresholds)
        assert len(set(thresholds)) == len(thresholds)


class TestAllFilters:
    """ALL_FILTERS registry is complete and consistent."""

    def test_contains_no_filter(self):
        assert "NO_FILTER" in ALL_FILTERS

    def test_contains_grid_size_filters(self):
        for key in ["ORB_G4", "ORB_G5", "ORB_G6", "ORB_G8"]:
            assert key in ALL_FILTERS, f"{key} missing from ALL_FILTERS"

    def test_l_filters_excluded_from_grid(self):
        for key in ["ORB_L2", "ORB_L3", "ORB_L4", "ORB_L6", "ORB_L8"]:
            assert key not in ALL_FILTERS, f"{key} should not be in ALL_FILTERS"

    def test_g2_g3_excluded_from_grid(self):
        for key in ["ORB_G2", "ORB_G3"]:
            assert key not in ALL_FILTERS, f"{key} should not be in ALL_FILTERS"

    def test_total_count(self):
        # NO_FILTER + 4 G-filters (G4,G5,G6,G8) + 1 VOL-filter = 6
        assert len(ALL_FILTERS) == 6

    def test_contains_volume_filter(self):
        assert "VOL_RV12_N20" in ALL_FILTERS

    def test_all_are_strategy_filters(self):
        for name, f in ALL_FILTERS.items():
            assert isinstance(f, StrategyFilter), f"{name} is not a StrategyFilter"

    def test_all_have_filter_type(self):
        for name, f in ALL_FILTERS.items():
            assert f.filter_type, f"{name} has empty filter_type"

    def test_all_serializable(self):
        for name, f in ALL_FILTERS.items():
            data = json.loads(f.to_json())
            assert "filter_type" in data, f"{name} JSON missing filter_type"


class TestVolumeFilter:
    """VolumeFilter matches by relative volume at break bar."""

    def test_matches_above_threshold(self):
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({"rel_vol_0900": 1.5}, "0900") is True

    def test_rejects_below_threshold(self):
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({"rel_vol_0900": 0.8}, "0900") is False

    def test_at_boundary_matches(self):
        """Exactly at min_rel_vol should match (>=)."""
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({"rel_vol_0900": 1.2}, "0900") is True

    def test_fail_closed_missing(self):
        """Missing rel_vol key -> ineligible (fail-closed)."""
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({}, "0900") is False

    def test_fail_closed_none(self):
        """None rel_vol -> ineligible (fail-closed)."""
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({"rel_vol_0900": None}, "0900") is False

    def test_uses_correct_orb_label(self):
        """Uses rel_vol for the specific ORB label."""
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        row = {"rel_vol_0900": 0.5, "rel_vol_1800": 2.0}
        assert f.matches_row(row, "0900") is False
        assert f.matches_row(row, "1800") is True

    def test_frozen(self):
        f = VolumeFilter(filter_type="TEST", description="test")
        with pytest.raises(AttributeError):
            f.min_rel_vol = 2.0

    def test_hashable(self):
        f1 = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        f2 = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert hash(f1) == hash(f2)

    def test_to_json_roundtrip(self):
        f = MGC_VOLUME_FILTERS["VOL_RV12_N20"]
        data = json.loads(f.to_json())
        assert data["filter_type"] == "VOL_RV12_N20"
        assert data["min_rel_vol"] == 1.2
        assert data["lookback_days"] == 20

    def test_predefined_vol_rv12_n20(self):
        """VOL_RV12_N20 has correct parameters."""
        f = ALL_FILTERS["VOL_RV12_N20"]
        assert isinstance(f, VolumeFilter)
        assert f.min_rel_vol == 1.2
        assert f.lookback_days == 20


class TestDayOfWeekSkipFilter:
    """DayOfWeekSkipFilter skips specified weekdays, fail-closed on missing data."""

    def test_skips_correct_day(self):
        """skip_days=(4,) rejects Friday (day_of_week=4)."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        assert f.matches_row({"day_of_week": 4}, "0900") is False

    def test_allows_other_days(self):
        """skip_days=(4,) allows Monday (day_of_week=0)."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        assert f.matches_row({"day_of_week": 0}, "0900") is True
        assert f.matches_row({"day_of_week": 1}, "0900") is True
        assert f.matches_row({"day_of_week": 3}, "0900") is True

    def test_fail_closed_missing(self):
        """Missing day_of_week returns False (fail-closed)."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        assert f.matches_row({}, "0900") is False

    def test_fail_closed_none(self):
        """None day_of_week returns False (fail-closed)."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        assert f.matches_row({"day_of_week": None}, "0900") is False

    def test_multiple_skip_days(self):
        """Can skip multiple days."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(0, 4))
        assert f.matches_row({"day_of_week": 0}, "0900") is False
        assert f.matches_row({"day_of_week": 4}, "0900") is False
        assert f.matches_row({"day_of_week": 2}, "0900") is True

    def test_frozen(self):
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        with pytest.raises(AttributeError):
            f.filter_type = "CHANGED"

    def test_to_json_roundtrip(self):
        f = DayOfWeekSkipFilter(filter_type="DOW_NOFRI", description="Skip Friday", skip_days=(4,))
        data = json.loads(f.to_json())
        assert data["filter_type"] == "DOW_NOFRI"
        assert data["skip_days"] == [4]


class TestDowComposites:
    """DOW composites in get_filters_for_grid."""

    def test_0900_has_nofri(self):
        filters = get_filters_for_grid("MGC", "0900")
        for key in ["ORB_G4_NOFRI", "ORB_G5_NOFRI", "ORB_G6_NOFRI", "ORB_G8_NOFRI"]:
            assert key in filters, f"{key} missing from 0900 grid"
            assert isinstance(filters[key], CompositeFilter)

    def test_1800_has_nomon(self):
        filters = get_filters_for_grid("MGC", "1800")
        for key in ["ORB_G4_NOMON", "ORB_G5_NOMON", "ORB_G6_NOMON", "ORB_G8_NOMON"]:
            assert key in filters, f"{key} missing from 1800 grid"
            assert isinstance(filters[key], CompositeFilter)

    def test_1000_has_notue_and_dir_long(self):
        filters = get_filters_for_grid("MGC", "1000")
        for key in ["ORB_G4_NOTUE", "ORB_G5_NOTUE", "ORB_G6_NOTUE", "ORB_G8_NOTUE"]:
            assert key in filters, f"{key} missing from 1000 grid"
        assert "DIR_LONG" in filters

    def test_1100_no_dow(self):
        filters = get_filters_for_grid("MGC", "1100")
        dow_keys = [k for k in filters if "NOFRI" in k or "NOMON" in k or "NOTUE" in k]
        assert dow_keys == [], f"1100 should have no DOW composites, got {dow_keys}"

    def test_composite_matches_row_correctly(self):
        """Composite(G4 + skip Friday) rejects Friday even with big ORB."""
        filters = get_filters_for_grid("MGC", "0900")
        comp = filters["ORB_G4_NOFRI"]
        # Big ORB on Friday -> rejected
        assert comp.matches_row({"orb_0900_size": 6.0, "day_of_week": 4}, "0900") is False
        # Big ORB on Monday -> accepted
        assert comp.matches_row({"orb_0900_size": 6.0, "day_of_week": 0}, "0900") is True
        # Small ORB on Monday -> rejected (base filter fails)
        assert comp.matches_row({"orb_0900_size": 2.0, "day_of_week": 0}, "0900") is False

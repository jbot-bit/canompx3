"""
Tests for trading_app.config filter definitions.
"""

import json
import pytest

from trading_app.config import (
    StrategyFilter,
    NoFilter,
    OrbSizeFilter,
    ALL_FILTERS,
    MGC_ORB_SIZE_FILTERS,
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

    def test_contains_all_size_filters(self):
        for key in ["ORB_L2", "ORB_L3", "ORB_L4", "ORB_L6", "ORB_L8",
                     "ORB_G2", "ORB_G3", "ORB_G4", "ORB_G5", "ORB_G6", "ORB_G8"]:
            assert key in ALL_FILTERS, f"{key} missing from ALL_FILTERS"

    def test_total_count(self):
        # NO_FILTER + 5 L-filters + 6 G-filters = 12
        assert len(ALL_FILTERS) == 12

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

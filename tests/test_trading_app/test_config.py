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
    BreakSpeedFilter,
    BreakBarContinuesFilter,
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
        assert f.matches_row({}, "CME_REOPEN") is True

    def test_matches_with_data(self):
        f = NoFilter()
        row = {"orb_CME_REOPEN_size": 3.5, "rsi_14_at_0900": 45.0}
        assert f.matches_row(row, "CME_REOPEN") is True

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
        assert f.matches_row({"orb_CME_REOPEN_size": 3.5}, "CME_REOPEN") is True

    def test_max_size_at_boundary(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        # max_size uses < (exclusive), so 4.0 should NOT match
        assert f.matches_row({"orb_CME_REOPEN_size": 4.0}, "CME_REOPEN") is False

    def test_min_size_above(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", min_size=4.0)
        assert f.matches_row({"orb_CME_REOPEN_size": 5.0}, "CME_REOPEN") is True

    def test_min_size_at_boundary(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", min_size=4.0)
        # min_size uses >= (inclusive at boundary check is <), so 4.0 matches
        assert f.matches_row({"orb_CME_REOPEN_size": 4.0}, "CME_REOPEN") is True

    def test_null_size_no_match(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        assert f.matches_row({"orb_CME_REOPEN_size": None}, "CME_REOPEN") is False

    def test_missing_size_no_match(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        assert f.matches_row({}, "CME_REOPEN") is False

    def test_uses_correct_orb_label(self):
        f = OrbSizeFilter(filter_type="TEST", description="test", max_size=4.0)
        row = {"orb_CME_REOPEN_size": 5.0, "orb_SINGAPORE_OPEN_size": 3.0}
        assert f.matches_row(row, "CME_REOPEN") is False
        assert f.matches_row(row, "SINGAPORE_OPEN") is True

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
        # NO_FILTER + 4 G-filters + 1 VOL-filter = 6
        # + 12 DOW composites (3 DOW x 4 G)
        # + 12 break quality composites (3 BRK x 4 G: FAST5, FAST10, CONT)
        # + 3 M6E pip-scaled size filters (M6E_G4/G6/G8)
        # + 2 direction filters (DIR_LONG, DIR_SHORT)
        # + 2 MES 1000 band filters (ORB_G4_L12, ORB_G5_L12)
        # = 37
        assert len(ALL_FILTERS) == 37

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
        assert f.matches_row({"rel_vol_CME_REOPEN": 1.5}, "CME_REOPEN") is True

    def test_rejects_below_threshold(self):
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({"rel_vol_CME_REOPEN": 0.8}, "CME_REOPEN") is False

    def test_at_boundary_matches(self):
        """Exactly at min_rel_vol should match (>=)."""
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({"rel_vol_CME_REOPEN": 1.2}, "CME_REOPEN") is True

    def test_fail_closed_missing(self):
        """Missing rel_vol key -> ineligible (fail-closed)."""
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({}, "CME_REOPEN") is False

    def test_fail_closed_none(self):
        """None rel_vol -> ineligible (fail-closed)."""
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        assert f.matches_row({"rel_vol_CME_REOPEN": None}, "CME_REOPEN") is False

    def test_uses_correct_orb_label(self):
        """Uses rel_vol for the specific ORB label."""
        f = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=1.2)
        row = {"rel_vol_CME_REOPEN": 0.5, "rel_vol_LONDON_METALS": 2.0}
        assert f.matches_row(row, "CME_REOPEN") is False
        assert f.matches_row(row, "LONDON_METALS") is True

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


class TestBreakSpeedFilter:
    """BreakSpeedFilter matches by break delay from ORB end."""

    def test_matches_fast_break(self):
        f = BreakSpeedFilter(filter_type="TEST", description="test", max_delay_min=5.0)
        assert f.matches_row({"orb_CME_REOPEN_break_delay_min": 2.0}, "CME_REOPEN") is True

    def test_rejects_slow_break(self):
        f = BreakSpeedFilter(filter_type="TEST", description="test", max_delay_min=5.0)
        assert f.matches_row({"orb_CME_REOPEN_break_delay_min": 10.0}, "CME_REOPEN") is False

    def test_at_boundary_matches(self):
        """Exactly at max_delay_min should match (<=)."""
        f = BreakSpeedFilter(filter_type="TEST", description="test", max_delay_min=5.0)
        assert f.matches_row({"orb_CME_REOPEN_break_delay_min": 5.0}, "CME_REOPEN") is True

    def test_fail_closed_missing(self):
        """Missing break_delay_min -> ineligible (fail-closed)."""
        f = BreakSpeedFilter(filter_type="TEST", description="test", max_delay_min=5.0)
        assert f.matches_row({}, "CME_REOPEN") is False

    def test_fail_closed_none(self):
        """None break_delay_min (no break) -> ineligible (fail-closed)."""
        f = BreakSpeedFilter(filter_type="TEST", description="test", max_delay_min=5.0)
        assert f.matches_row({"orb_CME_REOPEN_break_delay_min": None}, "CME_REOPEN") is False

    def test_uses_correct_orb_label(self):
        f = BreakSpeedFilter(filter_type="TEST", description="test", max_delay_min=5.0)
        row = {"orb_CME_REOPEN_break_delay_min": 10.0, "orb_TOKYO_OPEN_break_delay_min": 2.0}
        assert f.matches_row(row, "CME_REOPEN") is False
        assert f.matches_row(row, "TOKYO_OPEN") is True

    def test_zero_delay_matches(self):
        """Break on the first bar after ORB (delay=0) is the fastest."""
        f = BreakSpeedFilter(filter_type="TEST", description="test", max_delay_min=5.0)
        assert f.matches_row({"orb_CME_REOPEN_break_delay_min": 0.0}, "CME_REOPEN") is True

    def test_frozen(self):
        f = BreakSpeedFilter(filter_type="TEST", description="test")
        with pytest.raises(AttributeError):
            f.max_delay_min = 10.0


class TestBreakBarContinuesFilter:
    """BreakBarContinuesFilter matches by break bar direction relative to break."""

    def test_matches_continuation(self):
        f = BreakBarContinuesFilter(filter_type="TEST", description="test", require_continues=True)
        assert f.matches_row({"orb_CME_REOPEN_break_bar_continues": True}, "CME_REOPEN") is True

    def test_rejects_reversal(self):
        f = BreakBarContinuesFilter(filter_type="TEST", description="test", require_continues=True)
        assert f.matches_row({"orb_CME_REOPEN_break_bar_continues": False}, "CME_REOPEN") is False

    def test_fail_closed_missing(self):
        f = BreakBarContinuesFilter(filter_type="TEST", description="test", require_continues=True)
        assert f.matches_row({}, "CME_REOPEN") is False

    def test_fail_closed_none(self):
        f = BreakBarContinuesFilter(filter_type="TEST", description="test", require_continues=True)
        assert f.matches_row({"orb_CME_REOPEN_break_bar_continues": None}, "CME_REOPEN") is False

    def test_uses_correct_orb_label(self):
        f = BreakBarContinuesFilter(filter_type="TEST", description="test", require_continues=True)
        row = {"orb_CME_REOPEN_break_bar_continues": False, "orb_TOKYO_OPEN_break_bar_continues": True}
        assert f.matches_row(row, "CME_REOPEN") is False
        assert f.matches_row(row, "TOKYO_OPEN") is True

    def test_inverse_mode(self):
        """require_continues=False selects reversal candles."""
        f = BreakBarContinuesFilter(filter_type="TEST", description="test", require_continues=False)
        assert f.matches_row({"orb_CME_REOPEN_break_bar_continues": False}, "CME_REOPEN") is True
        assert f.matches_row({"orb_CME_REOPEN_break_bar_continues": True}, "CME_REOPEN") is False

    def test_frozen(self):
        f = BreakBarContinuesFilter(filter_type="TEST", description="test")
        with pytest.raises(AttributeError):
            f.require_continues = False


class TestDayOfWeekSkipFilter:
    """DayOfWeekSkipFilter skips specified weekdays, fail-closed on missing data."""

    def test_skips_correct_day(self):
        """skip_days=(4,) rejects Friday (day_of_week=4)."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        assert f.matches_row({"day_of_week": 4}, "CME_REOPEN") is False

    def test_allows_other_days(self):
        """skip_days=(4,) allows Monday (day_of_week=0)."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        assert f.matches_row({"day_of_week": 0}, "CME_REOPEN") is True
        assert f.matches_row({"day_of_week": 1}, "CME_REOPEN") is True
        assert f.matches_row({"day_of_week": 3}, "CME_REOPEN") is True

    def test_fail_closed_missing(self):
        """Missing day_of_week returns False (fail-closed)."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        assert f.matches_row({}, "CME_REOPEN") is False

    def test_fail_closed_none(self):
        """None day_of_week returns False (fail-closed)."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(4,))
        assert f.matches_row({"day_of_week": None}, "CME_REOPEN") is False

    def test_multiple_skip_days(self):
        """Can skip multiple days."""
        f = DayOfWeekSkipFilter(filter_type="TEST", description="test", skip_days=(0, 4))
        assert f.matches_row({"day_of_week": 0}, "CME_REOPEN") is False
        assert f.matches_row({"day_of_week": 4}, "CME_REOPEN") is False
        assert f.matches_row({"day_of_week": 2}, "CME_REOPEN") is True

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

    def test_cme_reopen_has_nofri(self):
        # MGC uses G6/G8 only (G4/G5 pass 85%+ of days — regime shift Feb 2026)
        filters = get_filters_for_grid("MGC", "CME_REOPEN")
        for key in ["ORB_G6_NOFRI", "ORB_G8_NOFRI"]:
            assert key in filters, f"{key} missing from MGC CME_REOPEN grid"
            assert isinstance(filters[key], CompositeFilter)
        # Non-MGC instruments still get full G4-G8
        mnq_filters = get_filters_for_grid("MNQ", "CME_REOPEN")
        for key in ["ORB_G4_NOFRI", "ORB_G5_NOFRI", "ORB_G6_NOFRI", "ORB_G8_NOFRI"]:
            assert key in mnq_filters, f"{key} missing from MNQ CME_REOPEN grid"
            assert isinstance(mnq_filters[key], CompositeFilter)

    def test_london_metals_has_nomon(self):
        # MGC uses G6/G8 only
        filters = get_filters_for_grid("MGC", "LONDON_METALS")
        for key in ["ORB_G6_NOMON", "ORB_G8_NOMON"]:
            assert key in filters, f"{key} missing from MGC LONDON_METALS grid"
            assert isinstance(filters[key], CompositeFilter)
        # Non-MGC instruments still get full G4-G8
        mnq_filters = get_filters_for_grid("MNQ", "LONDON_METALS")
        for key in ["ORB_G4_NOMON", "ORB_G5_NOMON", "ORB_G6_NOMON", "ORB_G8_NOMON"]:
            assert key in mnq_filters, f"{key} missing from MNQ LONDON_METALS grid"
            assert isinstance(mnq_filters[key], CompositeFilter)

    def test_tokyo_open_has_notue_and_dir_long(self):
        # MGC uses G6/G8 only
        filters = get_filters_for_grid("MGC", "TOKYO_OPEN")
        for key in ["ORB_G6_NOTUE", "ORB_G8_NOTUE"]:
            assert key in filters, f"{key} missing from MGC TOKYO_OPEN grid"
        assert "DIR_LONG" in filters
        # Non-MGC instruments still get full G4-G8
        mnq_filters = get_filters_for_grid("MNQ", "TOKYO_OPEN")
        for key in ["ORB_G4_NOTUE", "ORB_G5_NOTUE", "ORB_G6_NOTUE", "ORB_G8_NOTUE"]:
            assert key in mnq_filters, f"{key} missing from MNQ TOKYO_OPEN grid"

    def test_singapore_open_has_base_only(self):
        """SINGAPORE_OPEN gets base grid only — no DOW, no DIR_LONG, no NODBL.

        NODBL removed Feb 2026: double_break is look-ahead (computed over
        full session after trade entry). See config.py comment.
        """
        filters = get_filters_for_grid("MGC", "SINGAPORE_OPEN")
        dow_keys = [k for k in filters if "NOFRI" in k or "NOMON" in k or "NOTUE" in k]
        assert dow_keys == [], f"SINGAPORE_OPEN should have no DOW composites, got {dow_keys}"
        assert "DIR_LONG" not in filters
        nodbl_keys = [k for k in filters if "NODBL" in k or k == "NO_DBL_BREAK"]
        assert nodbl_keys == [], f"1100 should have no NODBL filters, got {nodbl_keys}"

    def test_composite_matches_row_correctly(self):
        """Composite(G6 + skip Friday) rejects Friday even with big ORB."""
        filters = get_filters_for_grid("MGC", "CME_REOPEN")
        comp = filters["ORB_G6_NOFRI"]
        # Big ORB on Friday -> rejected (DOW filter fails)
        assert comp.matches_row({"orb_CME_REOPEN_size": 8.0, "day_of_week": 4}, "CME_REOPEN") is False
        # Big ORB on Monday -> accepted
        assert comp.matches_row({"orb_CME_REOPEN_size": 8.0, "day_of_week": 0}, "CME_REOPEN") is True
        # Small ORB on Monday -> rejected (base G6 filter fails)
        assert comp.matches_row({"orb_CME_REOPEN_size": 2.0, "day_of_week": 0}, "CME_REOPEN") is False


class TestDoubleBreakFilterRemoved:
    """NODBL filters removed Feb 2026 — double_break is look-ahead.

    The DoubleBreakFilter class still exists in config.py (not deleted yet)
    but is no longer wired into any discovery grid or ALL_FILTERS registry.
    """

    def test_no_session_gets_nodbl(self):
        """No session gets NODBL filters in the discovery grid."""
        for session in ("CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS", "US_DATA_830", "NYSE_OPEN"):
            filters = get_filters_for_grid("MGC", session)
            nodbl_keys = [k for k in filters if "NODBL" in k or k == "NO_DBL_BREAK"]
            assert nodbl_keys == [], f"{session} has NODBL filters: {nodbl_keys}"

    def test_all_filters_no_nodbl(self):
        """ALL_FILTERS registry has no NODBL entries."""
        nodbl_keys = [k for k in ALL_FILTERS if "NODBL" in k or k == "NO_DBL_BREAK"]
        assert nodbl_keys == [], f"ALL_FILTERS has NODBL: {nodbl_keys}"


class TestBreakQualityComposites:
    """Break quality composites (FAST5, FAST10, CONT) in get_filters_for_grid."""

    def test_momentum_sessions_have_break_quality(self):
        """CME_REOPEN, TOKYO_OPEN, LONDON_METALS get break quality composites."""
        # MGC: G6/G8 only (regime shift Feb 2026)
        for session in ("CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS"):
            filters = get_filters_for_grid("MGC", session)
            for suffix in ("FAST5", "FAST10", "CONT"):
                for g in ("G6", "G8"):
                    key = f"ORB_{g}_{suffix}"
                    assert key in filters, f"{key} missing from MGC {session} grid"
                    assert isinstance(filters[key], CompositeFilter)
        # Non-MGC: full G4-G8
        for session in ("CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS"):
            filters = get_filters_for_grid("MNQ", session)
            for suffix in ("FAST5", "FAST10", "CONT"):
                for g in ("G4", "G5", "G6", "G8"):
                    key = f"ORB_{g}_{suffix}"
                    assert key in filters, f"{key} missing from MNQ {session} grid"
                    assert isinstance(filters[key], CompositeFilter)

    def test_non_momentum_sessions_no_break_quality(self):
        """SINGAPORE_OPEN, US_DATA_830, NYSE_OPEN do NOT get break quality composites."""
        for session in ("SINGAPORE_OPEN", "US_DATA_830", "NYSE_OPEN"):
            filters = get_filters_for_grid("MGC", session)
            bq_keys = [k for k in filters if "FAST5" in k or "FAST10" in k or "CONT" in k]
            assert bq_keys == [], f"{session} has break quality filters: {bq_keys}"

    def test_fast5_composite_matches_row(self):
        """Composite(G6 + FAST5) requires both big ORB and fast break."""
        filters = get_filters_for_grid("MGC", "CME_REOPEN")
        comp = filters["ORB_G6_FAST5"]
        # Big ORB + fast break -> accepted
        assert comp.matches_row(
            {"orb_CME_REOPEN_size": 8.0, "orb_CME_REOPEN_break_delay_min": 2.0}, "CME_REOPEN"
        ) is True
        # Big ORB + slow break -> rejected
        assert comp.matches_row(
            {"orb_CME_REOPEN_size": 8.0, "orb_CME_REOPEN_break_delay_min": 20.0}, "CME_REOPEN"
        ) is False
        # Small ORB + fast break -> rejected (base G6 filter fails)
        assert comp.matches_row(
            {"orb_CME_REOPEN_size": 2.0, "orb_CME_REOPEN_break_delay_min": 2.0}, "CME_REOPEN"
        ) is False

    def test_cont_composite_matches_row(self):
        """Composite(G6 + CONT) requires both big ORB and conviction candle."""
        filters = get_filters_for_grid("MGC", "CME_REOPEN")
        comp = filters["ORB_G6_CONT"]
        # Big ORB + continues -> accepted
        assert comp.matches_row(
            {"orb_CME_REOPEN_size": 8.0, "orb_CME_REOPEN_break_bar_continues": True}, "CME_REOPEN"
        ) is True
        # Big ORB + reversal -> rejected
        assert comp.matches_row(
            {"orb_CME_REOPEN_size": 8.0, "orb_CME_REOPEN_break_bar_continues": False}, "CME_REOPEN"
        ) is False

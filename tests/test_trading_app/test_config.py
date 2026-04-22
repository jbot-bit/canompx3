"""
Tests for trading_app.config filter definitions.
"""

import json

import pytest

from trading_app.config import (
    ALL_FILTERS,
    MGC_ORB_SIZE_FILTERS,
    MGC_VOLUME_FILTERS,
    BreakBarContinuesFilter,
    BreakSpeedFilter,
    CompositeFilter,
    DayOfWeekSkipFilter,
    NoFilter,
    OrbSizeFilter,
    PitRangeFilter,
    PrevDayGeometryFilter,
    StrategyFilter,
    VolumeFilter,
    VWAPBreakDirectionFilter,
    get_filters_for_grid,
    is_e2_lookahead_filter,
)


class TestNoFilter:
    """NoFilter always matches."""

    def test_matches_any_row(self):
        f = NoFilter()
        assert f.matches_row({}, "CME_REOPEN") is True

    def test_matches_with_data(self):
        f = NoFilter()
        row = {"orb_CME_REOPEN_size": 3.5, "rsi_14_at_CME_REOPEN": 45.0}
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
        # NO_FILTER + 4 G + 4 COST + 5 VOL_RV + ATR70_VOL + 4 ORB_VOL + 3 ATR_P = 22 (BASE_GRID_FILTERS)
        # + 12 DOW composites (3 DOW x 4 G)
        # + 12 break quality composites (3 BRK x 4 G: FAST5, FAST10, CONT)
        # + 3 M6E pip-scaled size filters (M6E_G4/G6/G8)
        # + 2 direction filters (DIR_LONG, DIR_SHORT)
        # + 2 MES 1000 band filters (ORB_G4_L12, ORB_G5_L12)
        # + 3 cross-asset ATR filters (X_MES_ATR70, X_MES_ATR60, X_MGC_ATR70)
        # + 4 overnight range absolute (OVNRNG_10/25/50/100 — US sessions only, NOT in BASE)
        # + 3 PDR prev-day-range/atr filters (PDR_R080/R105/R125)
        # + 2 GAP gap/atr filters (GAP_R005/R015)
        # + 8 COST_LT × FAST composites (4 COST × 2 FAST = 8)
        # + 8 OVNRNG × FAST composites (4 OVNRNG × 2 FAST = 8)
        # + 1 PIT_MIN (pit range/atr anti-filter, CME_REOPEN only — Apr 2026)
        # + 4 hypothesis-scoped filters — in ALL_FILTERS but NOT in BASE_GRID_FILTERS.
        #   Access via explicit routing or Phase 4 hypothesis-file injection.
        #     ATR_VEL_GE105 (Wave 4 Phase B, routed to MNQ TOKYO_OPEN + MES US_DATA_1000)
        #     ATR_VEL_GE110 (sensitivity variant, injection-only)
        #     ATR_VEL_GE115 (sensitivity variant, injection-only)
        #     GARCH_VOL_PCT_LT20 (Wave 5 G5, injection-only — MNQ NYSE_OPEN LOW regime)
        #     VWAP_MID_ALIGNED (Apr 2026 exhaustive audit, MNQ US_DATA_1000 O15)
        #     VWAP_BP_ALIGNED (Apr 2026 exhaustive audit, MNQ CME_PRECLOSE O5)
        #     CROSS_NYSE_MOMENTUM (Apr 2026 cross-session state, MNQ US_DATA_1000)
        #     CROSS_COMEX_MOMENTUM (Apr 2026, MNQ CME_PRECLOSE)
        #     CROSS_SGP_MOMENTUM (Apr 2026, MNQ EUROPE_FLOW)
        #     F5_BELOW_PDL (Apr 2026 P1 MNQ binary geometry bridge, long-only)
        #     F6_INSIDE_PDR (Apr 2026 P1 MNQ binary geometry bridge, long-only)
        # = 65 + 4 overnight + 3 PDR + 2 GAP + 8 COST×FAST + 8 OVNRNG×FAST + 1 PIT_MIN + 11 scoped = 93
        assert len(ALL_FILTERS) == 93

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


class TestPrevDayGeometryFilter:
    """Exact geometry predicates promoted from the P1 MNQ bridge."""

    def test_below_pdl_long_matches_only_long_breaks(self):
        f = PrevDayGeometryFilter(filter_type="F5_BELOW_PDL", description="test", mode="below_pdl_long")
        row = {
            "orb_US_DATA_1000_high": 99.0,
            "orb_US_DATA_1000_low": 97.0,
            "orb_US_DATA_1000_break_dir": "long",
            "prev_day_low": 98.5,
            "prev_day_high": 103.0,
        }
        assert f.matches_row(row, "US_DATA_1000") is True

        row["orb_US_DATA_1000_break_dir"] = "short"
        assert f.matches_row(row, "US_DATA_1000") is False

    def test_inside_pdr_long_requires_strict_inside(self):
        f = PrevDayGeometryFilter(filter_type="F6_INSIDE_PDR", description="test", mode="inside_pdr_long")
        row = {
            "orb_COMEX_SETTLE_high": 101.0,
            "orb_COMEX_SETTLE_low": 99.0,
            "orb_COMEX_SETTLE_break_dir": "long",
            "prev_day_low": 95.0,
            "prev_day_high": 105.0,
        }
        assert f.matches_row(row, "COMEX_SETTLE") is True

        row["prev_day_low"] = 100.0
        assert f.matches_row(row, "COMEX_SETTLE") is False

    def test_describe_marks_orb_formation(self):
        f = PrevDayGeometryFilter(filter_type="F5_BELOW_PDL", description="test", mode="below_pdl_long")
        atoms = f.describe(
            {
                "orb_US_DATA_1000_high": 99.0,
                "orb_US_DATA_1000_low": 97.0,
                "orb_US_DATA_1000_break_dir": "long",
                "prev_day_low": 98.5,
                "prev_day_high": 103.0,
            },
            "US_DATA_1000",
            "E2",
        )
        assert len(atoms) == 1
        assert atoms[0].resolves_at == "ORB_FORMATION"
        assert atoms[0].passes is True

    def test_higher_threshold_filters_registered(self):
        """Confluence program rel_vol filters (Mar 2026) are all registered."""
        for key, expected_rv in [
            ("VOL_RV15_N20", 1.5),
            ("VOL_RV20_N20", 2.0),
            ("VOL_RV25_N20", 2.5),
            ("VOL_RV30_N20", 3.0),
        ]:
            assert key in ALL_FILTERS, f"{key} missing from ALL_FILTERS"
            f = ALL_FILTERS[key]
            assert isinstance(f, VolumeFilter)
            assert f.min_rel_vol == expected_rv
            assert f.lookback_days == 20
            assert f.filter_type == key

    def test_higher_threshold_discriminates(self):
        """Higher thresholds correctly reject lower volumes."""
        f25 = VolumeFilter(filter_type="TEST", description="test", min_rel_vol=2.5)
        assert f25.matches_row({"rel_vol_NYSE_CLOSE": 3.0}, "NYSE_CLOSE") is True
        assert f25.matches_row({"rel_vol_NYSE_CLOSE": 2.0}, "NYSE_CLOSE") is False
        assert f25.matches_row({"rel_vol_NYSE_CLOSE": 2.5}, "NYSE_CLOSE") is True


class TestOrbVolumeFilter:
    """OrbVolumeFilter gates on total ORB window volume."""

    def test_matches_above_threshold(self):
        from trading_app.config import OrbVolumeFilter

        f = OrbVolumeFilter(filter_type="TEST", description="test", min_volume=4000.0)
        assert f.matches_row({"orb_NYSE_CLOSE_volume": 5000}, "NYSE_CLOSE") is True

    def test_rejects_below_threshold(self):
        from trading_app.config import OrbVolumeFilter

        f = OrbVolumeFilter(filter_type="TEST", description="test", min_volume=4000.0)
        assert f.matches_row({"orb_NYSE_CLOSE_volume": 3000}, "NYSE_CLOSE") is False

    def test_at_boundary_matches(self):
        from trading_app.config import OrbVolumeFilter

        f = OrbVolumeFilter(filter_type="TEST", description="test", min_volume=4000.0)
        assert f.matches_row({"orb_NYSE_CLOSE_volume": 4000}, "NYSE_CLOSE") is True

    def test_fail_closed_missing(self):
        from trading_app.config import OrbVolumeFilter

        f = OrbVolumeFilter(filter_type="TEST", description="test", min_volume=4000.0)
        assert f.matches_row({}, "NYSE_CLOSE") is False

    def test_fail_closed_none(self):
        from trading_app.config import OrbVolumeFilter

        f = OrbVolumeFilter(filter_type="TEST", description="test", min_volume=4000.0)
        assert f.matches_row({"orb_NYSE_CLOSE_volume": None}, "NYSE_CLOSE") is False

    def test_uses_correct_orb_label(self):
        from trading_app.config import OrbVolumeFilter

        f = OrbVolumeFilter(filter_type="TEST", description="test", min_volume=4000.0)
        row = {"orb_CME_PRECLOSE_volume": 5000, "orb_NYSE_CLOSE_volume": 2000}
        assert f.matches_row(row, "CME_PRECLOSE") is True
        assert f.matches_row(row, "NYSE_CLOSE") is False

    def test_instances_registered(self):
        for key, expected_vol in [
            ("ORB_VOL_2K", 2000.0),
            ("ORB_VOL_4K", 4000.0),
            ("ORB_VOL_8K", 8000.0),
            ("ORB_VOL_16K", 16000.0),
        ]:
            from trading_app.config import OrbVolumeFilter

            assert key in ALL_FILTERS, f"{key} missing from ALL_FILTERS"
            f = ALL_FILTERS[key]
            assert isinstance(f, OrbVolumeFilter)
            assert f.min_volume == expected_vol
            assert f.filter_type == key

    def test_to_json_roundtrip(self):
        from trading_app.config import OrbVolumeFilter

        f = OrbVolumeFilter(filter_type="ORB_VOL_4K", description="test", min_volume=4000.0)
        data = json.loads(f.to_json())
        assert data["filter_type"] == "ORB_VOL_4K"
        assert data["min_volume"] == 4000.0


class TestOvernightRangeAbsFilter:
    """OvernightRangeAbsFilter gates on absolute overnight range (US sessions only)."""

    def test_matches_above_threshold(self):
        from trading_app.config import OvernightRangeAbsFilter

        f = OvernightRangeAbsFilter(filter_type="TEST", description="test", min_range=50.0)
        assert f.matches_row({"overnight_range": 75.0}, "CME_PRECLOSE") is True

    def test_rejects_below_threshold(self):
        from trading_app.config import OvernightRangeAbsFilter

        f = OvernightRangeAbsFilter(filter_type="TEST", description="test", min_range=50.0)
        assert f.matches_row({"overnight_range": 30.0}, "CME_PRECLOSE") is False

    def test_at_boundary_matches(self):
        from trading_app.config import OvernightRangeAbsFilter

        f = OvernightRangeAbsFilter(filter_type="TEST", description="test", min_range=50.0)
        assert f.matches_row({"overnight_range": 50.0}, "CME_PRECLOSE") is True

    def test_fail_closed_missing(self):
        from trading_app.config import OvernightRangeAbsFilter

        f = OvernightRangeAbsFilter(filter_type="TEST", description="test", min_range=50.0)
        assert f.matches_row({}, "CME_PRECLOSE") is False

    def test_fail_closed_none(self):
        from trading_app.config import OvernightRangeAbsFilter

        f = OvernightRangeAbsFilter(filter_type="TEST", description="test", min_range=50.0)
        assert f.matches_row({"overnight_range": None}, "CME_PRECLOSE") is False

    def test_ignores_orb_label(self):
        """overnight_range is a global column, not session-specific."""
        from trading_app.config import OvernightRangeAbsFilter

        f = OvernightRangeAbsFilter(filter_type="TEST", description="test", min_range=50.0)
        row = {"overnight_range": 75.0}
        # Same value regardless of orb_label
        assert f.matches_row(row, "CME_PRECLOSE") is True
        assert f.matches_row(row, "NYSE_CLOSE") is True

    def test_instances_registered(self):
        from trading_app.config import OvernightRangeAbsFilter

        for key, expected_range in [
            ("OVNRNG_10", 10.0),
            ("OVNRNG_25", 25.0),
            ("OVNRNG_50", 50.0),
            ("OVNRNG_100", 100.0),
        ]:
            assert key in ALL_FILTERS, f"{key} missing from ALL_FILTERS"
            f = ALL_FILTERS[key]
            assert isinstance(f, OvernightRangeAbsFilter)
            assert f.min_range == expected_range
            assert f.filter_type == key

    def test_not_in_base_grid(self):
        """OVNRNG filters must NOT be in BASE_GRID_FILTERS (look-ahead for Asian sessions)."""
        from trading_app.config import BASE_GRID_FILTERS

        for key in ["OVNRNG_10", "OVNRNG_25", "OVNRNG_50", "OVNRNG_100"]:
            assert key not in BASE_GRID_FILTERS, (
                f"{key} must NOT be in BASE_GRID_FILTERS — look-ahead contamination for Asian sessions"
            )

    def test_routed_to_us_sessions(self):
        """OVNRNG filters must be present in get_filters_for_grid for US sessions."""
        from trading_app.config import get_filters_for_grid

        # US sessions where overnight_range is CLEAN
        for sess in ["CME_PRECLOSE", "COMEX_SETTLE", "US_DATA_1000", "NYSE_CLOSE"]:
            grid = get_filters_for_grid("MNQ", sess)
            assert "OVNRNG_25" in grid, f"OVNRNG_25 missing from {sess} grid"

    def test_not_routed_to_asian_sessions(self):
        """OVNRNG filters must NOT be in grid for Asian sessions (contaminated)."""
        from trading_app.config import get_filters_for_grid

        for sess in ["CME_REOPEN", "TOKYO_OPEN", "BRISBANE_1025", "SINGAPORE_OPEN"]:
            grid = get_filters_for_grid("MNQ", sess)
            assert "OVNRNG_25" not in grid, f"OVNRNG_25 should NOT be in {sess} grid"

    def test_to_json_roundtrip(self):
        from trading_app.config import OvernightRangeAbsFilter

        f = OvernightRangeAbsFilter(filter_type="OVNRNG_50", description="test", min_range=50.0)
        data = json.loads(f.to_json())
        assert data["filter_type"] == "OVNRNG_50"
        assert data["min_range"] == 50.0


class TestOwnATRPercentileFilter:
    """OwnATRPercentileFilter gates on instrument's own ATR(20) percentile."""

    def test_matches_above_threshold(self):
        from trading_app.config import OwnATRPercentileFilter

        f = OwnATRPercentileFilter(filter_type="TEST", description="test", min_pct=50.0)
        assert f.matches_row({"atr_20_pct": 75.0}, "CME_PRECLOSE") is True

    def test_rejects_below_threshold(self):
        from trading_app.config import OwnATRPercentileFilter

        f = OwnATRPercentileFilter(filter_type="TEST", description="test", min_pct=50.0)
        assert f.matches_row({"atr_20_pct": 30.0}, "CME_PRECLOSE") is False

    def test_fail_closed_missing(self):
        from trading_app.config import OwnATRPercentileFilter

        f = OwnATRPercentileFilter(filter_type="TEST", description="test", min_pct=50.0)
        assert f.matches_row({}, "CME_PRECLOSE") is False

    def test_instances_registered(self):
        from trading_app.config import OwnATRPercentileFilter

        for key, expected_pct in [("ATR_P30", 30.0), ("ATR_P50", 50.0), ("ATR_P70", 70.0)]:
            assert key in ALL_FILTERS, f"{key} missing from ALL_FILTERS"
            f = ALL_FILTERS[key]
            assert isinstance(f, OwnATRPercentileFilter)
            assert f.min_pct == expected_pct
            assert f.filter_type == key


class TestGARCHForecastVolPctFilter:
    """GARCHForecastVolPctFilter gates on GARCH forecast vol rolling percentile.

    Covers Wave 5 G5 deployment — direction="low" variant (MNQ NYSE_OPEN
    RR1.5 LOW-vol regime, Phase B T2-T8 survivor with in_ExpR +0.240).
    The class is direction-parameterized so future research findings
    in the high-vol regime can reuse the same filter class.
    """

    def test_low_direction_matches_below_threshold(self):
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        # 15th percentile <= 20th percentile → admit
        assert f.matches_row({"garch_forecast_vol_pct": 15.0}, "NYSE_OPEN") is True

    def test_low_direction_rejects_above_threshold(self):
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        # 50th percentile > 20th percentile → reject
        assert f.matches_row({"garch_forecast_vol_pct": 50.0}, "NYSE_OPEN") is False

    def test_low_direction_boundary_inclusive(self):
        """At the boundary (pct == pct_threshold), LOW direction admits."""
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        assert f.matches_row({"garch_forecast_vol_pct": 20.0}, "NYSE_OPEN") is True

    def test_high_direction_matches_above_threshold(self):
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=80.0,
            direction="high",
        )
        # 90th percentile >= 80th percentile → admit
        assert f.matches_row({"garch_forecast_vol_pct": 90.0}, "LONDON_METALS") is True

    def test_high_direction_rejects_below_threshold(self):
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=80.0,
            direction="high",
        )
        assert f.matches_row({"garch_forecast_vol_pct": 50.0}, "LONDON_METALS") is False

    def test_high_direction_boundary_inclusive(self):
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=80.0,
            direction="high",
        )
        assert f.matches_row({"garch_forecast_vol_pct": 80.0}, "LONDON_METALS") is True

    def test_fail_closed_missing_column(self):
        """Missing key in row → fail-closed reject. Day is ineligible."""
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        assert f.matches_row({}, "NYSE_OPEN") is False

    def test_fail_closed_none_value(self):
        """Explicit None value (warm-up window) → reject."""
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        assert f.matches_row({"garch_forecast_vol_pct": None}, "NYSE_OPEN") is False

    def test_fail_closed_nan_value(self):
        """NaN / pd.NA → reject (via _atom_numeric coercion)."""
        import math

        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        assert f.matches_row({"garch_forecast_vol_pct": math.nan}, "NYSE_OPEN") is False

    def test_invalid_direction_raises(self):
        from trading_app.config import GARCHForecastVolPctFilter

        with pytest.raises(ValueError, match="direction must be 'low' or 'high'"):
            GARCHForecastVolPctFilter(
                filter_type="TEST",
                description="test",
                pct_threshold=20.0,
                direction="medium",
            )

    def test_matches_df_low_direction(self):
        """Vectorized matches_df mirrors matches_row for the low direction."""
        import pandas as pd

        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        df = pd.DataFrame(
            {
                "garch_forecast_vol_pct": [5.0, 20.0, 21.0, None, 99.0],
                # A few other columns so df is realistic
                "symbol": ["MNQ"] * 5,
            }
        )
        result = f.matches_df(df, "NYSE_OPEN")
        # Expected: [True, True, False, False, False]
        assert list(result) == [True, True, False, False, False]

    def test_matches_df_missing_column(self):
        """Vectorized matches_df returns all-False when column is absent."""
        import pandas as pd

        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        df = pd.DataFrame({"symbol": ["MNQ", "MNQ", "MNQ"]})
        result = f.matches_df(df, "NYSE_OPEN")
        assert list(result) == [False, False, False]

    def test_instance_registered_lt20(self):
        """GARCH_VOL_PCT_LT20 must be in ALL_FILTERS with direction='low'."""
        from trading_app.config import GARCHForecastVolPctFilter

        assert "GARCH_VOL_PCT_LT20" in ALL_FILTERS
        f = ALL_FILTERS["GARCH_VOL_PCT_LT20"]
        assert isinstance(f, GARCHForecastVolPctFilter)
        assert f.filter_type == "GARCH_VOL_PCT_LT20"
        assert f.direction == "low"
        assert f.pct_threshold == 20.0

    def test_not_in_base_grid(self):
        """GARCH_VOL_PCT_LT20 must NOT be in BASE_GRID_FILTERS.

        Accessing GARCH via BASE would leak it into every session of every
        instrument's legacy discovery grid, violating the 'narrow-scope
        filters must be explicitly routed' invariant from
        docs/plans/2026-04-04-new-filter-type-design.md. Correct path:
        Phase 4 hypothesis-file injection via
        strategy_discovery._inject_hypothesis_filters().
        """
        from trading_app.config import BASE_GRID_FILTERS

        assert "GARCH_VOL_PCT_LT20" not in BASE_GRID_FILTERS, (
            "GARCH_VOL_PCT_LT20 must NOT be in BASE_GRID_FILTERS — narrow-scope filter, hypothesis-injection only"
        )

    def test_not_in_legacy_grid_for_any_session(self):
        """Regression: GARCH_VOL_PCT_LT20 must NOT appear in get_filters_for_grid
        for any MNQ session. Access is hypothesis-injection only.

        If a future edit routes GARCH into a specific (instrument, session)
        via explicit add, update this test to exclude that pair rather than
        silently accepting legacy-grid membership.
        """
        from trading_app.config import get_filters_for_grid

        mnq_sessions = [
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "BRISBANE_1025",
            "CME_REOPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
        ]
        for sess in mnq_sessions:
            grid = get_filters_for_grid("MNQ", sess)
            assert "GARCH_VOL_PCT_LT20" not in grid, (
                f"GARCH_VOL_PCT_LT20 leaked into MNQ {sess} legacy grid — "
                f"hypothesis-injection only, no explicit routing exists"
            )

    def test_describe_low_direction_includes_regime_word(self):
        """describe() surfaces the low-vol regime explanation."""
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="GARCH_VOL_PCT_LT20",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        atoms = f.describe({"garch_forecast_vol_pct": 15.0}, "NYSE_OPEN", "E2")
        assert len(atoms) == 1
        atom = atoms[0]
        assert atom.feature_column == "garch_forecast_vol_pct"
        assert atom.comparator == "<="
        assert atom.threshold == 20.0
        assert atom.passes is True
        assert "low-vol" in atom.explanation

    def test_describe_missing_data_passes_none(self):
        """describe() returns passes=None + is_data_missing=True on None."""
        from trading_app.config import GARCHForecastVolPctFilter

        f = GARCHForecastVolPctFilter(
            filter_type="TEST",
            description="test",
            pct_threshold=20.0,
            direction="low",
        )
        atoms = f.describe({"garch_forecast_vol_pct": None}, "NYSE_OPEN", "E2")
        assert len(atoms) == 1
        assert atoms[0].is_data_missing is True
        assert atoms[0].passes is None


class TestCombinedATRVolumeFilter:
    """CombinedATRVolumeFilter requires BOTH ATR pct and rel_vol."""

    def test_both_conditions_met(self):
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(filter_type="TEST", description="test")
        row = {"atr_20_pct": 80.0, "rel_vol_CME_REOPEN": 1.5}
        assert f.matches_row(row, "CME_REOPEN") is True

    def test_atr_below_threshold(self):
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(filter_type="TEST", description="test")
        row = {"atr_20_pct": 50.0, "rel_vol_CME_REOPEN": 1.5}
        assert f.matches_row(row, "CME_REOPEN") is False

    def test_vol_below_threshold(self):
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(filter_type="TEST", description="test")
        row = {"atr_20_pct": 80.0, "rel_vol_CME_REOPEN": 0.8}
        assert f.matches_row(row, "CME_REOPEN") is False

    def test_fail_closed_missing_atr(self):
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(filter_type="TEST", description="test")
        row = {"rel_vol_CME_REOPEN": 1.5}
        assert f.matches_row(row, "CME_REOPEN") is False

    def test_fail_closed_missing_vol(self):
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(filter_type="TEST", description="test")
        row = {"atr_20_pct": 80.0}
        assert f.matches_row(row, "CME_REOPEN") is False

    def test_fail_closed_none_atr(self):
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(filter_type="TEST", description="test")
        row = {"atr_20_pct": None, "rel_vol_CME_REOPEN": 1.5}
        assert f.matches_row(row, "CME_REOPEN") is False

    def test_at_boundary(self):
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(filter_type="TEST", description="test")
        # Exactly at thresholds (>= 70 and >= 1.2)
        row = {"atr_20_pct": 70.0, "rel_vol_CME_REOPEN": 1.2}
        assert f.matches_row(row, "CME_REOPEN") is True

    def test_is_volume_filter_subclass(self):
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(filter_type="TEST", description="test")
        assert isinstance(f, VolumeFilter)

    def test_predefined_atr70_vol(self):
        """ATR70_VOL in ALL_FILTERS has correct parameters."""
        f = ALL_FILTERS["ATR70_VOL"]
        from trading_app.config import CombinedATRVolumeFilter

        assert isinstance(f, CombinedATRVolumeFilter)
        assert f.min_atr_pct == 70.0
        assert f.min_rel_vol == 1.2
        assert f.lookback_days == 20


class TestCrossAssetATRFilter:
    """CrossAssetATRFilter matches by source instrument's ATR percentile."""

    def test_passes_when_above_threshold(self):
        from trading_app.config import CrossAssetATRFilter

        f = CrossAssetATRFilter(filter_type="TEST", description="test", source_instrument="MES", min_pct=70.0)
        row = {"cross_atr_MES_pct": 85.0}
        assert f.matches_row(row, "CME_PRECLOSE") is True

    def test_fails_when_below_threshold(self):
        from trading_app.config import CrossAssetATRFilter

        f = CrossAssetATRFilter(filter_type="TEST", description="test", source_instrument="MES", min_pct=70.0)
        row = {"cross_atr_MES_pct": 50.0}
        assert f.matches_row(row, "CME_PRECLOSE") is False

    def test_fail_closed_missing_key(self):
        from trading_app.config import CrossAssetATRFilter

        f = CrossAssetATRFilter(filter_type="TEST", description="test", source_instrument="MES", min_pct=70.0)
        assert f.matches_row({}, "CME_PRECLOSE") is False

    def test_fail_closed_none_value(self):
        from trading_app.config import CrossAssetATRFilter

        f = CrossAssetATRFilter(filter_type="TEST", description="test", source_instrument="MES", min_pct=70.0)
        row = {"cross_atr_MES_pct": None}
        assert f.matches_row(row, "CME_PRECLOSE") is False

    def test_at_boundary(self):
        from trading_app.config import CrossAssetATRFilter

        f = CrossAssetATRFilter(filter_type="TEST", description="test", source_instrument="MES", min_pct=70.0)
        row = {"cross_atr_MES_pct": 70.0}
        assert f.matches_row(row, "CME_PRECLOSE") is True

    def test_mgc_source(self):
        from trading_app.config import CrossAssetATRFilter

        f = CrossAssetATRFilter(filter_type="TEST", description="test", source_instrument="MGC", min_pct=70.0)
        row = {"cross_atr_MGC_pct": 75.0}
        assert f.matches_row(row, "COMEX_SETTLE") is True
        # Wrong source key → fail-closed
        row2 = {"cross_atr_MES_pct": 99.0}
        assert f.matches_row(row2, "COMEX_SETTLE") is False

    def test_predefined_x_mes_atr70(self):
        """X_MES_ATR70 in ALL_FILTERS has correct parameters."""
        from trading_app.config import CrossAssetATRFilter

        f = ALL_FILTERS["X_MES_ATR70"]
        assert isinstance(f, CrossAssetATRFilter)
        assert f.source_instrument == "MES"
        assert f.min_pct == 70.0

    def test_not_volume_filter_subclass(self):
        """CrossAssetATRFilter is NOT a VolumeFilter subclass (different enrichment path)."""
        from trading_app.config import CrossAssetATRFilter

        f = CrossAssetATRFilter(filter_type="TEST", description="test")
        assert not isinstance(f, VolumeFilter)


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
        f = DayOfWeekSkipFilter(filter_type="DOW_NOMON", description="Skip Monday", skip_days=(0,))
        data = json.loads(f.to_json())
        assert data["filter_type"] == "DOW_NOMON"
        assert data["skip_days"] == [0]


class TestDowComposites:
    """DOW composites in get_filters_for_grid."""

    def test_cme_reopen_no_nofri(self):
        """NOFRI removed from CME_REOPEN grid Mar 2026 — LIKELY NOISE (DOW stress test)."""
        filters = get_filters_for_grid("MGC", "CME_REOPEN")
        nofri_keys = [k for k in filters if "NOFRI" in k]
        assert nofri_keys == [], f"NOFRI should be removed from CME_REOPEN grid, got {nofri_keys}"

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

    def test_tokyo_open_has_dir_long_no_notue(self):
        """NOTUE removed from TOKYO_OPEN grid Mar 2026 — LIKELY NOISE. DIR_LONG stays."""
        filters = get_filters_for_grid("MGC", "TOKYO_OPEN")
        notue_keys = [k for k in filters if "NOTUE" in k]
        assert notue_keys == [], f"NOTUE should be removed from TOKYO_OPEN grid, got {notue_keys}"
        assert "DIR_LONG" in filters

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
        assert nodbl_keys == [], f"SINGAPORE_OPEN should have no NODBL filters, got {nodbl_keys}"

    def test_composite_matches_row_correctly(self):
        """Composite(G6 + skip Monday) rejects Monday even with big ORB."""
        filters = get_filters_for_grid("MGC", "LONDON_METALS")
        comp = filters["ORB_G6_NOMON"]
        # Big ORB on Monday -> rejected (DOW filter fails)
        assert comp.matches_row({"orb_LONDON_METALS_size": 8.0, "day_of_week": 0}, "LONDON_METALS") is False
        # Big ORB on Tuesday -> accepted
        assert comp.matches_row({"orb_LONDON_METALS_size": 8.0, "day_of_week": 1}, "LONDON_METALS") is True
        # Small ORB on Tuesday -> rejected (base G6 filter fails)
        assert comp.matches_row({"orb_LONDON_METALS_size": 2.0, "day_of_week": 1}, "LONDON_METALS") is False


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


class TestE2LookaheadHelper:
    """Canonical helper for E2-invalid break-bar-derived filter families."""

    @pytest.mark.parametrize(
        ("filter_type", "expected"),
        [
            ("VOL_RV12_N20", True),
            ("ATR70_VOL_RV12", True),
            ("ORB_G8_FAST5", True),
            ("ORB_G6_CONT", True),
            ("ORB_G5_NOMON_CONT", True),
            ("ORB_G5", False),
            ("OVNRNG_25", False),
            ("PDR_R105", False),
        ],
    )
    def test_classifies_filter_types(self, filter_type, expected):
        assert is_e2_lookahead_filter(filter_type) is expected


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

    def test_non_break_speed_sessions_no_break_quality(self):
        """Sessions without validated break-speed WR signal get no break quality composites."""
        for session in ("SINGAPORE_OPEN", "US_DATA_830", "COMEX_SETTLE", "EUROPE_FLOW"):
            filters = get_filters_for_grid("MGC", session)
            bq_keys = [k for k in filters if "FAST5" in k or "FAST10" in k or "CONT" in k]
            assert bq_keys == [], f"{session} has break quality filters: {bq_keys}"

    def test_nyse_sessions_have_break_quality(self):
        """NYSE_CLOSE and NYSE_OPEN get break quality composites (Apr 2026 retest)."""
        for session in ("NYSE_CLOSE", "NYSE_OPEN"):
            filters = get_filters_for_grid("MNQ", session)
            for suffix in ("FAST5", "FAST10", "CONT"):
                for g in ("G4", "G5", "G6", "G8"):
                    key = f"ORB_{g}_{suffix}"
                    assert key in filters, f"{key} missing from MNQ {session} grid"
                    assert isinstance(filters[key], CompositeFilter)

    def test_fast5_composite_matches_row(self):
        """Composite(G6 + FAST5) requires both big ORB and fast break."""
        filters = get_filters_for_grid("MGC", "CME_REOPEN")
        comp = filters["ORB_G6_FAST5"]
        # Big ORB + fast break -> accepted
        assert (
            comp.matches_row({"orb_CME_REOPEN_size": 8.0, "orb_CME_REOPEN_break_delay_min": 2.0}, "CME_REOPEN") is True
        )
        # Big ORB + slow break -> rejected
        assert (
            comp.matches_row({"orb_CME_REOPEN_size": 8.0, "orb_CME_REOPEN_break_delay_min": 20.0}, "CME_REOPEN")
            is False
        )
        # Small ORB + fast break -> rejected (base G6 filter fails)
        assert (
            comp.matches_row({"orb_CME_REOPEN_size": 2.0, "orb_CME_REOPEN_break_delay_min": 2.0}, "CME_REOPEN") is False
        )

    def test_cont_composite_matches_row(self):
        """Composite(G6 + CONT) requires both big ORB and conviction candle."""
        filters = get_filters_for_grid("MGC", "CME_REOPEN")
        comp = filters["ORB_G6_CONT"]
        # Big ORB + continues -> accepted
        assert (
            comp.matches_row({"orb_CME_REOPEN_size": 8.0, "orb_CME_REOPEN_break_bar_continues": True}, "CME_REOPEN")
            is True
        )
        # Big ORB + reversal -> rejected
        assert (
            comp.matches_row({"orb_CME_REOPEN_size": 8.0, "orb_CME_REOPEN_break_bar_continues": False}, "CME_REOPEN")
            is False
        )


class TestPitRangeFilter:
    """PitRangeFilter gates on pit_range_atr >= min_ratio (exchange pit range / ATR-20)."""

    def test_matches_above_threshold(self):
        f = PitRangeFilter(filter_type="TEST", description="test", min_ratio=0.10)
        assert f.matches_row({"pit_range_atr": 0.25}, "CME_REOPEN") is True

    def test_rejects_below_threshold(self):
        f = PitRangeFilter(filter_type="TEST", description="test", min_ratio=0.10)
        assert f.matches_row({"pit_range_atr": 0.05}, "CME_REOPEN") is False

    def test_at_boundary_matches(self):
        """Exactly at min_ratio should match (>=)."""
        f = PitRangeFilter(filter_type="TEST", description="test", min_ratio=0.10)
        assert f.matches_row({"pit_range_atr": 0.10}, "CME_REOPEN") is True

    def test_fail_closed_missing(self):
        """Missing pit_range_atr key -> ineligible (fail-closed)."""
        f = PitRangeFilter(filter_type="TEST", description="test", min_ratio=0.10)
        assert f.matches_row({}, "CME_REOPEN") is False

    def test_fail_closed_none(self):
        """None pit_range_atr -> ineligible (fail-closed)."""
        f = PitRangeFilter(filter_type="TEST", description="test", min_ratio=0.10)
        assert f.matches_row({"pit_range_atr": None}, "CME_REOPEN") is False

    def test_matches_df_above(self):
        import pandas as pd

        f = PitRangeFilter(filter_type="TEST", description="test", min_ratio=0.10)
        df = pd.DataFrame({"pit_range_atr": [0.25, 0.05, None, 0.10]})
        result = f.matches_df(df, "CME_REOPEN")
        assert list(result) == [True, False, False, True]

    def test_matches_df_missing_column(self):
        import pandas as pd

        f = PitRangeFilter(filter_type="TEST", description="test", min_ratio=0.10)
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = f.matches_df(df, "CME_REOPEN")
        assert not result.any()

    def test_pit_min_registered(self):
        """PIT_MIN instance is in ALL_FILTERS with correct parameters."""
        assert "PIT_MIN" in ALL_FILTERS
        f = ALL_FILTERS["PIT_MIN"]
        assert isinstance(f, PitRangeFilter)
        assert f.min_ratio == 0.10
        assert f.filter_type == "PIT_MIN"

    def test_routed_to_cme_reopen_only(self):
        """PIT_MIN must be in CME_REOPEN grid and absent from all other sessions."""
        grid = get_filters_for_grid("MNQ", "CME_REOPEN")
        assert "PIT_MIN" in grid, "PIT_MIN missing from CME_REOPEN grid"

        for sess in [
            "CME_PRECLOSE",
            "COMEX_SETTLE",
            "NYSE_OPEN",
            "NYSE_CLOSE",
            "SINGAPORE_OPEN",
            "TOKYO_OPEN",
            "EUROPE_FLOW",
            "LONDON_METALS",
        ]:
            grid = get_filters_for_grid("MNQ", sess)
            assert "PIT_MIN" not in grid, f"PIT_MIN should NOT be in {sess} grid"

    def test_frozen(self):
        f = PitRangeFilter(filter_type="TEST", description="test", min_ratio=0.10)
        with pytest.raises(AttributeError):
            f.min_ratio = 0.20

    def test_to_json_roundtrip(self):
        f = PitRangeFilter(filter_type="PIT_MIN", description="test", min_ratio=0.10)
        data = json.loads(f.to_json())
        assert data["filter_type"] == "PIT_MIN"
        assert data["min_ratio"] == 0.10


class TestRequiresMicroData:
    """Phase 3b: every StrategyFilter must self-describe whether it needs
    real-micro contract data in bars_1m (vs parent-proxy data).

    Default (price-based filters): False — work on any era.
    Volume-based (VolumeFilter family, OrbVolumeFilter): True — the volume
    signal is meaningless on parent-proxy data because MNQ/MES/MGC micro
    volume is NOT the same as NQ/ES/GC parent volume.
    CompositeFilter: dynamic — True iff any component requires it.

    Consumers (deferred to Stage 3c/3d):
    - Stage 3c: rebuild scope filter — only rebuild era-appropriate date ranges
    - Stage 3d: drift check — reject validated_setups where a volume filter
      references trades before micro_launch_day

    @rule canonical-filter-self-description (rule 4, integrity-guardian)
    """

    def test_base_strategy_filter_default_false(self):
        """Base class default: price-based, era-invariant."""
        # Use NoFilter as a stand-in for the base class default
        assert NoFilter().requires_micro_data is False

    def test_orb_size_filter_is_false(self):
        """ORB size is price-based (high-low of the ORB bars)."""
        f = OrbSizeFilter(filter_type="ORB_G5", description="test", min_size=5.0)
        assert f.requires_micro_data is False

    def test_volume_filter_is_true(self):
        """VolumeFilter (rel_vol) REQUIRES real micro volume data."""
        f = VolumeFilter(
            filter_type="VOL_RV12_N20",
            description="test",
            min_rel_vol=1.2,
            lookback_days=20,
        )
        assert f.requires_micro_data is True

    def test_orb_volume_filter_is_true(self):
        """OrbVolumeFilter (aggregate ORB-window volume) REQUIRES real micro."""
        from trading_app.config import OrbVolumeFilter

        f = OrbVolumeFilter(
            filter_type="ORB_VOL_2K",
            description="test",
            min_volume=2000.0,
        )
        assert f.requires_micro_data is True

    def test_combined_atr_volume_filter_inherits_true(self):
        """CombinedATRVolumeFilter subclasses VolumeFilter → inherits True."""
        from trading_app.config import CombinedATRVolumeFilter

        f = CombinedATRVolumeFilter(
            filter_type="ATR70_VOL",
            description="test",
            min_rel_vol=1.0,
            lookback_days=20,
            min_atr_pct=70.0,
        )
        assert f.requires_micro_data is True

    def test_composite_false_when_both_components_false(self):
        """CompositeFilter of two price-based filters → False."""
        base = OrbSizeFilter(filter_type="ORB_G5", description="test", min_size=5.0)
        overlay = NoFilter()
        comp = CompositeFilter(
            filter_type="COMPOSITE",
            description="test",
            base=base,
            overlay=overlay,
        )
        assert comp.requires_micro_data is False

    def test_composite_true_when_base_is_volume(self):
        """CompositeFilter with volume base → True (dynamic OR)."""
        base = VolumeFilter(
            filter_type="VOL_RV12_N20",
            description="test",
            min_rel_vol=1.2,
        )
        overlay = OrbSizeFilter(filter_type="ORB_G5", description="test", min_size=5.0)
        comp = CompositeFilter(
            filter_type="COMPOSITE",
            description="test",
            base=base,
            overlay=overlay,
        )
        assert comp.requires_micro_data is True

    def test_composite_true_when_overlay_is_volume(self):
        """CompositeFilter with volume overlay → True (dynamic OR)."""
        base = OrbSizeFilter(filter_type="ORB_G5", description="test", min_size=5.0)
        overlay = VolumeFilter(
            filter_type="VOL_RV12_N20",
            description="test",
            min_rel_vol=1.2,
        )
        comp = CompositeFilter(
            filter_type="COMPOSITE",
            description="test",
            base=base,
            overlay=overlay,
        )
        assert comp.requires_micro_data is True

    def test_nested_composite_propagates(self):
        """Composite of composite → dynamic OR through all nested levels."""
        inner = CompositeFilter(
            filter_type="INNER",
            description="test",
            base=VolumeFilter(filter_type="V", description="test", min_rel_vol=1.2),
            overlay=NoFilter(),
        )
        outer = CompositeFilter(
            filter_type="OUTER",
            description="test",
            base=OrbSizeFilter(filter_type="ORB_G5", description="test", min_size=5.0),
            overlay=inner,
        )
        assert outer.requires_micro_data is True

    def test_all_registered_filters_return_bool(self):
        """Every filter in ALL_FILTERS must have requires_micro_data returning bool.

        Drift guard — catches future filter additions that forget the attribute.
        """
        for filter_type, filter_instance in ALL_FILTERS.items():
            result = filter_instance.requires_micro_data
            assert isinstance(result, bool), (
                f"{filter_type}.requires_micro_data should return bool, got {type(result).__name__}"
            )

    def test_known_volume_filters_in_all_filters_are_true(self):
        """Sanity check: the volume filters we expect to be True actually are."""
        known_volume_filter_types = {
            "VOL_RV12_N20",
            "VOL_RV15_N20",
            "VOL_RV20_N20",
            "VOL_RV25_N20",
            "VOL_RV30_N20",
            "ATR70_VOL",
            "ORB_VOL_2K",
            "ORB_VOL_4K",
            "ORB_VOL_8K",
            "ORB_VOL_16K",
        }
        for ft in known_volume_filter_types:
            if ft not in ALL_FILTERS:
                continue  # Not all volume variants are always registered
            assert ALL_FILTERS[ft].requires_micro_data is True, (
                f"ALL_FILTERS[{ft}] is a known volume filter but requires_micro_data=False"
            )

    def test_known_price_filters_in_all_filters_are_false(self):
        """Sanity check: pure price-based filters should all be False."""
        known_price_filter_types = {
            "NO_FILTER",
            "ORB_G2",
            "ORB_G3",
            "ORB_G4",
            "ORB_G5",
            "ORB_G6",
            "ORB_G8",
            "ORB_L2",
            "ORB_L3",
            "ORB_L4",
            "ORB_L6",
            "ORB_L8",
            "COST_LT08",
            "COST_LT10",
            "COST_LT12",
        }
        for ft in known_price_filter_types:
            if ft not in ALL_FILTERS:
                continue
            assert ALL_FILTERS[ft].requires_micro_data is False, (
                f"ALL_FILTERS[{ft}] is a price filter but requires_micro_data=True"
            )

    def test_frozen_dataclass_still_frozen(self):
        """Adding requires_micro_data as @property must not break frozen semantics."""
        f = NoFilter()
        with pytest.raises(AttributeError):
            f.requires_micro_data = True  # type: ignore[misc]


class TestVWAPBreakDirectionFilter:
    """Tests for VWAPBreakDirectionFilter — VWAP break-direction alignment."""

    # --- orb_mid definition ---

    def test_orb_mid_long_aligned(self):
        """Long break with ORB midpoint above VWAP → passes."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_US_DATA_1000_vwap": 100.0,
            "orb_US_DATA_1000_high": 110.0,
            "orb_US_DATA_1000_low": 95.0,  # mid = 102.5 > 100
            "orb_US_DATA_1000_break_dir": "long",
        }
        assert f.matches_row(row, "US_DATA_1000") is True

    def test_orb_mid_long_counter(self):
        """Long break with ORB midpoint below VWAP → blocked."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_NYSE_OPEN_vwap": 100.0,
            "orb_NYSE_OPEN_high": 102.0,
            "orb_NYSE_OPEN_low": 95.0,  # mid = 98.5 < 100
            "orb_NYSE_OPEN_break_dir": "long",
        }
        assert f.matches_row(row, "NYSE_OPEN") is False

    def test_orb_mid_short_aligned(self):
        """Short break with ORB midpoint below VWAP → passes."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_CME_PRECLOSE_vwap": 100.0,
            "orb_CME_PRECLOSE_high": 99.0,
            "orb_CME_PRECLOSE_low": 95.0,  # mid = 97.0 < 100
            "orb_CME_PRECLOSE_break_dir": "short",
        }
        assert f.matches_row(row, "CME_PRECLOSE") is True

    def test_orb_mid_short_counter(self):
        """Short break with ORB midpoint above VWAP → blocked."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_CME_PRECLOSE_vwap": 100.0,
            "orb_CME_PRECLOSE_high": 105.0,
            "orb_CME_PRECLOSE_low": 99.0,  # mid = 102.0 > 100
            "orb_CME_PRECLOSE_break_dir": "short",
        }
        assert f.matches_row(row, "CME_PRECLOSE") is False

    # --- break_price definition ---

    def test_bp_long_aligned(self):
        """Long break with orb_high above VWAP → passes."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_BP_ALIGNED", description="test", definition="break_price")
        row = {
            "orb_CME_PRECLOSE_vwap": 100.0,
            "orb_CME_PRECLOSE_high": 105.0,  # break at 105 > 100
            "orb_CME_PRECLOSE_low": 95.0,
            "orb_CME_PRECLOSE_break_dir": "long",
        }
        assert f.matches_row(row, "CME_PRECLOSE") is True

    def test_bp_short_aligned(self):
        """Short break with orb_low below VWAP → passes."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_BP_ALIGNED", description="test", definition="break_price")
        row = {
            "orb_CME_PRECLOSE_vwap": 100.0,
            "orb_CME_PRECLOSE_high": 105.0,
            "orb_CME_PRECLOSE_low": 95.0,  # break at 95 < 100
            "orb_CME_PRECLOSE_break_dir": "short",
        }
        assert f.matches_row(row, "CME_PRECLOSE") is True

    def test_bp_long_counter(self):
        """Long break with orb_high below VWAP → blocked."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_BP_ALIGNED", description="test", definition="break_price")
        row = {
            "orb_TOKYO_OPEN_vwap": 100.0,
            "orb_TOKYO_OPEN_high": 99.0,  # break at 99 < 100
            "orb_TOKYO_OPEN_low": 95.0,
            "orb_TOKYO_OPEN_break_dir": "long",
        }
        assert f.matches_row(row, "TOKYO_OPEN") is False

    # --- fail-closed on missing data ---

    def test_fail_closed_no_vwap(self):
        """Missing VWAP → fail-closed (no trade)."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_NYSE_OPEN_vwap": None,
            "orb_NYSE_OPEN_high": 110.0,
            "orb_NYSE_OPEN_low": 95.0,
            "orb_NYSE_OPEN_break_dir": "long",
        }
        assert f.matches_row(row, "NYSE_OPEN") is False

    def test_fail_closed_no_break_dir(self):
        """Missing break_dir → fail-closed."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_NYSE_OPEN_vwap": 100.0,
            "orb_NYSE_OPEN_high": 110.0,
            "orb_NYSE_OPEN_low": 95.0,
            "orb_NYSE_OPEN_break_dir": None,
        }
        assert f.matches_row(row, "NYSE_OPEN") is False

    def test_fail_closed_no_orb_range(self):
        """Missing ORB high/low → fail-closed."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_NYSE_OPEN_vwap": 100.0,
            "orb_NYSE_OPEN_high": None,
            "orb_NYSE_OPEN_low": None,
            "orb_NYSE_OPEN_break_dir": "long",
        }
        assert f.matches_row(row, "NYSE_OPEN") is False

    def test_fail_closed_break_dir_none_string(self):
        """break_dir = 'none' (no break) → fail-closed."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_BP_ALIGNED", description="test", definition="break_price")
        row = {
            "orb_NYSE_OPEN_vwap": 100.0,
            "orb_NYSE_OPEN_high": 110.0,
            "orb_NYSE_OPEN_low": 95.0,
            "orb_NYSE_OPEN_break_dir": "none",
        }
        assert f.matches_row(row, "NYSE_OPEN") is False

    # --- edge cases ---

    def test_exact_equality_fails_closed(self):
        """ref == VWAP exactly → NOT aligned (strict inequality)."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_NYSE_OPEN_vwap": 100.0,
            "orb_NYSE_OPEN_high": 105.0,
            "orb_NYSE_OPEN_low": 95.0,  # mid = 100.0 == VWAP
            "orb_NYSE_OPEN_break_dir": "long",
        }
        assert f.matches_row(row, "NYSE_OPEN") is False

    def test_invalid_definition_raises(self):
        """Invalid definition string → ValueError at construction."""
        with pytest.raises(ValueError, match="must be"):
            VWAPBreakDirectionFilter(filter_type="BAD", description="test", definition="invalid")

    # --- registration ---

    def test_registered_in_all_filters(self):
        """Both VWAP filters must be in ALL_FILTERS."""
        assert "VWAP_MID_ALIGNED" in ALL_FILTERS
        assert "VWAP_BP_ALIGNED" in ALL_FILTERS

    def test_not_in_base_grid(self):
        """VWAP filters are hypothesis-scoped, not base grid."""
        from trading_app.config import BASE_GRID_FILTERS

        assert "VWAP_MID_ALIGNED" not in BASE_GRID_FILTERS
        assert "VWAP_BP_ALIGNED" not in BASE_GRID_FILTERS

    def test_describe_returns_atoms(self):
        """describe() returns non-empty atom list with correct category."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        row = {
            "orb_NYSE_OPEN_vwap": 100.0,
            "orb_NYSE_OPEN_high": 110.0,
            "orb_NYSE_OPEN_low": 95.0,
            "orb_NYSE_OPEN_break_dir": "long",
        }
        atoms = f.describe(row, "NYSE_OPEN", "E2")
        assert len(atoms) == 1
        assert atoms[0].category == "INTRA_SESSION"
        assert atoms[0].resolves_at == "BREAK_DETECTED"
        assert atoms[0].passes is True

    def test_frozen(self):
        """Filter is immutable."""
        f = VWAPBreakDirectionFilter(filter_type="VWAP_MID_ALIGNED", description="test", definition="orb_mid")
        with pytest.raises(AttributeError):
            f.definition = "break_price"  # type: ignore[misc]

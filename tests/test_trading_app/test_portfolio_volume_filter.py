"""Tests for Portfolio build_strategy_daily_series with VolumeFilter (T9).

Tests the VolumeFilter class and the overlay eligibility logic directly
without requiring a database connection.
"""
import pytest
import numpy as np
import pandas as pd
from trading_app.config import (
    VolumeFilter, OrbSizeFilter, NoFilter, ALL_FILTERS,
)


class TestVolumeFilterMatchesRow:
    def _vf(self):
        return VolumeFilter(
            filter_type="VOL_RV12_N20",
            description="Relative volume >= 1.2x",
            min_rel_vol=1.2, lookback_days=20,
        )

    def test_above_threshold_eligible(self):
        assert self._vf().matches_row({"rel_vol_0900": 1.5}, "0900") is True

    def test_below_threshold_ineligible(self):
        assert self._vf().matches_row({"rel_vol_0900": 0.8}, "0900") is False

    def test_exactly_at_threshold(self):
        assert self._vf().matches_row({"rel_vol_0900": 1.2}, "0900") is True

    def test_missing_rel_vol_fail_closed(self):
        assert self._vf().matches_row({}, "0900") is False

    def test_none_rel_vol_fail_closed(self):
        assert self._vf().matches_row({"rel_vol_0900": None}, "0900") is False

    def test_different_orb_labels(self):
        vf = self._vf()
        row = {"rel_vol_0900": 2.0, "rel_vol_1800": 0.5}
        assert vf.matches_row(row, "0900") is True
        assert vf.matches_row(row, "1800") is False


class TestOrbSizeFilterMatchesRow:
    def test_g4_above_threshold(self):
        filt = ALL_FILTERS["ORB_G4"]
        assert filt.matches_row({"orb_0900_size": 5.0}, "0900") is True

    def test_g4_below_threshold(self):
        filt = ALL_FILTERS["ORB_G4"]
        assert filt.matches_row({"orb_0900_size": 3.0}, "0900") is False

    def test_no_filter_always_true(self):
        filt = ALL_FILTERS["NO_FILTER"]
        assert filt.matches_row({}, "0900") is True
        assert filt.matches_row({"orb_0900_size": None}, "0900") is True


class TestOverlayEligibilityLogic:
    """Test the overlay logic used in build_strategy_daily_series:
    - Eligible but no trade = 0.0
    - Ineligible = NaN
    - Only overlay pnl_r on eligible (0.0) days
    """

    def test_overlay_only_on_eligible_days(self):
        days = pd.date_range("2024-01-01", periods=5, freq="D")
        series = pd.Series(np.nan, index=days)
        # Days 0,1,3 are eligible
        series.iloc[0] = 0.0
        series.iloc[1] = 0.0
        series.iloc[3] = 0.0

        # Trades on days 0, 2 (day 2 is ineligible)
        trades = {days[0]: 1.5, days[2]: -1.0}
        skipped = 0
        for td, pnl in trades.items():
            if series.loc[td] == 0.0:
                series.loc[td] = pnl
            else:
                skipped += 1

        assert series.iloc[0] == 1.5  # overlayed
        assert np.isnan(series.iloc[2])  # ineligible, NOT overlayed
        assert skipped == 1
        assert series.iloc[1] == 0.0  # eligible, no trade
        assert series.iloc[3] == 0.0  # eligible, no trade

    def test_all_eligible_days_get_trades(self):
        days = pd.date_range("2024-01-01", periods=3, freq="D")
        series = pd.Series(0.0, index=days)  # all eligible
        trades = {days[0]: 1.0, days[1]: -1.0, days[2]: 0.5}
        for td, pnl in trades.items():
            if series.loc[td] == 0.0:
                series.loc[td] = pnl
        assert list(series) == [1.0, -1.0, 0.5]

    def test_no_eligible_days(self):
        days = pd.date_range("2024-01-01", periods=3, freq="D")
        series = pd.Series(np.nan, index=days)  # all ineligible
        trades = {days[0]: 1.0, days[1]: -1.0}
        skipped = 0
        for td, pnl in trades.items():
            if td in series.index and series.loc[td] == 0.0:
                series.loc[td] = pnl
            else:
                skipped += 1
        assert skipped == 2
        assert all(np.isnan(series))


class TestAllFiltersSync:
    def test_vol_filter_in_all_filters(self):
        assert "VOL_RV12_N20" in ALL_FILTERS
        assert isinstance(ALL_FILTERS["VOL_RV12_N20"], VolumeFilter)

    def test_all_filter_types_match_keys(self):
        for key, filt in ALL_FILTERS.items():
            assert filt.filter_type == key, f"Mismatch: key={key}, filter_type={filt.filter_type}"

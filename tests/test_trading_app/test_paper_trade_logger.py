"""Tests for paper_trade_logger — profile-derived lanes + OOS boundary."""

from datetime import date

import pytest


class TestBuildLanes:
    """Lanes derived from prop_profiles must have valid structure."""

    def test_lanes_match_active_profile(self):
        """_build_lanes returns lanes matching the active profile's daily_lanes count."""
        from trading_app.paper_trade_logger import _build_lanes
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        for pid, p in ACCOUNT_PROFILES.items():
            if p.active and p.daily_lanes:
                lanes = _build_lanes(pid)
                assert len(lanes) == len(p.daily_lanes), (
                    f"Profile {pid}: _build_lanes returned {len(lanes)} but daily_lanes has {len(p.daily_lanes)}"
                )
                break

    def test_lane_strategy_ids_match_profile(self):
        """Every lane's strategy_id must come from the profile."""
        from trading_app.paper_trade_logger import _build_lanes
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        for pid, p in ACCOUNT_PROFILES.items():
            if p.active and p.daily_lanes:
                lanes = _build_lanes(pid)
                profile_ids = {spec.strategy_id for spec in p.daily_lanes}
                lane_ids = {lane.strategy_id for lane in lanes}
                assert lane_ids == profile_ids, f"Strategy ID mismatch in {pid}: {lane_ids ^ profile_ids}"
                break

    def test_all_lanes_have_required_fields(self):
        """Every lane must have valid trading parameters."""
        from trading_app.paper_trade_logger import _build_lanes

        lanes = _build_lanes()  # first active profile
        for lane in lanes:
            assert lane.strategy_id, "Lane missing strategy_id"
            assert lane.instrument, "Lane missing instrument"
            assert lane.orb_label, "Lane missing orb_label"
            assert lane.orb_minutes > 0, f"Lane {lane.strategy_id} bad orb_minutes"
            assert lane.rr_target > 0, f"Lane {lane.strategy_id} bad rr_target"
            assert lane.entry_model in ("E1", "E2"), f"Lane {lane.strategy_id} bad entry_model"

    def test_filter_type_registered(self):
        """Every lane's filter_type must exist in ALL_FILTERS."""
        from trading_app.config import ALL_FILTERS
        from trading_app.paper_trade_logger import _build_lanes

        lanes = _build_lanes()
        for lane in lanes:
            assert lane.filter_type in ALL_FILTERS, (
                f"Lane {lane.strategy_id} filter_type '{lane.filter_type}' not in ALL_FILTERS"
            )


class TestOOSBoundary:
    """OOS start date must be enforced."""

    def test_oos_start_is_2026(self):
        from trading_app.paper_trade_logger import OOS_START

        assert date(2026, 1, 1) == OOS_START, "OOS start must be 2026-01-01"

"""Tests for paper_trade_logger — lane sync + OOS boundary."""

from datetime import date

import pytest


class TestValidateLanes:
    """LANES in paper_trade_logger must match prop_profiles.daily_lanes."""

    def test_lanes_match_prop_profiles(self):
        """Strategy IDs in LANES must exactly match apex_50k_manual.daily_lanes."""
        from trading_app.paper_trade_logger import _validate_lanes

        # Should not raise
        _validate_lanes()

    def test_lane_count_matches(self):
        """Same number of lanes in both sources."""
        from trading_app.paper_trade_logger import LANES
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        profile = ACCOUNT_PROFILES["apex_50k_manual"]
        assert len(LANES) == len(profile.daily_lanes), (
            f"LANES has {len(LANES)} entries but daily_lanes has {len(profile.daily_lanes)}"
        )

    def test_lane_instruments_match(self):
        """Each lane's instrument should match the prop_profiles source."""
        from trading_app.paper_trade_logger import LANES
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        profile = ACCOUNT_PROFILES["apex_50k_manual"]
        profile_map = {la.strategy_id: la.instrument for la in profile.daily_lanes}
        for lane in LANES:
            assert lane.strategy_id in profile_map, f"Lane {lane.strategy_id} not in prop_profiles"
            assert lane.instrument == profile_map[lane.strategy_id], (
                f"Instrument mismatch for {lane.strategy_id}: "
                f"logger={lane.instrument}, profiles={profile_map[lane.strategy_id]}"
            )


class TestOOSBoundary:
    """OOS start date must be enforced."""

    def test_oos_start_is_2026(self):
        from trading_app.paper_trade_logger import OOS_START

        assert date(2026, 1, 1) == OOS_START, "OOS start must be 2026-01-01"


class TestLaneDef:
    """LaneDef basic properties."""

    def test_all_lanes_have_required_fields(self):
        from trading_app.paper_trade_logger import LANES

        for lane in LANES:
            assert lane.strategy_id, "Lane missing strategy_id"
            assert lane.instrument, "Lane missing instrument"
            assert lane.orb_label, "Lane missing orb_label"
            assert lane.orb_minutes > 0, f"Lane {lane.strategy_id} bad orb_minutes"
            assert lane.rr_target > 0, f"Lane {lane.strategy_id} bad rr_target"
            assert lane.entry_model in ("E1", "E2"), f"Lane {lane.strategy_id} bad entry_model"

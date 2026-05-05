"""Tests for paper_trade_logger — profile-derived lanes + OOS boundary + aperture routing."""

from datetime import date

import duckdb
import pytest

from pipeline.paths import GOLD_DB_PATH


class TestBuildLanes:
    """Lanes derived from prop_profiles must have valid structure."""

    def test_lanes_match_active_profile(self):
        """build_lanes returns lanes matching the active profile's daily_lanes count."""
        from trading_app.paper_trade_logger import build_lanes
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        for pid, p in ACCOUNT_PROFILES.items():
            if p.active and p.daily_lanes:
                lanes = build_lanes(pid)
                assert len(lanes) == len(p.daily_lanes), (
                    f"Profile {pid}: build_lanes returned {len(lanes)} but daily_lanes has {len(p.daily_lanes)}"
                )
                break

    def test_lane_strategy_ids_match_profile(self):
        """Every lane's strategy_id must come from the profile."""
        from trading_app.paper_trade_logger import build_lanes
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        for pid, p in ACCOUNT_PROFILES.items():
            if p.active and p.daily_lanes:
                lanes = build_lanes(pid)
                profile_ids = {spec.strategy_id for spec in p.daily_lanes}
                lane_ids = {lane.strategy_id for lane in lanes}
                assert lane_ids == profile_ids, f"Strategy ID mismatch in {pid}: {lane_ids ^ profile_ids}"
                break

    def test_all_lanes_have_required_fields(self):
        """Every lane must have valid trading parameters."""
        from trading_app.paper_trade_logger import build_lanes

        lanes = build_lanes()  # first active profile
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
        from trading_app.paper_trade_logger import build_lanes

        lanes = build_lanes()
        for lane in lanes:
            assert lane.filter_type in ALL_FILTERS, (
                f"Lane {lane.strategy_id} filter_type '{lane.filter_type}' not in ALL_FILTERS"
            )


class TestOOSBoundary:
    """OOS start date must be enforced."""

    def test_oos_start_is_2026(self):
        from trading_app.paper_trade_logger import OOS_START

        assert date(2026, 1, 1) == OOS_START, "OOS start must be 2026-01-01"


class TestApertureRouting:
    """_load_features and _inject_cross_asset_atrs MUST honour the lane's orb_minutes.

    Regression test for the PR #189 class bug recurrence in paper_trade_logger.py.
    daily_features carries 3 rows per (trading_day, symbol) — one per orb_minutes.
    Per-session columns (orb_NYSE_OPEN_size, etc.) carry materially different values
    across those rows because they describe ORB windows of different durations.
    Hardcoding orb_minutes=5 silently mis-scores every non-O5 lane.
    """

    @pytest.mark.skipif(not GOLD_DB_PATH.exists(), reason="gold.db not present")
    def test_load_features_returns_requested_aperture(self):
        """_load_features(orb_minutes=15) must return o15 rows, not o5 rows."""
        from trading_app.paper_trade_logger import _load_features

        with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
            # Find a (day, symbol) where o5 and o15 NYSE_OPEN size differ.
            sample = con.execute(
                """
                SELECT d5.trading_day, d5.symbol,
                       d5.orb_NYSE_OPEN_size AS s5,
                       d15.orb_NYSE_OPEN_size AS s15
                FROM daily_features d5
                JOIN daily_features d15
                  ON d5.trading_day = d15.trading_day
                 AND d5.symbol = d15.symbol
                WHERE d5.orb_minutes = 5 AND d15.orb_minutes = 15
                  AND d5.orb_NYSE_OPEN_size IS NOT NULL
                  AND d15.orb_NYSE_OPEN_size IS NOT NULL
                  AND d5.orb_NYSE_OPEN_size != d15.orb_NYSE_OPEN_size
                  AND d5.trading_day >= DATE '2026-01-01'
                LIMIT 1
                """
            ).fetchone()

            if sample is None:
                pytest.skip("No (day, symbol) with differing o5/o15 NYSE_OPEN_size in OOS — schema may have changed")

            trading_day, symbol, expected_o5, expected_o15 = sample

            features_o5 = _load_features(con, symbol, 5, since=trading_day)
            features_o15 = _load_features(con, symbol, 15, since=trading_day)

            assert trading_day in features_o5
            assert trading_day in features_o15

            # The o15 lane MUST see the o15 row's value, not the o5 row's value.
            assert features_o5[trading_day]["orb_NYSE_OPEN_size"] == pytest.approx(expected_o5)
            assert features_o15[trading_day]["orb_NYSE_OPEN_size"] == pytest.approx(expected_o15)
            assert features_o5[trading_day]["orb_NYSE_OPEN_size"] != features_o15[trading_day]["orb_NYSE_OPEN_size"]

    def test_load_features_signature_requires_orb_minutes(self):
        """_load_features must REQUIRE orb_minutes — no default that re-introduces the bug."""
        import inspect

        from trading_app.paper_trade_logger import _load_features

        sig = inspect.signature(_load_features)
        assert "orb_minutes" in sig.parameters, "orb_minutes parameter missing"
        assert sig.parameters["orb_minutes"].default is inspect.Parameter.empty, (
            "orb_minutes must NOT have a default — defaulting to 5 was the original bug"
        )

    def test_inject_cross_asset_atrs_signature_requires_orb_minutes(self):
        """_inject_cross_asset_atrs must REQUIRE orb_minutes — same reason."""
        import inspect

        from trading_app.paper_trade_logger import _inject_cross_asset_atrs

        sig = inspect.signature(_inject_cross_asset_atrs)
        assert "orb_minutes" in sig.parameters, "orb_minutes parameter missing"
        assert sig.parameters["orb_minutes"].default is inspect.Parameter.empty, "orb_minutes must NOT have a default"

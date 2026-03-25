# tests/test_trading_app/test_prop_profiles.py
"""Tests for trading_app.prop_profiles — prop firm config and data structures."""

import pytest

from trading_app.prop_profiles import (
    PropFirmSpec,
    PropFirmAccount,
    AccountProfile,
    DailyLaneSpec,
    TradingBookEntry,
    ExcludedEntry,
    TradingBook,
    PROP_FIRM_SPECS,
    ACCOUNT_TIERS,
    ACCOUNT_PROFILES,
    get_firm_spec,
    get_account_tier,
    get_profile,
    get_lane_registry,
    compute_profit_split_factor,
)


class TestPropFirmSpec:
    def test_topstep_exists(self):
        spec = get_firm_spec("topstep")
        assert spec.display_name == "TopStep"
        assert spec.dd_type == "eod_trailing"
        assert spec.auto_trading == "full"

    def test_tradeify_exists(self):
        spec = get_firm_spec("tradeify")
        assert spec.auto_trading == "full"
        assert spec.min_hold_seconds == 10

    def test_mffu_exists(self):
        spec = get_firm_spec("mffu")
        assert spec.auto_trading == "semi"

    def test_self_funded_no_firm(self):
        spec = get_firm_spec("self_funded")
        assert spec.consistency_rule is None
        assert spec.banned_instruments == frozenset()

    def test_unknown_firm_raises(self):
        with pytest.raises(KeyError):
            get_firm_spec("nonexistent")


class TestPropFirmAccount:
    def test_topstep_50k(self):
        tier = get_account_tier("topstep", 50_000)
        assert tier.max_dd == 2_000
        assert tier.max_contracts_mini == 5
        assert tier.max_contracts_micro == 50

    def test_topstep_150k(self):
        tier = get_account_tier("topstep", 150_000)
        assert tier.max_dd == 4_500
        assert tier.max_contracts_mini == 15

    def test_self_funded_50k(self):
        tier = get_account_tier("self_funded", 50_000)
        assert tier.max_dd == 5_000  # User-defined risk tolerance
        assert tier.max_contracts_micro == 500  # Effectively unlimited

    def test_unknown_tier_raises(self):
        with pytest.raises(KeyError):
            get_account_tier("topstep", 999_999)


class TestAccountProfile:
    def test_default_profiles_exist(self):
        p = get_profile("topstep_50k")
        assert p.firm == "topstep"
        assert p.account_size == 50_000
        assert p.stop_multiplier == 0.75
        assert p.copies == 5  # 5 Express accounts — MGC morning lane
        assert p.active is True

    def test_apex_manual_profile(self):
        p = get_profile("apex_50k_manual")
        assert p.firm == "apex"
        assert p.copies == 1  # Manual proof only
        assert p.active is True

    def test_tradeify_scaling_profile(self):
        p = get_profile("tradeify_50k")
        assert p.firm == "tradeify"
        assert p.copies == 5  # PRIMARY MNQ scaling lane
        assert p.active is True

    def test_self_funded_profile(self):
        p = get_profile("self_funded_50k")
        assert p.stop_multiplier == 1.0
        assert p.max_slots == 10
        assert p.active is False  # Phase 3 — not active yet

    def test_profile_copies(self):
        p = get_profile("topstep_50k")
        assert p.copies >= 1

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            get_profile("nonexistent")


class TestProfitSplitFactor:
    def test_topstep_below_threshold(self):
        """First $5K at 50% split."""
        spec = get_firm_spec("topstep")
        factor = compute_profit_split_factor(spec, cumulative_profit=0)
        assert factor == pytest.approx(0.50)

    def test_topstep_above_threshold(self):
        """After $5K at 90% split."""
        spec = get_firm_spec("topstep")
        factor = compute_profit_split_factor(spec, cumulative_profit=6000)
        assert factor == pytest.approx(0.90)

    def test_tradeify_flat_split(self):
        """Tradeify Select: flat 90/10."""
        spec = get_firm_spec("tradeify")
        factor = compute_profit_split_factor(spec, cumulative_profit=0)
        assert factor == pytest.approx(0.90)

    def test_self_funded_keeps_all(self):
        spec = get_firm_spec("self_funded")
        factor = compute_profit_split_factor(spec, cumulative_profit=0)
        assert factor == pytest.approx(1.0)


class TestTradingBook:
    def test_empty_book(self):
        book = TradingBook(profile_id="test", entries=[], excluded=[])
        assert book.total_slots == 0
        assert book.total_dd_used == 0.0

    def test_book_with_entries(self):
        entry = TradingBookEntry(
            strategy_id="MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5",
            instrument="MGC",
            orb_label="TOKYO_OPEN",
            session_time_brisbane="19:00",
            entry_model="E2",
            rr_target=2.0,
            confirm_bars=1,
            filter_type="ORB_G5",
            direction="long",
            contracts=1,
            stop_multiplier=0.75,
            effective_expr=0.18,
            sharpe_dd_ratio=1.5,
            dd_contribution=935.0,
        )
        book = TradingBook(profile_id="test", entries=[entry], excluded=[])
        assert book.total_slots == 1
        assert book.total_dd_used == 935.0


class TestDailyLaneSpecOrbCap:
    """Tests for the ORB size cap on DailyLaneSpec."""

    def test_default_max_orb_is_none(self):
        lane = DailyLaneSpec("TEST_ID", "MNQ", "NYSE_CLOSE")
        assert lane.max_orb_size_pts is None

    def test_explicit_max_orb(self):
        lane = DailyLaneSpec("TEST_ID", "MNQ", "NYSE_OPEN", max_orb_size_pts=150.0)
        assert lane.max_orb_size_pts == 150.0

    def test_nyse_open_has_cap(self):
        """NYSE_OPEN lane must have max_orb_size_pts=150.0 in the apex manual profile."""
        p = get_profile("apex_50k_manual")
        nyse_open_lanes = [l for l in p.daily_lanes if l.orb_label == "NYSE_OPEN"]
        assert len(nyse_open_lanes) == 1
        assert nyse_open_lanes[0].max_orb_size_pts == 150.0

    def test_other_lanes_no_cap(self):
        """Lanes 1-3 must have max_orb_size_pts=None (no cap)."""
        p = get_profile("apex_50k_manual")
        for lane in p.daily_lanes:
            if lane.orb_label != "NYSE_OPEN":
                assert lane.max_orb_size_pts is None, f"{lane.orb_label} should have no cap"


class TestLaneRegistryOrbCap:
    """Tests for ORB cap propagation through get_lane_registry."""

    def test_registry_has_max_orb_field(self):
        registry = get_lane_registry()
        for label, info in registry.items():
            assert "max_orb_size_pts" in info, f"{label} missing max_orb_size_pts"

    def test_nyse_open_cap_in_registry(self):
        registry = get_lane_registry()
        assert registry["NYSE_OPEN"]["max_orb_size_pts"] == 150.0

    def test_nyse_close_no_cap_in_registry(self):
        registry = get_lane_registry()
        assert registry["NYSE_CLOSE"]["max_orb_size_pts"] is None

    def test_singapore_open_no_cap_in_registry(self):
        registry = get_lane_registry()
        assert registry["SINGAPORE_OPEN"]["max_orb_size_pts"] is None

    def test_comex_settle_no_cap_in_registry(self):
        registry = get_lane_registry()
        assert registry["COMEX_SETTLE"]["max_orb_size_pts"] is None


class TestOrbCapLogic:
    """Unit tests for the ORB cap check logic (mirrors session_orchestrator gate)."""

    @staticmethod
    def _should_skip(risk_points, orb_cap):
        """Replicate the cap check from session_orchestrator._handle_event."""
        if orb_cap is not None and risk_points is not None:
            return risk_points >= orb_cap
        return False

    def test_149pt_under_cap_passes(self):
        assert not self._should_skip(149.0, 150.0)

    def test_150pt_at_cap_skipped(self):
        assert self._should_skip(150.0, 150.0)

    def test_151pt_over_cap_skipped(self):
        assert self._should_skip(151.0, 150.0)

    def test_no_cap_any_size_passes(self):
        assert not self._should_skip(999.0, None)

    def test_no_risk_points_passes(self):
        assert not self._should_skip(None, 150.0)

    def test_skip_counter_increments(self):
        """Verify the pattern: cap skip should increment a counter."""
        skips = 0
        for risk_pts in [100.0, 150.0, 200.0, 80.0, 160.0]:
            if self._should_skip(risk_pts, 150.0):
                skips += 1
        assert skips == 3  # 150, 200, 160 all >= 150

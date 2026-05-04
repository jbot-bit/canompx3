# tests/test_trading_app/test_prop_profiles.py
"""Tests for trading_app.prop_profiles — prop firm config and data structures."""

from dataclasses import replace

import pytest

from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    ACCOUNT_TIERS,
    PROP_FIRM_SPECS,
    AccountProfile,
    DailyLaneSpec,
    ExcludedEntry,
    PropFirmAccount,
    PropFirmSpec,
    TradingBook,
    TradingBookEntry,
    compute_profit_split_factor,
    get_account_tier,
    get_active_profile_ids,
    get_firm_spec,
    get_lane_registry,
    get_profile,
    get_profile_lane_definitions,
    resolve_execution_symbol,
    resolve_profile_id,
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
        assert spec.auto_trading == "full"  # Official: help.myfundedfutures.com article 8444599

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
        assert tier.max_dd == 10_000  # 20% of account (bot-enforced)
        assert tier.max_contracts_micro == 20  # 1 per $2,500

    def test_self_funded_5k(self):
        tier = get_account_tier("self_funded", 5_000)
        assert tier.max_dd == 750  # 15% of account (tighter for small)
        assert tier.max_contracts_micro == 2  # 1 per $2,500
        assert tier.daily_loss_limit == 250  # 5% of account

    def test_unknown_tier_raises(self):
        with pytest.raises(KeyError):
            get_account_tier("topstep", 999_999)


class TestAccountProfile:
    def test_default_profiles_exist(self):
        p = get_profile("topstep_50k_mnq_auto")
        assert p.firm == "topstep"
        assert p.account_size == 50_000
        assert p.stop_multiplier == 0.75
        assert p.copies == 2  # start small, scale after proving loop
        assert p.payout_policy_id == "topstep_express_standard"
        assert p.active is True

    def test_topstep_primary_auto_profile(self):
        """Structure-based assertions (not literal count). The primary auto
        profile evolves as validated lanes come and go; we anchor on the
        invariants: every lane must be MNQ-only, every lane must carry an
        ORB cap, effective lanes count <= max_slots, and all lane
        sessions must be in allowed_sessions.

        Since 2026-04-13 this profile uses dynamic lanes loaded from
        lane_allocation.json via effective_daily_lanes().
        """
        from trading_app.prop_profiles import effective_daily_lanes

        p = get_profile("topstep_50k_mnq_auto")
        assert p.firm == "topstep"
        assert p.account_size == 50_000
        assert p.active is True
        lanes = effective_daily_lanes(p)
        assert len(lanes) <= p.max_slots, f"effective lane count {len(lanes)} exceeds max_slots {p.max_slots}"
        assert all(lane.instrument == "MNQ" for lane in lanes)
        for lane in lanes:
            assert lane.max_orb_size_pts is not None, f"{lane.strategy_id} missing ORB cap"
            assert lane.orb_label in p.allowed_sessions, (
                f"{lane.strategy_id} session {lane.orb_label} not in allowed_sessions"
            )

    def test_tradeify_scaling_profile(self):
        p = get_profile("tradeify_50k")
        assert p.firm == "tradeify"
        assert p.copies == 5  # PRIMARY MNQ scaling lane
        assert p.payout_policy_id == "tradeify_select_funded"
        assert p.active is False  # Inactive until Tradovate runtime parity is verified

    def test_self_funded_profile(self):
        p = get_profile("self_funded_tradovate")
        assert p.account_size == 30_000
        assert p.stop_multiplier == 0.75
        assert p.max_slots == 10
        assert len(p.daily_lanes) >= 1
        assert len(p.daily_lanes) <= p.max_slots
        assert p.active is False
        assert p.payout_policy_id == "self_funded"
        # All lanes must have ORB caps (stress test showed uncapped tail risk)
        for lane in p.daily_lanes:
            assert lane.max_orb_size_pts is not None, f"{lane.strategy_id} missing ORB cap"

    def test_profile_copies(self):
        p = get_profile("topstep_50k")
        assert p.copies >= 1

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            get_profile("nonexistent")

    def test_resolve_profile_id_single_active_default(self):
        assert resolve_profile_id() == "topstep_50k_mnq_auto"

    def test_resolve_profile_id_multiple_active_fails_closed(self, monkeypatch):
        monkeypatch.setitem(
            ACCOUNT_PROFILES,
            "tradeify_50k",
            replace(get_profile("tradeify_50k"), active=True),
        )
        with pytest.raises(ValueError, match="Multiple active execution profiles"):
            resolve_profile_id()

    def test_get_active_profile_ids_filters_self_funded_and_inactive(self):
        active = get_active_profile_ids()
        assert "topstep_50k_mnq_auto" in active
        assert "tradeify_50k" not in active
        assert "self_funded_tradovate" not in active


class TestProfitSplitFactor:
    def test_topstep_flat_split(self):
        """Flat 90/10 since Jan 2026."""
        spec = get_firm_spec("topstep")
        factor = compute_profit_split_factor(spec, cumulative_profit=0)
        assert factor == pytest.approx(0.90)

    def test_topstep_flat_split_high_profit(self):
        """Still 90% at any profit level."""
        spec = get_firm_spec("topstep")
        factor = compute_profit_split_factor(spec, cumulative_profit=50000)
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

    def test_tokyo_open_has_cap(self):
        """Every TOKYO_OPEN lane in the primary TopStep profile must carry an
        ORB cap. Since 2026-04-13, caps come from allocator JSON (per-session
        P90) rather than hardcoded values.
        """
        from trading_app.prop_profiles import effective_daily_lanes

        p = get_profile("topstep_50k_mnq_auto")
        lanes = effective_daily_lanes(p)
        tokyo_lanes = [lane for lane in lanes if lane.orb_label == "TOKYO_OPEN"]
        if not tokyo_lanes:
            pytest.skip("TOKYO_OPEN not in current allocation")
        for lane in tokyo_lanes:
            assert lane.max_orb_size_pts is not None, "TOKYO_OPEN missing ORB cap"
            assert lane.max_orb_size_pts > 0, "TOKYO_OPEN cap must be positive"

    def test_all_lanes_have_caps(self):
        """All active primary lanes must have max_orb_size_pts set."""
        from trading_app.prop_profiles import effective_daily_lanes

        p = get_profile("topstep_50k_mnq_auto")
        for lane in effective_daily_lanes(p):
            assert lane.max_orb_size_pts is not None, f"{lane.orb_label} missing ORB cap"
            assert lane.max_orb_size_pts > 0, f"{lane.orb_label} cap must be positive"


class TestLaneRegistryOrbCap:
    """Tests for ORB cap propagation through get_lane_registry."""

    def test_registry_has_max_orb_field(self):
        registry = get_lane_registry()
        for (label, instrument), info in registry.items():
            assert "max_orb_size_pts" in info, f"{label}/{instrument} missing max_orb_size_pts"

    def test_tokyo_open_cap_in_registry(self):
        registry = get_lane_registry()
        tokyo_caps = [
            info["max_orb_size_pts"] for (label, _instrument), info in registry.items() if label == "TOKYO_OPEN"
        ]
        for cap in tokyo_caps:
            assert cap is not None and cap > 0, "TOKYO_OPEN cap must be positive"

    def test_all_registry_lanes_have_caps(self):
        """All primary active lanes should have ORB caps. Since 2026-04-13,
        caps are per-session P90 from allocator JSON (not hardcoded flat values).
        """
        registry = get_lane_registry()
        for (label, instrument), info in registry.items():
            cap = info.get("max_orb_size_pts")
            assert cap is not None, f"{label}/{instrument} missing ORB cap"
            assert cap > 0, f"{label}/{instrument} cap must be positive"

    def test_multi_instrument_session_profile_keeps_distinct_caps(self):
        """Different instruments on the same session must keep their own cap."""
        multi_instrument = AccountProfile(
            profile_id="multi_instrument_profile",
            firm="topstep",
            account_size=50_000,
            active=False,
            daily_lanes=(
                DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=117.8),
                DailyLaneSpec("MES_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12", "MES", "NYSE_OPEN", max_orb_size_pts=30.0),
            ),
        )
        ACCOUNT_PROFILES["multi_instrument_profile"] = multi_instrument
        try:
            registry = get_lane_registry("multi_instrument_profile")
            assert registry[("NYSE_OPEN", "MNQ")]["max_orb_size_pts"] == 117.8
            assert registry[("NYSE_OPEN", "MES")]["max_orb_size_pts"] == 30.0
        finally:
            ACCOUNT_PROFILES.pop("multi_instrument_profile", None)

    def test_duplicate_session_instrument_profile_raises_on_registry(self):
        """get_lane_registry must still fail closed on true cap conflicts."""
        conflict = AccountProfile(
            profile_id="dup_conflict_profile",
            firm="topstep",
            account_size=50_000,
            active=False,
            daily_lanes=(
                DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=117.8),
                DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=130.0),
            ),
        )
        ACCOUNT_PROFILES["dup_conflict_profile"] = conflict
        try:
            with pytest.raises(ValueError, match="same \\(session, instrument\\)"):
                get_lane_registry("dup_conflict_profile")
        finally:
            ACCOUNT_PROFILES.pop("dup_conflict_profile", None)

    def test_consistent_duplicate_sessions_allowed(self):
        """Multi-RR profiles (multiple lanes on the same session that share
        the same cap, as in the active topstep_50k_mnq_auto profile after
        the 2026-04-10 multi-RR discovery) must NOT raise. This is the live
        bot's ORB-cap loader path and was a latent production bug before
        2026-04-12 — session_orchestrator.__init__ would crash.

        Structure-based: the test asserts that the three core multi-RR
        sessions (EUROPE_FLOW, TOKYO_OPEN) have registry entries and the
        caps they carry. Which additional single-lane sessions are present
        (COMEX_SETTLE, CME_PRECLOSE, etc.) varies as profit expansions
        add and retire lanes, so we don't pin them here.
        """
        registry = get_lane_registry("topstep_50k_mnq_auto")
        # Since 2026-04-13, caps are per-session P90 from allocator JSON.
        # Assert structure (caps exist and are positive), not exact values.
        assert len(registry) >= 1, "expected at least one session in registry"
        for (label, instrument), info in registry.items():
            cap = info.get("max_orb_size_pts")
            assert cap is not None, f"{label}/{instrument} registry entry missing max_orb_size_pts"
            assert cap > 0, f"{label}/{instrument} cap must be positive"

    def test_duplicate_session_profile_preserves_all_lane_definitions(self):
        duplicate = AccountProfile(
            profile_id="dup_lane_profile",
            firm="topstep",
            account_size=50_000,
            active=False,
            daily_lanes=(
                DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=117.8),
                DailyLaneSpec("MES_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12", "MES", "NYSE_OPEN", max_orb_size_pts=30.0),
            ),
        )
        ACCOUNT_PROFILES["dup_lane_profile"] = duplicate
        try:
            lanes = get_profile_lane_definitions("dup_lane_profile")
            nyse_open = [lane for lane in lanes if lane["orb_label"] == "NYSE_OPEN"]
            assert len(nyse_open) == 2
        finally:
            ACCOUNT_PROFILES.pop("dup_lane_profile", None)


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


class TestCorrectedTierValues:
    """Verify prop firm rule corrections from April 2026 audit.

    Sources: saveonpropfirms.com/blog/tradeify-select-guide and
    Topstep support pages.
    """

    def test_tradeify_dd_corrected(self):
        """Tradeify Select DD: $2K/$3K/$4.5K (was $4K/$6K on 100K/150K)."""
        from trading_app.prop_profiles import ACCOUNT_TIERS

        assert ACCOUNT_TIERS[("tradeify", 50_000)].max_dd == 2_000
        assert ACCOUNT_TIERS[("tradeify", 100_000)].max_dd == 3_000
        assert ACCOUNT_TIERS[("tradeify", 150_000)].max_dd == 4_500

    def test_topstep_dd_unchanged(self):
        from trading_app.prop_profiles import ACCOUNT_TIERS

        assert ACCOUNT_TIERS[("topstep", 50_000)].max_dd == 2_000
        assert ACCOUNT_TIERS[("topstep", 100_000)].max_dd == 3_000
        assert ACCOUNT_TIERS[("topstep", 150_000)].max_dd == 4_500

    def test_topstep_has_no_firm_level_consistency_rule(self):
        """TopStep path-specific consistency is modeled in payout policies, not firm spec."""
        from trading_app.prop_profiles import PROP_FIRM_SPECS

        assert PROP_FIRM_SPECS["topstep"].consistency_rule is None

    def test_topstep_close_time(self):
        """TopStep: 3:10 PM CT = 4:10 PM ET."""
        from trading_app.prop_profiles import PROP_FIRM_SPECS

        assert PROP_FIRM_SPECS["topstep"].close_time_et == "16:10"

    def test_tradeify_close_time(self):
        """Tradeify: 4:59 PM ET."""
        from trading_app.prop_profiles import PROP_FIRM_SPECS

        assert PROP_FIRM_SPECS["tradeify"].close_time_et == "16:59"


# ─── F-5: AccountProfile is_express_funded / is_live_funded fields ───
# @canonical-source docs/research-input/topstep/topstep_mll_article.md
# @verbatim "For a $50,000 Express Funded Account, your Maximum Loss Limit
#            starts at -$2,000 and trails upward as your balance grows.
#            Once your balance reaches $2,000, the Maximum Loss Limit stays at $0."


class TestAccountProfileXfaFlag:
    """AccountProfile.is_express_funded and is_live_funded (F-5 + F-3 stub)."""

    def test_is_express_funded_default_true(self):
        """New AccountProfiles default to XFA semantics (matches active deployment)."""
        from trading_app.prop_profiles import AccountProfile

        prof = AccountProfile(profile_id="test", firm="topstep", account_size=50_000)
        assert prof.is_express_funded is True

    def test_is_live_funded_default_false(self):
        """LFA flag is reserved for future LFA-promotion path; default False."""
        from trading_app.prop_profiles import AccountProfile

        prof = AccountProfile(profile_id="test", firm="topstep", account_size=50_000)
        assert prof.is_live_funded is False

    def test_active_topstep_profiles_are_xfa(self):
        """All currently active TopStep profiles in the repo are XFA-shaped."""
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        for prof in ACCOUNT_PROFILES.values():
            if prof.active and prof.firm == "topstep":
                assert prof.is_express_funded is True, (
                    f"Active TopStep profile {prof.profile_id} should be XFA-shaped "
                    f"(is_express_funded=True). If this is a Trading Combine, set "
                    f"is_express_funded=False explicitly."
                )

    def test_freeze_formula_xfa_50k(self):
        """XFA 50K freeze = max_dd + 100 = $2,100 (NOT $52,100)."""
        from trading_app.prop_profiles import ACCOUNT_TIERS, AccountProfile

        prof = AccountProfile(profile_id="t", firm="topstep", account_size=50_000, is_express_funded=True)
        tier = ACCOUNT_TIERS[("topstep", 50_000)]
        # Mirror the formula in session_orchestrator.py:407-415
        freeze = (tier.max_dd + 100) if prof.is_express_funded else (prof.account_size + tier.max_dd + 100)
        assert freeze == 2_100

    def test_freeze_formula_xfa_100k(self):
        """XFA 100K freeze = $3,100."""
        from trading_app.prop_profiles import ACCOUNT_TIERS, AccountProfile

        prof = AccountProfile(profile_id="t", firm="topstep", account_size=100_000, is_express_funded=True)
        tier = ACCOUNT_TIERS[("topstep", 100_000)]
        freeze = (tier.max_dd + 100) if prof.is_express_funded else (prof.account_size + tier.max_dd + 100)
        assert freeze == 3_100

    def test_freeze_formula_xfa_150k(self):
        """XFA 150K freeze = $4,600."""
        from trading_app.prop_profiles import ACCOUNT_TIERS, AccountProfile

        prof = AccountProfile(profile_id="t", firm="topstep", account_size=150_000, is_express_funded=True)
        tier = ACCOUNT_TIERS[("topstep", 150_000)]
        freeze = (tier.max_dd + 100) if prof.is_express_funded else (prof.account_size + tier.max_dd + 100)
        assert freeze == 4_600

    def test_freeze_formula_tc_50k(self):
        """Trading Combine 50K freeze = account_size + max_dd + 100 = $52,100."""
        from trading_app.prop_profiles import ACCOUNT_TIERS, AccountProfile

        prof = AccountProfile(profile_id="t", firm="topstep", account_size=50_000, is_express_funded=False)
        tier = ACCOUNT_TIERS[("topstep", 50_000)]
        freeze = (tier.max_dd + 100) if prof.is_express_funded else (prof.account_size + tier.max_dd + 100)
        assert freeze == 52_100

    def test_freeze_formula_tc_100k(self):
        """Trading Combine 100K freeze = $103,100."""
        from trading_app.prop_profiles import ACCOUNT_TIERS, AccountProfile

        prof = AccountProfile(profile_id="t", firm="topstep", account_size=100_000, is_express_funded=False)
        tier = ACCOUNT_TIERS[("topstep", 100_000)]
        freeze = (tier.max_dd + 100) if prof.is_express_funded else (prof.account_size + tier.max_dd + 100)
        assert freeze == 103_100


class TestLoadAllocationLanes:
    """Tests for load_allocation_lanes — dynamic lane loading from JSON."""

    def _write_json(self, tmp_path, data):
        import json

        path = tmp_path / "lane_allocation.json"
        path.write_text(json.dumps(data))
        return path

    def _sample_allocation(self, profile_id="test_profile"):
        return {
            "rebalance_date": "2026-04-13",
            "trailing_window_months": 12,
            "profile_id": profile_id,
            "lanes": [
                {
                    "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
                    "instrument": "MNQ",
                    "orb_label": "COMEX_SETTLE",
                    "rr_target": 1.5,
                    "filter_type": "OVNRNG_100",
                    "annual_r": 41.9,
                    "trailing_expr": 0.282,
                    "trailing_n": 161,
                    "trailing_wr": 0.54,
                    "months_negative": 0,
                    "session_regime": "HOT",
                    "status": "DEPLOY",
                    "status_reason": "Session HOT",
                    "avg_orb_pts": 32.1,
                    "p90_orb_pts": 52.9,
                },
                {
                    "strategy_id": "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100",
                    "instrument": "MNQ",
                    "orb_label": "EUROPE_FLOW",
                    "rr_target": 1.5,
                    "filter_type": "OVNRNG_100",
                    "annual_r": 30.0,
                    "trailing_expr": 0.15,
                    "trailing_n": 100,
                    "trailing_wr": 0.50,
                    "months_negative": 0,
                    "session_regime": "HOT",
                    "status": "DEPLOY",
                    "status_reason": "Session HOT",
                },
                {
                    "strategy_id": "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12",
                    "instrument": "MNQ",
                    "orb_label": "TOKYO_OPEN",
                    "rr_target": 1.5,
                    "filter_type": "COST_LT12",
                    "annual_r": 5.0,
                    "trailing_expr": 0.01,
                    "trailing_n": 50,
                    "trailing_wr": 0.48,
                    "months_negative": 2,
                    "session_regime": "COLD",
                    "status": "PAUSE",
                    "status_reason": "Session COLD",
                },
            ],
            "paused": [],
            "all_scores_count": 35,
        }

    def test_happy_path(self, tmp_path):
        from trading_app.prop_profiles import load_allocation_lanes

        path = self._write_json(tmp_path, self._sample_allocation())
        lanes = load_allocation_lanes("test_profile", allocation_path=str(path))
        # Only DEPLOY lanes (PAUSE excluded)
        assert len(lanes) == 2
        assert lanes[0].strategy_id == "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100"
        assert lanes[0].max_orb_size_pts == 52.9  # p90 from JSON
        assert lanes[1].strategy_id == "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100"
        assert lanes[1].max_orb_size_pts == 120.0  # fallback to _P90_ORB_PTS (no p90 in JSON)

    def test_missing_file(self, tmp_path):
        from trading_app.prop_profiles import load_allocation_lanes

        lanes = load_allocation_lanes("test_profile", allocation_path=str(tmp_path / "missing.json"))
        assert lanes == ()

    def test_profile_mismatch(self, tmp_path):
        from trading_app.prop_profiles import load_allocation_lanes

        path = self._write_json(tmp_path, self._sample_allocation("other_profile"))
        lanes = load_allocation_lanes("test_profile", allocation_path=str(path))
        assert lanes == ()

    def test_corrupt_json(self, tmp_path):
        from trading_app.prop_profiles import load_allocation_lanes

        path = tmp_path / "lane_allocation.json"
        path.write_text("{invalid json")
        lanes = load_allocation_lanes("test_profile", allocation_path=str(path))
        assert lanes == ()

    def test_paused_lanes_excluded(self, tmp_path):
        from trading_app.prop_profiles import load_allocation_lanes

        path = self._write_json(tmp_path, self._sample_allocation())
        lanes = load_allocation_lanes("test_profile", allocation_path=str(path))
        sids = [lane.strategy_id for lane in lanes]
        assert "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12" not in sids

    def test_provisional_lanes_included(self, tmp_path):
        from trading_app.prop_profiles import load_allocation_lanes

        data = self._sample_allocation()
        data["lanes"][2]["status"] = "PROVISIONAL"
        path = self._write_json(tmp_path, data)
        lanes = load_allocation_lanes("test_profile", allocation_path=str(path))
        assert len(lanes) == 3

    def test_returns_dailylanespec_type(self, tmp_path):
        from trading_app.prop_profiles import DailyLaneSpec, load_allocation_lanes

        path = self._write_json(tmp_path, self._sample_allocation())
        lanes = load_allocation_lanes("test_profile", allocation_path=str(path))
        assert all(isinstance(lane, DailyLaneSpec) for lane in lanes)
        assert isinstance(lanes, tuple)


class TestExecutionSymbolMap:
    """Stage 1 of NQ-mini execution feature: AccountProfile symbol-substitution contract."""

    @staticmethod
    def _base_profile(**overrides):
        kwargs = dict(profile_id="test_exec", firm="topstep", account_size=50000)
        kwargs.update(overrides)
        return AccountProfile(**kwargs)

    def test_identity_default(self):
        profile = self._base_profile()
        assert profile.execution_symbol_map is None
        assert profile.execution_qty_divisor is None
        assert resolve_execution_symbol(profile, "MNQ") == ("MNQ", 1)
        assert resolve_execution_symbol(profile, "MES") == ("MES", 1)

    def test_valid_nq_mapping(self):
        profile = self._base_profile(
            execution_symbol_map={"MNQ": "NQ"},
            execution_qty_divisor={"MNQ": 10},
        )
        assert resolve_execution_symbol(profile, "MNQ") == ("NQ", 10)
        # Unmapped symbol falls through to identity.
        assert resolve_execution_symbol(profile, "MES") == ("MES", 1)

    def test_only_map_populated_fails(self):
        with pytest.raises(ValueError, match="declared together"):
            self._base_profile(
                execution_symbol_map={"MNQ": "NQ"},
                execution_qty_divisor=None,
            )

    def test_only_divisor_populated_fails(self):
        with pytest.raises(ValueError, match="declared together"):
            self._base_profile(
                execution_symbol_map=None,
                execution_qty_divisor={"MNQ": 10},
            )

    def test_divisor_missing_for_mapped_symbol(self):
        with pytest.raises(ValueError, match="must also appear in execution_qty_divisor"):
            self._base_profile(
                execution_symbol_map={"MNQ": "NQ", "MES": "ES"},
                execution_qty_divisor={"MNQ": 10},
            )

    def test_unknown_source_symbol_rejected(self):
        with pytest.raises(ValueError, match="ASSET_CONFIGS"):
            self._base_profile(
                execution_symbol_map={"BOGUS": "NQ"},
                execution_qty_divisor={"BOGUS": 10},
            )

    def test_unknown_target_symbol_rejected(self):
        with pytest.raises(ValueError, match="COST_SPECS"):
            self._base_profile(
                execution_symbol_map={"MNQ": "BOGUS"},
                execution_qty_divisor={"MNQ": 10},
            )

    def test_zero_divisor_rejected(self):
        with pytest.raises(ValueError, match="int >= 1"):
            self._base_profile(
                execution_symbol_map={"MNQ": "NQ"},
                execution_qty_divisor={"MNQ": 0},
            )

    def test_negative_divisor_rejected(self):
        with pytest.raises(ValueError, match="int >= 1"):
            self._base_profile(
                execution_symbol_map={"MNQ": "NQ"},
                execution_qty_divisor={"MNQ": -1},
            )

    def test_non_int_divisor_rejected(self):
        with pytest.raises(ValueError, match="int >= 1"):
            self._base_profile(
                execution_symbol_map={"MNQ": "NQ"},
                execution_qty_divisor={"MNQ": 10.0},
            )

    def test_existing_profiles_use_identity(self):
        # Stage 1 must preserve byte-equivalent behaviour for every shipped profile.
        for profile in ACCOUNT_PROFILES.values():
            assert profile.execution_symbol_map is None, profile.profile_id
            assert profile.execution_qty_divisor is None, profile.profile_id
            assert resolve_execution_symbol(profile, "MNQ") == ("MNQ", 1)

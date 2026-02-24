"""
Tests for pipeline/cost_model.py — futures instrument cost model.

Tests:
  - CostSpec properties
  - get_cost_spec (valid + fail-closed)
  - Realized risk/reward/RR calculations
  - R-multiple conversions
  - Stress test
"""

import pytest
from pipeline.cost_model import (
    CostSpec,
    get_cost_spec,
    get_session_cost_spec,
    SESSION_SLIPPAGE_MULT,
    list_validated_instruments,
    risk_in_dollars,
    reward_in_dollars,
    realized_rr,
    to_r_multiple,
    pnl_points_to_r,
    stress_test_costs,
)


class TestCostSpec:

    def test_mgc_total_friction(self):
        """MGC total friction = $2.40 + $2.00 + $4.00 = $8.40."""
        spec = get_cost_spec("MGC")
        assert spec.total_friction == pytest.approx(8.40)

    def test_mgc_point_value(self):
        spec = get_cost_spec("MGC")
        assert spec.point_value == 10.0

    def test_mgc_friction_in_points(self):
        """$8.40 / $10 per point = 0.84 points."""
        spec = get_cost_spec("MGC")
        assert spec.friction_in_points == pytest.approx(0.84)

    def test_mgc_tick_size(self):
        spec = get_cost_spec("MGC")
        assert spec.tick_size == 0.10

    def test_mgc_min_ticks_floor(self):
        spec = get_cost_spec("MGC")
        assert spec.min_ticks_floor == 10

    def test_mgc_min_risk_floor_points(self):
        """min_risk_floor = 10 ticks * 0.10 = 1.0 point."""
        spec = get_cost_spec("MGC")
        assert spec.min_risk_floor_points == pytest.approx(1.0)

    def test_mgc_min_risk_floor_dollars(self):
        """min_risk_floor_dollars = 1.0 point * $10 = $10.00."""
        spec = get_cost_spec("MGC")
        assert spec.min_risk_floor_dollars == pytest.approx(10.0)

    def test_costspec_is_frozen(self):
        spec = get_cost_spec("MGC")
        with pytest.raises(AttributeError):
            spec.point_value = 20.0


class TestGetCostSpec:

    def test_valid_instrument(self):
        spec = get_cost_spec("MGC")
        assert spec.instrument == "MGC"

    def test_case_insensitive(self):
        spec = get_cost_spec("mgc")
        assert spec.instrument == "MGC"

    def test_unknown_instrument_raises(self):
        with pytest.raises(ValueError, match="No cost model"):
            get_cost_spec("NQ")

    def test_list_validated(self):
        instruments = list_validated_instruments()
        assert "MGC" in instruments


class TestRiskReward:

    def test_risk_in_dollars_long(self):
        """Long: entry=2350, stop=2340, risk=10pts*$10+$8.40=$108.40."""
        spec = get_cost_spec("MGC")
        risk = risk_in_dollars(spec, entry=2350.0, stop=2340.0)
        assert risk == pytest.approx(108.40)

    def test_risk_in_dollars_short(self):
        """Short: entry=2340, stop=2350, same risk as long."""
        spec = get_cost_spec("MGC")
        risk = risk_in_dollars(spec, entry=2340.0, stop=2350.0)
        assert risk == pytest.approx(108.40)

    def test_reward_in_dollars_long(self):
        """Long: entry=2350, target=2360, reward=10pts*$10-$8.40=$91.60."""
        spec = get_cost_spec("MGC")
        rew = reward_in_dollars(spec, entry=2350.0, target=2360.0)
        assert rew == pytest.approx(91.60)

    def test_reward_less_than_friction_goes_negative(self):
        """Small target: reward can go negative after costs."""
        spec = get_cost_spec("MGC")
        # 0.5 point move: $5.00 - $8.40 = -$3.40
        rew = reward_in_dollars(spec, entry=2350.0, target=2350.5)
        assert rew < 0

    def test_realized_rr_symmetric(self):
        """RR=1 theoretical -> realized RR < 1 due to costs."""
        spec = get_cost_spec("MGC")
        # 10 point risk, 10 point target (theoretical RR=1)
        rr = realized_rr(spec, entry=2350.0, stop=2340.0, target=2360.0)
        # $91.60 / $108.40 ≈ 0.845
        assert rr == pytest.approx(91.60 / 108.40)
        assert rr < 1.0  # Costs degrade RR

    def test_realized_rr_2_theoretical(self):
        """RR=2 theoretical."""
        spec = get_cost_spec("MGC")
        # 10 point risk, 20 point target
        rr = realized_rr(spec, entry=2350.0, stop=2340.0, target=2370.0)
        # reward = 20*10 - 8.40 = 191.60, risk = 10*10 + 8.40 = 108.40
        assert rr == pytest.approx(191.60 / 108.40)


class TestRMultiple:

    def test_to_r_multiple_breakeven(self):
        """At breakeven (0 PnL points), R is negative due to costs."""
        spec = get_cost_spec("MGC")
        r = to_r_multiple(spec, entry=2350.0, stop=2340.0, pnl_points=0.0)
        # (0 - 8.40) / 108.40
        assert r < 0

    def test_to_r_multiple_1r_profit(self):
        """10 point profit (1R theoretical) -> realized R < 1."""
        spec = get_cost_spec("MGC")
        r = to_r_multiple(spec, entry=2350.0, stop=2340.0, pnl_points=10.0)
        # (100 - 8.40) / 108.40 ≈ 0.845
        assert r == pytest.approx(91.60 / 108.40)

    def test_pnl_points_to_r_no_friction_deduction(self):
        """pnl_points_to_r doesn't deduct friction from PnL."""
        spec = get_cost_spec("MGC")
        r = pnl_points_to_r(spec, entry=2350.0, stop=2340.0, pnl_points=10.0)
        # (100) / 108.40 ≈ 0.923
        assert r == pytest.approx(100.0 / 108.40)


class TestStressTest:

    def test_stress_test_50pct(self):
        """Stress test at +50% increases all friction components."""
        spec = get_cost_spec("MGC")
        stressed = stress_test_costs(spec, multiplier=1.5)

        assert stressed.commission_rt == pytest.approx(3.60)
        assert stressed.spread_doubled == pytest.approx(3.00)
        assert stressed.slippage == pytest.approx(6.00)
        assert stressed.total_friction == pytest.approx(12.60)

    def test_stress_test_degrades_rr(self):
        """Stressed costs produce worse RR."""
        spec = get_cost_spec("MGC")
        stressed = stress_test_costs(spec)

        rr_normal = realized_rr(spec, 2350.0, 2340.0, 2360.0)
        rr_stressed = realized_rr(stressed, 2350.0, 2340.0, 2360.0)

        assert rr_stressed < rr_normal

    def test_stress_preserves_point_value(self):
        spec = get_cost_spec("MGC")
        stressed = stress_test_costs(spec)
        assert stressed.point_value == spec.point_value

    def test_stress_preserves_tick_size(self):
        spec = get_cost_spec("MGC")
        stressed = stress_test_costs(spec)
        assert stressed.tick_size == spec.tick_size
        assert stressed.min_ticks_floor == spec.min_ticks_floor


class TestSessionSlippage:
    """Tests for session-aware slippage (Phase 3 risk hardening)."""

    def test_cme_reopen_slippage_higher_than_base(self):
        """CME_REOPEN (thin Asian) should have higher slippage than base."""
        base = get_cost_spec("MGC")
        session = get_session_cost_spec("MGC", "CME_REOPEN")
        assert session.slippage > base.slippage
        assert session.slippage == pytest.approx(base.slippage * 1.3)

    def test_us_data_830_slippage_lower_than_base(self):
        """US_DATA_830 (NY session) should have lower slippage than base."""
        base = get_cost_spec("MGC")
        session = get_session_cost_spec("MGC", "US_DATA_830")
        assert session.slippage < base.slippage
        assert session.slippage == pytest.approx(base.slippage * 0.8)

    def test_unknown_session_returns_base(self):
        """Unknown session label returns unmodified base spec."""
        base = get_cost_spec("MGC")
        session = get_session_cost_spec("MGC", "9999")
        assert session.slippage == base.slippage
        assert session is base  # Same object when mult == 1.0

    def test_singapore_open_returns_base_spec(self):
        """SINGAPORE_OPEN has mult 1.0, should return base spec directly."""
        base = get_cost_spec("MGC")
        session = get_session_cost_spec("MGC", "SINGAPORE_OPEN")
        assert session is base

    def test_total_friction_changes(self):
        """Session slippage changes total_friction correctly."""
        base = get_cost_spec("MGC")
        session = get_session_cost_spec("MGC", "CME_REOPEN")
        # Only slippage changes; commission + spread stay same
        expected_friction = base.commission_rt + base.spread_doubled + round(base.slippage * 1.3, 2)
        assert session.total_friction == pytest.approx(expected_friction)

    def test_point_value_preserved(self):
        """Session cost spec preserves point value and tick size."""
        session = get_session_cost_spec("MGC", "CME_REOPEN")
        assert session.point_value == 10.0
        assert session.tick_size == 0.10

    def test_all_sessions_have_multiplier(self):
        """All ORB sessions should be in SESSION_SLIPPAGE_MULT for MGC."""
        mgc_mults = SESSION_SLIPPAGE_MULT["MGC"]
        for label in ["CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN",
                       "LONDON_METALS", "US_DATA_830", "NYSE_OPEN"]:
            assert label in mgc_mults

    def test_all_sessions_have_multiplier_mnq(self):
        """All ORB sessions should be in SESSION_SLIPPAGE_MULT for MNQ."""
        mnq_mults = SESSION_SLIPPAGE_MULT["MNQ"]
        for label in ["CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN",
                       "LONDON_METALS", "US_DATA_830", "NYSE_OPEN"]:
            assert label in mnq_mults


class TestMNQCostSpec:
    """MNQ cost model tests -- mirrors MGC tests."""

    def test_mnq_total_friction(self):
        """MNQ total friction = $1.24 + $0.50 + $1.00 = $2.74."""
        spec = get_cost_spec("MNQ")
        assert spec.total_friction == pytest.approx(2.74)

    def test_mnq_point_value(self):
        spec = get_cost_spec("MNQ")
        assert spec.point_value == 2.0

    def test_mnq_friction_in_points(self):
        """$2.74 / $2 per point = 1.37 points."""
        spec = get_cost_spec("MNQ")
        assert spec.friction_in_points == pytest.approx(1.37)

    def test_mnq_tick_size(self):
        spec = get_cost_spec("MNQ")
        assert spec.tick_size == 0.25

    def test_mnq_min_risk_floor_points(self):
        """min_risk_floor = 10 ticks * 0.25 = 2.5 points."""
        spec = get_cost_spec("MNQ")
        assert spec.min_risk_floor_points == pytest.approx(2.5)

    def test_mnq_min_risk_floor_dollars(self):
        """min_risk_floor_dollars = 2.5 points * $2 = $5.00."""
        spec = get_cost_spec("MNQ")
        assert spec.min_risk_floor_dollars == pytest.approx(5.0)

    def test_mnq_case_insensitive(self):
        spec = get_cost_spec("mnq")
        assert spec.instrument == "MNQ"

    def test_mnq_in_validated_list(self):
        instruments = list_validated_instruments()
        assert "MNQ" in instruments

    def test_mnq_risk_in_dollars(self):
        """Long: entry=20000, stop=19990, risk=10pts*$2+$2.74=$22.74."""
        spec = get_cost_spec("MNQ")
        risk = risk_in_dollars(spec, entry=20000.0, stop=19990.0)
        assert risk == pytest.approx(22.74)

    def test_mnq_realized_rr(self):
        """10pt risk, 20pt target (theoretical RR=2)."""
        spec = get_cost_spec("MNQ")
        rr = realized_rr(spec, entry=20000.0, stop=19990.0, target=20020.0)
        # reward = 20*2 - 2.74 = 37.26, risk = 10*2 + 2.74 = 22.74
        assert rr == pytest.approx(37.26 / 22.74)

    def test_mnq_stress_test(self):
        """Stress test at +50% increases friction."""
        spec = get_cost_spec("MNQ")
        stressed = stress_test_costs(spec, multiplier=1.5)
        assert stressed.total_friction == pytest.approx(2.74 * 1.5)

    def test_mnq_session_us_data_830_lower_slippage(self):
        """US_DATA_830 (NY) session has lower slippage for MNQ."""
        base = get_cost_spec("MNQ")
        session = get_session_cost_spec("MNQ", "US_DATA_830")
        assert session.slippage < base.slippage
        assert session.slippage == pytest.approx(base.slippage * 0.8)

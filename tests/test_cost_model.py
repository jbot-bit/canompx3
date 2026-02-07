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

"""Tests for tight stop (Option B) feature.

Tests cover:
- apply_tight_stop simulation logic
- Strategy ID encoding with stop_multiplier
- Friction-adjusted MAE threshold correctness
- Edge cases: zero risk, missing fields, 1.0x passthrough
"""

import pytest
from trading_app.config import apply_tight_stop, STOP_MULTIPLIERS
from trading_app.strategy_discovery import make_strategy_id, parse_dst_regime, parse_stop_multiplier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeCostSpec:
    """Minimal CostSpec for testing."""
    def __init__(self, point_value=10.0, total_friction=5.74):
        self.point_value = point_value
        self.total_friction = total_friction


def _make_outcome(entry_price, stop_price, mae_r, pnl_r, outcome="win"):
    """Helper to build an outcome dict."""
    return {
        "entry_price": entry_price,
        "stop_price": stop_price,
        "mae_r": mae_r,
        "pnl_r": pnl_r,
        "outcome": outcome,
        "trading_day": "2025-01-15",
    }


# ---------------------------------------------------------------------------
# apply_tight_stop tests
# ---------------------------------------------------------------------------

class TestApplyTightStop:

    def test_passthrough_at_1x(self):
        """stop_multiplier=1.0 returns outcomes unchanged."""
        outcomes = [_make_outcome(2700.0, 2695.0, 0.5, 1.5)]
        result = apply_tight_stop(outcomes, 1.0, FakeCostSpec())
        assert result[0]["pnl_r"] == 1.5

    def test_winner_not_killed(self):
        """Winner with small MAE survives tight stop."""
        # MGC: entry=2700, stop=2695, risk_pts=5, raw_risk_d=50, risk_d=55.74
        # mae_r=0.3: max_adv_pts = 0.3 * 55.74 / 10 = 1.6722
        # threshold = 0.75 * 5 = 3.75
        # 1.6722 < 3.75 → NOT killed
        spec = FakeCostSpec(point_value=10.0, total_friction=5.74)
        outcomes = [_make_outcome(2700.0, 2695.0, 0.3, 2.0)]
        result = apply_tight_stop(outcomes, 0.75, spec)
        assert result[0]["pnl_r"] == 2.0
        assert result[0]["outcome"] == "win"

    def test_loss_killed_to_075r(self):
        """Loss with MAE past 0.75x threshold gets capped to -0.75R."""
        # entry=2700, stop=2695, risk_pts=5, raw_risk_d=50, risk_d=55.74
        # mae_r=0.87: max_adv_pts = 0.87 * 55.74 / 10 = 4.849
        # threshold = 0.75 * 5 = 3.75
        # 4.849 >= 3.75 → KILLED
        spec = FakeCostSpec(point_value=10.0, total_friction=5.74)
        outcomes = [_make_outcome(2700.0, 2695.0, 0.87, -1.0, "loss")]
        result = apply_tight_stop(outcomes, 0.75, spec)
        assert result[0]["pnl_r"] == -0.75
        assert result[0]["outcome"] == "loss"

    def test_winner_killed_becomes_loss(self):
        """Winner whose MAE exceeds threshold gets killed to -0.75R loss."""
        # A trade that was ultimately a winner but went past 0.75x first
        spec = FakeCostSpec(point_value=10.0, total_friction=5.74)
        # mae_r=0.9 → max_adv_pts = 0.9 * 55.74 / 10 = 5.017, threshold=3.75
        outcomes = [_make_outcome(2700.0, 2695.0, 0.9, 1.5, "win")]
        result = apply_tight_stop(outcomes, 0.75, spec)
        assert result[0]["pnl_r"] == -0.75
        assert result[0]["outcome"] == "loss"

    def test_mnq_high_friction(self):
        """MNQ has highest friction ratio — verify threshold math."""
        # MNQ: point_value=2.0, friction=2.74
        # entry=21000, stop=20990, risk_pts=10, raw_risk_d=20, risk_d=22.74
        # mae_r=0.5: max_adv_pts = 0.5 * 22.74 / 2 = 5.685
        # threshold = 0.75 * 10 = 7.5
        # 5.685 < 7.5 → NOT killed
        spec = FakeCostSpec(point_value=2.0, total_friction=2.74)
        outcomes = [_make_outcome(21000.0, 20990.0, 0.5, 1.0)]
        result = apply_tight_stop(outcomes, 0.75, spec)
        assert result[0]["pnl_r"] == 1.0  # survived

        # mae_r=0.8: max_adv_pts = 0.8 * 22.74 / 2 = 9.096
        # 9.096 >= 7.5 → KILLED
        outcomes2 = [_make_outcome(21000.0, 20990.0, 0.8, -1.0, "loss")]
        result2 = apply_tight_stop(outcomes2, 0.75, spec)
        assert result2[0]["pnl_r"] == -0.75

    def test_zero_risk_passthrough(self):
        """Trades with zero risk_pts pass through unchanged."""
        spec = FakeCostSpec()
        outcomes = [_make_outcome(2700.0, 2700.0, 0.5, 0.0)]
        result = apply_tight_stop(outcomes, 0.75, spec)
        assert result[0]["pnl_r"] == 0.0

    def test_missing_fields_passthrough(self):
        """Outcomes with None mae_r/entry_price/stop_price pass through."""
        spec = FakeCostSpec()
        outcomes = [{"pnl_r": -1.0, "mae_r": None, "entry_price": None,
                     "stop_price": None, "outcome": "loss"}]
        result = apply_tight_stop(outcomes, 0.75, spec)
        assert result[0]["pnl_r"] == -1.0

    def test_original_not_mutated(self):
        """apply_tight_stop must not mutate the input list."""
        spec = FakeCostSpec(point_value=10.0, total_friction=5.74)
        original = _make_outcome(2700.0, 2695.0, 0.87, -1.0, "loss")
        outcomes = [original]
        result = apply_tight_stop(outcomes, 0.75, spec)
        assert original["pnl_r"] == -1.0  # original unchanged
        assert result[0]["pnl_r"] == -0.75  # copy modified

    def test_batch_mixed(self):
        """Batch with mix of killed/surviving trades."""
        spec = FakeCostSpec(point_value=10.0, total_friction=5.74)
        outcomes = [
            _make_outcome(2700.0, 2695.0, 0.2, 2.0, "win"),   # survives
            _make_outcome(2700.0, 2695.0, 0.87, -1.0, "loss"),  # killed
            _make_outcome(2700.0, 2695.0, 0.5, 1.5, "win"),   # survives
            _make_outcome(2700.0, 2695.0, 0.95, -1.0, "loss"),  # killed
        ]
        result = apply_tight_stop(outcomes, 0.75, spec)
        assert [r["pnl_r"] for r in result] == [2.0, -0.75, 1.5, -0.75]


# ---------------------------------------------------------------------------
# Strategy ID encoding tests
# ---------------------------------------------------------------------------

class TestStrategyIdEncoding:

    def test_no_suffix_at_1x(self):
        sid = make_strategy_id("MGC", "TOKYO_OPEN", "E2", 2.5, 1, "ORB_G4")
        assert "_S0" not in sid
        assert sid == "MGC_TOKYO_OPEN_E2_RR2.5_CB1_ORB_G4"

    def test_s075_suffix(self):
        sid = make_strategy_id("MGC", "TOKYO_OPEN", "E2", 2.5, 1, "ORB_G4",
                               stop_multiplier=0.75)
        assert sid.endswith("_S075")

    def test_s075_before_dst(self):
        sid = make_strategy_id("MGC", "CME_REOPEN", "E2", 2.5, 1, "ORB_G4",
                               stop_multiplier=0.75, dst_regime="winter")
        assert "_S075_W" in sid

    def test_s075_with_o15(self):
        sid = make_strategy_id("MGC", "TOKYO_OPEN", "E2", 2.5, 1, "ORB_G4",
                               orb_minutes=15, stop_multiplier=0.75)
        assert "_O15_S075" in sid

    def test_parse_dst_not_confused_by_s075(self):
        """_S075 suffix must NOT be parsed as DST summer."""
        sid = "MGC_TOKYO_OPEN_E2_RR2.5_CB1_ORB_G4_S075"
        assert parse_dst_regime(sid) is None

    def test_parse_dst_works_with_s075_and_summer(self):
        sid = "MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G4_S075_S"
        assert parse_dst_regime(sid) == "summer"

    def test_parse_stop_multiplier_075(self):
        assert parse_stop_multiplier("MGC_TOKYO_OPEN_E2_RR2.5_CB1_ORB_G4_S075") == 0.75

    def test_parse_stop_multiplier_default(self):
        assert parse_stop_multiplier("MGC_TOKYO_OPEN_E2_RR2.5_CB1_ORB_G4") == 1.0

    def test_parse_stop_multiplier_with_dst(self):
        assert parse_stop_multiplier("MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G4_S075_W") == 0.75

    def test_parse_stop_multiplier_roundtrip(self):
        """make_strategy_id -> parse_stop_multiplier roundtrips correctly."""
        sid = make_strategy_id("MGC", "TOKYO_OPEN", "E2", 2.5, 1, "ORB_G4",
                               stop_multiplier=0.75)
        assert parse_stop_multiplier(sid) == 0.75

    def test_parse_stop_multiplier_dst_summer_not_confused(self):
        """Standard strategy with _S (summer) suffix should NOT parse as stop multiplier."""
        assert parse_stop_multiplier("MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G4_S") == 1.0


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

class TestConfigSanity:

    def test_stop_multipliers_includes_baseline(self):
        assert 1.0 in STOP_MULTIPLIERS

    def test_stop_multipliers_includes_075(self):
        assert 0.75 in STOP_MULTIPLIERS

    def test_stop_multipliers_sorted_descending(self):
        assert STOP_MULTIPLIERS == sorted(STOP_MULTIPLIERS, reverse=True)

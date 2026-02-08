"""Tests for nested/audit_outcomes._outcomes_match() (T2)."""
import pytest
from trading_app.nested.audit_outcomes import _outcomes_match


class TestOutcomesMatchExact:
    def test_both_none_outcome(self):
        a = {"outcome": None, "pnl_r": None, "entry_price": None}
        b = {"outcome": None, "pnl_r": None, "entry_price": None}
        assert _outcomes_match(a, b) is True

    def test_identical_win(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is True

    def test_identical_loss(self):
        a = {"outcome": "loss", "pnl_r": -1.0, "entry_price": 2700.0}
        b = {"outcome": "loss", "pnl_r": -1.0, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is True


class TestOutcomesMatchTolerance:
    def test_pnl_within_tolerance(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.505, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is True

    def test_pnl_exceeds_tolerance(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.52, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is False

    def test_entry_price_within_tolerance(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.005}
        assert _outcomes_match(a, b) is True

    def test_entry_price_exceeds_tolerance(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.02}
        assert _outcomes_match(a, b) is False


class TestOutcomesMatchMismatch:
    def test_different_outcome(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        b = {"outcome": "loss", "pnl_r": -1.0, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is False

    def test_one_pnl_none(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": None, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is False

    def test_one_entry_price_none(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.5, "entry_price": None}
        assert _outcomes_match(a, b) is False

    def test_both_pnl_none_matches(self):
        a = {"outcome": "win", "pnl_r": None, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": None, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is True

    def test_both_entry_price_none_matches(self):
        a = {"outcome": "win", "pnl_r": 1.5, "entry_price": None}
        b = {"outcome": "win", "pnl_r": 1.5, "entry_price": None}
        assert _outcomes_match(a, b) is True


class TestOutcomesMatchEdgeCases:
    def test_pnl_zero_vs_zero(self):
        a = {"outcome": "scratch", "pnl_r": 0.0, "entry_price": 2700.0}
        b = {"outcome": "scratch", "pnl_r": 0.0, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is True

    def test_pnl_boundary_exactly_0_01(self):
        # _outcomes_match uses abs(diff) > 0.01, so 0.01 is NOT > 0.01, it passes
        a = {"outcome": "win", "pnl_r": 1.0, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.01, "entry_price": 2700.0}
        # abs(1.01 - 1.0) = 0.01, 0.01 > 0.01 is False, so it matches
        assert _outcomes_match(a, b) is False  # Actually 0.01 triggers > 0.01 due to float

    def test_pnl_within_0_009(self):
        a = {"outcome": "win", "pnl_r": 1.0, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.009, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is True

    def test_pnl_boundary_just_over_0_01(self):
        a = {"outcome": "win", "pnl_r": 1.0, "entry_price": 2700.0}
        b = {"outcome": "win", "pnl_r": 1.011, "entry_price": 2700.0}
        assert _outcomes_match(a, b) is False

    def test_missing_keys_uses_get_default(self):
        a = {"outcome": "win"}
        b = {"outcome": "win"}
        assert _outcomes_match(a, b) is True

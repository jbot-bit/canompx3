"""Tests for nested/validator.py and nested/compare.py (T4).

These test pure logic without requiring a database.
"""
import pytest
from trading_app.strategy_validator import validate_strategy
from trading_app.nested.compare import _make_comparison_key, _load_strategies
from pipeline.cost_model import get_cost_spec
import json


COST_SPEC = get_cost_spec("MGC")


class TestValidateStrategy:
    def _make_row(self, **overrides):
        base = {
            "strategy_id": "TEST_MGC_0900_15m_E1_RR2.5_CB2_ORB_G4",
            "instrument": "MGC",
            "orb_label": "0900",
            "orb_minutes": 15,
            "entry_resolution": 5,
            "rr_target": 2.5,
            "confirm_bars": 2,
            "entry_model": "E1",
            "filter_type": "ORB_G4",
            "sample_size": 150,
            "win_rate": 0.40,
            "expectancy_r": 0.60,
            "sharpe_ratio": 0.5,
            "max_drawdown_r": -8.0,
            "yearly_results": json.dumps({
                "2022": {"avg_r": 0.1, "trades": 30},
                "2023": {"avg_r": 0.2, "trades": 35},
                "2024": {"avg_r": 0.15, "trades": 40},
                "2025": {"avg_r": 0.3, "trades": 45},
            }),
        }
        base.update(overrides)
        return base

    def test_passes_valid_strategy(self):
        status, notes, _ = validate_strategy(self._make_row(), COST_SPEC)
        assert status == "PASSED"

    def test_rejects_small_sample(self):
        status, notes, _ = validate_strategy(self._make_row(sample_size=10), COST_SPEC, min_sample=30)
        assert status == "REJECTED"
        assert "Sample size" in notes

    def test_rejects_negative_expectancy(self):
        status, notes, _ = validate_strategy(self._make_row(expectancy_r=-0.1), COST_SPEC)
        assert status == "REJECTED"
        assert "ExpR" in notes

    def test_rejects_zero_expectancy(self):
        status, notes, _ = validate_strategy(self._make_row(expectancy_r=0.0), COST_SPEC)
        assert status == "REJECTED"

    def test_exclude_years(self):
        row = self._make_row(yearly_results=json.dumps({
            "2021": {"avg_r": -0.5, "trades": 20},
            "2022": {"avg_r": 0.1, "trades": 30},
            "2023": {"avg_r": 0.2, "trades": 35},
            "2024": {"avg_r": 0.15, "trades": 40},
        }))
        # With 2021 excluded, all remaining years positive
        status, _, _ = validate_strategy(row, COST_SPEC, exclude_years={2021})
        assert status == "PASSED"

    def test_min_years_positive_pct(self):
        row = self._make_row(yearly_results=json.dumps({
            "2022": {"avg_r": 0.1, "trades": 30},
            "2023": {"avg_r": -0.1, "trades": 35},
            "2024": {"avg_r": 0.15, "trades": 40},
            "2025": {"avg_r": 0.3, "trades": 45},
        }))
        # 3/4 years positive = 75%. With min_years_positive_pct=0.8, should reject
        status, _, _ = validate_strategy(row, COST_SPEC, min_years_positive_pct=0.8)
        assert status == "REJECTED"
        # With 0.7, should pass
        status, _, _ = validate_strategy(row, COST_SPEC, min_years_positive_pct=0.7)
        assert status == "PASSED"


class TestComparisonKey:
    def test_key_structure(self):
        row = {
            "orb_label": "0900",
            "entry_model": "E1",
            "rr_target": 2.5,
            "confirm_bars": 2,
            "filter_type": "ORB_G4",
        }
        key = _make_comparison_key(row)
        assert key == ("0900", "E1", 2.5, 2, "ORB_G4")

    def test_same_strategy_different_orb_minutes_matches(self):
        baseline = {
            "orb_label": "0900", "entry_model": "E1",
            "rr_target": 2.5, "confirm_bars": 2, "filter_type": "ORB_G4",
            "orb_minutes": 5,
        }
        nested = {
            "orb_label": "0900", "entry_model": "E1",
            "rr_target": 2.5, "confirm_bars": 2, "filter_type": "ORB_G4",
            "orb_minutes": 15,
        }
        assert _make_comparison_key(baseline) == _make_comparison_key(nested)

    def test_different_filter_does_not_match(self):
        a = {
            "orb_label": "0900", "entry_model": "E1",
            "rr_target": 2.5, "confirm_bars": 2, "filter_type": "ORB_G4",
        }
        b = {
            "orb_label": "0900", "entry_model": "E1",
            "rr_target": 2.5, "confirm_bars": 2, "filter_type": "ORB_G5",
        }
        assert _make_comparison_key(a) != _make_comparison_key(b)

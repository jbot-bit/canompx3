"""
Tests for trading_app.execution_spec.
"""

import json
import pytest

from trading_app.execution_spec import ExecutionSpec


class TestExecutionSpecCreation:
    """ExecutionSpec creation and validation."""

    def test_basic_creation(self):
        spec = ExecutionSpec(confirm_bars=1)
        assert spec.confirm_bars == 1
        assert spec.order_type == "market"
        assert spec.entry_model == "E1"

    def test_all_confirm_bars_values(self):
        for cb in [1, 2, 3, 4, 5]:
            spec = ExecutionSpec(confirm_bars=cb)
            assert spec.confirm_bars == cb

    def test_invalid_confirm_bars_zero(self):
        with pytest.raises(ValueError, match="confirm_bars must be 1-5"):
            ExecutionSpec(confirm_bars=0)

    def test_invalid_confirm_bars_six(self):
        with pytest.raises(ValueError, match="confirm_bars must be 1-5"):
            ExecutionSpec(confirm_bars=6)

    def test_invalid_confirm_bars_negative(self):
        with pytest.raises(ValueError, match="confirm_bars must be 1-5"):
            ExecutionSpec(confirm_bars=-1)

    def test_valid_entry_models(self):
        for em in ["E1", "E2", "E3"]:
            spec = ExecutionSpec(confirm_bars=1, entry_model=em)
            assert spec.entry_model == em

    def test_invalid_entry_model(self):
        with pytest.raises(ValueError, match="entry_model must be E1/E2/E3"):
            ExecutionSpec(confirm_bars=1, entry_model="E4")

    def test_invalid_entry_model_legacy(self):
        with pytest.raises(ValueError, match="entry_model must be E1/E2/E3"):
            ExecutionSpec(confirm_bars=1, entry_model="E_LEGACY")

    def test_valid_order_types(self):
        for ot in ["market", "limit", "stop"]:
            spec = ExecutionSpec(confirm_bars=1, order_type=ot)
            assert spec.order_type == ot

    def test_invalid_order_type(self):
        with pytest.raises(ValueError, match="order_type must be market/limit/stop"):
            ExecutionSpec(confirm_bars=1, order_type="FOK")

    def test_negative_limit_offset(self):
        with pytest.raises(ValueError, match="limit_offset_pct must be >= 0"):
            ExecutionSpec(confirm_bars=1, limit_offset_pct=-0.01)

    def test_invalid_benchmark(self):
        with pytest.raises(ValueError, match="benchmark must be arrival/vwap/twap"):
            ExecutionSpec(confirm_bars=1, benchmark="close")

    def test_valid_benchmark(self):
        for bm in ["arrival", "vwap", "twap"]:
            spec = ExecutionSpec(confirm_bars=1, benchmark=bm)
            assert spec.benchmark == bm

    def test_frozen(self):
        spec = ExecutionSpec(confirm_bars=2)
        with pytest.raises(AttributeError):
            spec.confirm_bars = 3


class TestExecutionSpecSerialization:
    """JSON serialization and deserialization."""

    def test_to_json(self):
        spec = ExecutionSpec(confirm_bars=3)
        data = json.loads(spec.to_json())
        assert data["confirm_bars"] == 3
        assert data["order_type"] == "market"
        assert data["entry_model"] == "E1"

    def test_from_json(self):
        spec = ExecutionSpec(confirm_bars=2, order_type="limit", entry_model="E2")
        json_str = spec.to_json()
        restored = ExecutionSpec.from_json(json_str)
        assert restored.confirm_bars == 2
        assert restored.order_type == "limit"
        assert restored.entry_model == "E2"

    def test_roundtrip(self):
        spec = ExecutionSpec(confirm_bars=3, entry_model="E3", order_type="stop", benchmark="vwap")
        restored = ExecutionSpec.from_json(spec.to_json())
        assert spec == restored

    def test_str_representation(self):
        spec = ExecutionSpec(confirm_bars=3)
        assert str(spec) == "CB3_E1_MARKET"

    def test_str_with_limit(self):
        spec = ExecutionSpec(confirm_bars=1, order_type="limit")
        assert str(spec) == "CB1_E1_LIMIT"

    def test_str_with_benchmark(self):
        spec = ExecutionSpec(confirm_bars=2, benchmark="vwap")
        assert "bench=vwap" in str(spec)

    def test_str_includes_entry_model(self):
        spec = ExecutionSpec(confirm_bars=2, entry_model="E3")
        assert "E3" in str(spec)

"""Tests for research._alt_strategy_utils NaN handling."""

import json
import math

import numpy as np
import pytest

from research._alt_strategy_utils import compute_strategy_metrics
from research.cross_validate_strategies import _sanitize


class TestComputeStrategyMetricsNaN:
    """Bug 1: NaN values in pnls must be stripped before computation."""

    def test_nan_stripped(self):
        """NaN values removed, metrics computed on remaining values."""
        result = compute_strategy_metrics(np.array([1.0, float("nan"), -0.5]))
        assert result is not None
        assert result["n"] == 2
        # Verify JSON-serializable (no NaN tokens)
        serialized = json.dumps(result)
        assert "NaN" not in serialized

    def test_all_nan_returns_none(self):
        """Array of all NaN returns None."""
        result = compute_strategy_metrics(np.array([float("nan"), float("nan")]))
        assert result is None

    def test_empty_returns_none(self):
        """Empty array returns None."""
        result = compute_strategy_metrics(np.array([]))
        assert result is None

    def test_no_nan_unchanged(self):
        """Array without NaN works as before."""
        result = compute_strategy_metrics(np.array([2.0, -1.0, 1.5]))
        assert result is not None
        assert result["n"] == 3

    def test_single_valid_after_nan_strip(self):
        """Single non-NaN value after stripping produces valid result."""
        result = compute_strategy_metrics(np.array([float("nan"), 2.0, float("nan")]))
        assert result is not None
        assert result["n"] == 1
        assert result["wr"] == 1.0


class TestSanitize:
    """Bug 1: _sanitize() must convert NaN/inf to None for valid JSON."""

    def test_nan_to_none(self):
        assert _sanitize(float("nan")) is None

    def test_inf_to_none(self):
        assert _sanitize(float("inf")) is None

    def test_neg_inf_to_none(self):
        assert _sanitize(float("-inf")) is None

    def test_normal_float_unchanged(self):
        assert _sanitize(1.5) == 1.5

    def test_nested_dict(self):
        data = {"a": 1, "b": float("nan"), "c": {"d": float("inf")}}
        result = _sanitize(data)
        assert result == {"a": 1, "b": None, "c": {"d": None}}

    def test_nested_list(self):
        data = [1.0, float("nan"), [float("inf"), 2.0]]
        result = _sanitize(data)
        assert result == [1.0, None, [None, 2.0]]

    def test_string_unchanged(self):
        assert _sanitize("hello") == "hello"

    def test_none_unchanged(self):
        assert _sanitize(None) is None

    def test_full_round_trip(self):
        """Sanitized data must produce valid JSON."""
        data = {"sharpe": float("nan"), "values": [1.0, float("inf")]}
        sanitized = _sanitize(data)
        serialized = json.dumps(sanitized)
        assert "NaN" not in serialized
        assert "Infinity" not in serialized
        parsed = json.loads(serialized)
        assert parsed["sharpe"] is None

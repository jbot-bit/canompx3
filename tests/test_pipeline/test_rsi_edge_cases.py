"""Tests for RSI edge cases — avg_loss near zero after smoothing (T6)."""
import pytest
import numpy as np
from pipeline.build_daily_features import _wilders_rsi


class TestWildersRsiEdgeCases:
    def test_all_gains_returns_100(self):
        closes = np.array([100.0 + i for i in range(20)])  # monotonically increasing
        rsi = _wilders_rsi(closes, period=14)
        assert rsi == 100.0

    def test_all_losses_returns_near_zero(self):
        closes = np.array([120.0 - i for i in range(20)])  # monotonically decreasing
        rsi = _wilders_rsi(closes, period=14)
        assert rsi is not None
        assert rsi < 5.0

    def test_insufficient_data_returns_none(self):
        closes = np.array([100.0, 101.0, 102.0])
        assert _wilders_rsi(closes, period=14) is None

    def test_exactly_15_bars(self):
        closes = np.array([100.0 + i * 0.5 for i in range(15)])
        rsi = _wilders_rsi(closes, period=14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_14_bars_insufficient(self):
        closes = np.array([100.0 + i * 0.5 for i in range(14)])
        assert _wilders_rsi(closes, period=14) is None

    def test_flat_prices_returns_none_or_50(self):
        closes = np.array([100.0] * 20)  # no movement
        rsi = _wilders_rsi(closes, period=14)
        # avg_gain=0, avg_loss=0 => avg_loss < 1e-12 => returns 100.0
        # This is the "zero avg_loss" edge case
        assert rsi == 100.0

    def test_single_loss_then_all_gains(self):
        closes = np.concatenate([
            np.array([101.0, 100.0]),  # 1 loss
            np.array([100.0 + i * 0.5 for i in range(18)]),  # 18 gains
        ])
        rsi = _wilders_rsi(closes, period=14)
        assert rsi is not None
        assert rsi > 80.0  # mostly gains

    def test_alternating_gains_losses_near_50(self):
        closes = np.array([100.0 + (i % 2) for i in range(20)])
        rsi = _wilders_rsi(closes, period=14)
        assert rsi is not None
        assert 30 < rsi < 70  # should be near 50

    def test_avg_loss_smoothed_to_near_zero(self):
        """After initial period with one loss, Wilder's smoothing decays avg_loss
        toward zero. Verify no division by zero."""
        closes = np.concatenate([
            np.array([101.0, 100.0]),  # tiny loss at start
            np.array([100.0 + i for i in range(200)]),  # 200 bars of gains
        ])
        rsi = _wilders_rsi(closes, period=14)
        assert rsi is not None
        assert rsi > 99.0  # avg_loss decayed to near zero, RSI near 100

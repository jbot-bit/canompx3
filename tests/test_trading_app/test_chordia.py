"""Tests for trading_app/chordia.py — Criterion 4 t-statistic gate.

Covers:
- compute_chordia_t identity ``t = sharpe * sqrt(N)`` against hand-computed values
- chordia_threshold returns the locked thresholds
- chordia_gate verdict at exact boundary, above, below
- N < 2 raises ValueError
- Three synthetic distributions: high-variance, low-variance, noise
- Theory vs no-theory threshold differs
"""

from __future__ import annotations

import math

import pytest

from trading_app.chordia import (
    CHORDIA_T_WITH_THEORY,
    CHORDIA_T_WITHOUT_THEORY,
    chordia_gate,
    chordia_threshold,
    compute_chordia_t,
)


class TestChordiaThreshold:
    """The locked threshold lookup."""

    def test_with_theory_returns_3_00(self):
        assert chordia_threshold(has_theory=True) == 3.00

    def test_without_theory_returns_3_79(self):
        assert chordia_threshold(has_theory=False) == 3.79

    def test_threshold_constants_match(self):
        # Sanity check the module-level constants are not accidentally rebound.
        assert CHORDIA_T_WITH_THEORY == 3.00
        assert CHORDIA_T_WITHOUT_THEORY == 3.79


class TestComputeChordiaT:
    """The t-statistic identity ``t = sharpe * sqrt(N)``."""

    def test_identity_at_n_100(self):
        # sharpe=0.30, N=100 → t = 0.30 * 10 = 3.0
        assert compute_chordia_t(0.30, 100) == pytest.approx(3.0, abs=1e-9)

    def test_identity_at_n_64(self):
        # sharpe=0.50, N=64 → t = 0.50 * 8 = 4.0
        assert compute_chordia_t(0.50, 64) == pytest.approx(4.0, abs=1e-9)

    def test_identity_at_n_400(self):
        # sharpe=0.20, N=400 → t = 0.20 * 20 = 4.0
        assert compute_chordia_t(0.20, 400) == pytest.approx(4.0, abs=1e-9)

    def test_negative_sharpe_negative_t(self):
        # A losing strategy produces a negative t.
        assert compute_chordia_t(-0.30, 100) == pytest.approx(-3.0, abs=1e-9)

    def test_zero_sharpe_zero_t(self):
        assert compute_chordia_t(0.0, 100) == 0.0

    def test_sample_size_below_2_raises(self):
        with pytest.raises(ValueError, match="sample_size >= 2"):
            compute_chordia_t(0.5, 1)

    def test_sample_size_zero_raises(self):
        with pytest.raises(ValueError, match="sample_size >= 2"):
            compute_chordia_t(0.5, 0)

    def test_sample_size_2_minimum_works(self):
        # N=2 is the smallest valid sample.
        result = compute_chordia_t(1.0, 2)
        assert result == pytest.approx(math.sqrt(2), abs=1e-9)


class TestChordiaGate:
    """End-to-end gate verdict + boundary behavior."""

    def test_with_theory_at_exact_boundary_passes(self):
        # sharpe=0.30, N=100 → t=3.0 == threshold (with theory) → INCLUSIVE pass
        passed, t_stat, threshold = chordia_gate(0.30, 100, has_theory=True)
        assert passed is True
        assert t_stat == pytest.approx(3.00, abs=1e-9)
        assert threshold == 3.00

    def test_with_theory_above_boundary_passes(self):
        # sharpe=0.40, N=100 → t=4.0 > 3.00
        passed, t_stat, threshold = chordia_gate(0.40, 100, has_theory=True)
        assert passed is True
        assert t_stat == pytest.approx(4.0, abs=1e-9)
        assert threshold == 3.00

    def test_with_theory_below_boundary_fails(self):
        # sharpe=0.25, N=100 → t=2.5 < 3.00
        passed, t_stat, _threshold = chordia_gate(0.25, 100, has_theory=True)
        assert passed is False
        assert t_stat == pytest.approx(2.5, abs=1e-9)

    def test_without_theory_at_boundary_passes(self):
        # sharpe=0.379, N=100 → t=3.79 == threshold (no theory)
        passed, t_stat, threshold = chordia_gate(0.379, 100, has_theory=False)
        assert passed is True
        assert t_stat == pytest.approx(3.79, abs=1e-9)
        assert threshold == 3.79

    def test_without_theory_just_below_boundary_fails(self):
        # sharpe=0.378, N=100 → t=3.78 < 3.79
        passed, t_stat, threshold = chordia_gate(0.378, 100, has_theory=False)
        assert passed is False

    def test_threshold_changes_verdict(self):
        # A strategy with sharpe=0.32, N=100 → t=3.2:
        # - PASSES with theory (3.2 >= 3.00)
        # - FAILS without theory (3.2 < 3.79)
        passed_with, _t1, _th1 = chordia_gate(0.32, 100, has_theory=True)
        passed_without, _t2, _th2 = chordia_gate(0.32, 100, has_theory=False)
        assert passed_with is True
        assert passed_without is False


class TestSyntheticDistributions:
    """Realistic per-trade-Sharpe scenarios from the project's distribution."""

    def test_high_variance_strategy(self):
        # High-variance: many trades, low per-trade Sharpe but enough N to clear
        # the t-statistic. sharpe=0.15, N=600 → t = 0.15 * sqrt(600) ≈ 3.674
        passed_with, t_stat, _ = chordia_gate(0.15, 600, has_theory=True)
        assert passed_with is True  # 3.674 >= 3.00
        assert t_stat == pytest.approx(0.15 * math.sqrt(600), abs=1e-9)
        # Without theory, the same N is not enough.
        passed_without, _, _ = chordia_gate(0.15, 600, has_theory=False)
        assert passed_without is False  # 3.674 < 3.79

    def test_low_variance_strategy(self):
        # Low-variance: fewer trades but stronger per-trade Sharpe.
        # sharpe=0.45, N=80 → t = 0.45 * sqrt(80) ≈ 4.025
        passed, t_stat, _ = chordia_gate(0.45, 80, has_theory=True)
        assert passed is True  # 4.025 >= 3.00
        assert t_stat == pytest.approx(0.45 * math.sqrt(80), abs=1e-9)

    def test_noise_strategy(self):
        # A noise-floor strategy: sharpe=0.05, N=300 → t ≈ 0.866
        # Should fail BOTH thresholds.
        passed_with, t_stat, _ = chordia_gate(0.05, 300, has_theory=True)
        passed_without, _, _ = chordia_gate(0.05, 300, has_theory=False)
        assert passed_with is False
        assert passed_without is False
        assert t_stat < CHORDIA_T_WITH_THEORY

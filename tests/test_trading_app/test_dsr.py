"""Tests for trading_app.dsr — Deflated Sharpe Ratio."""

import math

import pytest

from trading_app.dsr import _norm_cdf, _norm_ppf, compute_dsr, compute_sr0


class TestSR0:
    """Tests for the False Strategy Theorem expected max Sharpe."""

    def test_sr0_increases_with_n(self):
        """More trials → higher expected noise max."""
        sr0_10 = compute_sr0(10, 0.05)
        sr0_100 = compute_sr0(100, 0.05)
        sr0_1000 = compute_sr0(1000, 0.05)
        assert sr0_10 < sr0_100 < sr0_1000

    def test_sr0_increases_with_variance(self):
        """Higher variance → higher expected noise max."""
        sr0_low = compute_sr0(100, 0.01)
        sr0_high = compute_sr0(100, 0.10)
        assert sr0_low < sr0_high

    def test_sr0_zero_variance(self):
        """Zero variance → SR0 = 0."""
        assert compute_sr0(100, 0.0) == 0.0

    def test_sr0_n_less_than_2(self):
        """N < 2 → SR0 = 0."""
        assert compute_sr0(1, 0.05) == 0.0

    def test_sr0_known_value(self):
        """Cross-check against manual calculation.

        N=100, V[SR]=0.05 → std=0.2236
        z1 = Phi^{-1}(0.99) ≈ 2.326
        z2 = Phi^{-1}(1 - 1/(100*e)) ≈ Phi^{-1}(0.99632) ≈ 2.675
        SR0 ≈ 0.2236 * ((1-0.5772)*2.326 + 0.5772*2.675) ≈ 0.2236 * 2.527 ≈ 0.565
        """
        sr0 = compute_sr0(100, 0.05)
        assert 0.50 < sr0 < 0.60  # approximate


class TestDSR:
    """Tests for the Deflated Sharpe Ratio."""

    def test_high_sharpe_passes(self):
        """Strategy with SR well above SR0 → DSR near 1."""
        dsr = compute_dsr(sr_hat=1.0, sr0=0.3, t_obs=500)
        assert dsr > 0.99

    def test_low_sharpe_fails(self):
        """Strategy with SR well below SR0 → DSR near 0."""
        dsr = compute_dsr(sr_hat=0.1, sr0=0.6, t_obs=500)
        assert dsr < 0.01

    def test_equal_to_sr0(self):
        """Strategy at exactly SR0 → DSR ≈ 0.5."""
        dsr = compute_dsr(sr_hat=0.5, sr0=0.5, t_obs=1000)
        assert 0.45 < dsr < 0.55

    def test_more_observations_helps(self):
        """More trades → narrower CI → stronger DSR signal."""
        dsr_few = compute_dsr(sr_hat=0.5, sr0=0.3, t_obs=50)
        dsr_many = compute_dsr(sr_hat=0.5, sr0=0.3, t_obs=500)
        assert dsr_few < dsr_many

    def test_negative_skewness_hurts(self):
        """Negative skewness (common in trading) reduces DSR."""
        dsr_zero_skew = compute_dsr(sr_hat=0.5, sr0=0.3, t_obs=200, skewness=0.0)
        dsr_neg_skew = compute_dsr(sr_hat=0.5, sr0=0.3, t_obs=200, skewness=-1.0)
        # Negative skew inflates denominator when SR > 0
        # Actually skewness term is -skew*SR, so negative skew → +SR → larger denom → lower z
        assert dsr_neg_skew < dsr_zero_skew

    def test_t_obs_1_returns_zero(self):
        """Single observation → DSR = 0."""
        assert compute_dsr(sr_hat=1.0, sr0=0.0, t_obs=1) == 0.0

    def test_returns_probability(self):
        """DSR is always in [0, 1]."""
        for sr in [-1.0, 0.0, 0.5, 1.0, 5.0]:
            dsr = compute_dsr(sr_hat=sr, sr0=0.3, t_obs=100)
            assert 0.0 <= dsr <= 1.0


class TestNormApproximations:
    """Test the no-scipy norm CDF/PPF implementations."""

    def test_cdf_at_zero(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10

    def test_cdf_at_extremes(self):
        assert _norm_cdf(-10) < 1e-15
        assert _norm_cdf(10) > 1 - 1e-15

    def test_cdf_known_values(self):
        assert abs(_norm_cdf(1.96) - 0.975) < 0.001
        assert abs(_norm_cdf(-1.96) - 0.025) < 0.001

    def test_ppf_roundtrip(self):
        """PPF(CDF(x)) ≈ x."""
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            assert abs(_norm_ppf(_norm_cdf(x)) - x) < 0.01

    def test_ppf_known_values(self):
        assert abs(_norm_ppf(0.975) - 1.96) < 0.02
        assert abs(_norm_ppf(0.5) - 0.0) < 0.01


# NOTE: The compute_dsr_gate wrapper was removed from trading_app/dsr.py
# as part of Phase 4 Stage 4.0 institutional review. The first-draft
# implementation had a semantic type confusion (``observed_n`` was passed
# trade count when the Bailey-LdP 2014 N argument is a strategy-trial
# count) and a monotonicity-direction error in its docstring (it claimed
# "never softer than actual sample" when the composition ``min(declared,
# observed)`` produces a LOOSER gate than pure observed N in realistic
# cases). Amendment 2.1 of docs/institutional/pre_registered_criteria.md
# already declares DSR to be cross-check only, so a hard-gate wrapper was
# out of scope for Stage 4.0 anyway. The existing informational DSR block
# at the bottom of strategy_validator.run_validation correctly implements
# Amendment 2.1 by computing and storing dsr_score without gating.
# Stage 4.0b will revisit DSR enforcement if and when N_eff is formally
# solved per Bailey-LdP 2014 Equation 9.

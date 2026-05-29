"""Tests for trading_app.dsr — Deflated Sharpe Ratio."""

import math

import pytest

import numpy as np

from trading_app.dsr import (
    _norm_cdf,
    _norm_ppf,
    bailey_neff_correlation,
    compute_dsr,
    compute_sr0,
    estimate_n_eff_onc,
)


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


class TestBaileyNeffCorrelation:
    """Closed-form Bailey-LdP 2014 Eq. 9: N̂ = ρ̂ + (1-ρ̂)·M."""

    def test_independence_bound(self):
        """ρ̂ → 0 ⇒ N̂ = M (every trial independent)."""
        assert bailey_neff_correlation(0.0, 100) == 100.0

    def test_full_dependence_bound(self):
        """ρ̂ → 1 ⇒ N̂ = 1 (all trials are the same bet)."""
        assert bailey_neff_correlation(1.0, 100) == 1.0

    def test_interpolates_monotonically(self):
        """Higher ρ̂ ⇒ fewer effective trials."""
        low = bailey_neff_correlation(0.2, 100)
        high = bailey_neff_correlation(0.8, 100)
        assert 1.0 < high < low < 100.0

    def test_negative_rho_floored(self):
        """Negative ρ̂ would push N̂ > M (meaningless) — floored to ρ̂=0 ⇒ N̂=M."""
        assert bailey_neff_correlation(-0.5, 50) == 50.0

    def test_clamped_to_one_minimum(self):
        """N̂ never drops below 1."""
        assert bailey_neff_correlation(1.0, 1) == 1.0
        assert bailey_neff_correlation(0.99, 2) >= 1.0

    def test_single_trial(self):
        assert bailey_neff_correlation(0.5, 1) == 1.0


def _block_correlated_matrix(n_obs, blocks, noise=0.05, seed=0):
    """Build an (n_obs × sum(blocks)) matrix where each block is a set of
    near-clones (shared latent factor + small idiosyncratic noise) and blocks
    are mutually independent. ONC should recover len(blocks) clusters."""
    rng = np.random.default_rng(seed)
    cols = []
    for size in blocks:
        factor = rng.standard_normal(n_obs)
        for _ in range(size):
            cols.append(factor + noise * rng.standard_normal(n_obs))
    return np.column_stack(cols)


class TestONC:
    """López de Prado Optimal Number of Clusters N̂ estimation."""

    def test_recovers_three_clusters(self):
        """3 blocks of clones → N̂ ≈ 3, NOT the raw column count (9)."""
        mat = _block_correlated_matrix(400, blocks=[3, 3, 3], seed=1)
        out = estimate_n_eff_onc(mat, min_overlap=30)
        assert out["m"] == 9
        assert out["method"] == "onc_silhouette"
        assert out["n_eff"] == 3
        # Heavily correlated within block → high off-diagonal mean would be
        # diluted by independent cross-block pairs; just assert it's defined.
        assert -1.0 <= out["rho_hat"] <= 1.0

    def test_neff_never_exceeds_columns(self):
        """Clustering cannot manufacture more independent trials than columns."""
        mat = _block_correlated_matrix(300, blocks=[2, 2], seed=2)
        out = estimate_n_eff_onc(mat)
        assert 1 <= out["n_eff"] <= out["m"] == 4

    def test_single_column_fallback(self):
        """M < 2 → conservative raw-count fallback."""
        out = estimate_n_eff_onc(np.random.default_rng(3).standard_normal((100, 1)))
        assert out["method"] == "fallback_raw_count"
        assert out["n_eff"] == 1
        assert out["best_silhouette"] is None

    def test_all_nan_fallback(self):
        """All-NaN matrix → no finite correlations → fallback to raw count."""
        mat = np.full((50, 4), np.nan)
        out = estimate_n_eff_onc(mat)
        assert out["method"] == "fallback_raw_count"
        assert out["n_eff"] == 4

    def test_sparse_overlap_handled(self):
        """Disjoint non-NaN regions (no pair shares min_overlap obs) → fallback,
        not a crash."""
        mat = np.full((100, 3), np.nan)
        mat[0:40, 0] = np.random.default_rng(4).standard_normal(40)
        mat[40:80, 1] = np.random.default_rng(5).standard_normal(40)
        mat[60:100, 2] = np.random.default_rng(6).standard_normal(40)
        out = estimate_n_eff_onc(mat, min_overlap=30)
        # No pair overlaps ≥30 → corr all-NaN off-diagonal → fallback.
        assert out["method"] == "fallback_raw_count"
        assert out["n_eff"] == 3

    def test_deterministic(self):
        """Same input + seed → identical N̂ (reproducibility contract)."""
        mat = _block_correlated_matrix(300, blocks=[2, 3], seed=7)
        a = estimate_n_eff_onc(mat, random_state=0)
        b = estimate_n_eff_onc(mat, random_state=0)
        assert a["n_eff"] == b["n_eff"]

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            estimate_n_eff_onc(np.array([1.0, 2.0, 3.0]))

    def test_neff_feeds_sr0_lower_than_raw(self):
        """The whole point: clustered N̂ < raw M ⇒ lower SR_0 ⇒ higher DSR.

        A correlated universe of 9 clone-ish cells (true N̂≈3) must produce a
        LOWER noise floor than treating all 9 as independent.
        """
        mat = _block_correlated_matrix(400, blocks=[3, 3, 3], seed=8)
        out = estimate_n_eff_onc(mat)
        sr0_clustered = compute_sr0(out["n_eff"], var_sr=0.05)
        sr0_raw = compute_sr0(out["m"], var_sr=0.05)
        assert sr0_clustered < sr0_raw

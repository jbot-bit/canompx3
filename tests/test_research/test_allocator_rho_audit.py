"""Tests for research.audit_allocator_rho_excluded novel helpers.

Scope: 3 helpers that are NEW in this audit (not canonical imports):
  - bootstrap_rho_ci
  - fisher_z_p_value_rho_below
  - bh_fdr_adjust

All other logic (compute_lane_scores, build_allocation, etc.) is canonical
from `trading_app.lane_allocator` — not re-tested here.
"""

from __future__ import annotations

import numpy as np

from research.audit_allocator_rho_excluded import (
    bh_fdr_adjust,
    bootstrap_rho_ci,
    fisher_z_p_value_rho_below,
)


# ─────────────────────────────────────────────────────────────────────────
# bootstrap_rho_ci
# ─────────────────────────────────────────────────────────────────────────


class TestBootstrapRhoCi:
    def test_brackets_known_positive_rho(self):
        rng = np.random.default_rng(1)
        n = 300
        x = rng.standard_normal(n)
        y = 0.7 * x + 0.3 * rng.standard_normal(n)
        # True rho ~ 0.7 / sqrt(0.7^2 + 0.3^2) ~ 0.92
        point, lo, hi = bootstrap_rho_ci(x, y, B=500, seed=42)
        assert 0.8 < point < 1.0
        assert lo < point < hi
        # CI should contain the point; width should be tight for n=300
        assert (hi - lo) < 0.15

    def test_brackets_known_near_zero_rho(self):
        rng = np.random.default_rng(2)
        n = 300
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        point, lo, hi = bootstrap_rho_ci(x, y, B=500, seed=42)
        assert -0.2 < point < 0.2
        assert lo < point < hi
        # CI contains zero
        assert lo < 0 < hi

    def test_small_sample_returns_nan_ci(self):
        # n < 10 should return (point, NaN, NaN)
        x = np.arange(5, dtype=float)
        y = np.arange(5, dtype=float)
        point, lo, hi = bootstrap_rho_ci(x, y, B=100, seed=42)
        assert np.isnan(lo)
        assert np.isnan(hi)
        # point should still be computable for n>=2
        assert abs(point - 1.0) < 1e-9

    def test_n_equals_1_returns_nan_point(self):
        x = np.array([1.0])
        y = np.array([2.0])
        point, lo, hi = bootstrap_rho_ci(x, y, B=100, seed=42)
        assert np.isnan(point)
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_zero_variance_returns_zero_point(self):
        x = np.ones(100)
        y = np.arange(100, dtype=float)
        point, lo, hi = bootstrap_rho_ci(x, y, B=100, seed=42)
        assert point == 0.0
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(200)
        y = 0.4 * x + 0.6 * rng.standard_normal(200)
        r1 = bootstrap_rho_ci(x, y, B=200, seed=99)
        r2 = bootstrap_rho_ci(x, y, B=200, seed=99)
        assert r1 == r2


# ─────────────────────────────────────────────────────────────────────────
# fisher_z_p_value_rho_below
# ─────────────────────────────────────────────────────────────────────────


class TestFisherZPValue:
    def test_rho_at_threshold_gives_p_half(self):
        """When observed rho equals threshold, p-value should be 0.5 (two-sided 50/50)."""
        p = fisher_z_p_value_rho_below(rho=0.70, n=100, threshold=0.70)
        assert abs(p - 0.5) < 1e-9

    def test_rho_well_below_threshold_gives_small_p(self):
        """Large sample, observed rho far below threshold → strong evidence rho<threshold → p small."""
        p = fisher_z_p_value_rho_below(rho=0.2, n=500, threshold=0.70)
        assert p < 0.001

    def test_rho_well_above_threshold_gives_large_p(self):
        """Observed rho above threshold → weak/no evidence rho<threshold → p large."""
        p = fisher_z_p_value_rho_below(rho=0.9, n=500, threshold=0.70)
        assert p > 0.99

    def test_small_sample_returns_nan(self):
        p = fisher_z_p_value_rho_below(rho=0.5, n=3, threshold=0.70)
        assert np.isnan(p)

    def test_rho_exactly_one_returns_nan(self):
        p = fisher_z_p_value_rho_below(rho=1.0, n=100, threshold=0.70)
        assert np.isnan(p)

    def test_monotonic_in_n(self):
        """Holding observed rho constant below threshold, larger n gives smaller p."""
        rhos = [0.5] * 3
        ns = [30, 100, 500]
        ps = [fisher_z_p_value_rho_below(r, n, 0.70) for r, n in zip(rhos, ns, strict=False)]
        assert ps[0] > ps[1] > ps[2]


# ─────────────────────────────────────────────────────────────────────────
# bh_fdr_adjust
# ─────────────────────────────────────────────────────────────────────────


class TestBhFdrAdjust:
    def test_all_significant_passes(self):
        """All p-values very small → all pass FDR."""
        pvals = [1e-6, 1e-5, 1e-4]
        result = bh_fdr_adjust(pvals, q=0.05)
        for _, passes in result:
            assert passes is True

    def test_none_significant(self):
        """All p-values large → none pass."""
        pvals = [0.9, 0.8, 0.7]
        result = bh_fdr_adjust(pvals, q=0.05)
        for _, passes in result:
            assert passes is False

    def test_bh_step_up_critical_value(self):
        """Classic BH example: m=4, q=0.05, p-values sorted = 0.005, 0.02, 0.04, 0.05.
        Critical: 0.05*1/4=0.0125, 0.05*2/4=0.025, 0.05*3/4=0.0375, 0.05*4/4=0.05
        p_(1)=0.005 ≤ 0.0125 ✓; p_(2)=0.02 ≤ 0.025 ✓; p_(3)=0.04 > 0.0375 ✗;
        p_(4)=0.05 ≤ 0.05 ✓ — but BH says largest passing rank was 2 (since p_(3) fails).
        Wait: BH rule is largest k where p_(k) <= k/m * q; ALL smaller pass.
        Here p_(4)=0.05 ≤ 0.05 so k=4 passes → ALL pass.
        """
        pvals = [0.005, 0.02, 0.04, 0.05]
        result = bh_fdr_adjust(pvals, q=0.05)
        for _, passes in result:
            assert passes is True

    def test_bh_mixed_pass(self):
        """Mixed: small p's pass, large ones don't."""
        pvals = [0.001, 0.01, 0.5, 0.9]
        result = bh_fdr_adjust(pvals, q=0.05)
        passes_list = [p for _, p in result]
        # 0.001 and 0.01 should pass (p_(1)=0.001<=0.0125, p_(2)=0.01<=0.025)
        # p_(3)=0.5 > 0.0375 fail, p_(4)=0.9 > 0.05 fail
        assert passes_list[0] is True  # 0.001
        assert passes_list[1] is True  # 0.01
        assert passes_list[2] is False  # 0.5
        assert passes_list[3] is False  # 0.9

    def test_nan_inputs_fail_but_count_preserved(self):
        """NaN p-values fail and don't inflate m."""
        pvals = [0.001, float("nan"), 0.01, float("nan")]
        result = bh_fdr_adjust(pvals, q=0.05)
        assert result[0][1] is True  # 0.001 passes with m=2 -> 0.05*1/2=0.025
        assert result[1][1] is False  # NaN auto-fail
        assert result[2][1] is True  # 0.01 <= 0.05*2/2=0.05
        assert result[3][1] is False

    def test_adjusted_p_values_monotone(self):
        """Standard BH step-up: adjusted p-values are monotone increasing (smallest p_adj for smallest raw p)."""
        pvals = [0.001, 0.01, 0.04, 0.5]
        result = bh_fdr_adjust(pvals, q=0.05)
        adj = [r[0] for r in result]
        # Adjusted p-values in order of input should be monotone if input is sorted
        assert adj[0] <= adj[1] <= adj[2] <= adj[3]

    def test_input_order_preserved(self):
        """Output list is in input order, not sorted."""
        pvals = [0.5, 0.001, 0.9, 0.01]
        result = bh_fdr_adjust(pvals, q=0.05)
        passes = [r[1] for r in result]
        # 0.001 is at index 1 in input, should pass
        # 0.01 at index 3, should pass
        # 0.5 at index 0 and 0.9 at index 2 should fail
        assert passes[1] is True
        assert passes[3] is True
        assert passes[0] is False
        assert passes[2] is False

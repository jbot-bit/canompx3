"""Unit tests for pure helpers in research_mnq_nyse_close_long_direction_locked_v1.

No DB calls — helpers are tested with synthetic inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.research_mnq_nyse_close_long_direction_locked_v1 import (
    SINGLE_YEAR_DOMINANCE_MAX,
    dollar_ev_estimate,
    era_stability,
    jaccard,
    long_short_parity,
    moving_block_bootstrap_p,
    p_from_t_two_tailed,
    single_year_dominance,
    walk_forward_efficiency,
)


class TestPTwoTailed:
    def test_t_threshold_3_yields_small_p(self):
        p = p_from_t_two_tailed(3.0, n=500)
        assert 0.0 < p < 0.01

    def test_t_zero_yields_p_near_one(self):
        p = p_from_t_two_tailed(0.0, n=500)
        assert 0.95 < p <= 1.0

    def test_nan_t_returns_nan(self):
        assert np.isnan(p_from_t_two_tailed(float("nan"), n=500))

    def test_tiny_n_returns_nan(self):
        assert np.isnan(p_from_t_two_tailed(3.0, n=1))


class TestBootstrap:
    def test_strong_positive_signal_low_p(self):
        rng = np.random.default_rng(0)
        pnl = rng.normal(loc=0.2, scale=1.0, size=500)
        p = moving_block_bootstrap_p(pnl, n_boot=2000, seed=1)
        assert p < 0.05

    def test_null_signal_p_near_0_5(self):
        rng = np.random.default_rng(0)
        pnl = rng.normal(loc=0.0, scale=1.0, size=500)
        p = moving_block_bootstrap_p(pnl, n_boot=2000, seed=1)
        assert 0.2 < p < 0.8

    def test_tiny_sample_returns_nan(self):
        pnl = np.array([0.1, -0.2, 0.3])
        assert np.isnan(moving_block_bootstrap_p(pnl, n_boot=100, block_size=5))


class TestSingleYearDominance:
    def test_even_distribution_passes(self):
        year_sum = {2019: 1.0, 2020: 1.0, 2021: 1.0, 2022: 1.0,
                    2023: 1.0, 2024: 1.0, 2025: 1.0}
        share, _ = single_year_dominance(year_sum)
        assert share == pytest.approx(1.0 / 7)
        assert share <= SINGLE_YEAR_DOMINANCE_MAX

    def test_one_year_dominates_fails(self):
        year_sum = {2019: 0.1, 2020: 10.0, 2021: 0.1, 2022: 0.1,
                    2023: 0.1, 2024: 0.1, 2025: 0.1}
        share, yr = single_year_dominance(year_sum)
        assert yr == 2020
        assert share > SINGLE_YEAR_DOMINANCE_MAX

    def test_negative_years_do_not_count(self):
        year_sum = {2019: -5.0, 2020: 1.0, 2021: 1.0}
        share, yr = single_year_dominance(year_sum)
        # Only positive contributors counted → 2020 and 2021 split
        assert yr in (2020, 2021)
        assert share == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        share, yr = single_year_dominance({})
        assert share == 0.0
        assert yr is None


class TestEraStability:
    def _make_df(self, rows):
        return pd.DataFrame(rows)

    def test_all_eras_positive_passes(self):
        df = self._make_df([{"yr": y, "pnl_r": 0.1} for y in range(2019, 2026)] * 20)
        results = era_stability(df, [(2019, 2022), (2023, 2023), (2024, 2025)])
        assert all(e["passes"] for e in results)

    def test_era_below_threshold_fails(self):
        rows = []
        for y in range(2019, 2023):
            rows.extend([{"yr": y, "pnl_r": -0.10}] * 20)  # bad era: -0.10
        for y in range(2023, 2026):
            rows.extend([{"yr": y, "pnl_r": 0.20}] * 20)
        df = self._make_df(rows)
        results = era_stability(df, [(2019, 2022), (2023, 2023), (2024, 2025)])
        assert results[0]["passes"] is False
        assert results[2]["passes"] is True

    def test_small_era_is_exempt(self):
        rows = [{"yr": 2019, "pnl_r": -5.0}] * 10  # below threshold but only 10 trades
        rows += [{"yr": y, "pnl_r": 0.1} for y in range(2023, 2026) for _ in range(50)]
        df = self._make_df(rows)
        results = era_stability(df, [(2019, 2022), (2023, 2023), (2024, 2025)])
        assert results[0]["exempt"] is True
        assert results[0]["passes"] is True


class TestWalkForward:
    def test_missing_oos_returns_insufficient(self):
        rows = [{"trading_day": pd.Timestamp("2020-01-01"), "pnl_r": 0.1}] * 100
        df = pd.DataFrame(rows)
        res = walk_forward_efficiency(df, "2024-01-01")
        assert res["n_oos"] == 0
        assert res["passes"] is False

    def test_strong_consistency_passes(self):
        rng = np.random.default_rng(0)
        is_rows = [{"trading_day": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "pnl_r": 0.15 + rng.normal(0, 0.3)} for i in range(400)]
        oos_rows = [{"trading_day": pd.Timestamp("2024-02-01") + pd.Timedelta(days=i),
                     "pnl_r": 0.15 + rng.normal(0, 0.3)} for i in range(200)]
        df = pd.DataFrame(is_rows + oos_rows)
        res = walk_forward_efficiency(df, "2024-01-01")
        assert res["wfe"] > 0.5

    def test_total_collapse_fails(self):
        rng = np.random.default_rng(0)
        is_rows = [{"trading_day": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "pnl_r": 0.2 + rng.normal(0, 0.3)} for i in range(400)]
        oos_rows = [{"trading_day": pd.Timestamp("2024-02-01") + pd.Timedelta(days=i),
                     "pnl_r": -0.2 + rng.normal(0, 0.3)} for i in range(200)]
        df = pd.DataFrame(is_rows + oos_rows)
        res = walk_forward_efficiency(df, "2024-01-01")
        assert res["passes"] is False


class TestLongShortParity:
    def test_clear_asymmetry_passes(self):
        res = long_short_parity(long_t=4.5, short_t=1.5)
        assert res["passes"] is True
        assert res["short_also_significant"] is False

    def test_both_significant_fails(self):
        res = long_short_parity(long_t=4.5, short_t=4.0)
        assert res["short_also_significant"] is True
        assert res["passes"] is False

    def test_narrow_gap_fails(self):
        res = long_short_parity(long_t=3.5, short_t=3.0)
        assert res["passes"] is False  # gap 0.5 < 1.5

    def test_nan_short_handled(self):
        res = long_short_parity(long_t=4.5, short_t=float("nan"))
        assert res["passes"] is False


class TestJaccard:
    def test_empty_sets(self):
        assert jaccard(set(), set()) == 0.0

    def test_identical(self):
        s = {1, 2, 3}
        assert jaccard(s, s) == 1.0

    def test_half_overlap(self):
        assert jaccard({1, 2, 3, 4}, {3, 4, 5, 6}) == pytest.approx(2 / 6)


class TestDollarEV:
    def test_known_values(self):
        res = dollar_ev_estimate(avg_r=0.12, n=740, years=7.0,
                                  median_risk_dollars=60.0, copies=5, contracts=3)
        # per_trade = 0.12 * 60 = 7.2
        # n_per_year = 740 / 7 ≈ 105.7
        # annual gross at scale = 105.7 * 7.2 * 5 * 3 ≈ 11,417
        assert res["per_trade_dollars"] == pytest.approx(7.2)
        assert 10000 < res["annual_gross_at_scale"] < 15000

    def test_nan_risk_returns_nan(self):
        res = dollar_ev_estimate(avg_r=0.12, n=100, years=1.0,
                                  median_risk_dollars=float("nan"))
        assert np.isnan(res["annual_gross_at_scale"])

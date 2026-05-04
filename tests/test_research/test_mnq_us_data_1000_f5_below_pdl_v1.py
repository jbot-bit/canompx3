from __future__ import annotations

import math

import pandas as pd

from research.mnq_us_data_1000_f5_below_pdl_v1 import _one_sample_t, _t0_tautology, _t7_per_year


def test_one_sample_t_returns_finite_values_for_non_constant_series() -> None:
    t_stat, p_val = _one_sample_t(pd.Series([0.2, 0.4, -0.1, 0.3]))
    assert math.isfinite(t_stat)
    assert math.isfinite(p_val)


def test_t0_tautology_uses_expected_proxy_columns() -> None:
    df = pd.DataFrame(
        {
            "feature_fire": [1, 0, 1, 0],
            "prev_day_range": [2.0, 1.0, 3.0, 1.5],
            "gap_open_points": [0.1, 0.0, 0.2, 0.1],
            "atr_20": [2.0, 2.0, 2.0, 2.0],
            "atr_20_pct": [80.0, 40.0, 90.0, 20.0],
            "overnight_range_pct": [90.0, 20.0, 85.0, 10.0],
        }
    )
    max_corr, max_filter, corrs = _t0_tautology(df)
    assert max_filter in {"pdr_r105_fire", "gap_r015_fire", "atr70_fire", "ovn80_fire"}
    assert set(corrs) == {"pdr_r105_fire", "gap_r015_fire", "atr70_fire", "ovn80_fire"}
    assert math.isfinite(max_corr)


def test_t7_per_year_uses_on_signal_mean_not_on_off_delta() -> None:
    df = pd.DataFrame(
        {
            "is_is": [True] * 10,
            "year": [2021] * 10,
            "feature_fire": [1] * 5 + [0] * 5,
            "pnl_r": [0.2, 0.1, 0.3, 0.0, 0.1, 0.5, 0.6, 0.4, 0.5, 0.5],
        }
    )
    positive_years, testable_years, year_values = _t7_per_year(df)
    assert positive_years == 1
    assert testable_years == 1
    assert math.isclose(year_values[2021], 0.14)

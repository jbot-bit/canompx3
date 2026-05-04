from __future__ import annotations

import math
from datetime import date

import pandas as pd

from research.mgc_us_data_830_to_mnq_nyse_open_context_v1 import (
    _chronology_audit,
    _is_threshold,
    _one_sample_t,
    _t0_tautology,
    _t7_per_year,
)


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


def test_is_threshold_uses_upper_quartile_on_is_only() -> None:
    df = pd.DataFrame(
        {
            "is_is": ([True] * 20) + [False],
            "source_orb_size_norm": list(range(1, 22)),
        }
    )
    threshold = _is_threshold(df)
    assert math.isclose(threshold, 15.25)


def test_chronology_audit_proves_source_precedes_target() -> None:
    df = pd.DataFrame({"trading_day": pd.to_datetime([date(2025, 1, 15), date(2025, 6, 16)])})
    audit = _chronology_audit(df)
    assert audit["all_rows_safe"] is True
    assert audit["min_gap_min"] > 0
    assert audit["n_days_checked"] == 2


def test_t7_per_year_uses_is_slice_only() -> None:
    df = pd.DataFrame(
        {
            "is_is": [True, True, True, True, True, False, False],
            "year": [2025, 2025, 2025, 2025, 2025, 2025, 2025],
            "feature_fire": [1, 1, 1, 1, 1, 1, 1],
            "pnl_r": [0.1, 0.2, 0.0, 0.3, 0.1, -10.0, -10.0],
        }
    )
    positive_years, testable_years, year_values = _t7_per_year(df)
    assert positive_years == 1
    assert testable_years == 1
    assert math.isclose(year_values[2025], 0.14)

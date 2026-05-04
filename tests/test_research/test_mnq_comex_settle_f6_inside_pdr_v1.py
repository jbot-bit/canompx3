from __future__ import annotations

import math

import pandas as pd

from research.mnq_comex_settle_f6_inside_pdr_v1 import (
    _t0_tautology,
    _t1_accounting_consistency,
    _t7_per_year_delta,
)


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


def test_t1_accounting_consistency_keeps_sign_across_resolved_and_scratch_zero() -> None:
    df = pd.DataFrame(
        {
            "is_is": [True] * 6,
            "is_scratch": [0, 0, 1, 0, 0, 1],
            "feature_fire": [1, 1, 1, 0, 0, 0],
            "pnl_r": [-0.5, -0.2, None, 0.2, 0.4, None],
            "pnl_r0": [-0.5, -0.2, 0.0, 0.2, 0.4, 0.0],
        }
    )
    resolved_delta, scratch_delta = _t1_accounting_consistency(df)
    assert resolved_delta < 0
    assert scratch_delta < 0


def test_t7_per_year_delta_uses_on_minus_off_sign() -> None:
    df = pd.DataFrame(
        {
            "is_is": [True] * 10,
            "is_scratch": [0] * 10,
            "year": [2021] * 10,
            "feature_fire": [1] * 5 + [0] * 5,
            "pnl_r": [-0.4, -0.2, -0.3, -0.1, -0.2, 0.3, 0.4, 0.2, 0.5, 0.1],
        }
    )
    negative_years, testable_years, year_delta = _t7_per_year_delta(df)
    assert negative_years == 1
    assert testable_years == 1
    assert year_delta[2021] < 0

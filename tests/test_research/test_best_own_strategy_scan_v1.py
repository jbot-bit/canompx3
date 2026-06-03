import math

import pandas as pd

from research.best_own_strategy_scan_v1 import _filter_mask, _score_returns


def test_cost_filter_uses_canonical_raw_orb_risk() -> None:
    # MNQ friction is $2.92. COST_LT10 should pass when
    # 2.92 / (orb_size * $2 + 2.92) < 10%.
    df = pd.DataFrame(
        {
            "symbol": ["MNQ", "MNQ"],
            "orb_NYSE_OPEN_size": [13.0, 14.0],
            "orb_size": [13.0, 14.0],
        }
    )

    mask = _filter_mask(df, "COST_LT10", "NYSE_OPEN")

    assert mask.tolist() == [False, True]


def test_score_returns_emits_non_normal_shape_for_dsr() -> None:
    df = pd.DataFrame(
        {
            "trading_day": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-08",
                    "2025-01-09",
                    "2025-01-10",
                    "2025-01-13",
                    "2025-01-14",
                    "2025-01-15",
                ]
            ).date,
            "pnl_r": [2.0, 1.5, 1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0, -4.0],
        }
    )

    row = _score_returns(df, "pnl_r", strategy="TEST")

    assert math.isfinite(row["skewness"])
    assert math.isfinite(row["kurtosis_excess"])
    assert abs(float(row["skewness"])) > 0.01

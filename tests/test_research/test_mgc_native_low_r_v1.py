from __future__ import annotations

import pandas as pd

from research.research_mgc_native_low_r_v1 import evaluate_variants


def test_evaluate_variants_flags_primary_survivor_with_positive_bh_row() -> None:
    trades = pd.DataFrame(
        {
            "family_id": ["NYSE_OPEN_BROAD_RR1"] * 120,
            "family_kind": ["broad"] * 120,
            "orb_label": ["NYSE_OPEN"] * 120,
            "filter_type": ["NO_FILTER"] * 120,
            "trading_day": pd.date_range("2025-01-01", periods=120, freq="D"),
            "lower_0_5_pnl_r": [0.2] * 120,
            "lower_0_75_pnl_r": [0.15] * 120,
        }
    )
    result = evaluate_variants(trades)
    assert len(result) == 2
    assert result["bh_survive"].all()
    assert result["primary_survivor"].all()


def test_evaluate_variants_does_not_promote_negative_mean() -> None:
    trades = pd.DataFrame(
        {
            "family_id": ["EUROPE_FLOW_BROAD_RR1"] * 120,
            "family_kind": ["broad"] * 120,
            "orb_label": ["EUROPE_FLOW"] * 120,
            "filter_type": ["NO_FILTER"] * 120,
            "trading_day": pd.date_range("2025-01-01", periods=120, freq="D"),
            "lower_0_5_pnl_r": [-0.1] * 120,
            "lower_0_75_pnl_r": [-0.1] * 120,
        }
    )
    result = evaluate_variants(trades)
    assert not result["primary_survivor"].any()

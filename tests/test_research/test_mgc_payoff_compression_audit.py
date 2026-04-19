from __future__ import annotations

import pandas as pd

from research.research_mgc_payoff_compression_audit import (
    FamilySpec,
    conservative_lower_target_pnl,
    passes_filter,
    target_net_r,
)


def test_passes_filter_matches_session_and_threshold() -> None:
    row = pd.Series(
        {
            "orb_label": "US_DATA_1000",
            "atr_20_pct": 72.0,
            "overnight_range": 14.0,
            "orb_US_DATA_1000_size": 5.5,
        }
    )
    assert passes_filter(row, FamilySpec("x", "US_DATA_1000", "warm", "ATR_P70")) is True
    assert passes_filter(row, FamilySpec("x", "US_DATA_1000", "warm", "OVNRNG_10")) is True
    assert passes_filter(row, FamilySpec("x", "US_DATA_1000", "warm", "ORB_G5")) is True
    assert passes_filter(row, FamilySpec("x", "NYSE_OPEN", "warm", "ATR_P70")) is False


def test_conservative_lower_target_rescues_non_ambiguous_trade_only() -> None:
    row = pd.Series(
        {
            "pnl_r": -1.0,
            "mfe_r": 0.20,
            "outcome": "loss",
            "ambiguous_bar": False,
            "entry_price": 2500.0,
            "stop_price": 2499.0,
        }
    )
    rescued = conservative_lower_target_pnl(row, 0.5)
    assert rescued is not None
    assert rescued > row["pnl_r"]


def test_conservative_lower_target_keeps_ambiguous_loss_fail_closed() -> None:
    row = pd.Series(
        {
            "pnl_r": -1.0,
            "mfe_r": 0.50,
            "outcome": "loss",
            "ambiguous_bar": True,
            "entry_price": 2500.0,
            "stop_price": 2499.0,
        }
    )
    assert conservative_lower_target_pnl(row, 0.5) == -1.0


def test_target_net_r_declines_with_smaller_target() -> None:
    low = target_net_r(2500.0, 2499.0, 0.5)
    high = target_net_r(2500.0, 2499.0, 0.75)
    assert low is not None and high is not None
    assert low < high

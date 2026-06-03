from __future__ import annotations

import pandas as pd

from research.mnq_open_book_risk_overlay_v1 import BOOK_SHAPES, DECLARED_K, OVERLAYS, _apply_realized_loss_throttle


def test_declared_k_matches_fixed_book_overlay_universe() -> None:
    assert len(BOOK_SHAPES) * len(OVERLAYS) == DECLARED_K
    assert DECLARED_K == 28


def test_current_cost_book_shape_remains_in_universe() -> None:
    assert any(shape.name == "CURRENT_RR2_COST_LT10" and shape.us_data_filter == "COST_LT10" for shape in BOOK_SHAPES)


def test_realized_loss_throttle_skips_only_after_known_loss_exit() -> None:
    day = pd.Timestamp("2025-01-02").date()
    trades = pd.DataFrame(
        [
            {
                "trading_day": day,
                "entry_ts": pd.Timestamp("2025-01-02T14:30:00Z"),
                "exit_ts": pd.Timestamp("2025-01-02T15:00:00Z"),
                "pnl_dollars_1ct": -100.0,
            },
            {
                "trading_day": day,
                "entry_ts": pd.Timestamp("2025-01-02T14:45:00Z"),
                "exit_ts": pd.Timestamp("2025-01-02T16:00:00Z"),
                "pnl_dollars_1ct": 100.0,
            },
            {
                "trading_day": day,
                "entry_ts": pd.Timestamp("2025-01-02T15:30:00Z"),
                "exit_ts": pd.Timestamp("2025-01-02T16:15:00Z"),
                "pnl_dollars_1ct": 100.0,
            },
        ]
    )

    kept = _apply_realized_loss_throttle(trades)

    assert kept["entry_ts"].tolist() == [
        pd.Timestamp("2025-01-02T14:30:00Z"),
        pd.Timestamp("2025-01-02T14:45:00Z"),
    ]

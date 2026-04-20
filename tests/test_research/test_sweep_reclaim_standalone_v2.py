"""Tests for sweep_reclaim_standalone_v2 helper semantics."""

import pandas as pd

from research.sweep_reclaim_standalone_v2 import (
    reclaim_direction,
    stop_price_from_sweep_extreme,
    sweep_extreme_from_window,
    target_price_from_rr,
)


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts_utc", "open", "high", "low", "close"]).assign(
        ts_utc=lambda df: pd.to_datetime(df["ts_utc"], utc=True)
    )


def test_reclaim_direction_matches_level_side():
    assert reclaim_direction("below") == -1
    assert reclaim_direction("above") == 1


def test_sweep_extreme_short_uses_max_high_through_reclaim():
    bars = _bars(
        [
            ("2026-04-21T00:00:00Z", 99.0, 100.8, 98.9, 100.4),
            ("2026-04-21T00:01:00Z", 100.4, 101.1, 99.7, 99.8),
        ]
    )
    assert sweep_extreme_from_window(bars, event_bar_index=0, reclaim_bar_index=1, direction=-1) == 101.1


def test_sweep_extreme_long_uses_min_low_through_reclaim():
    bars = _bars(
        [
            ("2026-04-21T00:00:00Z", 101.0, 101.1, 99.2, 99.5),
            ("2026-04-21T00:01:00Z", 99.5, 100.6, 98.7, 100.3),
        ]
    )
    assert sweep_extreme_from_window(bars, event_bar_index=0, reclaim_bar_index=1, direction=1) == 98.7


def test_stop_price_offsets_one_tick_beyond_extreme():
    assert stop_price_from_sweep_extreme(101.0, direction=-1, tick_size=0.25) == 101.25
    assert stop_price_from_sweep_extreme(99.0, direction=1, tick_size=0.25) == 98.75


def test_target_price_uses_fixed_rr():
    assert target_price_from_rr(100.0, 99.0, direction=1, rr=1.5) == 101.5
    assert target_price_from_rr(100.0, 101.0, direction=-1, rr=1.5) == 98.5

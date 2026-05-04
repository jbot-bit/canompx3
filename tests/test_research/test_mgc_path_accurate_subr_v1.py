from __future__ import annotations

import pandas as pd

from research.research_mgc_path_accurate_subr_v1 import rebuild_outcome


def _row(**overrides):
    base = {
        "entry_ts": pd.Timestamp("2025-01-02 00:00:00+00:00"),
        "entry_price": 2500.0,
        "stop_price": 2499.0,
        "target_r": 0.5,
        "trading_day": pd.Timestamp("2025-01-02").date(),
        "rr1_pnl_r": -1.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_rebuild_outcome_fill_bar_ambiguous_fails_closed() -> None:
    bars = pd.DataFrame(
        {
            "ts_utc": [pd.Timestamp("2025-01-02 00:00:00+00:00")],
            "high": [2500.6],
            "low": [2498.8],
        }
    )
    out = rebuild_outcome(bars, _row())
    assert out["path_outcome"] == "loss"
    assert out["path_pnl_r"] == -1.0
    assert out["path_ambiguous_bar"] is True


def test_rebuild_outcome_post_entry_target_win() -> None:
    bars = pd.DataFrame(
        {
            "ts_utc": [
                pd.Timestamp("2025-01-02 00:00:00+00:00"),
                pd.Timestamp("2025-01-02 00:01:00+00:00"),
            ],
            "high": [2500.2, 2501.1],
            "low": [2499.8, 2500.0],
        }
    )
    out = rebuild_outcome(bars, _row(stop_price=2498.0))
    assert out["path_outcome"] == "win"
    assert out["path_pnl_r"] > 0
    assert out["path_ambiguous_bar"] is False


def test_rebuild_outcome_scratch_when_neither_hit() -> None:
    bars = pd.DataFrame(
        {
            "ts_utc": [
                pd.Timestamp("2025-01-02 00:00:00+00:00"),
                pd.Timestamp("2025-01-02 00:01:00+00:00"),
                pd.Timestamp("2025-01-02 00:02:00+00:00"),
            ],
            "high": [2500.2, 2500.3, 2500.35],
            "low": [2499.8, 2499.7, 2499.65],
        }
    )
    out = rebuild_outcome(bars, _row(target_r=1.0))
    assert out["path_outcome"] == "scratch"
    assert out["path_pnl_r"] is None

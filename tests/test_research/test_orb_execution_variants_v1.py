from __future__ import annotations

import math

import pandas as pd
import pytest

from pipeline.cost_model import get_cost_spec, to_r_multiple
from research.orb_execution_variants_v1 import (
    E2_LOOKAHEAD_BANNED_PATTERNS,
    ReentryConfig,
    TradePathResult,
    bh_q_values,
    infer_direction,
    reject_e2_lookahead_columns,
    same_direction_reentry_policy_pnl,
    simulate_trade_path,
)


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_utc": pd.Timestamp(ts, tz="UTC"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
            for ts, open_, high, low, close in rows
        ]
    )


def test_reject_e2_lookahead_columns_blocks_known_bad_predictors() -> None:
    with pytest.raises(ValueError, match="E2 lookahead"):
        reject_e2_lookahead_columns(
            [
                "orb_NYSE_OPEN_size",
                "orb_NYSE_OPEN_break_ts",
                "rel_vol_NYSE_OPEN",
                "pnl_r",
            ]
        )


def test_reject_e2_lookahead_columns_allows_pre_entry_safe_fields() -> None:
    reject_e2_lookahead_columns(["orb_NYSE_OPEN_size", "orb_NYSE_OPEN_volume", "atr_20_pct"])
    assert E2_LOOKAHEAD_BANNED_PATTERNS


def test_simulate_trade_path_uses_conservative_ambiguous_bar_loss() -> None:
    bars = _bars(
        [
            ("2025-01-02T14:36:00Z", 101.0, 112.0, 89.0, 100.0),
            ("2025-01-02T14:37:00Z", 100.0, 102.0, 99.0, 101.0),
        ]
    )

    result = simulate_trade_path(
        bars,
        entry_ts=pd.Timestamp("2025-01-02T14:36:00Z", tz="UTC"),
        entry_price=101.0,
        stop_price=90.0,
        target_price=111.0,
        direction="long",
        cost_spec=get_cost_spec("MNQ"),
    )

    assert result == TradePathResult(outcome="loss", pnl_r=-1.0, exit_ts=pd.Timestamp("2025-01-02T14:36:00Z", tz="UTC"))


def test_simulate_trade_path_books_scratch_eod_mtm_not_null() -> None:
    spec = get_cost_spec("MNQ")
    bars = _bars(
        [
            ("2025-01-02T14:36:00Z", 101.0, 104.0, 100.5, 102.0),
            ("2025-01-02T14:37:00Z", 102.0, 105.0, 101.0, 104.0),
        ]
    )

    result = simulate_trade_path(
        bars,
        entry_ts=pd.Timestamp("2025-01-02T14:36:00Z", tz="UTC"),
        entry_price=101.0,
        stop_price=90.0,
        target_price=120.0,
        direction="long",
        cost_spec=spec,
    )

    assert result.outcome == "scratch"
    assert result.exit_ts == pd.Timestamp("2025-01-02T14:37:00Z", tz="UTC")
    assert math.isclose(result.pnl_r, round(to_r_multiple(spec, 101.0, 90.0, 3.0), 4))


def test_same_direction_reentry_policy_includes_initial_stop_loss() -> None:
    spec = get_cost_spec("MNQ")
    bars = _bars(
        [
            ("2025-01-02T14:36:00Z", 100.0, 101.0, 89.0, 90.0),
            ("2025-01-02T14:37:00Z", 91.0, 102.0, 91.0, 101.0),
            ("2025-01-02T14:38:00Z", 101.0, 112.0, 100.0, 111.0),
        ]
    )
    parent = pd.Series(
        {
            "pnl_r": -1.0,
            "outcome": "loss",
            "entry_price": 101.0,
            "stop_price": 90.0,
            "exit_ts": pd.Timestamp("2025-01-02T14:36:00Z", tz="UTC"),
            "orb_high": 100.0,
            "orb_low": 90.0,
            "rr_target": 1.0,
        }
    )

    policy, reentry = same_direction_reentry_policy_pnl(
        parent,
        bars,
        ReentryConfig(wait_bars=0, size=1.0, max_reentries=1),
        cost_spec=spec,
    )

    assert reentry is not None
    assert policy == pytest.approx(-1.0 + reentry.pnl_r)
    assert reentry.pnl_r > 0.0


def test_non_loss_parent_is_unchanged_by_reentry_rule() -> None:
    parent = pd.Series(
        {
            "pnl_r": 1.2,
            "outcome": "win",
            "entry_price": 101.0,
            "stop_price": 90.0,
            "exit_ts": pd.Timestamp("2025-01-02T14:36:00Z", tz="UTC"),
            "orb_high": 100.0,
            "orb_low": 90.0,
            "rr_target": 1.0,
        }
    )
    policy, reentry = same_direction_reentry_policy_pnl(
        parent,
        _bars([]),
        ReentryConfig(wait_bars=0, size=1.0, max_reentries=1),
        cost_spec=get_cost_spec("MNQ"),
    )

    assert policy == 1.2
    assert reentry is None


def test_infer_direction_from_entry_and_stop() -> None:
    assert infer_direction(entry_price=101.0, stop_price=90.0) == "long"
    assert infer_direction(entry_price=89.0, stop_price=100.0) == "short"


def test_bh_q_values_are_monotone() -> None:
    q_map = bh_q_values(
        [
            ("h2", 0.04),
            ("h1", 0.01),
            ("h3", 0.03),
        ]
    )
    assert math.isclose(q_map["h1"], 0.03)
    assert math.isclose(q_map["h3"], 0.04)
    assert math.isclose(q_map["h2"], 0.04)

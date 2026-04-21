from __future__ import annotations

import pandas as pd

from trading_app.meta_labeling.profile_monotonic import (
    LaneAllocatorSpec,
    LaneFeatureSpec,
    apply_lane_allocator,
    fit_lane_allocator,
    translate_weight_to_contracts,
)


def test_translate_weight_to_contracts() -> None:
    assert translate_weight_to_contracts(0.5) == 0
    assert translate_weight_to_contracts(0.75) == 1
    assert translate_weight_to_contracts(1.49) == 1
    assert translate_weight_to_contracts(1.50) == 2


def test_fit_and_apply_lane_allocator() -> None:
    frame = pd.DataFrame(
        {
            "trading_day": pd.date_range("2024-01-01", periods=120, freq="D"),
            "pnl_r": [0.5 if i % 2 == 0 else -0.25 for i in range(120)],
            "feature_a": [float(i) for i in range(120)],
            "feature_b": [float(120 - i) for i in range(120)],
        }
    )
    spec = LaneAllocatorSpec(
        strategy_id="TEST",
        orb_label="TOKYO_OPEN",
        features=(
            LaneFeatureSpec("feature_a", "high_better"),
            LaneFeatureSpec("feature_b", "low_better"),
        ),
    )

    allocator = fit_lane_allocator(frame, spec)
    assert allocator.fallback_reason is None
    assert allocator.train_rows == 120
    assert len(allocator.desired_multipliers) == 5

    scored = apply_lane_allocator(allocator, frame.tail(10).reset_index(drop=True))
    assert set(scored["contracts"].unique()).issubset({0, 1, 2})
    assert scored["desired_weight"].between(0.5, 1.5).all()


def test_fit_lane_allocator_fails_closed_on_small_sample() -> None:
    frame = pd.DataFrame(
        {
            "trading_day": pd.date_range("2024-01-01", periods=20, freq="D"),
            "pnl_r": [0.1] * 20,
            "feature_a": [float(i) for i in range(20)],
        }
    )
    spec = LaneAllocatorSpec(
        strategy_id="TEST",
        orb_label="TOKYO_OPEN",
        features=(LaneFeatureSpec("feature_a", "high_better"),),
        min_train_trades=100,
    )

    allocator = fit_lane_allocator(frame, spec)
    assert allocator.fallback_reason == "insufficient_complete_train_rows:20"

    scored = apply_lane_allocator(allocator, frame)
    assert (scored["contracts"] == 1).all()
    assert (scored["desired_weight"] == 1.0).all()

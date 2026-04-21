from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd

from trading_app.meta_labeling.dataset import DatasetBundle, MetaLabelFamilySpec
from trading_app.meta_labeling.feature_contract import MetaLabelFeatureContract
from trading_app.meta_labeling.monotonic import (
    TOKYO_OPEN_MONOTONIC_V1,
    apply_monotonic_allocator,
    fit_monotonic_allocator,
    run_monotonic_walkforward,
)


def _make_frame() -> pd.DataFrame:
    rows = []
    for year, base in [(2021, 1.0), (2022, 2.0), (2023, 3.0), (2024, 4.0), (2025, 5.0)]:
        for offset in range(12):
            strength = base + offset / 20.0
            pnl_r = -0.10 + strength * 0.08
            rows.append(
                {
                    "trading_day": date(year, 1 + (offset % 6), 1 + (offset % 20)),
                    "pnl_r": pnl_r,
                    "target": int(pnl_r > 0),
                    "prev_day_range": 8.0 * strength,
                    "atr_20": 4.0,
                    "atr_vel_ratio": 0.8 + strength / 5.0,
                }
            )
    return pd.DataFrame(rows)


def _make_bundle() -> DatasetBundle:
    full_frame = _make_frame()
    forward_rows = []
    for idx in range(6):
        strength = 6.0 + idx / 10.0
        pnl_r = -0.10 + strength * 0.08
        forward_rows.append(
            {
                "trading_day": date(2026, 1, 1 + idx),
                "pnl_r": pnl_r,
                "target": int(pnl_r > 0),
                "prev_day_range": 8.0 * strength,
                "atr_20": 4.0,
                "atr_vel_ratio": 0.8 + strength / 5.0,
            }
        )
    forward_frame = pd.DataFrame(forward_rows)
    family = MetaLabelFamilySpec(
        symbol="MNQ",
        orb_label="TOKYO_OPEN",
        entry_model="E2",
        rr_target=1.5,
        confirm_bars=1,
        direction="long",
    )
    contract = MetaLabelFeatureContract(
        target_session="TOKYO_OPEN",
        numeric_features=("atr_20", "atr_vel_ratio", "prev_day_range"),
        categorical_features=(),
    )
    return DatasetBundle(
        family=family,
        feature_contract=contract,
        full_frame=pd.concat([full_frame, forward_frame], ignore_index=True),
        development_frame=full_frame.reset_index(drop=True),
        forward_frame=forward_frame.reset_index(drop=True),
    )


def test_fit_uses_train_only_quantile_edges():
    train = _make_frame().iloc[:24].reset_index(drop=True)
    allocator = fit_monotonic_allocator(train, config=TOKYO_OPEN_MONOTONIC_V1)

    expected_edges = pd.qcut(
        train["prev_day_range"] / train["atr_20"], q=5, retbins=True, duplicates="drop"
    )[1]
    observed_edges = allocator.feature_edges["pdr_atr_ratio"]
    assert tuple(round(x, 10) for x in observed_edges) == tuple(round(float(x), 10) for x in expected_edges)


def test_scalar_map_is_monotonic_non_decreasing():
    train = _make_frame().reset_index(drop=True)
    allocator = fit_monotonic_allocator(train, config=TOKYO_OPEN_MONOTONIC_V1)
    scalars = [stat.size_scalar for stat in allocator.bucket_stats]
    assert scalars == sorted(scalars)


def test_apply_allocator_scores_and_sizes_rows():
    train = _make_frame().iloc[:36].reset_index(drop=True)
    test = _make_frame().iloc[36:48].reset_index(drop=True)
    allocator = fit_monotonic_allocator(train, config=TOKYO_OPEN_MONOTONIC_V1)
    scored = apply_monotonic_allocator(allocator, test)

    assert "monotonic_score" in scored.columns
    assert "score_bucket" in scored.columns
    assert "size_scalar" in scored.columns
    assert "sized_pnl_r" in scored.columns
    assert scored["size_scalar"].between(0.0, 2.0).all()


def test_walkforward_keeps_2026_as_forward_monitor_only():
    bundle = _make_bundle()
    config = replace(TOKYO_OPEN_MONOTONIC_V1, min_train_years=3)
    report = run_monotonic_walkforward(bundle, config=config)

    test_years = [fold["test_year"] for fold in report["development_walkforward"]["folds"]]
    assert test_years == [2024, 2025]
    assert report["forward_monitor"]["rows"] == len(bundle.forward_frame)
    assert report["forward_monitor"]["status"] == "INFORMATIONAL_ONLY"

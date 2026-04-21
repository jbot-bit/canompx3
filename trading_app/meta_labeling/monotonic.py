"""Simple monotonic allocator baseline for clean-room meta-label research."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading_app.meta_labeling.dataset import (
    DatasetBundle,
    PHASE1_V1_FAMILY,
    build_family_dataset,
)


@dataclass(frozen=True)
class MonotonicFeatureSpec:
    """Locked feature transform for the simple scorecard baseline."""

    name: str
    transform: str
    source_columns: tuple[str, ...]
    direction: str = "high_better"


@dataclass(frozen=True)
class MonotonicAllocatorConfig:
    """Pre-registered config for a monotonic scorecard allocator."""

    name: str
    feature_specs: tuple[MonotonicFeatureSpec, ...]
    feature_bin_count: int = 5
    score_bucket_count: int = 5
    max_scalar: float = 2.0
    min_scalar: float = 0.0
    min_train_years: int = 3


@dataclass(frozen=True)
class ScoreBucketStat:
    """Train-only score bucket summary."""

    bucket: int
    trade_count: int
    win_rate: float
    expectancy_r: float
    raw_scalar: float
    size_scalar: float


@dataclass(frozen=True)
class FittedMonotonicAllocator:
    """Frozen train-only monotonic scorecard."""

    config: MonotonicAllocatorConfig
    baseline_expectancy_r: float
    feature_edges: dict[str, tuple[float, ...]]
    score_edges: tuple[float, ...]
    bucket_stats: tuple[ScoreBucketStat, ...]


TOKYO_OPEN_MONOTONIC_V1 = MonotonicAllocatorConfig(
    name="mnq_tokyo_open_monotonic_v1",
    feature_specs=(
        MonotonicFeatureSpec(
            name="pdr_atr_ratio",
            transform="ratio",
            source_columns=("prev_day_range", "atr_20"),
            direction="high_better",
        ),
        MonotonicFeatureSpec(
            name="atr_vel_ratio",
            transform="identity",
            source_columns=("atr_vel_ratio",),
            direction="high_better",
        ),
    ),
)


def _validate_direction(direction: str) -> None:
    if direction not in {"high_better", "low_better"}:
        raise ValueError(f"Unsupported monotonic direction: {direction}")


def _transform_feature(frame: pd.DataFrame, spec: MonotonicFeatureSpec) -> pd.Series:
    if spec.transform == "identity":
        if len(spec.source_columns) != 1:
            raise ValueError(f"identity transform requires 1 source column: {spec.name}")
        return pd.to_numeric(frame[spec.source_columns[0]], errors="coerce")

    if spec.transform == "ratio":
        if len(spec.source_columns) != 2:
            raise ValueError(f"ratio transform requires 2 source columns: {spec.name}")
        numerator = pd.to_numeric(frame[spec.source_columns[0]], errors="coerce")
        denominator = pd.to_numeric(frame[spec.source_columns[1]], errors="coerce")
        denominator = denominator.where(denominator > 0)
        return numerator / denominator

    if spec.transform == "abs_ratio":
        if len(spec.source_columns) != 2:
            raise ValueError(f"abs_ratio transform requires 2 source columns: {spec.name}")
        numerator = pd.to_numeric(frame[spec.source_columns[0]], errors="coerce").abs()
        denominator = pd.to_numeric(frame[spec.source_columns[1]], errors="coerce")
        denominator = denominator.where(denominator > 0)
        return numerator / denominator

    raise ValueError(f"Unsupported monotonic transform: {spec.transform}")


def enrich_monotonic_features(
    frame: pd.DataFrame,
    config: MonotonicAllocatorConfig = TOKYO_OPEN_MONOTONIC_V1,
) -> pd.DataFrame:
    """Add locked transformed features used by the monotonic scorecard."""
    enriched = frame.copy()
    for spec in config.feature_specs:
        _validate_direction(spec.direction)
        enriched[spec.name] = _transform_feature(enriched, spec)
    return enriched


def _fit_quantile_edges(series: pd.Series, requested_bins: int) -> tuple[float, ...]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        raise ValueError("Cannot fit quantile edges on empty series")
    _, bins = pd.qcut(valid, q=requested_bins, retbins=True, duplicates="drop")
    bins = np.unique(np.asarray(bins, dtype=float))
    if len(bins) < 2:
        raise ValueError("Quantile fit collapsed to fewer than 2 edges")
    return tuple(float(x) for x in bins)


def _apply_quantile_edges(series: pd.Series, finite_edges: tuple[float, ...]) -> pd.Series:
    if len(finite_edges) < 2:
        raise ValueError("Need at least 2 finite edges")
    bins = np.asarray(finite_edges, dtype=float).copy()
    bins[0] = -np.inf
    bins[-1] = np.inf
    bucketed = pd.cut(series, bins=bins, labels=False, include_lowest=True, duplicates="drop")
    return pd.Series(bucketed, index=series.index, dtype="Float64")


def _build_score_components(
    frame: pd.DataFrame,
    feature_edges: dict[str, tuple[float, ...]],
    config: MonotonicAllocatorConfig,
) -> pd.DataFrame:
    enriched = enrich_monotonic_features(frame, config=config)
    score_parts: list[pd.Series] = []

    for spec in config.feature_specs:
        buckets = _apply_quantile_edges(enriched[spec.name], feature_edges[spec.name])
        max_bucket = max(len(feature_edges[spec.name]) - 2, 1)
        if spec.direction == "high_better":
            ranked = buckets / max_bucket
        else:
            ranked = (max_bucket - buckets) / max_bucket
        enriched[f"{spec.name}_bucket"] = buckets.astype("Int64")
        enriched[f"{spec.name}_rank"] = ranked
        score_parts.append(ranked)

    score_frame = pd.concat(score_parts, axis=1)
    enriched["monotonic_score"] = score_frame.mean(axis=1, skipna=False)
    return enriched


def _pava_non_decreasing(values: list[float], weights: list[int]) -> list[float]:
    """Weighted pooled-adjacent-violators algorithm for monotonic repair."""
    if len(values) != len(weights):
        raise ValueError("values and weights must have identical length")

    blocks: list[dict[str, float | int]] = []
    for idx, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append(
            {
                "start": idx,
                "end": idx,
                "weight": float(weight),
                "value": float(value),
            }
        )
        while len(blocks) >= 2 and float(blocks[-2]["value"]) > float(blocks[-1]["value"]):
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left["weight"]) + float(right["weight"])
            merged_value = (
                float(left["value"]) * float(left["weight"])
                + float(right["value"]) * float(right["weight"])
            ) / merged_weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": merged_weight,
                    "value": merged_value,
                }
            )

    repaired = [0.0] * len(values)
    for block in blocks:
        for idx in range(int(block["start"]), int(block["end"]) + 1):
            repaired[idx] = float(block["value"])
    return repaired


def fit_monotonic_allocator(
    train_frame: pd.DataFrame,
    config: MonotonicAllocatorConfig = TOKYO_OPEN_MONOTONIC_V1,
) -> FittedMonotonicAllocator:
    """Fit a simple train-only monotonic scorecard and scalar map."""
    if train_frame.empty:
        raise ValueError("Cannot fit monotonic allocator on empty frame")

    enriched = enrich_monotonic_features(train_frame, config=config)
    feature_edges = {
        spec.name: _fit_quantile_edges(enriched[spec.name], config.feature_bin_count)
        for spec in config.feature_specs
    }
    scored = _build_score_components(train_frame, feature_edges, config)
    score_edges = _fit_quantile_edges(scored["monotonic_score"], config.score_bucket_count)
    scored["score_bucket"] = _apply_quantile_edges(scored["monotonic_score"], score_edges).astype("Int64")

    baseline_expr = float(pd.to_numeric(scored["pnl_r"], errors="coerce").mean())
    if baseline_expr <= 0:
        raise ValueError(
            "Monotonic allocator requires positive train expectancy, got "
            f"{baseline_expr:+.6f}R"
        )

    raw_bucket_stats = []
    for bucket in sorted(int(x) for x in scored["score_bucket"].dropna().unique()):
        subset = scored.loc[scored["score_bucket"] == bucket]
        expectancy_r = float(pd.to_numeric(subset["pnl_r"], errors="coerce").mean())
        raw_scalar = expectancy_r / baseline_expr if expectancy_r > 0 else 0.0
        raw_scalar = float(np.clip(raw_scalar, config.min_scalar, config.max_scalar))
        raw_bucket_stats.append(
            {
                "bucket": bucket,
                "trade_count": int(len(subset)),
                "win_rate": float(pd.to_numeric(subset["target"], errors="coerce").mean()),
                "expectancy_r": expectancy_r,
                "raw_scalar": raw_scalar,
            }
        )

    repaired_scalars = _pava_non_decreasing(
        [stat["raw_scalar"] for stat in raw_bucket_stats],
        [stat["trade_count"] for stat in raw_bucket_stats],
    )

    bucket_stats: list[ScoreBucketStat] = []
    for stat, repaired_scalar in zip(raw_bucket_stats, repaired_scalars, strict=True):
        bucket_stats.append(
            ScoreBucketStat(
                bucket=stat["bucket"],
                trade_count=stat["trade_count"],
                win_rate=stat["win_rate"],
                expectancy_r=stat["expectancy_r"],
                raw_scalar=stat["raw_scalar"],
                size_scalar=float(np.clip(repaired_scalar, config.min_scalar, config.max_scalar)),
            )
        )

    return FittedMonotonicAllocator(
        config=config,
        baseline_expectancy_r=baseline_expr,
        feature_edges=feature_edges,
        score_edges=score_edges,
        bucket_stats=tuple(bucket_stats),
    )


def apply_monotonic_allocator(
    allocator: FittedMonotonicAllocator,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a frozen train-only monotonic scorecard to any frame."""
    scored = _build_score_components(frame, allocator.feature_edges, allocator.config)
    scored["score_bucket"] = _apply_quantile_edges(
        scored["monotonic_score"], allocator.score_edges
    ).astype("Int64")
    scalar_map = {stat.bucket: stat.size_scalar for stat in allocator.bucket_stats}
    scored["size_scalar"] = scored["score_bucket"].map(scalar_map).astype(float)
    scored["size_scalar"] = scored["size_scalar"].fillna(allocator.config.min_scalar)
    scored["sized_pnl_r"] = pd.to_numeric(scored["pnl_r"], errors="coerce") * scored["size_scalar"]
    return scored


def _build_expanding_year_folds(
    development_frame: pd.DataFrame,
    min_train_years: int,
) -> list[dict[str, object]]:
    years = sorted({td.year for td in development_frame["trading_day"]})
    folds: list[dict[str, object]] = []
    for year_idx in range(min_train_years, len(years)):
        train_years = years[:year_idx]
        test_year = years[year_idx]
        train_mask = development_frame["trading_day"].map(lambda td: td.year in train_years)
        test_mask = development_frame["trading_day"].map(lambda td: td.year == test_year)
        train_frame = development_frame.loc[train_mask].reset_index(drop=True)
        test_frame = development_frame.loc[test_mask].reset_index(drop=True)
        if train_frame.empty or test_frame.empty:
            continue
        folds.append(
            {
                "train_years": tuple(train_years),
                "test_year": test_year,
                "train_frame": train_frame,
                "test_frame": test_frame,
            }
        )
    return folds


def _safe_sharpe(values: pd.Series) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if len(series) < 2:
        return None
    std = float(series.std(ddof=1))
    if std <= 0:
        return None
    return float(series.mean() / std)


def _summarize_scored_frame(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "baseline_expr_r": round(float(pd.to_numeric(frame["pnl_r"], errors="coerce").mean()), 6),
        "allocator_expr_r": round(float(pd.to_numeric(frame["sized_pnl_r"], errors="coerce").mean()), 6),
        "delta_expr_r": round(
            float(pd.to_numeric(frame["sized_pnl_r"], errors="coerce").mean())
            - float(pd.to_numeric(frame["pnl_r"], errors="coerce").mean()),
            6,
        ),
        "baseline_sharpe": _safe_sharpe(frame["pnl_r"]),
        "allocator_sharpe": _safe_sharpe(frame["sized_pnl_r"]),
        "avg_scalar": round(float(pd.to_numeric(frame["size_scalar"], errors="coerce").mean()), 6),
        "skip_rate": round(
            float((pd.to_numeric(frame["size_scalar"], errors="coerce") <= 0).mean()),
            6,
        ),
    }


def run_monotonic_walkforward(
    bundle: DatasetBundle,
    config: MonotonicAllocatorConfig = TOKYO_OPEN_MONOTONIC_V1,
) -> dict[str, object]:
    """Run expanding-year walk-forward inside pre-2026, plus forward monitor."""
    development = bundle.development_frame.copy()
    development = development.sort_values("trading_day").reset_index(drop=True)
    folds = _build_expanding_year_folds(development, min_train_years=config.min_train_years)
    if not folds:
        raise ValueError("No walk-forward folds available for monotonic allocator")

    fold_results: list[dict[str, object]] = []
    scored_folds: list[pd.DataFrame] = []

    for fold in folds:
        allocator = fit_monotonic_allocator(fold["train_frame"], config=config)
        scored_test = apply_monotonic_allocator(allocator, fold["test_frame"])
        scored_folds.append(scored_test.assign(test_year=fold["test_year"]))
        fold_results.append(
            {
                "train_years": list(fold["train_years"]),
                "test_year": int(fold["test_year"]),
                "train_rows": int(len(fold["train_frame"])),
                "test_rows": int(len(fold["test_frame"])),
                "bucket_stats": [asdict(stat) for stat in allocator.bucket_stats],
                **_summarize_scored_frame(scored_test),
            }
        )

    scored_development = pd.concat(scored_folds, ignore_index=True)
    full_allocator = fit_monotonic_allocator(development, config=config)
    forward_scored = apply_monotonic_allocator(full_allocator, bundle.forward_frame.copy())

    return {
        "allocator_name": config.name,
        "family": {
            "symbol": bundle.family.symbol,
            "orb_label": bundle.family.orb_label,
            "entry_model": bundle.family.entry_model,
            "rr_target": bundle.family.rr_target,
            "confirm_bars": bundle.family.confirm_bars,
            "direction": bundle.family.direction,
            "orb_minutes": bundle.family.orb_minutes,
        },
        "feature_specs": [asdict(spec) for spec in config.feature_specs],
        "development_walkforward": {
            "folds": fold_results,
            "aggregate": _summarize_scored_frame(scored_development),
        },
        "full_development_fit": {
            "baseline_expectancy_r": round(full_allocator.baseline_expectancy_r, 6),
            "bucket_stats": [asdict(stat) for stat in full_allocator.bucket_stats],
        },
        "forward_monitor": {
            "status": "INFORMATIONAL_ONLY",
            "reason": "2026+ remains sacred forward monitoring and is too thin for promotion.",
            **_summarize_scored_frame(forward_scored),
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the clean-room monotonic allocator baseline on the locked family"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override DB path (defaults to pipeline.paths.GOLD_DB_PATH)",
    )
    args = parser.parse_args()

    bundle = build_family_dataset(family=PHASE1_V1_FAMILY, db_path=args.db_path)
    report = run_monotonic_walkforward(bundle, config=TOKYO_OPEN_MONOTONIC_V1)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

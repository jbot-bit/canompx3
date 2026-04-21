"""Lane-aware monotonic allocator primitives for profile-level replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["high_better", "low_better"]

RAW_BUCKET_MULTIPLIERS = np.asarray([0.5, 0.75, 1.0, 1.25, 1.5], dtype=float)


@dataclass(frozen=True)
class LaneFeatureSpec:
    """Pre-registered per-lane feature direction."""

    name: str
    direction: Direction


@dataclass(frozen=True)
class LaneAllocatorSpec:
    """Locked lane-level scorecard definition."""

    strategy_id: str
    orb_label: str
    features: tuple[LaneFeatureSpec, ...]
    min_train_trades: int = 100


@dataclass(frozen=True)
class FittedLaneAllocator:
    """Frozen train-only lane allocator."""

    strategy_id: str
    orb_label: str
    features: tuple[LaneFeatureSpec, ...]
    sorted_feature_values: dict[str, tuple[float, ...]]
    score_edges: tuple[float, ...]
    desired_multipliers: tuple[float, ...]
    train_rows: int
    fallback_reason: str | None = None


def translate_weight_to_contracts(weight: float) -> int:
    """Translate desired weight into live-feasible integer contracts."""
    if weight < 0.75:
        return 0
    if weight < 1.50:
        return 1
    return 2


def _empirical_cdf(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Train-only empirical CDF scores in [0, 1]."""
    if len(sorted_values) == 0:
        raise ValueError("Cannot score against empty sorted_values")
    ranks = np.searchsorted(sorted_values, values, side="right")
    return ranks.astype(float) / float(len(sorted_values))


def _apply_direction(cdf_values: np.ndarray, direction: Direction) -> np.ndarray:
    if direction == "high_better":
        return cdf_values
    if direction == "low_better":
        return 1.0 - cdf_values
    raise ValueError(f"Unsupported direction: {direction}")


def _complete_cases(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    required = ["pnl_r", *feature_names]
    return frame.loc[:, required + ["trading_day"]].dropna().reset_index(drop=True)


def _fit_score_edges(scores: pd.Series) -> tuple[float, ...]:
    _, bins = pd.qcut(scores, q=5, retbins=True, duplicates="drop")
    bins = np.unique(np.asarray(bins, dtype=float))
    if len(bins) != 6:
        raise ValueError(f"Expected 5 monotonic buckets, got {len(bins) - 1}")
    return tuple(float(x) for x in bins)


def _bucketize(scores: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    bins = np.asarray(edges, dtype=float).copy()
    bins[0] = -np.inf
    bins[-1] = np.inf
    return pd.Series(
        pd.cut(scores, bins=bins, labels=False, include_lowest=True).astype("Int64"),
        index=scores.index,
    )


def fit_lane_allocator(
    train_frame: pd.DataFrame,
    spec: LaneAllocatorSpec,
) -> FittedLaneAllocator:
    """Fit a train-only monotonic lane allocator."""
    feature_names = [feature.name for feature in spec.features]
    complete = _complete_cases(train_frame, feature_names)

    if len(complete) < spec.min_train_trades:
        return FittedLaneAllocator(
            strategy_id=spec.strategy_id,
            orb_label=spec.orb_label,
            features=spec.features,
            sorted_feature_values={},
            score_edges=(),
            desired_multipliers=(),
            train_rows=int(len(complete)),
            fallback_reason=f"insufficient_complete_train_rows:{len(complete)}",
        )

    sorted_feature_values: dict[str, tuple[float, ...]] = {}
    feature_scores: list[np.ndarray] = []
    for feature in spec.features:
        sorted_values = np.sort(pd.to_numeric(complete[feature.name], errors="coerce").to_numpy(dtype=float))
        if len(sorted_values) == 0:
            return FittedLaneAllocator(
                strategy_id=spec.strategy_id,
                orb_label=spec.orb_label,
                features=spec.features,
                sorted_feature_values={},
                score_edges=(),
                desired_multipliers=(),
                train_rows=int(len(complete)),
                fallback_reason=f"empty_feature:{feature.name}",
            )
        sorted_feature_values[feature.name] = tuple(float(x) for x in sorted_values)
        cdf = _empirical_cdf(sorted_values, complete[feature.name].to_numpy(dtype=float))
        feature_scores.append(_apply_direction(cdf, feature.direction))

    composite = np.mean(np.vstack(feature_scores), axis=0)
    complete["composite_score"] = composite

    try:
        score_edges = _fit_score_edges(complete["composite_score"])
    except ValueError as exc:
        return FittedLaneAllocator(
            strategy_id=spec.strategy_id,
            orb_label=spec.orb_label,
            features=spec.features,
            sorted_feature_values=sorted_feature_values,
            score_edges=(),
            desired_multipliers=(),
            train_rows=int(len(complete)),
            fallback_reason=str(exc),
        )

    score_bucket = _bucketize(complete["composite_score"], score_edges)
    raw = RAW_BUCKET_MULTIPLIERS[score_bucket.to_numpy(dtype=int)]
    mean_raw = float(np.mean(raw))
    if mean_raw <= 0:
        return FittedLaneAllocator(
            strategy_id=spec.strategy_id,
            orb_label=spec.orb_label,
            features=spec.features,
            sorted_feature_values=sorted_feature_values,
            score_edges=(),
            desired_multipliers=(),
            train_rows=int(len(complete)),
            fallback_reason="non_positive_mean_raw_weight",
        )

    recentered = np.clip(RAW_BUCKET_MULTIPLIERS / mean_raw, RAW_BUCKET_MULTIPLIERS.min(), RAW_BUCKET_MULTIPLIERS.max())

    return FittedLaneAllocator(
        strategy_id=spec.strategy_id,
        orb_label=spec.orb_label,
        features=spec.features,
        sorted_feature_values=sorted_feature_values,
        score_edges=score_edges,
        desired_multipliers=tuple(float(x) for x in recentered),
        train_rows=int(len(complete)),
        fallback_reason=None,
    )


def apply_lane_allocator(
    allocator: FittedLaneAllocator,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a frozen lane allocator to any frame."""
    out = frame.copy()
    out["composite_score"] = np.nan
    out["score_bucket"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["desired_weight"] = 1.0
    out["contracts"] = 1

    if allocator.fallback_reason:
        return out

    feature_names = [feature.name for feature in allocator.features]
    complete_mask = out[feature_names].notna().all(axis=1)
    if not bool(complete_mask.any()):
        return out

    feature_scores: list[np.ndarray] = []
    complete = out.loc[complete_mask, feature_names]
    for feature in allocator.features:
        sorted_values = np.asarray(allocator.sorted_feature_values[feature.name], dtype=float)
        values = pd.to_numeric(complete[feature.name], errors="coerce").to_numpy(dtype=float)
        cdf = _empirical_cdf(sorted_values, values)
        feature_scores.append(_apply_direction(cdf, feature.direction))

    composite = np.mean(np.vstack(feature_scores), axis=0)
    out.loc[complete_mask, "composite_score"] = composite
    out.loc[complete_mask, "score_bucket"] = _bucketize(
        pd.Series(composite, index=out.index[complete_mask]),
        allocator.score_edges,
    )

    bucket_to_weight = {idx: weight for idx, weight in enumerate(allocator.desired_multipliers)}
    weights = out.loc[complete_mask, "score_bucket"].map(bucket_to_weight).astype(float)
    out.loc[complete_mask, "desired_weight"] = weights
    out.loc[complete_mask, "contracts"] = weights.map(translate_weight_to_contracts).astype(int)
    return out

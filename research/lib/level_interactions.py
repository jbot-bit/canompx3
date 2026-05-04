"""Research-only level interaction primitives.

This module is intentionally narrow:
- canonical level resolution from daily_features-style rows
- chronology-safe level access via pipeline.session_guard
- first interaction classification for pass/fail and sweep/reclaim studies

It is not a live-trading surface and not a new pipeline truth layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pipeline.session_guard import is_feature_safe

ReferenceSide = Literal["below", "above"]
InteractionKind = Literal["touch_only", "wick_fail", "close_through"]

SUPPORTED_LEVELS = frozenset(
    {
        "prev_day_high",
        "prev_day_low",
        "pivot",
        "overnight_high",
        "overnight_low",
        "session_asia_high",
        "session_asia_low",
        "session_london_high",
        "session_london_low",
        "session_ny_high",
        "session_ny_low",
    }
)

REQUIRED_BAR_COLUMNS = frozenset({"open", "high", "low", "close"})


@dataclass(frozen=True)
class LevelReference:
    level_name: str
    price: float | None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class LevelInteractionEvent:
    level_name: str
    level_price: float | None
    reference_side: ReferenceSide | None
    interaction_kind: InteractionKind | None
    bar_index: int | None
    ts_utc: pd.Timestamp | None
    swept: bool
    reclaimed: bool
    reclaim_bar_index: int | None
    reclaim_ts_utc: pd.Timestamp | None
    unavailable_reason: str | None = None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(fval):
        return None
    return fval


def resolve_level_reference(
    feature_row: Mapping[str, object],
    level_name: str,
    *,
    target_session: str | None = None,
) -> LevelReference:
    """Resolve a supported level from a daily_features-style row.

    Returns fail-closed with an explicit reason when the level is unsupported,
    unavailable, or chronologically unsafe for the requested target session.
    """
    if level_name not in SUPPORTED_LEVELS:
        return LevelReference(level_name=level_name, price=None, unavailable_reason="unsupported_level")

    if target_session is not None and not _is_level_safe(level_name, target_session):
        return LevelReference(level_name=level_name, price=None, unavailable_reason="level_not_safe_for_session")

    if level_name == "pivot":
        pdh = _coerce_float(feature_row.get("prev_day_high"))
        pdl = _coerce_float(feature_row.get("prev_day_low"))
        pdc = _coerce_float(feature_row.get("prev_day_close"))
        if pdh is None or pdl is None or pdc is None:
            return LevelReference(level_name=level_name, price=None, unavailable_reason="missing_level_value")
        return LevelReference(level_name=level_name, price=(pdh + pdl + pdc) / 3.0)

    price = _coerce_float(feature_row.get(level_name))
    if price is None:
        return LevelReference(level_name=level_name, price=None, unavailable_reason="missing_level_value")
    return LevelReference(level_name=level_name, price=price)


def _is_level_safe(level_name: str, target_session: str) -> bool:
    if level_name == "pivot":
        return all(
            is_feature_safe(col, target_session)
            for col in ("prev_day_high", "prev_day_low", "prev_day_close")
        )
    return is_feature_safe(level_name, target_session)


def classify_level_interaction(
    bars: pd.DataFrame,
    *,
    level_name: str,
    level_price: float | None,
    reference_side: ReferenceSide,
    sweep_epsilon: float = 0.0,
    reclaim_lookahead_bars: int = 3,
) -> LevelInteractionEvent:
    """Classify the first eligible interaction with a level.

    The first eligible bar must:
    - approach from the stated reference side
    - touch the level with its intrabar range

    Classification is intentionally narrow:
    - touch_only
    - wick_fail
    - close_through
    """
    if reference_side not in {"below", "above"}:
        return LevelInteractionEvent(
            level_name=level_name,
            level_price=level_price,
            reference_side=None,
            interaction_kind=None,
            bar_index=None,
            ts_utc=None,
            swept=False,
            reclaimed=False,
            reclaim_bar_index=None,
            reclaim_ts_utc=None,
            unavailable_reason="invalid_reference_side",
        )

    if level_price is None or pd.isna(level_price):
        return LevelInteractionEvent(
            level_name=level_name,
            level_price=None,
            reference_side=reference_side,
            interaction_kind=None,
            bar_index=None,
            ts_utc=None,
            swept=False,
            reclaimed=False,
            reclaim_bar_index=None,
            reclaim_ts_utc=None,
            unavailable_reason="missing_level_value",
        )

    missing = REQUIRED_BAR_COLUMNS.difference(bars.columns)
    if missing:
        return LevelInteractionEvent(
            level_name=level_name,
            level_price=level_price,
            reference_side=reference_side,
            interaction_kind=None,
            bar_index=None,
            ts_utc=None,
            swept=False,
            reclaimed=False,
            reclaim_bar_index=None,
            reclaim_ts_utc=None,
            unavailable_reason="missing_bar_columns",
        )

    if bars.empty:
        return LevelInteractionEvent(
            level_name=level_name,
            level_price=level_price,
            reference_side=reference_side,
            interaction_kind=None,
            bar_index=None,
            ts_utc=None,
            swept=False,
            reclaimed=False,
            reclaim_bar_index=None,
            reclaim_ts_utc=None,
            unavailable_reason="empty_bars",
        )

    ts_present = "ts_utc" in bars.columns

    for i in range(len(bars)):
        bar = bars.iloc[i]
        pre_price = bars.iloc[i - 1]["close"] if i > 0 else bar["open"]
        if not _is_on_reference_side(pre_price, level_price, reference_side):
            continue

        touched = bool(bar["low"] <= level_price <= bar["high"])
        if not touched:
            continue

        crossed = _crossed_level(bar, level_price, reference_side)
        close_other = _is_on_opposite_side(bar["close"], level_price, reference_side)
        close_ref = _is_on_reference_side(bar["close"], level_price, reference_side)
        swept = _is_sweep(bar, level_price, reference_side, sweep_epsilon)

        if close_other:
            kind: InteractionKind = "close_through"
        elif crossed and close_ref:
            kind = "wick_fail"
        else:
            kind = "touch_only"

        reclaim_bar_index = None
        reclaim_ts = None
        reclaimed = False
        if kind == "close_through" and swept and reclaim_lookahead_bars > 0:
            reclaim_bar_index, reclaim_ts = _find_reclaim(
                bars,
                start_idx=i + 1,
                stop_idx=min(len(bars), i + 1 + reclaim_lookahead_bars),
                level_price=level_price,
                reference_side=reference_side,
            )
            reclaimed = reclaim_bar_index is not None

        return LevelInteractionEvent(
            level_name=level_name,
            level_price=level_price,
            reference_side=reference_side,
            interaction_kind=kind,
            bar_index=i,
            ts_utc=bar["ts_utc"] if ts_present else None,
            swept=swept,
            reclaimed=reclaimed,
            reclaim_bar_index=reclaim_bar_index,
            reclaim_ts_utc=reclaim_ts,
            unavailable_reason=None,
        )

    return LevelInteractionEvent(
        level_name=level_name,
        level_price=level_price,
        reference_side=reference_side,
        interaction_kind=None,
        bar_index=None,
        ts_utc=None,
        swept=False,
        reclaimed=False,
        reclaim_bar_index=None,
        reclaim_ts_utc=None,
        unavailable_reason="no_interaction",
    )


def _is_on_reference_side(price: float, level_price: float, reference_side: ReferenceSide) -> bool:
    if reference_side == "below":
        return price <= level_price
    return price >= level_price


def _is_on_opposite_side(price: float, level_price: float, reference_side: ReferenceSide) -> bool:
    if reference_side == "below":
        return price > level_price
    return price < level_price


def _crossed_level(bar: pd.Series, level_price: float, reference_side: ReferenceSide) -> bool:
    if reference_side == "below":
        return bool(bar["high"] > level_price)
    return bool(bar["low"] < level_price)


def _is_sweep(
    bar: pd.Series,
    level_price: float,
    reference_side: ReferenceSide,
    sweep_epsilon: float,
) -> bool:
    if reference_side == "below":
        return bool((bar["high"] - level_price) > sweep_epsilon)
    return bool((level_price - bar["low"]) > sweep_epsilon)


def _find_reclaim(
    bars: pd.DataFrame,
    *,
    start_idx: int,
    stop_idx: int,
    level_price: float,
    reference_side: ReferenceSide,
) -> tuple[int | None, pd.Timestamp | None]:
    ts_present = "ts_utc" in bars.columns
    for i in range(start_idx, stop_idx):
        bar = bars.iloc[i]
        if _is_on_reference_side(bar["close"], level_price, reference_side):
            return i, (bar["ts_utc"] if ts_present else None)
    return None, None

"""Research-side helpers for canonical filter signal computation.

Created 2026-04-18 to eliminate the re-encoding pattern caught in the VWAP
comprehensive family scan code review: research scripts were implementing
`StrategyFilter.matches_df` logic inline rather than calling the canonical
filter instance. Per `.claude/rules/institutional-rigor.md` Rule 4, parallel
models drift; research must delegate to canonical sources.

Every filter in `trading_app.config.ALL_FILTERS` implements a uniform
`matches_df(df: pd.DataFrame, orb_label: str) -> pd.Series` signature
(audited across 26 classes, `trading_app/config.py` lines 428-2733). This
module provides a thin wrapper that returns a 0/1 numpy array — the
representation most scan code actually wants.

Authority
---------
- Canonical filter registry: `trading_app.config.ALL_FILTERS`
- Canonical filter protocol: `trading_app.config.StrategyFilter` base class
- Institutional rule: `.claude/rules/institutional-rigor.md` Rule 4
  ("delegate to canonical sources — never re-encode")

Why not just call `ALL_FILTERS[key].matches_df(...).to_numpy().astype(int)`
directly in every scan? That's the correct call — but it requires the scan
to (a) know how to handle the key-not-in-registry case, (b) handle empty
DataFrames uniformly, (c) know that `.fillna(False)` is needed because
`matches_df` may return a Series with NaN for missing-data rows. This
module centralizes those three concerns so the scan code is one line.

Input contract
--------------
- `df` must contain the canonical `orb_{orb_label}_*` columns the filter
  reads (e.g., `orb_COMEX_SETTLE_vwap`, `orb_COMEX_SETTLE_break_dir`).
  Do NOT pre-alias these columns in research SQL — load them raw and
  let the canonical filter's matches_df look them up. See the refactored
  `research/vwap_comprehensive_family_scan.py` for an example.

Usage
-----
    from research.filter_utils import filter_signal, filter_signals

    # Single filter
    sig = filter_signal(df, "VWAP_MID_ALIGNED", orb_label="US_DATA_1000")
    # sig is np.ndarray[int] of shape (len(df),), values in {0, 1}

    # Batch for T0 tautology checks
    deployed_sigs = filter_signals(df, ["ORB_G5", "OVNRNG_100"], orb_label=session)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from trading_app.config import ALL_FILTERS, StrategyFilter


def filter_signal(df: pd.DataFrame, filter_key: str, orb_label: str) -> np.ndarray:
    """Return canonical filter's fire-pattern as 0/1 numpy array.

    Parameters
    ----------
    df
        DataFrame containing the canonical `orb_{orb_label}_*` columns the
        filter requires. Row count = len(df). NaN in feature columns is
        treated as filter-fails (fail-closed) per canonical filter contract.
    filter_key
        Key into `trading_app.config.ALL_FILTERS`. Must exist; unknown keys
        raise KeyError.
    orb_label
        ORB session label passed to `filter.matches_df(...)`. Must match the
        suffix of the `orb_*` columns in `df`.

    Returns
    -------
    np.ndarray
        Shape `(len(df),)`, dtype int. `1` where filter fires, `0` otherwise.
        NaN outputs from the canonical `matches_df` are coerced to `0`
        (fail-closed per StrategyFilter protocol).

    Raises
    ------
    KeyError
        If `filter_key` is not in `ALL_FILTERS`.
    """
    if filter_key not in ALL_FILTERS:
        raise KeyError(
            f"filter_key {filter_key!r} not registered in "
            f"trading_app.config.ALL_FILTERS. Available keys: "
            f"{sorted(ALL_FILTERS.keys())[:5]}... "
            f"(total {len(ALL_FILTERS)})"
        )
    filt: StrategyFilter = ALL_FILTERS[filter_key]
    if len(df) == 0:
        return np.zeros(0, dtype=int)
    series = filt.matches_df(df, orb_label).fillna(False)
    return series.to_numpy().astype(int)


def filter_signals(
    df: pd.DataFrame,
    filter_keys: Iterable[str],
    orb_label: str,
) -> dict[str, np.ndarray]:
    """Compute multiple canonical filter signals in one call.

    Convenience wrapper for T0 tautology pre-screens that check several
    deployed filters at once. Returns a dict keyed by filter_key.

    Parameters
    ----------
    df, orb_label
        Same as `filter_signal`.
    filter_keys
        Iterable of keys into `ALL_FILTERS`.

    Returns
    -------
    dict[str, np.ndarray]
        One entry per key, each a 0/1 int array of length `len(df)`.
    """
    return {key: filter_signal(df, key, orb_label) for key in filter_keys}


__all__ = ["filter_signal", "filter_signals"]

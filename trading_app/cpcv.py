"""Combinatorial Purged Cross-Validation with embargo.

**STATUS: PARKED (2026-04-21).**  Pre-registered calibration hypothesis
H3 (embargo sensitivity) KILLED per its locked kill criterion.  Embargo
provides no benefit over plain k-fold when evaluating a fixed,
already-realised return stream — the use case Amendment 3.2 wanted CPCV
for — because ``cpcv_evaluate`` never consults the train fold; it just
computes a per-fold test-set t-statistic, and within-test serial
correlation cannot be mitigated by excluding train indices.  See
``docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1-postmortem.md``
for the full interpretation and the alternative techniques (block
bootstrap, hierarchical pooling) that would actually address the
underlying contamination.

**DO NOT IMPORT FROM PRODUCTION CODE.**  Wiring this module into
``trading_app.strategy_validator._check_criterion_8_oos`` (the
``cpcv_fallback`` kwarg referenced in the pre-reg) is explicitly
forbidden by the postmortem.  The module is retained as an audit
artifact so future readers do not re-speculate in the same direction.

Methodology: Lopez de Prado 2020 ``Machine Learning for Asset Managers``
Chapter 7 (Cross-Validation in Finance).  See extract at
``docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md``.

Pre-registered infrastructure: ``docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml``.
Authority: ``docs/institutional/pre_registered_criteria.md`` Amendment 3.2.

Purpose
-------
Amendment 3.2 (2026-04-21) identifies CPCV as the second-opinion OOS
mechanism that generates probabilistic OOS evidence from the
pre-2026-01-01 in-sample data without consuming the sacred 2026 holdout.
The sacred holdout remains immovable (``trading_app.holdout_policy``).
CPCV provides coverage the 3-month / ~15-30-trade filtered live OOS
cannot provide at current fire rates.

Design
------
Trades are partitioned into ``n_splits`` contiguous chunks.  Each CPCV
fold selects ``n_test_splits`` of those chunks as the test set; the
remaining chunks (minus an embargo window after each test chunk
boundary) form the training set.  This produces ``C(n_splits,
n_test_splits)`` folds rather than ``n_splits`` — the combinatorial part
— giving more coverage with better-balanced test/train overlap
properties than plain k-fold.

Embargo (``embargo_trades``) excludes trades within ``embargo_trades``
positions *after* each test chunk's last index from the training set.
This handles residual serial correlation in per-trade returns
(e.g. overnight regime persistence) that simple index-disjoint
splitting does not.  Embargo length is calibrated by pre-registered
hypothesis H3.

Purge (by trading_day) is implemented by the caller when running on
real `orb_outcomes` data: a training-fold trade sharing a trading_day
with any test-fold trade must be dropped.  This module exposes
trade-index primitives; trading_day-level purging is left to the
caller because it requires access to the outcome timestamps.

This module contains NO filter logic, NO strategy selection, NO
size-multiplier derivation.  It is a pure cross-validation primitive.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from typing import Any


def cpcv_splits(
    n_trades: int,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_trades: int = 5,
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield ``(train_idx, test_idx)`` tuples for each CPCV fold.

    Parameters
    ----------
    n_trades
        Total number of trades in the returns series.
    n_splits
        Number of contiguous chunks to partition the index range into.
        Must be >= 2.
    n_test_splits
        Number of chunks per fold that belong to the test set.  Must be
        >= 1 and < ``n_splits``.
    embargo_trades
        Number of indices to exclude from the train set AFTER each test
        chunk's last index, to buffer against serial correlation.  Must
        be >= 0.

    Yields
    ------
    (train_idx, test_idx)
        Two disjoint index lists per fold.  ``train_idx`` excludes test
        indices AND the embargo window.  Union over all folds of
        ``test_idx`` covers ``range(n_trades)``; a given test index may
        appear in multiple folds (that is the combinatorial property).

    Raises
    ------
    ValueError
        If parameters are out of range.

    Notes
    -----
    Chunks are produced by ``_contiguous_chunks`` below, which is a
    deterministic floor-based partition (equivalent to
    ``numpy.array_split`` without the numpy dependency).  Identical
    inputs always produce identical fold index layouts — the determinism
    check under H1 calibration relies on this.
    """
    if n_trades < n_splits:
        raise ValueError(f"n_trades={n_trades} must be >= n_splits={n_splits}")
    if n_splits < 2:
        raise ValueError(f"n_splits={n_splits} must be >= 2")
    if n_test_splits < 1 or n_test_splits >= n_splits:
        raise ValueError(f"n_test_splits={n_test_splits} must satisfy 1 <= n_test_splits < n_splits={n_splits}")
    if embargo_trades < 0:
        raise ValueError(f"embargo_trades={embargo_trades} must be >= 0")

    chunks = _contiguous_chunks(n_trades, n_splits)

    for test_chunk_ids in combinations(range(n_splits), n_test_splits):
        test_idx: list[int] = []
        embargo_set: set[int] = set()
        for i in test_chunk_ids:
            test_idx.extend(chunks[i])
            # Embargo: the `embargo_trades` indices AFTER this test
            # chunk's last index (bounded at n_trades).
            chunk_last = chunks[i][-1]
            for j in range(chunk_last + 1, min(chunk_last + 1 + embargo_trades, n_trades)):
                embargo_set.add(j)

        test_set = set(test_idx)
        train_idx = [k for k in range(n_trades) if k not in test_set and k not in embargo_set]
        # Sort test_idx for caller convenience (folds with multiple
        # chunks would otherwise have unsorted concatenation).
        yield train_idx, sorted(test_idx)


def _contiguous_chunks(n: int, k: int) -> list[list[int]]:
    """Partition ``range(n)`` into ``k`` contiguous chunks of near-equal size.

    Deterministic floor-based allocation: the first ``n % k`` chunks get
    one extra element.  Equivalent to ``numpy.array_split(range(n), k)``
    index layout.
    """
    base = n // k
    remainder = n % k
    chunks: list[list[int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


def cpcv_fold_t_statistic(returns: list[float]) -> float | None:
    """One-sample t-statistic for H0: mean = 0 on a returns vector.

    Returns ``None`` when ``len(returns) < 2`` (no variance) or when the
    sample standard deviation is effectively zero (< 1e-12 — floating
    point dust from uniform input).  This is the same degeneracy guard
    used by ``_estimate_oos_power`` in ``strategy_validator``.
    """
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    variance = sum((x - mean) ** 2 for x in returns) / (n - 1)
    sd = variance**0.5
    if sd < 1e-12:
        return None
    return mean / (sd / n**0.5)


def cpcv_evaluate(
    returns: list[float],
    *,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_trades: int = 5,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run CPCV on ``returns`` and return per-fold + aggregate statistics.

    Per-fold: one-sample t-test of mean = 0 (two-tailed) at ``alpha``.

    Aggregate: ``reject_fraction`` — fraction of folds where the null is
    rejected.  Under true H0, ``reject_fraction`` should concentrate
    near ``alpha``; systematic deviation indicates purge/embargo leak
    (over-rejection) or over-conservative partitioning (under-rejection).

    Returns a dict with keys: ``n_folds``, ``reject_fraction``,
    ``mean_r_across_folds``, ``folds`` (per-fold list of dicts with
    ``n_test``, ``mean_r``, ``t``, ``p_two_tailed``, ``rejects_h0``),
    ``params`` (echoes input parameters for audit).

    Raises
    ------
    ValueError
        Propagated from ``cpcv_splits`` parameter checks.
    """
    from scipy import stats

    n = len(returns)
    fold_stats: list[dict[str, Any]] = []
    for _train_idx, test_idx in cpcv_splits(n, n_splits, n_test_splits, embargo_trades):
        test_returns = [returns[i] for i in test_idx]
        t = cpcv_fold_t_statistic(test_returns)
        n_test = len(test_returns)
        if t is None or n_test < 2:
            p_two = None
            rejects = False
        else:
            p_two = float(2.0 * stats.t.sf(abs(t), df=n_test - 1))
            rejects = p_two < alpha
        fold_stats.append(
            {
                "n_test": n_test,
                "mean_r": sum(test_returns) / n_test if n_test else 0.0,
                "t": t,
                "p_two_tailed": p_two,
                "rejects_h0": rejects,
            }
        )

    n_folds = len(fold_stats)
    reject_fraction = sum(1 for f in fold_stats if f["rejects_h0"]) / n_folds if n_folds else 0.0
    mean_r_agg = sum(f["mean_r"] for f in fold_stats) / n_folds if n_folds else 0.0
    return {
        "n_folds": n_folds,
        "reject_fraction": reject_fraction,
        "mean_r_across_folds": mean_r_agg,
        "folds": fold_stats,
        "params": {
            "n_splits": n_splits,
            "n_test_splits": n_test_splits,
            "embargo_trades": embargo_trades,
            "alpha": alpha,
        },
    }

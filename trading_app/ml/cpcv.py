"""Combinatorial Purged Cross-Validation (CPCV).

Per de Prado (AIFML): "Makes overfitting practically impossible by generating
thousands of test sets through resampling combinatorial splits."

Implementation:
  - Split data into N time-ordered groups (by trading_day)
  - For each combination of k groups as test set: C(N, k) splits
  - Purge: remove training samples within purge_days of test boundaries
  - Embargo: remove training samples within embargo_days after test end
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trading_app.ml.config import (
    CPCV_EMBARGO_DAYS,
    CPCV_K_TEST,
    CPCV_N_GROUPS,
    CPCV_PURGE_DAYS,
)

logger = logging.getLogger(__name__)


def _assign_groups(trading_days: pd.Series, n_groups: int) -> np.ndarray:
    """Assign each row to a time-ordered group based on trading_day."""
    unique_days = np.sort(trading_days.unique())
    day_to_group = {}
    group_size = len(unique_days) / n_groups
    for i, day in enumerate(unique_days):
        day_to_group[day] = min(int(i / group_size), n_groups - 1)
    return trading_days.map(day_to_group).values


def _purge_embargo_mask(
    trading_days: pd.Series,
    test_days: set,
    purge_days: int,
    embargo_days: int,
) -> np.ndarray:
    """Return boolean mask of training rows to KEEP (True = keep).

    Removes rows within purge_days of any test day boundary,
    and within embargo_days after the last test day.
    """
    all_days_sorted = np.sort(trading_days.unique())
    day_to_idx = {d: i for i, d in enumerate(all_days_sorted)}

    # Find test day indices
    test_indices = sorted([day_to_idx[d] for d in test_days if d in day_to_idx])
    if not test_indices:
        return np.ones(len(trading_days), dtype=bool)

    # Days to exclude from training
    exclude_days = set()
    for ti in test_indices:
        # Purge: days within purge_days before test
        for offset in range(1, purge_days + 1):
            idx = ti - offset
            if 0 <= idx < len(all_days_sorted):
                exclude_days.add(all_days_sorted[idx])
        # Embargo: days within embargo_days after test
        for offset in range(1, embargo_days + 1):
            idx = ti + offset
            if 0 <= idx < len(all_days_sorted):
                exclude_days.add(all_days_sorted[idx])

    # Build mask: keep rows NOT in exclude_days and NOT in test_days
    keep = ~trading_days.isin(exclude_days | test_days)
    return keep.values


def cpcv_score(
    model_class,
    model_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    trading_days: pd.Series,
    *,
    n_groups: int = CPCV_N_GROUPS,
    k_test: int = CPCV_K_TEST,
    purge_days: int = CPCV_PURGE_DAYS,
    embargo_days: int = CPCV_EMBARGO_DAYS,
    max_splits: int | None = None,
) -> dict:
    """Run CPCV and return OOS AUC scores across all combinatorial splits.

    Args:
        model_class: sklearn classifier class (e.g., RandomForestClassifier)
        model_params: kwargs for model_class
        X: Feature matrix
        y: Binary target
        trading_days: Trading day for each row (for time-ordering)
        n_groups: Number of time-ordered groups
        k_test: Number of groups to hold out as test
        purge_days: Days to purge around test boundaries
        embargo_days: Days to embargo after test end
        max_splits: Cap on number of splits (for speed during dev)

    Returns:
        dict with keys: auc_mean, auc_std, auc_scores, n_splits, n_train_avg, n_test_avg
    """
    groups = _assign_groups(trading_days, n_groups)
    all_combos = list(combinations(range(n_groups), k_test))

    if max_splits is not None and len(all_combos) > max_splits:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_combos), max_splits, replace=False)
        all_combos = [all_combos[i] for i in sorted(indices)]

    auc_scores = []
    n_trains = []
    n_tests = []

    for combo in all_combos:
        test_groups = set(combo)
        test_mask = np.isin(groups, list(test_groups))
        test_days = set(trading_days[test_mask].unique())

        # Train = NOT test, with purge+embargo applied
        train_base = ~test_mask
        purge_keep = _purge_embargo_mask(trading_days, test_days, purge_days, embargo_days)
        train_mask = train_base & purge_keep

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue

        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]

        # Skip if only one class in train or test
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
            auc_scores.append(auc)
            n_trains.append(train_mask.sum())
            n_tests.append(test_mask.sum())
        except ValueError:
            continue

    if not auc_scores:
        logger.warning("CPCV produced 0 valid splits")
        return {
            "auc_mean": 0.5,
            "auc_std": 0.0,
            "auc_scores": [],
            "n_splits": 0,
            "n_train_avg": 0,
            "n_test_avg": 0,
        }

    return {
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "auc_scores": auc_scores,
        "n_splits": len(auc_scores),
        "n_train_avg": int(np.mean(n_trains)),
        "n_test_avg": int(np.mean(n_tests)),
    }

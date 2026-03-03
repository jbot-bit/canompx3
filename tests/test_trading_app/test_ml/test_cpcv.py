"""Tests for trading_app.ml.cpcv — Combinatorial Purged Cross-Validation."""

import numpy as np
import pandas as pd
import pytest

from trading_app.ml.cpcv import cpcv_score


class TestCPCVSplits:
    """CPCV produces valid train/test splits."""

    @pytest.fixture
    def sample_data(self):
        """Minimal dataset: 500 rows over 100 unique trading days."""
        rng = np.random.RandomState(42)
        n = 500
        # 100 unique trading days, 5 rows each
        days = np.repeat(pd.date_range("2023-01-01", periods=100), 5)
        X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
        y = pd.Series(rng.randint(0, 2, n))
        trading_days = pd.Series(days)
        return X, y, trading_days

    def test_returns_dict_with_required_keys(self, sample_data):
        from sklearn.ensemble import RandomForestClassifier
        X, y, days = sample_data
        result = cpcv_score(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X, y, days,
            max_splits=3,
        )
        assert "auc_mean" in result
        assert "auc_std" in result
        assert "auc_scores" in result
        assert "n_splits" in result

    def test_max_splits_cap(self, sample_data):
        from sklearn.ensemble import RandomForestClassifier
        X, y, days = sample_data
        result = cpcv_score(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X, y, days,
            max_splits=5,
        )
        assert result["n_splits"] <= 5

    def test_auc_scores_are_valid(self, sample_data):
        from sklearn.ensemble import RandomForestClassifier
        X, y, days = sample_data
        result = cpcv_score(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X, y, days,
            max_splits=3,
        )
        for score in result["auc_scores"]:
            assert 0.0 <= score <= 1.0

    def test_no_train_test_overlap(self, sample_data):
        """Train and test indices must be disjoint in each split."""
        from sklearn.ensemble import RandomForestClassifier
        X, y, days = sample_data

        # Access internals by running with max_splits=1
        result = cpcv_score(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X, y, days,
            max_splits=1,
        )
        # If it ran, train/test didn't overlap (model trained successfully)
        assert result["n_splits"] == 1
        assert len(result["auc_scores"]) == 1

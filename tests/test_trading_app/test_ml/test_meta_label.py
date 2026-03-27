"""Tests for trading_app.ml.meta_label — training pipeline and quality gates."""

import numpy as np
import pytest

from trading_app.ml.config import THRESHOLD_MAX, THRESHOLD_MIN, THRESHOLD_STEP
from trading_app.ml.meta_label import _optimize_threshold_profit


class TestOptimizeThresholdProfit:
    """Threshold optimizer: finds threshold that maximizes total R delta."""

    def test_finds_best_threshold(self):
        """Simple case: one threshold clearly better than others."""
        # 20 trades: first 10 have p_win >= 0.60 and positive pnl,
        # last 10 have p_win < 0.50 and negative pnl.
        y_prob = np.array([0.7] * 10 + [0.3] * 10)
        pnl_r = np.array([1.0] * 10 + [-1.0] * 10)

        best_t, best_delta = _optimize_threshold_profit(y_prob, pnl_r, min_kept=5)

        # Best threshold should filter out the losing trades
        assert best_t is not None
        # Threshold between 0.3 and 0.7 should keep only winners
        assert 0.35 <= best_t <= 0.65
        # Delta should be positive (removing losing trades = +10R)
        assert best_delta > 0

    def test_returns_none_when_no_improvement(self):
        """All trades equally good → no threshold beats baseline."""
        y_prob = np.array([0.50] * 100)
        pnl_r = np.array([1.0] * 100)

        best_t, best_delta = _optimize_threshold_profit(y_prob, pnl_r, min_kept=10)

        # Any threshold that keeps trades still sums to ≤ baseline
        # (removing positive trades always hurts). No threshold beats baseline.
        assert best_t is None
        assert best_delta == 0.0

    def test_respects_min_kept(self):
        """Threshold is rejected if it keeps fewer than min_kept trades."""
        # 5 good trades at p=0.8, 95 bad trades at p=0.2
        # All thresholds in [0.35, 0.70] keep only the 5 good trades (p=0.8),
        # which is < min_kept=10, so ALL thresholds are rejected.
        y_prob = np.array([0.8] * 5 + [0.2] * 95)
        pnl_r = np.array([10.0] * 5 + [-1.0] * 95)

        best_t, best_delta = _optimize_threshold_profit(y_prob, pnl_r, min_kept=10)

        # No threshold keeps >= 10 trades → None
        assert best_t is None
        assert best_delta == 0.0

    def test_nan_pnl_excluded(self):
        """NaN pnl_r values are excluded from optimization."""
        # 2 winners at p=0.7, 1 loser at p=0.3, 1 NaN (ignored)
        y_prob = np.array([0.7, 0.7, 0.3, np.nan])
        pnl_r = np.array([1.0, 1.0, -1.0, np.nan])

        best_t, best_delta = _optimize_threshold_profit(y_prob, pnl_r, min_kept=1)

        # Optimizer should find threshold filtering out the loser (+1.0R delta)
        assert best_t is not None
        assert best_delta > 0

    def test_threshold_within_configured_range(self):
        """Returned threshold is within [THRESHOLD_MIN, THRESHOLD_MAX]."""
        y_prob = np.linspace(0.0, 1.0, 200)
        pnl_r = np.where(y_prob > 0.55, 2.0, -1.0)

        best_t, _ = _optimize_threshold_profit(y_prob, pnl_r, min_kept=10)

        if best_t is not None:
            # round(t, 2) in production code guarantees exact range
            assert THRESHOLD_MIN <= best_t <= THRESHOLD_MAX


class TestQualityGateContracts:
    """Contract tests: quality gate thresholds are what we expect.

    These guard against accidental changes to gate logic. If a threshold
    changes intentionally, update both the code AND this test.
    """

    def test_gate_thresholds_stable(self):
        """Quality gate threshold constants haven't drifted."""
        # Gate 1: OOS positive (implicit — checked via honest_delta_r < 0)
        # Gate 2: CPCV >= 0.50
        # Gate 3: AUC > 0.52
        # Gate 4: Skip < 85%
        #
        # These are checked inline in meta_label.py. We verify by
        # importing the function and checking the docstring + source.
        import inspect

        from trading_app.ml.meta_label import train_per_session_meta_label

        source = inspect.getsource(train_per_session_meta_label)

        # Gate 1: honest_delta_r < 0
        assert "honest_delta_r < 0" in source
        # Gate 2: cpcv_auc < 0.50
        assert "cpcv_auc < 0.50" in source
        # Gate 3: test_auc < 0.52
        assert "test_auc < 0.52" in source
        # Gate 4: skip_pct > 0.85
        assert "skip_pct > 0.85" in source

    def test_four_gates_exist(self):
        """All 4 quality gates are present in the training function."""
        import inspect

        from trading_app.ml.meta_label import train_per_session_meta_label

        source = inspect.getsource(train_per_session_meta_label)

        # Count gate comments
        assert source.count("Gate 1:") == 1
        assert source.count("Gate 2:") == 1
        assert source.count("Gate 3:") == 1
        assert source.count("Gate 4:") == 1

    def test_threshold_search_range(self):
        """Threshold search configuration is correct."""
        assert THRESHOLD_MIN == 0.35
        assert THRESHOLD_MAX == 0.70
        assert THRESHOLD_STEP == 0.01

    def test_min_test_trades(self):
        """Threshold rejection requires n_kept_test < 10."""
        import inspect

        from trading_app.ml.meta_label import train_per_session_meta_label

        source = inspect.getsource(train_per_session_meta_label)

        assert "n_kept_test < 10" in source

    def test_three_way_split(self):
        """60/20/20 split is encoded in the config hash."""
        import inspect

        from trading_app.ml.config import compute_config_hash

        source = inspect.getsource(compute_config_hash)
        assert "split=60/20/20" in source

    def test_positive_baseline_gate_exists(self):
        """Fix E: negative baseline sessions must be skipped."""
        import inspect

        from trading_app.ml.meta_label import train_per_session_meta_label

        source = inspect.getsource(train_per_session_meta_label)
        assert "train_expr <= 0" in source
        assert "negative_baseline" in source

    def test_ef_lm_cross_session_guard(self):
        """Fix B: EUROPE_FLOW and LONDON_METALS drop cross-session features."""
        import pandas as pd

        from trading_app.ml.meta_label import _get_session_features

        X = pd.DataFrame({
            "orb_size_norm": [0.5],
            "prior_sessions_broken": [2],
            "nearest_level_to_high_R": [0.5],
        })
        X_ef = _get_session_features(X, "EUROPE_FLOW")
        assert "prior_sessions_broken" not in X_ef.columns
        assert "nearest_level_to_high_R" not in X_ef.columns
        assert "orb_size_norm" in X_ef.columns

    def test_core_feature_selection(self):
        """Fix F: training uses only ML_CORE_FEATURES."""
        import inspect

        from trading_app.ml.meta_label import train_per_session_meta_label

        source = inspect.getsource(train_per_session_meta_label)
        assert "ML_CORE_FEATURES" in source

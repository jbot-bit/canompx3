"""Tests for trading_app.ml.predict_live — live ML prediction module."""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trading_app.ml.predict_live import LiveMLPredictor, MLPrediction, _compute_config_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_bundle(
    n_features: int = 10,
    threshold: float = 0.55,
    config_hash: str | None = None,
    trained_at: str | None = None,
) -> dict:
    """Create a mock model bundle matching joblib structure."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])

    feature_names = [f"feat_{i}" for i in range(n_features)]
    # Add one-hot columns to test alignment
    feature_names.extend(["orb_label_CME_REOPEN", "entry_model_E2"])

    return {
        "model": model,
        "feature_names": feature_names,
        "instrument": "MGC",
        "n_train": 5000,
        "oos_auc": 0.65,
        "optimal_threshold": threshold,
        "cpcv_auc": 0.64,
        "trained_at": trained_at or datetime.now(timezone.utc).isoformat(),
        "data_date_range": ("2020-01-01", "2025-12-31"),
        "config_hash": config_hash or _compute_config_hash(),
    }


def _make_mock_daily_features_row() -> dict:
    """Create a mock daily_features row."""
    return {
        "trading_day": date(2025, 12, 1),
        "symbol": "MGC",
        "orb_minutes": 5,
        "atr_20": 25.0,
        "atr_vel_ratio": 1.1,
        "rsi_14_at_CME_REOPEN": 55.0,
        "gap_open_points": 2.5,
        "garch_atr_ratio": 0.95,
        "garch_forecast_vol": 24.0,
        "prev_day_range": 30.0,
        "prev_day_direction": "UP",
        "overnight_range": 15.0,
        "day_of_week": 2,
        "is_friday": False,
        "is_monday": False,
        "orb_CME_REOPEN_size": 5.0,
        "orb_CME_REOPEN_volume": 1200,
        "orb_CME_REOPEN_break_bar_volume": 300,
        "orb_CME_REOPEN_break_delay_min": 3.0,
        "orb_CME_REOPEN_break_bar_continues": True,
        "orb_CME_REOPEN_break_dir": "LONG",
        "rel_vol_CME_REOPEN": 1.2,
        "gap_type": "SMALL_UP",
        "atr_vel_regime": "EXPANDING",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFailOpen:
    """Fail-open: every error returns (0.5, True, 0.5)."""

    def test_fail_open_missing_model(self):
        """No model for instrument → fail-open."""
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        # _models is empty — no model loaded
        result = predictor.predict(
            instrument="MGC",
            trading_day=date(2025, 12, 1),
            orb_label="CME_REOPEN",
            orb_minutes=5,
            entry_model="E2",
            rr_target=1.5,
            confirm_bars=1,
        )
        assert result == MLPrediction(p_win=0.5, take=True, threshold=0.5)
        assert predictor.fail_open_count == 1

    def test_fail_open_missing_daily_features(self):
        """No daily_features row → fail-open."""
        bundle = _make_mock_bundle()
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        # Mock DB to return None
        with patch.object(predictor, "_get_daily_features", return_value=None):
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="CME_REOPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=1.5,
                confirm_bars=1,
            )
        assert result.take is True
        assert result.p_win == 0.5
        assert predictor.fail_open_count == 1

    def test_fail_open_on_prediction_exception(self):
        """Exception during prediction → fail-open."""
        bundle = _make_mock_bundle()
        bundle["model"].predict_proba.side_effect = ValueError("boom")

        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        daily_row = _make_mock_daily_features_row()
        with patch.object(predictor, "_get_daily_features", return_value=daily_row):
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="CME_REOPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=1.5,
                confirm_bars=1,
            )
        assert result.take is True
        assert result.p_win == 0.5
        assert predictor.fail_open_count == 1


class TestFeatureAlignment:
    """Feature alignment handles missing/extra columns correctly."""

    def test_missing_onehot_filled_with_zero(self):
        """One-hot columns not in X → filled with 0.0."""
        X = pd.DataFrame({"feat_0": [1.0], "feat_1": [2.0]})
        model_features = ["feat_0", "feat_1", "orb_label_TOKYO_OPEN"]
        result = LiveMLPredictor._align_features(X, model_features)
        assert list(result.columns) == model_features
        assert result["orb_label_TOKYO_OPEN"].iloc[0] == 0.0

    def test_missing_numeric_filled_with_neg999(self):
        """Numeric columns not in X → filled with -999."""
        X = pd.DataFrame({"feat_0": [1.0]})
        model_features = ["feat_0", "atr_20"]
        result = LiveMLPredictor._align_features(X, model_features)
        assert result["atr_20"].iloc[0] == -999.0

    def test_extra_columns_dropped(self):
        """Columns in X but not in model → dropped."""
        X = pd.DataFrame({"feat_0": [1.0], "new_feat": [99.0]})
        model_features = ["feat_0"]
        result = LiveMLPredictor._align_features(X, model_features)
        assert list(result.columns) == ["feat_0"]
        assert "new_feat" not in result.columns

    def test_column_order_matches_model(self):
        """Output column order matches model's feature_names exactly."""
        X = pd.DataFrame({"c": [3.0], "a": [1.0], "b": [2.0]})
        model_features = ["a", "b", "c"]
        result = LiveMLPredictor._align_features(X, model_features)
        assert list(result.columns) == ["a", "b", "c"]


class TestCaching:
    """Cache behavior: prediction results and daily features."""

    def test_prediction_cache_hit(self):
        """Second identical call returns cached result."""
        bundle = _make_mock_bundle()
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        daily_row = _make_mock_daily_features_row()
        with patch.object(predictor, "_get_daily_features", return_value=daily_row):
            result1 = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="CME_REOPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=1.5,
                confirm_bars=1,
            )
            result2 = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="CME_REOPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=1.5,
                confirm_bars=1,
            )
        assert result1 == result2
        assert predictor.predictions_cached == 1

    def test_clear_daily_cache(self):
        """clear_daily_cache empties both caches."""
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._daily_cache[("MGC", date(2025, 12, 1), 5)] = {"foo": 1}
        predictor._prediction_cache[("key",)] = MLPrediction(0.6, True, 0.55)

        predictor.clear_daily_cache()
        assert len(predictor._daily_cache) == 0
        assert len(predictor._prediction_cache) == 0


class TestConfigHash:
    """Config hash consistency."""

    def test_config_hash_deterministic(self):
        """Same config → same hash."""
        h1 = _compute_config_hash()
        h2 = _compute_config_hash()
        assert h1 == h2
        assert len(h1) == 12

    def test_config_hash_matches_shared_function(self):
        """predict_live._compute_config_hash matches config.compute_config_hash."""
        from trading_app.ml.config import compute_config_hash
        assert _compute_config_hash() == compute_config_hash()


class TestPredictionResult:
    """Prediction produces correct take/skip decisions."""

    def test_above_threshold_take(self):
        """P(win) >= threshold → take=True."""
        bundle = _make_mock_bundle(threshold=0.55)
        bundle["model"].predict_proba.return_value = np.array([[0.4, 0.6]])

        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        daily_row = _make_mock_daily_features_row()
        with patch.object(predictor, "_get_daily_features", return_value=daily_row):
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="CME_REOPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=1.5,
                confirm_bars=1,
            )
        assert result.take is True
        assert result.p_win == pytest.approx(0.6, abs=0.01)

    def test_below_threshold_skip(self):
        """P(win) < threshold → take=False."""
        bundle = _make_mock_bundle(threshold=0.55)
        bundle["model"].predict_proba.return_value = np.array([[0.6, 0.4]])

        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        daily_row = _make_mock_daily_features_row()
        with patch.object(predictor, "_get_daily_features", return_value=daily_row):
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="CME_REOPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=1.5,
                confirm_bars=1,
            )
        assert result.take is False
        assert result.p_win == pytest.approx(0.4, abs=0.01)


class TestModelInfo:
    """get_model_info returns correct metadata."""

    def test_model_info_loaded(self):
        bundle = _make_mock_bundle()
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        info = predictor.get_model_info("MGC")
        assert info is not None
        assert info["instrument"] == "MGC"
        assert info["n_features"] == 12  # 10 + 2 one-hot

    def test_model_info_missing(self):
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        assert predictor.get_model_info("MGC") is None

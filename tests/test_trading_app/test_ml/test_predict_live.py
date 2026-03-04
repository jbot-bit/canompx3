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


# ---------------------------------------------------------------------------
# Hybrid per-session model tests
# ---------------------------------------------------------------------------

def _make_hybrid_bundle(
    sessions: dict[str, dict | None] | None = None,
    model_type: str = "single_config_per_session",
    total_honest_delta_r: float = 97.2,
) -> dict:
    """Create a hybrid per-session model bundle.

    Args:
        sessions: Dict of session_name -> session_info. None entries mean
            no model for that session (fail-open). If None, creates a
            default bundle with SINGAPORE_OPEN having a model.
    """
    if sessions is None:
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.35, 0.65]])
        sessions = {
            "SINGAPORE_OPEN": {
                "model": model,
                "feature_names": ["feat_0", "feat_1", "orb_label_CME_REOPEN"],
                "optimal_threshold": 0.50,
                "test_auc": 0.62,
                "cpcv_auc": 0.58,
                "n_train": 300,
            },
            "TOKYO_OPEN": {
                "model": None,  # No model for this session
                "reason": "oos_negative",
            },
        }

    return {
        "model_type": model_type,
        "sessions": sessions,
        "rr_target": 2.5,
        "total_honest_delta_r": total_honest_delta_r,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": _compute_config_hash(),
        "data_date_range": ("2020-01-01", "2025-12-31"),
    }


class TestHybridModel:
    """Hybrid per-session model: routes to correct session sub-model."""

    def test_hybrid_predict_with_session_model(self):
        """Session with a sub-model returns real prediction."""
        bundle = _make_hybrid_bundle()
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        daily_row = _make_mock_daily_features_row()
        with patch.object(predictor, "_get_daily_features", return_value=daily_row):
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="SINGAPORE_OPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=2.5,
                confirm_bars=1,
            )
        # Should use the model's prediction (0.65), not fail-open
        assert result.p_win == pytest.approx(0.65, abs=0.01)
        assert result.threshold == 0.50
        assert predictor.predictions_made == 1

    def test_hybrid_fail_open_no_session_model(self):
        """Session without a sub-model (model=None) → fail-open.

        Must short-circuit BEFORE hitting _get_daily_features (no DB call).
        """
        bundle = _make_hybrid_bundle()
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        with patch.object(predictor, "_get_daily_features") as mock_gdf:
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="TOKYO_OPEN",  # has no model
                orb_minutes=5,
                entry_model="E2",
                rr_target=2.5,
                confirm_bars=1,
            )
            mock_gdf.assert_not_called()
        assert result == MLPrediction(p_win=0.5, take=True, threshold=0.5)

    def test_hybrid_fail_open_unknown_session(self):
        """Session not in bundle at all → fail-open."""
        bundle = _make_hybrid_bundle()
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        result = predictor.predict(
            instrument="MGC",
            trading_day=date(2025, 12, 1),
            orb_label="COMEX_SETTLE",  # not in sessions dict
            orb_minutes=5,
            entry_model="E2",
            rr_target=2.5,
            confirm_bars=1,
        )
        assert result == MLPrediction(p_win=0.5, take=True, threshold=0.5)

    def test_hybrid_predict_with_calibrator(self):
        """Calibrator transforms raw P(win) for display, but threshold uses raw."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.35, 0.65]])

        # Calibrator maps 0.65 → 0.72 (isotonic output)
        mock_calibrator = MagicMock()
        mock_calibrator.predict.return_value = np.array([0.72])

        sessions = {
            "SINGAPORE_OPEN": {
                "model": mock_model,
                "calibrator": mock_calibrator,
                "feature_names": ["feat_0", "feat_1", "orb_label_CME_REOPEN"],
                "optimal_threshold": 0.50,
                "test_auc": 0.62,
            },
        }
        bundle = _make_hybrid_bundle(sessions=sessions)

        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        daily_row = _make_mock_daily_features_row()
        with patch.object(predictor, "_get_daily_features", return_value=daily_row):
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="SINGAPORE_OPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=2.5,
                confirm_bars=1,
            )
        # p_win should be CALIBRATED (0.72), but take decision uses RAW (0.65 >= 0.50)
        assert result.p_win == pytest.approx(0.72, abs=0.01)
        assert result.take is True
        assert result.threshold == 0.50
        mock_calibrator.predict.assert_called_once()

    def test_hybrid_predict_without_calibrator_returns_raw(self):
        """No calibrator in bundle → p_win equals raw probability."""
        bundle = _make_hybrid_bundle()  # default has no calibrator
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        daily_row = _make_mock_daily_features_row()
        with patch.object(predictor, "_get_daily_features", return_value=daily_row):
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="SINGAPORE_OPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=2.5,
                confirm_bars=1,
            )
        # No calibrator → p_win equals raw probability
        assert result.p_win == pytest.approx(0.65, abs=0.01)

    def test_legacy_bundle_no_calibrator(self):
        """Legacy (non-hybrid) bundle without calibrator returns raw p_win."""
        bundle = _make_mock_bundle(threshold=0.55)
        bundle["model"].predict_proba.return_value = np.array([[0.4, 0.6]])
        # Legacy bundles have no "calibrator" key at all

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
        # Legacy: no calibrator → p_win == raw (0.6), take based on raw >= 0.55
        assert result.p_win == pytest.approx(0.6, abs=0.01)
        assert result.take is True

    def test_single_config_per_session_type_recognized(self):
        """model_type='single_config_per_session' is treated as hybrid."""
        bundle = _make_hybrid_bundle(model_type="single_config_per_session")
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        daily_row = _make_mock_daily_features_row()
        with patch.object(predictor, "_get_daily_features", return_value=daily_row):
            result = predictor.predict(
                instrument="MGC",
                trading_day=date(2025, 12, 1),
                orb_label="SINGAPORE_OPEN",
                orb_minutes=5,
                entry_model="E2",
                rr_target=2.5,
                confirm_bars=1,
            )
        # Real prediction, not fail-open
        assert result.p_win != 0.5
        assert predictor.predictions_made == 1


class TestHybridModelInfo:
    """get_model_info for hybrid models returns session details."""

    def test_hybrid_model_info_sessions(self):
        bundle = _make_hybrid_bundle(total_honest_delta_r=97.2)
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        info = predictor.get_model_info("MGC")
        assert info is not None
        assert info["model_type"] == "hybrid_per_session"
        assert info["n_ml_sessions"] == 1  # Only SINGAPORE_OPEN has a model
        assert "SINGAPORE_OPEN" in info["ml_sessions"]
        assert info["total_delta_r"] == 97.2

    def test_hybrid_info_correct_key_name(self):
        """total_delta_r in info reads from total_honest_delta_r in bundle."""
        bundle = _make_hybrid_bundle(total_honest_delta_r=251.9)
        with patch("trading_app.ml.predict_live.LiveMLPredictor._load_models"):
            predictor = LiveMLPredictor(db_path="dummy.db", instruments=["MGC"])
        predictor._models["MGC"] = bundle

        info = predictor.get_model_info("MGC")
        # Verifies the fix: reads "total_honest_delta_r" not "total_delta_r"
        assert info["total_delta_r"] == 251.9

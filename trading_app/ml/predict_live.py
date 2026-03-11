"""Live ML prediction: P(win) for trade entry decisions.

Loads trained meta-label models and predicts P(win) for individual trade
setups. Used by ExecutionEngine to skip low-confidence trades.

Design principles (per de Prado AIFML, per ML_LIVE_INTEGRATION.md):
  1. Fail-open — missing model = trade anyway (0.5, True)
  2. Shared feature pipeline — uses transform_to_features() from features.py
     to guarantee training/serving parity
  3. Hybrid per-session models — each session gets its own RF + threshold
     Sessions without a model fall-open (take all trades).
     Falls back to per-instrument model if no hybrid model exists.
  4. Config hash + freshness checks for drift detection
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import NamedTuple

import duckdb
import joblib
import pandas as pd

from pipeline.db_config import configure_connection
from trading_app.ml.config import (
    ACTIVE_INSTRUMENTS,
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    compute_config_hash,
)
from trading_app.ml.features import apply_e6_filter, transform_to_features

logger = logging.getLogger(__name__)

# Cache eviction: prevent unbounded growth in long backtests.
# 10K entries ≈ 4 instruments × 12 sessions × ~230 trading days.
_MAX_CACHE_SIZE = 10_000


class MLPrediction(NamedTuple):
    """Result of an ML prediction."""

    p_win: float  # Probability of win (0.0 to 1.0)
    take: bool  # True = trade, False = skip
    threshold: float  # Threshold used for the decision


# Prefixes for one-hot encoded columns — used to determine fill values
# during feature alignment. One-hot columns get 0.0, numeric get -999.
_CAT_PREFIXES = tuple(f"{c}_" for c in CATEGORICAL_FEATURES)


def _compute_config_hash() -> str:
    """Compute config hash via shared function in config.py."""
    return compute_config_hash()


class LiveMLPredictor:
    """Predict P(win) for trade setups using trained meta-label models.

    Thread safety: NOT thread-safe. Use one instance per thread/process.
    DuckDB connections are opened per-query and closed immediately.

    Caching:
      - Models: loaded once at init, cached per instrument
      - Daily features: cached per (instrument, trading_day, orb_minutes)
      - Predictions: cached per full combo key (deterministic)
    """

    def __init__(
        self,
        db_path: str,
        instruments: list[str] | None = None,
    ):
        self.db_path = str(db_path)
        self.instruments = instruments or list(ACTIVE_INSTRUMENTS)

        # Caches
        self._models: dict[str, dict] = {}  # instrument → model bundle
        self._daily_cache: dict[tuple, dict] = {}  # (inst, day, orb_min) → row
        self._prediction_cache: dict[tuple, MLPrediction] = {}  # combo → result

        # Stats for summary reporting
        self.predictions_made: int = 0
        self.predictions_cached: int = 0
        self.fail_open_count: int = 0
        self.aperture_mismatch_count: int = 0
        self.rr_mismatch_count: int = 0

        self._load_models()

    def _load_models(self) -> None:
        """Load model .joblib files for all configured instruments.

        Prefers hybrid per-session models (_hybrid.joblib) when available,
        falls back to per-instrument models. This enables gradual migration.
        """
        current_hash = _compute_config_hash()

        for inst in self.instruments:
            # Prefer hybrid model
            hybrid_path = MODEL_DIR / f"meta_label_{inst}_hybrid.joblib"
            legacy_path = MODEL_DIR / f"meta_label_{inst}.joblib"

            path = hybrid_path if hybrid_path.exists() else legacy_path
            if not path.exists():
                logger.warning(
                    "No ML model for %s (checked %s, %s) — will fail-open",
                    inst,
                    hybrid_path,
                    legacy_path,
                )
                continue

            try:
                bundle = joblib.load(path)
                self._models[inst] = bundle

                is_hybrid = bundle.get("model_type") in (
                    "hybrid_per_session",
                    "single_config_per_session",
                )
                if is_hybrid:
                    is_pa = bundle.get("bundle_format") == "per_aperture"
                    if is_pa:
                        n_ml = sum(
                            1
                            for s_data in bundle.get("sessions", {}).values()
                            for a_info in s_data.values()
                            if isinstance(a_info, dict) and a_info.get("model") is not None
                        )
                        n_total = sum(len(s_data) for s_data in bundle.get("sessions", {}).values())
                    else:
                        n_ml = sum(1 for s in bundle.get("sessions", {}).values() if s.get("model") is not None)
                        n_total = len(bundle.get("sessions", {}))
                    fmt_label = "per-aperture" if is_pa else "per-session"
                    logger.info(
                        "Loaded HYBRID ML model for %s (%s): %d/%d models, trained=%s",
                        inst,
                        fmt_label,
                        n_ml,
                        n_total,
                        bundle.get("trained_at", "unknown"),
                    )
                else:
                    logger.info(
                        "Loaded ML model for %s: threshold=%.2f, AUC=%.4f, %d features, trained=%s",
                        inst,
                        bundle["optimal_threshold"],
                        bundle.get("oos_auc", 0),
                        len(bundle["feature_names"]),
                        bundle.get("trained_at", "unknown"),
                    )

                # Config hash check
                model_hash = bundle.get("config_hash")
                if model_hash and model_hash != current_hash:
                    logger.warning(
                        "ML config hash MISMATCH for %s: model=%s, current=%s. "
                        "Model may be stale — consider retraining.",
                        inst,
                        model_hash,
                        current_hash,
                    )

                # Freshness check: warn at >60 days, fail-closed at >90 days
                trained_at_str = bundle.get("trained_at")
                if trained_at_str:
                    try:
                        trained_at = datetime.fromisoformat(trained_at_str)
                        age_days = (datetime.now(UTC) - trained_at).days
                        if age_days > 90:
                            logger.warning(
                                "ML model for %s is %d days old (>90 day limit) — "
                                "REMOVING from active models (fail-closed). Retrain to re-enable.",
                                inst,
                                age_days,
                            )
                            del self._models[inst]
                            continue
                        elif age_days > 60:
                            logger.warning(
                                "ML model for %s is %d days old (>60 day threshold)",
                                inst,
                                age_days,
                            )
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not parse trained_at for %s: %s",
                            inst,
                            trained_at_str,
                        )

            except Exception:
                logger.warning("Failed to load ML model for %s", inst, exc_info=True)

    def predict(
        self,
        instrument: str,
        trading_day: date,
        orb_label: str,
        orb_minutes: int,
        entry_model: str,
        rr_target: float,
        confirm_bars: int,
    ) -> MLPrediction:
        """Predict P(win) for a specific trade setup.

        Args:
            instrument: e.g. "MNQ"
            trading_day: The trading day
            orb_label: e.g. "SINGAPORE_OPEN"
            orb_minutes: ORB aperture (5, 15, or 30)
            entry_model: e.g. "E2"
            rr_target: Risk-reward target (e.g. 1.5)
            confirm_bars: Number of confirm bars (e.g. 1)

        Returns:
            MLPrediction(p_win, take, threshold)
            Fail-open: returns (0.5, True, 0.5) on any error.
        """
        self._evict_caches_if_needed()

        cache_key = (
            instrument,
            trading_day,
            orb_label,
            orb_minutes,
            entry_model,
            rr_target,
            confirm_bars,
        )
        if cache_key in self._prediction_cache:
            self.predictions_cached += 1
            return self._prediction_cache[cache_key]

        # Fail-open: no model for this instrument
        if instrument not in self._models:
            self.fail_open_count += 1
            result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
            self._prediction_cache[cache_key] = result
            return result

        bundle = self._models[instrument]
        is_hybrid = bundle.get("model_type") in (
            "hybrid_per_session",
            "single_config_per_session",
        )

        # Hybrid model: check if this session has a sub-model
        if is_hybrid:
            session_data = bundle.get("sessions", {}).get(orb_label)
            if session_data is None:
                # No model for this session — fail-open (take all trades)
                result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
                self._prediction_cache[cache_key] = result
                return result

            # Detect per-aperture vs flat bundle format
            is_per_aperture = bundle.get("bundle_format") == "per_aperture"
            if is_per_aperture:
                aperture_key = f"O{orb_minutes}"
                session_info = session_data.get(aperture_key, {})
            else:
                session_info = session_data

            if session_info.get("model") is None:
                # No model for this (session, aperture) — fail-open
                result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
                self._prediction_cache[cache_key] = result
                return result

            # Aperture guard: check training aperture matches prediction aperture.
            # Old bundles without training_aperture → fail-open (safe default).
            training_aperture = session_info.get("training_aperture")
            if training_aperture is None:
                logger.warning(
                    "Model for %s %s lacks training_aperture metadata — "
                    "cannot verify aperture match (old model format)",
                    instrument,
                    orb_label,
                )
            elif training_aperture != orb_minutes:
                logger.info(
                    "Aperture mismatch for %s %s: model trained on O%d, prediction for O%d — fail-open",
                    instrument,
                    orb_label,
                    training_aperture,
                    orb_minutes,
                )
                self.aperture_mismatch_count += 1
                result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
                self._prediction_cache[cache_key] = result
                return result

            # RR guard: with family_rr_locks, trade RR should always match
            # training RR. Mismatches indicate stale model or config drift.
            #
            # Aggressive (trade_rr > training_rr) → skip trade. Model trained
            # at lower RR overestimates P(win) at higher RR — dangerous.
            # Conservative (trade_rr < training_rr) → proceed with warning.
            # P(win) is understated (actual win rate is higher) — safe direction.
            training_rr = session_info.get("training_rr")
            if training_rr is not None and rr_target > training_rr:
                logger.debug(
                    "RR mismatch for %s %s: model trained on RR%.1f, "
                    "prediction for RR%.1f (AGGRESSIVE) — skipping trade",
                    instrument,
                    orb_label,
                    training_rr,
                    rr_target,
                )
                self.rr_mismatch_count += 1
                result = MLPrediction(p_win=0.5, take=False, threshold=0.5)
                self._prediction_cache[cache_key] = result
                return result

            if training_rr is not None and rr_target < training_rr:
                logger.debug(
                    "RR conservative for %s %s: model RR%.1f, trade RR%.1f — P(win) may be understated, proceeding",
                    instrument,
                    orb_label,
                    training_rr,
                    rr_target,
                )

        try:
            # Step 1: Get daily features from DB
            daily_row = self._get_daily_features(instrument, trading_day, orb_minutes)
            if daily_row is None:
                logger.warning(
                    "No daily_features for %s %s orb_minutes=%d — fail-open",
                    instrument,
                    trading_day,
                    orb_minutes,
                )
                self.fail_open_count += 1
                result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
                self._prediction_cache[cache_key] = result
                return result

            # Step 2: Build single-row DataFrame matching training format
            df = pd.DataFrame([daily_row])
            df["orb_label"] = orb_label
            df["entry_model"] = entry_model
            df["rr_target"] = float(rr_target)
            df["confirm_bars"] = int(confirm_bars)
            df["orb_minutes"] = int(orb_minutes)

            # Step 3: Transform using SHARED pipeline (same as training)
            X = transform_to_features(df)

            if is_hybrid:
                # Hybrid path: E6 filter + session-specific model
                # session_info already resolved above (flat or per-aperture)
                X = apply_e6_filter(X)
                model = session_info["model"]
                feature_names = session_info["feature_names"]
                threshold = float(session_info["optimal_threshold"])
            else:
                # Legacy path: per-instrument model
                model = bundle.get("model")
                feature_names = bundle.get("feature_names")
                opt_threshold = bundle.get("optimal_threshold")
                if model is None or feature_names is None or opt_threshold is None:
                    missing = [k for k in ("model", "feature_names", "optimal_threshold") if k not in bundle]
                    logger.warning(
                        "Corrupt model bundle for %s: missing keys %s — fail-open",
                        instrument,
                        missing,
                    )
                    self.fail_open_count += 1
                    result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
                    self._prediction_cache[cache_key] = result
                    return result
                threshold = float(opt_threshold)

            # Step 4: Align to model's stored feature_names
            X_aligned = self._align_features(X, feature_names)

            # Step 5: Predict (raw probability)
            p_win_raw = float(model.predict_proba(X_aligned)[:, 1][0])

            # Step 6: Take/skip decision on RAW probability
            # Threshold was optimized on raw RF probabilities during training.
            # Must compare raw probability to raw threshold for consistency.
            take = p_win_raw >= threshold

            # Step 7: Calibrate probability for display/Kelly sizing
            # Isotonic regression makes P(win) meaningful (0.60 ≈ 60% actual).
            # Old bundles without calibrator fall back to raw probability.
            calibrator = None
            if is_hybrid:
                calibrator = session_info.get("calibrator")
            else:
                calibrator = bundle.get("calibrator")

            if calibrator is not None:
                p_win = float(calibrator.predict([p_win_raw])[0])
            else:
                p_win = p_win_raw

            result = MLPrediction(p_win=p_win, take=take, threshold=threshold)
            self._prediction_cache[cache_key] = result
            self.predictions_made += 1
            return result

        except Exception:
            logger.warning(
                "ML prediction failed for %s %s %s — fail-open",
                instrument,
                orb_label,
                trading_day,
                exc_info=True,
            )
            self.fail_open_count += 1
            result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
            self._prediction_cache[cache_key] = result
            return result

    def _get_daily_features(
        self,
        instrument: str,
        trading_day: date,
        orb_minutes: int,
    ) -> dict | None:
        """Fetch daily_features row from DB. Cached per (inst, day, orb_min).

        Backfills global features from orb_minutes=5 when they're NULL
        (pipeline stores global features only at orb_minutes=5 for some
        instruments).
        """
        cache_key = (instrument, trading_day, orb_minutes)
        if cache_key in self._daily_cache:
            return self._daily_cache[cache_key]

        con = duckdb.connect(self.db_path, read_only=True)
        configure_connection(con)
        try:
            result = con.execute(
                """SELECT * FROM daily_features
                   WHERE symbol = ? AND orb_minutes = ? AND trading_day = ?
                   LIMIT 1""",
                [instrument, orb_minutes, trading_day],
            )
            row = result.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in result.description]
            row_dict = dict(zip(columns, row, strict=False))

            # Backfill global features from orb_minutes=5 if needed.
            # Check multiple features — atr_20 may exist at O15 while
            # overnight_range/prev_day_range are NULL.
            if orb_minutes != 5 and any(
                row_dict.get(c) is None for c in ("overnight_range", "prev_day_range", "atr_vel_ratio")
            ):
                g5_result = con.execute(
                    """SELECT * FROM daily_features
                       WHERE symbol = ? AND orb_minutes = 5 AND trading_day = ?
                       LIMIT 1""",
                    [instrument, trading_day],
                )
                g5_row = g5_result.fetchone()
                if g5_row is not None:
                    g5_cols = [desc[0] for desc in g5_result.description]
                    g5_dict = dict(zip(g5_cols, g5_row, strict=False))
                    from trading_app.ml.config import GLOBAL_FEATURES

                    for col in GLOBAL_FEATURES:
                        if row_dict.get(col) is None and g5_dict.get(col) is not None:
                            row_dict[col] = g5_dict[col]

            self._daily_cache[cache_key] = row_dict
            return row_dict
        finally:
            con.close()

    @staticmethod
    def _align_features(
        X: pd.DataFrame,
        model_feature_names: list[str],
    ) -> pd.DataFrame:
        """Align prediction features to model's training feature order.

        - Columns in model but NOT in X: filled with 0.0 (one-hot) or -999 (numeric)
        - Columns in X but NOT in model: dropped (new feature model wasn't trained on)
        - Order matches model exactly (sklearn RF is order-sensitive)
        """
        X_aligned = pd.DataFrame(index=X.index)
        n_filled = 0
        for col in model_feature_names:
            if col in X.columns:
                X_aligned[col] = X[col].values
            elif col.startswith(_CAT_PREFIXES):
                # One-hot column for a category not present → 0.0
                X_aligned[col] = 0.0
                n_filled += 1
            else:
                # Numeric feature missing → -999 (matches training fill)
                X_aligned[col] = -999.0
                n_filled += 1

        n_total = len(model_feature_names)
        if n_total > 0 and n_filled / n_total > 0.30:
            logger.warning(
                "Feature drift: %d/%d columns (%d%%) filled with sentinel values — "
                "model may be stale or feature pipeline changed",
                n_filled,
                n_total,
                int(n_filled / n_total * 100),
            )
        return X_aligned

    def get_model_info(self, instrument: str) -> dict | None:
        """Return model metadata for an instrument (for logging/display)."""
        if instrument not in self._models:
            return None
        b = self._models[instrument]

        if b.get("model_type") in ("hybrid_per_session", "single_config_per_session"):
            is_pa = b.get("bundle_format") == "per_aperture"
            if is_pa:
                models_with_model = [
                    f"{s} {ak}"
                    for s, s_data in b.get("sessions", {}).items()
                    for ak, info in s_data.items()
                    if isinstance(info, dict) and info.get("model") is not None
                ]
            else:
                models_with_model = [s for s, info in b.get("sessions", {}).items() if info.get("model") is not None]
            return {
                "instrument": instrument,
                "model_type": "hybrid_per_session",
                "bundle_format": b.get("bundle_format", "flat"),
                "n_ml_sessions": len(models_with_model),
                "ml_sessions": models_with_model,
                "total_delta_r": b.get("total_honest_delta_r"),
                "total_full_delta_r": b.get("total_full_delta_r"),
                "trained_at": b.get("trained_at"),
                "config_hash": b.get("config_hash"),
                "data_date_range": b.get("data_date_range"),
            }

        return {
            "instrument": instrument,
            "model_type": "per_instrument",
            "threshold": b["optimal_threshold"],
            "n_train": b["n_train"],
            "oos_auc": b.get("oos_auc"),
            "cpcv_auc": b.get("cpcv_auc"),
            "trained_at": b.get("trained_at"),
            "n_features": len(b["feature_names"]),
            "config_hash": b.get("config_hash"),
            "data_date_range": b.get("data_date_range"),
        }

    def _evict_caches_if_needed(self) -> None:
        """Evict caches when they exceed size limit (prevents unbounded growth)."""
        if len(self._prediction_cache) > _MAX_CACHE_SIZE:
            logger.info(
                "Prediction cache exceeded %d entries (%d) — clearing",
                _MAX_CACHE_SIZE,
                len(self._prediction_cache),
            )
            self._prediction_cache.clear()
        if len(self._daily_cache) > _MAX_CACHE_SIZE:
            self._daily_cache.clear()

    def clear_daily_cache(self) -> None:
        """Clear daily features cache (call on new trading day)."""
        self._daily_cache.clear()
        self._prediction_cache.clear()

    def summary(self) -> dict:
        """Return prediction stats for logging."""
        return {
            "models_loaded": list(self._models.keys()),
            "predictions_made": self.predictions_made,
            "predictions_cached": self.predictions_cached,
            "fail_open_count": self.fail_open_count,
            "aperture_mismatch_count": self.aperture_mismatch_count,
            "rr_mismatch_count": self.rr_mismatch_count,
            "daily_cache_size": len(self._daily_cache),
        }

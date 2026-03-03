"""Live ML prediction: P(win) for trade entry decisions.

Loads trained meta-label models and predicts P(win) for individual trade
setups. Used by ExecutionEngine to skip low-confidence trades.

Design principles (per de Prado AIFML, per ML_LIVE_INTEGRATION.md):
  1. Fail-open — missing model = trade anyway (0.5, True)
  2. Shared feature pipeline — uses transform_to_features() from features.py
     to guarantee training/serving parity
  3. One model per instrument — session/entry/rr are INPUT features
  4. Config hash + freshness checks for drift detection
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
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
from trading_app.ml.features import transform_to_features

logger = logging.getLogger(__name__)


class MLPrediction(NamedTuple):
    """Result of an ML prediction."""
    p_win: float     # Probability of win (0.0 to 1.0)
    take: bool       # True = trade, False = skip
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

        self._load_models()

    def _load_models(self) -> None:
        """Load model .joblib files for all configured instruments."""
        current_hash = _compute_config_hash()

        for inst in self.instruments:
            path = MODEL_DIR / f"meta_label_{inst}.joblib"
            if not path.exists():
                logger.warning(
                    "No ML model for %s (expected %s) — will fail-open", inst, path
                )
                continue

            try:
                bundle = joblib.load(path)
                self._models[inst] = bundle
                logger.info(
                    "Loaded ML model for %s: threshold=%.2f, AUC=%.4f, "
                    "%d features, trained=%s",
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
                        inst, model_hash, current_hash,
                    )

                # Freshness check (isolated — parsing failure must not prevent model use)
                trained_at_str = bundle.get("trained_at")
                if trained_at_str:
                    try:
                        trained_at = datetime.fromisoformat(trained_at_str)
                        age_days = (datetime.now(timezone.utc) - trained_at).days
                        if age_days > 90:
                            logger.warning(
                                "ML model for %s is %d days old (>90 day threshold)",
                                inst, age_days,
                            )
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not parse trained_at for %s: %s",
                            inst, trained_at_str,
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
        cache_key = (
            instrument, trading_day, orb_label, orb_minutes,
            entry_model, rr_target, confirm_bars,
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

        try:
            # Step 1: Get daily features from DB
            daily_row = self._get_daily_features(instrument, trading_day, orb_minutes)
            if daily_row is None:
                logger.warning(
                    "No daily_features for %s %s orb_minutes=%d — fail-open",
                    instrument, trading_day, orb_minutes,
                )
                self.fail_open_count += 1
                result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
                self._prediction_cache[cache_key] = result
                return result

            # Step 2: Build single-row DataFrame matching training format
            df = pd.DataFrame([daily_row])
            # Overlay trade config (these come from the strategy, not DB)
            df["orb_label"] = orb_label
            df["entry_model"] = entry_model
            df["rr_target"] = float(rr_target)
            df["confirm_bars"] = int(confirm_bars)
            # orb_minutes is already in daily_features row, ensure consistency
            df["orb_minutes"] = int(orb_minutes)

            # Step 3: Transform using SHARED pipeline (same as training)
            X = transform_to_features(df)

            # Step 4: Align to model's stored feature_names
            X_aligned = self._align_features(X, bundle["feature_names"])

            # Step 5: Predict
            model = bundle["model"]
            p_win = float(model.predict_proba(X_aligned)[:, 1][0])
            threshold = float(bundle["optimal_threshold"])
            take = p_win >= threshold

            result = MLPrediction(p_win=p_win, take=take, threshold=threshold)
            self._prediction_cache[cache_key] = result
            self.predictions_made += 1
            return result

        except Exception:
            logger.warning(
                "ML prediction failed for %s %s %s — fail-open",
                instrument, orb_label, trading_day,
                exc_info=True,
            )
            self.fail_open_count += 1
            result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
            self._prediction_cache[cache_key] = result
            return result

    def _get_daily_features(
        self, instrument: str, trading_day: date, orb_minutes: int,
    ) -> dict | None:
        """Fetch daily_features row from DB. Cached per (inst, day, orb_min)."""
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
            row_dict = dict(zip(columns, row))
            self._daily_cache[cache_key] = row_dict
            return row_dict
        finally:
            con.close()

    @staticmethod
    def _align_features(
        X: pd.DataFrame, model_feature_names: list[str],
    ) -> pd.DataFrame:
        """Align prediction features to model's training feature order.

        - Columns in model but NOT in X: filled with 0.0 (one-hot) or -999 (numeric)
        - Columns in X but NOT in model: dropped (new feature model wasn't trained on)
        - Order matches model exactly (sklearn RF is order-sensitive)
        """
        X_aligned = pd.DataFrame(index=X.index)
        for col in model_feature_names:
            if col in X.columns:
                X_aligned[col] = X[col].values
            elif col.startswith(_CAT_PREFIXES):
                # One-hot column for a category not present → 0.0
                X_aligned[col] = 0.0
            else:
                # Numeric feature missing → -999 (matches training fill)
                X_aligned[col] = -999.0
        return X_aligned

    def get_model_info(self, instrument: str) -> dict | None:
        """Return model metadata for an instrument (for logging/display)."""
        if instrument not in self._models:
            return None
        b = self._models[instrument]
        return {
            "instrument": instrument,
            "threshold": b["optimal_threshold"],
            "n_train": b["n_train"],
            "oos_auc": b.get("oos_auc"),
            "cpcv_auc": b.get("cpcv_auc"),
            "trained_at": b.get("trained_at"),
            "n_features": len(b["feature_names"]),
            "config_hash": b.get("config_hash"),
            "data_date_range": b.get("data_date_range"),
        }

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
            "daily_cache_size": len(self._daily_cache),
        }

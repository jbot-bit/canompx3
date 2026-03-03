"""Feature extraction: daily_features + orb_outcomes → ML feature matrix.

Single feature extraction path used by BOTH training and live prediction.
No redundant code — add a feature in config.py, it propagates everywhere.

Key function:
  transform_to_features(df) — shared transformation used by BOTH:
    1. load_feature_matrix() for bulk training data
    2. LiveMLPredictor.predict() for single-row live prediction
  This guarantees training/serving parity. Any feature change in config.py
  propagates to both paths automatically.
"""

from __future__ import annotations

import logging
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from pipeline.db_config import configure_connection
from trading_app.ml.config import (
    ATR_NORMALIZE,
    CATEGORICAL_FEATURES,
    GLOBAL_FEATURES,
    LOOKAHEAD_BLACKLIST,
    SESSION_FEATURE_SUFFIXES,
    TRADE_CONFIG_FEATURES,
)

logger = logging.getLogger(__name__)


def _extract_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-session ORB features based on each row's orb_label.

    Instead of carrying all 11 sessions' columns, we extract only the
    traded session's features into generic columns: orb_size, orb_volume,
    rel_vol, break_delay_min, break_bar_continues, break_dir.
    """
    result = pd.DataFrame(index=df.index)

    for suffix in SESSION_FEATURE_SUFFIXES:
        generic_col = f"orb_{suffix}" if suffix != "break_dir" else "break_dir"
        is_bool_feat = suffix == "break_bar_continues"
        is_str_feat = suffix == "break_dir"
        if is_str_feat:
            values = pd.Series(np.nan, index=df.index, dtype="object")
        elif is_bool_feat:
            values = pd.Series(np.nan, index=df.index, dtype="float64")
        else:
            values = pd.Series(np.nan, index=df.index, dtype="float64")

        for session in df["orb_label"].unique():
            mask = df["orb_label"] == session
            col_name = f"orb_{session}_{suffix}" if suffix != "break_dir" else f"orb_{session}_break_dir"
            # rel_vol has different naming pattern
            if suffix == "volume":
                col_name = f"orb_{session}_volume"
            elif suffix == "break_bar_volume":
                col_name = f"orb_{session}_break_bar_volume"

            if col_name not in df.columns:
                # Compression_z and some features only exist for certain sessions
                continue
            raw = df.loc[mask, col_name]
            if is_bool_feat:
                # Convert boolean/nullable boolean to float (True=1, False=0, NA=NaN)
                raw = pd.to_numeric(raw, errors="coerce").astype("float64")
            elif not is_str_feat:
                raw = raw.astype("float64", errors="ignore")
            values.loc[mask] = raw

        result[generic_col] = values

    # Also extract rel_vol for the traded session
    rel_vol = pd.Series(np.nan, index=df.index)
    for session in df["orb_label"].unique():
        mask = df["orb_label"] == session
        col_name = f"rel_vol_{session}"
        if col_name in df.columns:
            rel_vol.loc[mask] = df.loc[mask, col_name]
    result["rel_vol"] = rel_vol

    return result


def _normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize price-level features by ATR for stationarity."""
    df = df.copy()
    atr = df["atr_20"].replace(0, np.nan)

    for col in ATR_NORMALIZE:
        if col in df.columns:
            df[f"{col}_norm"] = df[col] / atr
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features, drop originals."""
    cats_present = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if not cats_present:
        return df

    # Convert to string to handle NaN gracefully
    for c in cats_present:
        df[c] = df[c].astype(str).replace({"nan": "UNKNOWN", "None": "UNKNOWN"})

    df = pd.get_dummies(df, columns=cats_present, drop_first=False, dtype=float)
    return df


def transform_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a DataFrame into an ML feature matrix.

    Shared between training (load_feature_matrix) and live prediction
    (LiveMLPredictor). This is the SINGLE feature pipeline — any feature
    change in config.py propagates to both callers automatically.

    Input DataFrame must have:
      - orb_label column (for session feature extraction)
      - All daily_features columns (global features, session ORB columns)
      - Trade config columns: confirm_bars, orb_minutes
      - entry_model column (for categorical encoding)

    Returns:
        Feature matrix (float, ready for sklearn). Columns may vary based on
        which categories are present — alignment to model's feature_names is
        the caller's responsibility.
    """
    # --- Extract session-specific features ---
    session_feats = _extract_session_features(df)

    # --- Build feature set ---
    # Start with global features
    feature_cols = [c for c in GLOBAL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()

    # Add session-specific features
    for col in session_feats.columns:
        X[col] = session_feats[col]

    # Add trade config features
    for col in TRADE_CONFIG_FEATURES:
        if col in df.columns:
            X[col] = df[col]

    # Add categoricals that need encoding
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and col not in X.columns:
            X[col] = df[col]
        elif col in session_feats.columns and col not in X.columns:
            X[col] = session_feats[col]

    # Add gap_type and atr_vel_regime from daily_features
    for col in ["gap_type", "atr_vel_regime"]:
        if col in df.columns and col not in X.columns:
            X[col] = df[col]

    # --- Safety: remove any look-ahead columns ---
    # Exact match OR substring containment (catches orb_{SESSION}_mae_r etc.)
    for col in list(X.columns):
        if col in LOOKAHEAD_BLACKLIST or any(bl in col for bl in LOOKAHEAD_BLACKLIST):
            X.drop(columns=col, inplace=True)
            logger.warning(f"Dropped look-ahead column: {col}")

    # --- Normalize ---
    X = _normalize_features(X)

    # --- Encode categoricals ---
    X = _encode_categoricals(X)

    # --- Final cleanup ---
    # Drop duplicate columns from the join (d.trading_day, d.symbol, etc.)
    drop_cols = [c for c in X.columns if c in ("trading_day", "symbol",
                 "trading_day:1", "symbol:1", "orb_minutes:1")]
    X.drop(columns=[c for c in drop_cols if c in X.columns], inplace=True, errors="ignore")

    # Convert bool columns to float
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(float)

    # Fill NaN with -999 (sklearn RF handles missing values poorly)
    X = X.fillna(-999.0)

    # Ensure all columns are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"Dropping non-numeric columns: {non_numeric}")
        X.drop(columns=non_numeric, inplace=True)

    return X


def load_feature_matrix(
    db_path: str,
    instrument: str,
    *,
    orb_minutes: Optional[int] = None,
    entry_model: Optional[str] = None,
    orb_label: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load and prepare the full ML feature matrix for an instrument.

    Returns:
        X: Feature matrix (float, ready for sklearn)
        y: Binary target (1=win, 0=loss)
        meta: DataFrame with trading_day, pnl_r, orb_label, etc. for evaluation
    """
    con = duckdb.connect(db_path, read_only=True)
    configure_connection(con)
    try:
        # Build WHERE clause
        where_parts = ["o.symbol = $instrument", "o.pnl_r IS NOT NULL"]
        params: dict = {"instrument": instrument}

        if orb_minutes is not None:
            where_parts.append("o.orb_minutes = $orb_minutes")
            params["orb_minutes"] = orb_minutes
        if entry_model is not None:
            where_parts.append("o.entry_model = $entry_model")
            params["entry_model"] = entry_model
        if orb_label is not None:
            where_parts.append("o.orb_label = $orb_label")
            params["orb_label"] = orb_label
        if min_date is not None:
            where_parts.append("o.trading_day >= $min_date")
            params["min_date"] = min_date
        if max_date is not None:
            where_parts.append("o.trading_day <= $max_date")
            params["max_date"] = max_date

        where_clause = " AND ".join(where_parts)

        query = f"""
            SELECT
                o.trading_day,
                o.symbol,
                o.orb_label,
                o.orb_minutes,
                o.entry_model,
                o.rr_target,
                o.confirm_bars,
                o.pnl_r,
                o.outcome,
                d.*
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            WHERE {where_clause}
            ORDER BY o.trading_day
        """

        df = con.execute(query, params).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise ValueError(f"No data for {instrument} with filters: {params}")

    logger.info(f"Loaded {len(df):,d} outcomes for {instrument}")

    # --- Target ---
    y = (df["pnl_r"] > 0).astype(int)

    # --- Meta (for evaluation, not features) ---
    meta = df[["trading_day", "symbol", "orb_label", "orb_minutes",
               "entry_model", "rr_target", "confirm_bars", "pnl_r", "outcome"]].copy()

    # --- Transform to features (shared with live prediction) ---
    X = transform_to_features(df)

    logger.info(f"Feature matrix: {X.shape[0]:,d} rows x {X.shape[1]} features")
    return X, y, meta

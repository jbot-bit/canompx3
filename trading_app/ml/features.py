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
    E6_NOISE_EXACT,
    E6_NOISE_PREFIXES,
    GLOBAL_FEATURES,
    LOOKAHEAD_BLACKLIST,
    SESSION_CHRONOLOGICAL_ORDER,
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


def _extract_cross_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cross-session features: prior session ORB break counts.

    For each trade, looks at sessions EARLIER in the day to count:
    - How many prior sessions had an ORB break
    - Directional counts (LONG vs SHORT breaks)

    These features are genuine pre-entry information — a trader knows
    which prior sessions broke before deciding on the current session.

    WARNING: In a per-instrument model, these features leak session
    identity (prior_sessions_broken is nearly deterministic per session).
    Use per-session models to avoid this bias.

    Vectorized: iterates per session pair (O(sessions^2)), not per row.
    """
    n = len(df)
    broken = np.zeros(n)
    long_count = np.zeros(n)
    short_count = np.zeros(n)

    # For each current session, accumulate break counts from prior sessions
    for session_idx, session in enumerate(SESSION_CHRONOLOGICAL_ORDER):
        mask = (df["orb_label"] == session).values
        if not mask.any():
            continue

        for ps in SESSION_CHRONOLOGICAL_ORDER[:session_idx]:
            break_col = f"orb_{ps}_break_dir"
            if break_col not in df.columns:
                continue

            bd = df.loc[mask, break_col].astype(str).str.lower()
            is_long = (bd == "long").values.astype(float)
            is_short = (bd == "short").values.astype(float)

            broken[mask] += is_long + is_short
            long_count[mask] += is_long
            short_count[mask] += is_short

    return pd.DataFrame({
        "prior_sessions_broken": broken,
        "prior_sessions_long": long_count,
        "prior_sessions_short": short_count,
    }, index=df.index)


def _extract_level_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract level proximity features: prior session ORB levels as magnets.

    For each trade, computes distances from current ORB boundaries to
    prior session ORB highs/lows (in R = current ORB size units):
    - Nearest prior level distance from ORB high/low
    - Count of prior levels within 1R and 2R
    - Whether current ORB is nested inside a prior ORB
    - Max prior ORB size relative to current (bigger = stronger magnet)

    All distances are in R (multiples of current ORB size) for stationarity.

    Vectorized: iterates per session pair (O(sessions^2)), not per row.
    """
    n = len(df)
    nearest_to_high = np.full(n, np.inf)
    nearest_to_low = np.full(n, np.inf)
    levels_within_1r = np.zeros(n)
    levels_within_2r = np.zeros(n)
    is_nested = np.zeros(n)
    prior_size_ratio_max = np.full(n, -999.0)
    has_prior_levels = np.zeros(n, dtype=bool)

    for session_idx, session in enumerate(SESSION_CHRONOLOGICAL_ORDER):
        mask = (df["orb_label"] == session).values
        if not mask.any() or session_idx == 0:
            continue

        # Current session's ORB boundaries
        h_col = f"orb_{session}_high"
        l_col = f"orb_{session}_low"
        s_col = f"orb_{session}_size"
        if h_col not in df.columns or l_col not in df.columns or s_col not in df.columns:
            continue

        cur_high = df.loc[mask, h_col].values.astype(float)
        cur_low = df.loc[mask, l_col].values.astype(float)
        cur_size = df.loc[mask, s_col].values.astype(float)

        # Valid rows: non-NaN and positive size
        valid_base = ~np.isnan(cur_high) & ~np.isnan(cur_low) & ~np.isnan(cur_size) & (cur_size > 0)
        if not valid_base.any():
            continue

        R = np.where(valid_base, cur_size, np.nan)
        mask_indices = np.where(mask)[0]

        for ps in SESSION_CHRONOLOGICAL_ORDER[:session_idx]:
            ps_h_col = f"orb_{ps}_high"
            ps_l_col = f"orb_{ps}_low"
            ps_s_col = f"orb_{ps}_size"

            # Process each prior level (high and low separately)
            for level_col in [ps_h_col, ps_l_col]:
                if level_col not in df.columns:
                    continue

                level = df.loc[mask, level_col].values.astype(float)
                valid = valid_base & ~np.isnan(level)
                if not valid.any():
                    continue

                has_prior_levels[mask_indices[valid]] = True

                d_to_high = np.where(valid, np.abs(level - cur_high) / R, np.inf)
                d_to_low = np.where(valid, np.abs(level - cur_low) / R, np.inf)
                d_min = np.minimum(d_to_high, d_to_low)

                # Update running minimums (scatter into global arrays)
                nearest_to_high[mask_indices] = np.minimum(
                    nearest_to_high[mask_indices], d_to_high)
                nearest_to_low[mask_indices] = np.minimum(
                    nearest_to_low[mask_indices], d_to_low)

                # Count levels within thresholds
                levels_within_1r[mask_indices] += np.where(valid & (d_min <= 1.0), 1.0, 0.0)
                levels_within_2r[mask_indices] += np.where(valid & (d_min <= 2.0), 1.0, 0.0)

            # Nesting check: current inside prior (need both high and low)
            if ps_h_col in df.columns and ps_l_col in df.columns:
                ps_high = df.loc[mask, ps_h_col].values.astype(float)
                ps_low = df.loc[mask, ps_l_col].values.astype(float)
                valid_nest = valid_base & ~np.isnan(ps_high) & ~np.isnan(ps_low)
                nested = valid_nest & (cur_high <= ps_high) & (cur_low >= ps_low)
                is_nested[mask_indices] = np.maximum(
                    is_nested[mask_indices], nested.astype(float))

            # Size ratio: prior size / current size
            if ps_s_col in df.columns:
                ps_size = df.loc[mask, ps_s_col].values.astype(float)
                valid_size = valid_base & ~np.isnan(ps_size) & (ps_size > 0)
                if valid_size.any():
                    ratio = np.where(valid_size, ps_size / R, -999.0)
                    prior_size_ratio_max[mask_indices] = np.maximum(
                        prior_size_ratio_max[mask_indices], ratio)

    # Replace inf with -999 for rows without prior levels
    nearest_to_high = np.where(has_prior_levels, nearest_to_high, -999.0)
    nearest_to_low = np.where(has_prior_levels, nearest_to_low, -999.0)

    return pd.DataFrame({
        "nearest_level_to_high_R": nearest_to_high,
        "nearest_level_to_low_R": nearest_to_low,
        "levels_within_1R": levels_within_1r,
        "levels_within_2R": levels_within_2r,
        "orb_nested_in_prior": is_nested,
        "prior_orb_size_ratio_max": prior_size_ratio_max,
    }, index=df.index)


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


def apply_e6_filter(X: pd.DataFrame) -> pd.DataFrame:
    """Remove E6 noise features from a feature matrix.

    Drops columns matching E6_NOISE_PREFIXES (one-hot encoded categoricals
    with <1% importance) and E6_NOISE_EXACT (near-constant features).

    @research-source: ml_level_proximity_experiment.py (Mar 4 2026)
        All had <1% importance across all 4 instruments in E3/E6 experiments.
        orb_label one-hots cause session identity leakage (11-13% importance
        but tautological — the model learns "which session" not "which regime").
    """
    drop_cols = []
    for col in X.columns:
        if any(col.startswith(prefix) for prefix in E6_NOISE_PREFIXES):
            drop_cols.append(col)
        elif col in E6_NOISE_EXACT:
            drop_cols.append(col)

    if drop_cols:
        logger.info(f"E6 filter: dropping {len(drop_cols)} noise columns")
        return X.drop(columns=drop_cols)
    return X


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

    # --- Extract cross-session + level proximity features ---
    cross_feats = _extract_cross_session_features(df)
    level_feats = _extract_level_proximity_features(df)

    # --- Build feature set ---
    # Start with global features
    feature_cols = [c for c in GLOBAL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()

    # Add session-specific features
    for col in session_feats.columns:
        X[col] = session_feats[col]

    # Add cross-session features
    for col in cross_feats.columns:
        X[col] = cross_feats[col]

    # Add level proximity features
    for col in level_feats.columns:
        X[col] = level_feats[col]

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


def load_validated_feature_matrix(
    db_path: str,
    instrument: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load ML feature matrix filtered to ONLY validated strategy outcomes.

    Per de Prado meta-labeling methodology: the meta-model must train only
    on outcomes where the primary model (validated ORB rules) would trigger.
    This means:
      1. Only (orb_label, entry_model, rr_target, confirm_bars, orb_minutes) combos
         that appear in validated_setups
      2. Only trading days where the strategy's filter_type condition passes

    This eliminates population mismatch: training distribution matches
    deployment distribution (paper_trader / live engine).

    Returns:
        X: Feature matrix (float, ready for sklearn)
        y: Binary target (1=win, 0=loss)
        meta: DataFrame with trading_day, pnl_r, orb_label, filter_type, etc.
    """
    from trading_app.config import ALL_FILTERS

    con = duckdb.connect(db_path, read_only=True)
    configure_connection(con)
    try:
        # Join orb_outcomes to validated_setups to get only validated combos,
        # and to daily_features for market-condition features + filter columns.
        query = """
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
                v.filter_type,
                d.*
            FROM orb_outcomes o
            JOIN validated_setups v
                ON o.symbol = v.instrument
                AND o.orb_label = v.orb_label
                AND o.entry_model = v.entry_model
                AND o.rr_target = v.rr_target
                AND o.confirm_bars = v.confirm_bars
                AND o.orb_minutes = v.orb_minutes
            JOIN daily_features d
                ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = $instrument
                AND o.pnl_r IS NOT NULL
                AND v.status = 'active'
            ORDER BY o.trading_day
        """

        df = con.execute(query, {"instrument": instrument}).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise ValueError(f"No validated outcomes for {instrument}")

    n_before_filter = len(df)
    logger.info(f"Loaded {n_before_filter:,d} validated-combo outcomes for {instrument}")

    # Apply filter_type eligibility conditions per row.
    # Each row has a filter_type from validated_setups — check if the
    # daily_features conditions for that filter pass on that trading day.
    keep_mask = np.zeros(len(df), dtype=bool)
    filter_cache: dict[str, object] = {}

    for idx, row in df.iterrows():
        ft = row["filter_type"]
        orb_label = row["orb_label"]

        if ft not in filter_cache:
            filter_cache[ft] = ALL_FILTERS.get(ft)

        filt = filter_cache[ft]
        if filt is None:
            # Unknown filter — fail-closed, skip
            logger.warning(f"Unknown filter_type '{ft}' — skipping")
            continue

        if filt.matches_row(row.to_dict(), orb_label):
            keep_mask[idx] = True

    df = df[keep_mask].reset_index(drop=True)
    n_after_filter = len(df)
    logger.info(f"After filter eligibility: {n_after_filter:,d} rows "
                f"({n_before_filter - n_after_filter:,d} filtered out, "
                f"{n_after_filter / max(n_before_filter, 1):.1%} kept)")

    # Deduplicate: same (trading_day, orb_label, entry_model, rr_target,
    # confirm_bars, orb_minutes) outcome may appear multiple times if
    # multiple validated strategies share those params but differ in filter_type.
    # Keep only unique outcomes (the outcome itself is identical).
    dedup_cols = ["trading_day", "orb_label", "entry_model", "rr_target",
                  "confirm_bars", "orb_minutes"]
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    if len(df) < n_before_dedup:
        logger.info(f"Deduped: {n_before_dedup:,d} → {len(df):,d} unique outcomes")

    # --- Target ---
    y = (df["pnl_r"] > 0).astype(int)

    # --- Meta (for evaluation, not features) ---
    meta = df[["trading_day", "symbol", "orb_label", "orb_minutes",
               "entry_model", "rr_target", "confirm_bars", "pnl_r",
               "outcome", "filter_type"]].copy()

    # --- Transform to features (shared with live prediction) ---
    X = transform_to_features(df)

    logger.info(f"Validated feature matrix: {X.shape[0]:,d} rows x {X.shape[1]} features "
                f"(win rate: {y.mean():.1%})")
    return X, y, meta

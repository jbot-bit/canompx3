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


def _backfill_global_features(
    df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> pd.DataFrame:
    """Fill missing global features from orb_minutes=5 rows.

    Some instruments have NULL global features for orb_minutes=15/30 in
    daily_features (pipeline computes them only at orb_minutes=5). Global
    features don't depend on ORB aperture, so orb_minutes=5 values are
    a valid fallback.

    Called BEFORE the DB connection is closed — requires the open connection.
    """
    # Quick check: if ALL global features are present, skip.
    # Must check every feature — some (atr_20) may be populated at O15/O30
    # while others (prev_day_range, atr_vel_ratio) are NULL.
    any_missing = False
    for col in GLOBAL_FEATURES:
        if col in df.columns and df[col].isna().any():
            any_missing = True
            break
    if not any_missing:
        return df

    n_missing = max(df[col].isna().sum() for col in GLOBAL_FEATURES if col in df.columns)
    logger.info(f"Backfilling global features: {n_missing:,d}/{len(df):,d} rows missing (orb_minutes≠5)")

    global_df = con.execute(
        """SELECT trading_day, """
        + ", ".join(GLOBAL_FEATURES)
        + """ FROM daily_features
            WHERE symbol = $instrument AND orb_minutes = 5""",
        {"instrument": instrument},
    ).fetchdf()

    # Merge on trading_day, fill only NULL values
    df = df.merge(global_df, on="trading_day", how="left", suffixes=("", "_g5"))
    for col in GLOBAL_FEATURES:
        g5_col = f"{col}_g5"
        if g5_col in df.columns:
            mask = df[col].isna() & df[g5_col].notna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, g5_col].values
            df.drop(columns=[g5_col], inplace=True)

    n_still_missing = max(
        (df[col].isna().sum() for col in GLOBAL_FEATURES if col in df.columns),
        default=0,
    )
    if n_still_missing > 0:
        logger.warning(
            f"  {n_still_missing:,d} rows still missing global features "
            f"after backfill (no orb_minutes=5 row for those days)"
        )
    return df


def _extract_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-session ORB features based on each row's orb_label.

    Instead of carrying all 12 sessions' columns, we extract only the
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

    return pd.DataFrame(
        {
            "prior_sessions_broken": broken,
            "prior_sessions_long": long_count,
            "prior_sessions_short": short_count,
        },
        index=df.index,
    )


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
                nearest_to_high[mask_indices] = np.minimum(nearest_to_high[mask_indices], d_to_high)
                nearest_to_low[mask_indices] = np.minimum(nearest_to_low[mask_indices], d_to_low)

                # Count levels within thresholds
                levels_within_1r[mask_indices] += np.where(valid & (d_min <= 1.0), 1.0, 0.0)
                levels_within_2r[mask_indices] += np.where(valid & (d_min <= 2.0), 1.0, 0.0)

            # Nesting check: current inside prior (need both high and low)
            if ps_h_col in df.columns and ps_l_col in df.columns:
                ps_high = df.loc[mask, ps_h_col].values.astype(float)
                ps_low = df.loc[mask, ps_l_col].values.astype(float)
                valid_nest = valid_base & ~np.isnan(ps_high) & ~np.isnan(ps_low)
                nested = valid_nest & (cur_high <= ps_high) & (cur_low >= ps_low)
                is_nested[mask_indices] = np.maximum(is_nested[mask_indices], nested.astype(float))

            # Size ratio: prior size / current size
            if ps_s_col in df.columns:
                ps_size = df.loc[mask, ps_s_col].values.astype(float)
                valid_size = valid_base & ~np.isnan(ps_size) & (ps_size > 0)
                if valid_size.any():
                    ratio = np.where(valid_size, ps_size / R, -999.0)
                    prior_size_ratio_max[mask_indices] = np.maximum(prior_size_ratio_max[mask_indices], ratio)

    # Replace inf with -999 for rows without prior levels
    nearest_to_high = np.where(has_prior_levels, nearest_to_high, -999.0)
    nearest_to_low = np.where(has_prior_levels, nearest_to_low, -999.0)

    return pd.DataFrame(
        {
            "nearest_level_to_high_R": nearest_to_high,
            "nearest_level_to_low_R": nearest_to_low,
            "levels_within_1R": levels_within_1r,
            "levels_within_2R": levels_within_2r,
            "orb_nested_in_prior": is_nested,
            "prior_orb_size_ratio_max": prior_size_ratio_max,
        },
        index=df.index,
    )


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
        if any(col.startswith(prefix) for prefix in E6_NOISE_PREFIXES) or col in E6_NOISE_EXACT:
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

    # --- Session chronological guard (catches cross-session look-ahead) ---
    # For per-session models, mask features from sessions that haven't happened yet.
    # This is the RUNTIME enforcement of pipeline.session_guard.
    #
    # NOTE: By this point, _extract_session_features() has already converted
    # session-prefixed columns (orb_TOKYO_OPEN_size) into generic names (orb_size).
    # Generic names are SAFE — the extraction guarantees same-session-only data.
    # Cross-session features (prior_sessions_broken, levels_within_2R) are also
    # safe — _extract_cross_session_features uses SESSION_CHRONOLOGICAL_ORDER[:idx].
    #
    # The guard here catches any RAW daily_features columns that leaked through
    # without going through extraction (e.g. from df[feature_cols].copy() above).
    _GENERIC_SAFE = {
        "orb_size", "orb_volume", "rel_vol", "break_dir",
        "orb_break_bar_continues", "orb_size_norm",
        "orb_vwap", "orb_vwap_norm", "orb_pre_velocity", "orb_pre_velocity_norm",
        "prior_sessions_broken", "prior_sessions_long", "prior_sessions_short",
        "nearest_level_to_high_R", "nearest_level_to_low_R",
        "levels_within_1R", "levels_within_2R",
        "orb_nested_in_prior", "prior_orb_size_ratio_max",
    }
    if "orb_label" in df.columns:
        try:
            from pipeline.session_guard import is_feature_safe
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for session in df["orb_label"].unique():
                session_mask = df["orb_label"] == session
                for col in numeric_cols:
                    if col in _GENERIC_SAFE:
                        continue  # extracted by same-session/chronological functions
                    if not is_feature_safe(col, session):
                        X.loc[session_mask, col] = -999.0  # sentinel = unknown
                        logger.warning(f"Session guard: masked {col} for {session}")
        except ImportError:
            logger.error("session_guard import failed — look-ahead protection DISABLED")

    # --- Normalize ---
    X = _normalize_features(X)

    # --- Encode categoricals ---
    X = _encode_categoricals(X)

    # --- Final cleanup ---
    # Drop duplicate columns from the join (d.trading_day, d.symbol, etc.)
    drop_cols = [c for c in X.columns if c in ("trading_day", "symbol", "trading_day:1", "symbol:1", "orb_minutes:1")]
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
    orb_minutes: int | None = None,
    entry_model: str | None = None,
    orb_label: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
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

        # Backfill global features for orb_minutes != 5 (pipeline gap)
        df = _backfill_global_features(df, con, instrument)
    finally:
        con.close()

    if df.empty:
        raise ValueError(f"No data for {instrument} with filters: {params}")

    logger.info(f"Loaded {len(df):,d} outcomes for {instrument}")

    # --- Target ---
    y = (df["pnl_r"] > 0).astype(int)

    # --- Meta (for evaluation, not features) ---
    meta = df[
        [
            "trading_day",
            "symbol",
            "orb_label",
            "orb_minutes",
            "entry_model",
            "rr_target",
            "confirm_bars",
            "pnl_r",
            "outcome",
        ]
    ].copy()

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
            LEFT JOIN family_rr_locks frl
                ON v.instrument = frl.instrument
                AND v.orb_label = frl.orb_label
                AND v.filter_type = frl.filter_type
                AND v.entry_model = frl.entry_model
                AND v.orb_minutes = frl.orb_minutes
                AND v.confirm_bars = frl.confirm_bars
            JOIN daily_features d
                ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = $instrument
                AND o.pnl_r IS NOT NULL
                AND v.status = 'active'
                AND (frl.locked_rr IS NULL OR v.rr_target = frl.locked_rr)
            ORDER BY o.trading_day
        """

        df = con.execute(query, {"instrument": instrument}).fetchdf()

        # Backfill global features for orb_minutes != 5 (pipeline gap)
        df = _backfill_global_features(df, con, instrument)
    finally:
        con.close()

    if df.empty:
        raise ValueError(f"No validated outcomes for {instrument}")

    n_before_filter = len(df)
    logger.info(f"Loaded {n_before_filter:,d} validated-combo outcomes for {instrument}")

    # Apply filter_type eligibility conditions per group (vectorized).
    # Group by (filter_type, orb_label) to apply each filter's matches_df
    # to its session's data in bulk — O(groups) pandas ops vs O(N) Python loop.
    keep_mask = pd.Series(False, index=df.index)
    filter_cache: dict[str, object] = {}

    for (ft_name, orb_label), group_idx in df.groupby(["filter_type", "orb_label"]).groups.items():
        if ft_name not in filter_cache:
            filter_cache[ft_name] = ALL_FILTERS.get(ft_name)

        filt = filter_cache[ft_name]
        if filt is None:
            logger.warning(f"Unknown filter_type '{ft_name}' — skipping")
            continue

        keep_mask.loc[group_idx] = filt.matches_df(df.loc[group_idx], orb_label)

    df = df[keep_mask].reset_index(drop=True)
    n_after_filter = len(df)
    logger.info(
        f"After filter eligibility: {n_after_filter:,d} rows "
        f"({n_before_filter - n_after_filter:,d} filtered out, "
        f"{n_after_filter / max(n_before_filter, 1):.1%} kept)"
    )

    # Deduplicate: same (trading_day, orb_label, entry_model, rr_target,
    # confirm_bars, orb_minutes) outcome may appear multiple times if
    # multiple validated strategies share those params but differ in filter_type.
    # Keep only unique outcomes (the outcome itself is identical).
    dedup_cols = ["trading_day", "orb_label", "entry_model", "rr_target", "confirm_bars", "orb_minutes"]
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    if len(df) < n_before_dedup:
        logger.info(f"Deduped: {n_before_dedup:,d} → {len(df):,d} unique outcomes")

    # --- Target ---
    y = (df["pnl_r"] > 0).astype(int)

    # --- Meta (for evaluation, not features) ---
    meta = df[
        [
            "trading_day",
            "symbol",
            "orb_label",
            "orb_minutes",
            "entry_model",
            "rr_target",
            "confirm_bars",
            "pnl_r",
            "outcome",
            "filter_type",
        ]
    ].copy()

    # --- Transform to features (shared with live prediction) ---
    X = transform_to_features(df)

    logger.info(f"Validated feature matrix: {X.shape[0]:,d} rows x {X.shape[1]} features (win rate: {y.mean():.1%})")
    return X, y, meta


def load_single_config_feature_matrix(
    db_path: str,
    instrument: str,
    *,
    rr_target: float | None = None,
    config_selection: str = "max_samples",
    skip_filter: bool = False,
    per_aperture: bool = False,
    apply_rr_lock: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load ML feature matrix with ONE config per session — clean labels.

    Fixes the contradictory-label problem: instead of loading ALL validated
    configs (N rows per day per session with different RR targets producing
    different WIN/LOSS labels on identical features), picks ONE config per
    session and loads only those outcomes.

    Config selection modes:
      - "max_samples": Pick config with MOST data per session (recommended for ML).
        Avoids filter-selection bias where tight filters inflate Sharpe but starve ML.
      - "best_sharpe": Pick config with highest Sharpe (original behavior).
        WARNING: tends to select tightest filters = fewest samples.

    Args:
        rr_target: If set, only consider configs at this RR. If None, picks
            across all RR targets per session.
        config_selection: "max_samples" (default) or "best_sharpe".
        skip_filter: If True, load ALL break days (no filter eligibility check).
            ML sees full dataset and learns to discriminate via features like
            orb_size and atr_20 instead of being pre-filtered.
        per_aperture: If True, pick ONE config per (session, aperture) instead of
            per session. Returns multiple apertures per session for per-aperture
            model training.
        apply_rr_lock: If True (default), filter validated_setups through
            family_rr_locks to only use locked RR per family. Set False for ML
            training — decouples training RR from portfolio RR lock so the model
            picks whatever RR gives best label balance for discrimination.

    Returns:
        X: Feature matrix (float, ready for sklearn)
        y: Binary target (1=win, 0=loss)
        meta: DataFrame with trading_day, pnl_r, orb_label, filter_type, etc.

    @research-source: Aronson Ch.6 data-mining bias analysis (Mar 4 2026)
    @revalidated-for: E2
    """
    from trading_app.config import ALL_FILTERS

    valid_selections = ("max_samples", "best_sharpe")
    if config_selection not in valid_selections:
        raise ValueError(f"config_selection must be one of {valid_selections}, got '{config_selection}'")

    con = duckdb.connect(db_path, read_only=True)
    configure_connection(con)
    try:
        # Build optional RR filter
        rr_clause = ""
        rr_clause_bare = ""  # same without table alias, for fallback subqueries
        params: dict = {"instrument": instrument}
        if rr_target is not None:
            rr_clause = "AND v.rr_target = $rr_target"
            rr_clause_bare = "AND rr_target = $rr_target"
            params["rr_target"] = rr_target

        # Config picker: max_samples picks loosest filter (most data for ML),
        # best_sharpe picks tightest filter (highest Sharpe but least data).
        if config_selection == "max_samples":
            order_clause = "ORDER BY v.sample_size DESC NULLS LAST"
        else:
            order_clause = "ORDER BY v.sharpe_ratio DESC NULLS LAST"

        # Per-aperture: pick one config per (session, aperture) instead of per session.
        # This gives each aperture its own best config, enabling per-aperture models.
        partition_clause = "PARTITION BY v.orb_label, v.orb_minutes" if per_aperture else "PARTITION BY v.orb_label"

        # Single query: pick one config per session (or per session+aperture), load outcomes
        # When apply_rr_lock=True: LEFT JOIN family_rr_locks to restrict to locked RR
        # When apply_rr_lock=False: ML training — skip lock, let ROW_NUMBER pick
        #   whatever RR gives best data for label balance/discrimination
        if apply_rr_lock:
            frl_join = """
                LEFT JOIN family_rr_locks frl
                    ON v.instrument = frl.instrument
                    AND v.orb_label = frl.orb_label
                    AND v.filter_type = frl.filter_type
                    AND v.entry_model = frl.entry_model
                    AND v.orb_minutes = frl.orb_minutes
                    AND v.confirm_bars = frl.confirm_bars"""
            frl_where = "AND (frl.locked_rr IS NULL OR v.rr_target = frl.locked_rr)"
        else:
            frl_join = ""
            frl_where = ""

        # Check if validated_setups has configs for this instrument+RR.
        # If not, fall back to orb_outcomes directly — ML needs trade
        # mechanics (entry/exit structure), not validated edge.
        _check_params = dict(params)
        _check_query = (
            "SELECT COUNT(*) FROM validated_setups "
            "WHERE instrument = $instrument AND status = 'active'"
        )
        if rr_target is not None:
            _check_query += " AND rr_target = $rr_target"
        _n_validated = con.execute(_check_query, _check_params).fetchone()[0]

        if _n_validated > 0:
            # Normal path: pick configs from validated_setups
            config_cte = f"""
                WITH best_configs AS (
                    SELECT
                        v.orb_label, v.entry_model, v.rr_target, v.confirm_bars,
                        v.orb_minutes, v.filter_type, v.sharpe_ratio, v.sample_size,
                        ROW_NUMBER() OVER (
                            {partition_clause}
                            {order_clause}
                        ) AS rn
                    FROM validated_setups v
                    {frl_join}
                    WHERE v.instrument = $instrument
                        AND v.status = 'active'
                        {frl_where}
                        {rr_clause}
                )"""
        else:
            # Fallback: pick configs from orb_outcomes directly.
            # ML only needs trade structure — group by config, pick max samples.
            logger.warning(
                f"No validated_setups for {instrument}{f' at RR {rr_target}' if rr_target else ''} — "
                f"picking configs from orb_outcomes (ML needs trade mechanics only)"
            )
            # Force skip_filter — orb_outcomes has no filter concept, applying
            # filter eligibility would wipe all data (filter_type='NONE' has no
            # matching entry in ALL_FILTERS).
            if not skip_filter:
                logger.info("  Forcing skip_filter=True for fallback path (no filter to apply)")
                skip_filter = True
            if config_selection == "best_sharpe":
                logger.warning("  config_selection='best_sharpe' unavailable in fallback — using max_samples")
            _fb_partition_v = (
                "PARTITION BY v.orb_label, v.orb_minutes" if per_aperture
                else "PARTITION BY v.orb_label"
            )
            _fb_partition_bare = (
                "PARTITION BY orb_label, orb_minutes" if per_aperture
                else "PARTITION BY orb_label"
            )
            config_cte = f"""
                WITH best_configs AS (
                    SELECT
                        v.orb_label, v.entry_model, v.rr_target, v.confirm_bars,
                        v.orb_minutes,
                        'NONE' AS filter_type,
                        NULL AS sharpe_ratio,
                        v.cnt AS sample_size,
                        ROW_NUMBER() OVER (
                            {_fb_partition_v}
                            ORDER BY v.cnt DESC
                        ) AS rn
                    FROM (
                        SELECT orb_label, entry_model, rr_target, confirm_bars,
                               orb_minutes, COUNT(*) AS cnt
                        FROM orb_outcomes
                        WHERE symbol = $instrument
                            AND pnl_r IS NOT NULL
                            {rr_clause_bare}
                        GROUP BY orb_label, entry_model, rr_target, confirm_bars, orb_minutes
                    ) v
                )"""

        query = f"""
            {config_cte}
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
                bc.filter_type,
                bc.sharpe_ratio AS config_sharpe,
                bc.sample_size AS config_n,
                d.*
            FROM orb_outcomes o
            JOIN best_configs bc
                ON o.orb_label = bc.orb_label
                AND o.entry_model = bc.entry_model
                AND o.rr_target = bc.rr_target
                AND o.confirm_bars = bc.confirm_bars
                AND o.orb_minutes = bc.orb_minutes
                AND bc.rn = 1
            JOIN daily_features d
                ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = $instrument
                AND o.pnl_r IS NOT NULL
            ORDER BY o.trading_day
        """

        df = con.execute(query, params).fetchdf()

        # Fetch selected configs for reporting (same source as main query)
        if _n_validated > 0:
            configs_query = f"""
                WITH ranked AS (
                    SELECT
                        v.orb_label, v.entry_model, v.rr_target, v.confirm_bars,
                        v.orb_minutes, v.filter_type, v.sharpe_ratio, v.sample_size,
                        ROW_NUMBER() OVER (
                            {partition_clause}
                            {order_clause}
                        ) AS rn
                    FROM validated_setups v
                    {frl_join}
                    WHERE v.instrument = $instrument
                        AND v.status = 'active'
                        {frl_where}
                        {rr_clause}
                )
                SELECT orb_label, entry_model, rr_target, confirm_bars,
                       orb_minutes, filter_type, sharpe_ratio, sample_size
                FROM ranked WHERE rn = 1
                ORDER BY orb_label
            """
        else:
            configs_query = f"""
                WITH ranked AS (
                    SELECT orb_label, entry_model, rr_target, confirm_bars,
                           orb_minutes, 'NONE' AS filter_type, NULL AS sharpe_ratio,
                           COUNT(*) AS sample_size,
                           ROW_NUMBER() OVER (
                               {_fb_partition_bare}
                               ORDER BY COUNT(*) DESC
                           ) AS rn
                    FROM orb_outcomes
                    WHERE symbol = $instrument AND pnl_r IS NOT NULL
                        {rr_clause_bare}
                    GROUP BY orb_label, entry_model, rr_target, confirm_bars, orb_minutes
                )
                SELECT orb_label, entry_model, rr_target, confirm_bars,
                       orb_minutes, filter_type, sharpe_ratio, sample_size
                FROM ranked WHERE rn = 1
                ORDER BY orb_label
            """
        configs_df = con.execute(configs_query, params).fetchdf()

        # Backfill global features for orb_minutes != 5 (pipeline gap)
        df = _backfill_global_features(df, con, instrument)
    finally:
        con.close()

    if df.empty:
        rr_str = f" at RR {rr_target}" if rr_target is not None else ""
        raise ValueError(f"No single-config outcomes for {instrument}{rr_str}")

    # Log selected configs
    sel_str = config_selection.upper()
    rr_str = f" RR={rr_target}" if rr_target is not None else ""
    filt_str = " UNFILTERED" if skip_filter else ""
    aperture_str = " PER-APERTURE" if per_aperture else ""
    logger.info(
        f"Single-config ({sel_str}{rr_str}{filt_str}{aperture_str}): "
        f"{len(configs_df)} configs, {len(df):,d} raw outcomes for {instrument}"
    )
    for _, cfg in configs_df.iterrows():
        logger.info(
            f"  {cfg['orb_label']:<22} E{cfg['entry_model'][-1]} "
            f"RR{cfg['rr_target']:.1f} CB{cfg['confirm_bars']} "
            f"O{cfg['orb_minutes']} {cfg['filter_type']:<20} "
            f"Sharpe={cfg['sharpe_ratio']:.3f} N={cfg['sample_size']}"
        )

    if skip_filter:
        # ML sees ALL break days — learns filter boundary from features
        # (orb_size, atr_20, etc.) instead of being pre-filtered
        logger.info(f"Filter SKIPPED: ML trains on all {len(df):,d} break days")
    else:
        # Apply filter eligibility (vectorized — same logic as load_validated_feature_matrix)
        n_before_filter = len(df)
        keep_mask = pd.Series(False, index=df.index)
        filter_cache: dict[str, object] = {}

        for (ft_name, orb_label_g), group_idx in df.groupby(["filter_type", "orb_label"]).groups.items():
            if ft_name not in filter_cache:
                filter_cache[ft_name] = ALL_FILTERS.get(ft_name)

            filt = filter_cache[ft_name]
            if filt is None:
                logger.warning(f"Unknown filter_type '{ft_name}' — skipping")
                continue

            keep_mask.loc[group_idx] = filt.matches_df(df.loc[group_idx], orb_label_g)

        df = df[keep_mask].reset_index(drop=True)
        n_after_filter = len(df)
        logger.info(f"After filter: {n_after_filter:,d} rows ({n_before_filter - n_after_filter:,d} filtered out)")

    # Safety dedup: 1 row per (trading_day, orb_label[, orb_minutes]) already
    dedup_cols = ["trading_day", "orb_label", "orb_minutes"] if per_aperture else ["trading_day", "orb_label"]
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    if len(df) < n_before_dedup:
        logger.warning(f"Single-config had {n_before_dedup - len(df)} unexpected duplicates — investigate")

    # Drop config metadata columns before feature extraction
    df = df.drop(columns=["config_sharpe", "config_n"], errors="ignore")

    # --- Target ---
    y = (df["pnl_r"] > 0).astype(int)

    # --- Meta (for evaluation, not features) ---
    meta = df[
        [
            "trading_day",
            "symbol",
            "orb_label",
            "orb_minutes",
            "entry_model",
            "rr_target",
            "confirm_bars",
            "pnl_r",
            "outcome",
            "filter_type",
        ]
    ].copy()

    # --- Transform to features (shared with live prediction) ---
    X = transform_to_features(df)

    logger.info(
        f"Single-config feature matrix: {X.shape[0]:,d} rows x {X.shape[1]} features (win rate: {y.mean():.1%})"
    )
    return X, y, meta

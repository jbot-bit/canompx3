"""ML meta-labeling configuration: features, hyperparameters, constants.

Single source of truth for all ML feature lists and model parameters.
Add new features here → they propagate to training + live prediction.
"""

from __future__ import annotations

import pathlib

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Global market-state features (same across all sessions on a given day)
# @research-source: NO-GO removals (Mar 3 2026):
#   is_nfp_day, is_opex_day — calendar overlays, 0 BH survivors at q=0.10
#   (NFP/OPEX/FOMC all confirmed NO-GO, re-verified Mar 3 with TZ fix)
GLOBAL_FEATURES: list[str] = [
    "atr_20",
    "atr_vel_ratio",
    "rsi_14_at_CME_REOPEN",
    "gap_open_points",
    "garch_atr_ratio",
    "garch_forecast_vol",
    "prev_day_range",
    "prev_day_direction",
    "overnight_range",
    "day_of_week",
    "is_friday",
    "is_monday",
]

# Per-session features (extracted dynamically from orb_{SESSION}_{field})
# These column suffixes are appended to the traded session's orb_label.
# @research-source: compression_z removed (Mar 3 2026) — pre-ORB compression
#   had 90+ tests, 0 BH survivors for break quality prediction.
#   all_narrow AVOID works as a binary gate, NOT a continuous predictor.
SESSION_FEATURE_SUFFIXES: list[str] = [
    "size",               # ORB range in points
    "volume",             # Total ORB-window volume
    "break_bar_volume",   # Volume on the break bar
    "break_delay_min",    # Minutes from ORB close to break
    "break_bar_continues",  # Break bar closed in break direction
    "break_dir",          # LONG / SHORT / NONE
]

# Features to normalize by atr_20 for stationarity
ATR_NORMALIZE: list[str] = [
    "gap_open_points",
    "prev_day_range",
    "overnight_range",
    "orb_size",  # Session-specific, normalized after extraction
]

# Categorical features that get one-hot encoded
CATEGORICAL_FEATURES: list[str] = [
    "orb_label",
    "entry_model",
    "prev_day_direction",
    "gap_type",
    "atr_vel_regime",
    "break_dir",
]

# Outcome columns from orb_outcomes that become trade-config features
TRADE_CONFIG_FEATURES: list[str] = [
    "rr_target",
    "confirm_bars",
    "orb_minutes",
]

# LOOK-AHEAD BLACKLIST — NEVER use as ML features
# These depend on information not available at trade-entry time.
LOOKAHEAD_BLACKLIST: set[str] = {
    "double_break",       # Full-session look-ahead
    "day_type",           # Full-session look-ahead
    "outcome",            # IS the target
    "mae_r",              # Post-trade
    "mfe_r",              # Post-trade
    "pnl_r",              # IS the target
    "pnl_dollars",        # IS the target
    "risk_dollars",       # Post-entry computation
    "exit_ts",            # Post-trade
    "exit_price",         # Post-trade
    "ts_outcome",         # Time-stop outcome (post-trade)
    "ts_pnl_r",           # Time-stop PnL (post-trade)
    "ts_exit_ts",         # Time-stop exit (post-trade)
    "took_pdh_before_1000",  # Time-dependent within day
    "took_pdl_before_1000",  # Time-dependent within day
}

# Sessions available as rel_vol features
REL_VOL_SESSIONS: list[str] = [
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "US_DATA_830", "NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE",
    "CME_PRECLOSE", "NYSE_CLOSE", "BRISBANE_1025",
]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

RF_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,            # Conservative to prevent overfitting
    "min_samples_leaf": 50,    # Stable leaf predictions
    "max_features": "sqrt",    # Feature subsampling
    "class_weight": "balanced",  # Handle 20-33% win rate imbalance
    "random_state": 42,
    "n_jobs": -1,              # Use all cores
}

# CPCV configuration (per de Prado)
CPCV_N_GROUPS: int = 10       # Split data into 10 time-ordered groups
CPCV_K_TEST: int = 2          # Use 2 groups as test = C(10,2) = 45 splits
CPCV_PURGE_DAYS: int = 1      # Remove 1 day between train/test
CPCV_EMBARGO_DAYS: int = 1    # Embargo 1 day after test set

# Threshold search range
THRESHOLD_MIN: float = 0.35
THRESHOLD_MAX: float = 0.70
THRESHOLD_STEP: float = 0.01

# Minimum samples per instrument to train a model
MIN_SAMPLES_TRAIN: int = 1000

# Active instruments for ML — derived from pipeline canonical source.
# Excludes instruments with 0 validated strategies (MBT has no ORB edge).
# Drift check #48 guards against staleness.
_ML_EXCLUDED: set[str] = {"MBT"}
ACTIVE_INSTRUMENTS: list[str] = [i for i in ACTIVE_ORB_INSTRUMENTS if i not in _ML_EXCLUDED]

# Model persistence directory
MODEL_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent.parent / "models" / "ml"

"""ML meta-labeling configuration: features, hyperparameters, constants.

Single source of truth for all ML feature lists and model parameters.
Add new features here → they propagate to training + live prediction.
"""

from __future__ import annotations

import hashlib
import pathlib

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Global market-state features (same across all sessions on a given day)
# @research-source: NO-GO removals (Mar 3 2026):
#   is_nfp_day, is_opex_day — calendar overlays, 0 BH survivors at q=0.10
#   (NFP/OPEX/FOMC all confirmed NO-GO, re-verified Mar 3 with TZ fix)
# @research-source: Theory audit (Mar 4 2026) — removed features without
#   structural mechanism for ORB breakouts:
#   garch_atr_ratio, garch_forecast_vol — statistical artifact, no structural
#     theory for WHY GARCH ratio predicts ORB quality. Correlated with atr_20
#     and atr_vel_ratio (substitution effect). 5.0% and 4.2% avg importance.
#   rsi_14_at_CME_REOPEN — RESEARCH_RULES: indicators guilty until proven.
#     RSI is mean-reverting (opposite of breakout logic). Computed at CME_REOPEN
#     only — no mechanism for predicting LONDON_METALS ORB hours later. 4.6%.
#   day_of_week, is_friday, is_monday — DOW effects: 0 BH survivors at
#     FDR=0.05. NO-GO confirmed (re-verified Mar 3 with TZ fix). 1.3%.
#   prev_day_direction — already E6-dropped (one-hot prefix in E6_NOISE_PREFIXES).
#     Removed from list for clarity.
# @revalidated-for: E1, E2
GLOBAL_FEATURES: list[str] = [
    "atr_20",  # Volatility regime — defines the environment
    "atr_20_pct",  # ATR percentile (0-100) — vol regime rank vs trailing 252d
    "atr_vel_ratio",  # Vol acceleration — compressed spring (confirmed)
    "gap_open_points",  # Overnight institutional repositioning
    "prev_day_range",  # Prior day activity level — regime context
    "overnight_range",  # Asian session range — #1 feature (6.5% avg imp)
]

# Per-session features (extracted dynamically from orb_{SESSION}_{field})
# These column suffixes are appended to the traded session's orb_label.
# @research-source: compression_z removed (Mar 3 2026) — pre-ORB compression
#   had 90+ tests, 0 BH survivors for break quality prediction.
#   all_narrow AVOID works as a binary gate, NOT a continuous predictor.
#
# ML PREDICTION ARCHITECTURE: PRE-BREAK (decide before placing stop)
# The ML meta-label answers "should I place this stop?" BEFORE the break.
# Cost of skip = zero (no fill, no slippage). This means only features known
# at ORB close (before break) can be used as ML features.
#
# AT-BREAK features (break_delay_min, break_bar_volume, break_bar_continues)
# were removed Mar 4 2026 after M2.5 audit + trading theory review:
#   break_delay_min  — theory: slow break = weak momentum. VALID theory but
#                      unknown pre-break. Was 1-7% importance across sessions.
#   break_bar_volume — theory: low vol = weak conviction. VALID theory but
#                      unknown pre-break. Was 2-10% importance across sessions.
#   break_bar_continues — already E6-dropped (<1% importance)
# These features have real structural theory but conflict with pre-break architecture.
# If architecture changes to at-break filtering, restore them.
# @research-source: M2.5 audit 2026-03-04, verified by timing analysis
# @revalidated-for: E1, E2
SESSION_FEATURE_SUFFIXES: list[str] = [
    "size",  # ORB range in points — known at ORB close (pre-break)
    "volume",  # Total ORB-window volume — known at ORB close (pre-break)
]

# Features to normalize by atr_20 for stationarity
ATR_NORMALIZE: list[str] = [
    "gap_open_points",
    "prev_day_range",
    "overnight_range",
    "orb_size",  # Session-specific, normalized after extraction
]

# Categorical features that get one-hot encoded
# NOTE: break_dir removed (pre-break blacklist), prev_day_direction removed
# (no structural mechanism, was already E6-dropped as one-hot prefix).
CATEGORICAL_FEATURES: list[str] = [
    "orb_label",
    "entry_model",
    "gap_type",
    "atr_vel_regime",
]

# Outcome columns from orb_outcomes that become trade-config features.
# NOTE: rr_target was REMOVED from this list (Mar 4 2026) after ML audit
# showed it dominated 56-69% of feature importance across all instruments.
# The model was learning "low RR = higher win rate" (tautological) instead
# of market-regime signals. Removing it forces the model to discriminate
# based on daily market conditions.
# @research-source: scripts/tools/ml_audit.py, all 4 instruments
# @revalidated-for: E1, E2
TRADE_CONFIG_FEATURES: list[str] = [
    "confirm_bars",
    "orb_minutes",
]

# LOOK-AHEAD BLACKLIST — NEVER use as ML features
# These depend on information not available at trade-entry time.
# Architecture: ML predicts PRE-BREAK (before placing stop). Features must be
# known at ORB close, before the break event occurs.
LOOKAHEAD_BLACKLIST: set[str] = {
    "double_break",  # Full-session look-ahead
    "day_type",  # Full-session look-ahead
    "outcome",  # IS the target
    "mae_r",  # Post-trade
    "mfe_r",  # Post-trade
    "pnl_r",  # IS the target
    "pnl_dollars",  # IS the target
    "risk_dollars",  # Post-entry computation
    "exit_ts",  # Post-trade
    "exit_price",  # Post-trade
    "ts_outcome",  # Time-stop outcome (post-trade)
    "ts_pnl_r",  # Time-stop PnL (post-trade)
    "ts_exit_ts",  # Time-stop exit (post-trade)
    "took_pdh_before_1000",  # Time-dependent within day
    "took_pdl_before_1000",  # Time-dependent within day
    # AT-BREAK features — valid theory but unknown pre-break (added Mar 4 2026)
    # @research-source: M2.5 audit + trading theory review
    "break_delay_min",  # Minutes from ORB close to break — unknown pre-break
    "break_bar_volume",  # Volume on the break bar — unknown pre-break
    "break_bar_continues",  # Break bar close direction — unknown pre-break
    "break_dir",  # Direction — unknown pre-break (DIR_BOTH) or constant (DIR_LONG/SHORT)
}

# Sessions available as rel_vol features
REL_VOL_SESSIONS: list[str] = [
    "CME_REOPEN",
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
    "BRISBANE_1025",
]

# Session chronological order (Brisbane time) — used for cross-session features.
# Note: EUROPE_FLOW/LONDON_METALS swap order by season (winter EF=17:00 before LM=18:00,
# summer LM=17:00 before EF=18:00). Static ordering here uses summer convention.
# @research-source: pipeline/dst.py SESSION_CATALOG ordering
SESSION_CHRONOLOGICAL_ORDER: list[str] = [
    "CME_REOPEN",
    "TOKYO_OPEN",
    "BRISBANE_1025",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
]

# Cross-session features: prior session ORB break counts and level proximity.
# @research-source: ml_hybrid_experiment.py (Mar 4 2026)
#   prior_sessions_broken = #1 feature for MES (12.2%), #3 for MNQ (8.3%)
#   levels_within_2R = #2 for MES (7.2%), #5 for M2K (4.6%)
# WARNING: In a per-instrument model, these features leak session identity
# (79.7% accuracy at predicting session from cross-features alone for MES).
# They MUST be used with per-session models to avoid session-position bias.
CROSS_SESSION_FEATURES: list[str] = [
    "prior_sessions_broken",  # Count of prior sessions with ORB break today
    "prior_sessions_long",  # Prior sessions with LONG break
    "prior_sessions_short",  # Prior sessions with SHORT break
]

LEVEL_PROXIMITY_FEATURES: list[str] = [
    "nearest_level_to_high_R",  # Distance from ORB high to nearest prior level (in R)
    "nearest_level_to_low_R",  # Distance from ORB low to nearest prior level (in R)
    "levels_within_1R",  # Count of prior levels within 1R of ORB boundaries
    "levels_within_2R",  # Count of prior levels within 2R
    "orb_nested_in_prior",  # Current ORB nested inside a prior ORB (0/1)
    "prior_orb_size_ratio_max",  # Max(prior ORB size / current ORB size)
]

# E6 noise features to exclude from clean feature set.
# @research-source: ml_level_proximity_experiment.py (Mar 4 2026)
#   All had <1% importance across all 4 instruments in E3/E6 experiments.
#   orb_label one-hots are the #1 problem — cause session identity leakage.
E6_NOISE_PREFIXES: list[str] = [
    "orb_label_",  # Session identity leakage (11-13% importance = tautological)
    "gap_type_",  # <1% importance, noise
    "atr_vel_regime_",  # <1% importance, noise
    "prev_day_direction_",  # <1% importance, noise
]
E6_NOISE_EXACT: list[str] = [
    "confirm_bars",  # <1% importance, near-constant
    "orb_break_bar_continues",  # <1% importance
    "orb_minutes",  # <1% importance, near-constant
]

# ---------------------------------------------------------------------------
# Per-session model configuration
# ---------------------------------------------------------------------------

# Minimum samples to train a per-session model.
# Sessions below this threshold use no-model (take all trades).
MIN_SESSION_SAMPLES: int = 500

# Sessions with <= this index in SESSION_CHRONOLOGICAL_ORDER have
# near-constant cross-session features (0 or 1 prior sessions).
# Cross-session features are dropped for these sessions.
MAX_EARLY_SESSION_INDEX: int = 1  # CME_REOPEN (0) and TOKYO_OPEN (1)

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

RF_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 6,  # Conservative to prevent overfitting
    "min_samples_leaf": 100,  # Big leaf — stable predictions, prevents memorization
    "max_features": "sqrt",  # Feature subsampling
    "class_weight": "balanced",  # Handle 20-33% win rate imbalance
    "random_state": 42,
    "n_jobs": -1,  # Use all cores
}
# @research-source: n_estimators convergence test (Mar 4 2026)
#   100=0.6270, 200=0.6266, 300=0.6254, 500=0.6252 AUC on MGC test set.
#   All within noise (<0.002). 200 is sweet spot: same AUC, 60% less time.
# @research-source: min_samples_leaf=100 > 50 in parameter sweep (Mar 4 2026)
# AUC 0.5717→0.5718 (E3), more robust across thresholds. Big leaf reduces
# overfitting risk with 103K samples. sweep: scripts/tools/ml_level_proximity_experiment.py

# CPCV configuration (per de Prado)
CPCV_N_GROUPS: int = 10  # Split data into 10 time-ordered groups
CPCV_K_TEST: int = 2  # Use 2 groups as test = C(10,2) = 45 splits
CPCV_PURGE_DAYS: int = 1  # Remove 1 day between train/test
CPCV_EMBARGO_DAYS: int = 1  # Embargo 1 day after test set

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


def compute_config_hash() -> str:
    """Compute SHA-256 hash of ML config for drift detection.

    Used by BOTH training (meta_label.py) and prediction (predict_live.py)
    to detect when models were trained with different config. Must be called
    from a single source to prevent hash formula divergence.
    """
    config_str = (
        f"{RF_PARAMS}|{THRESHOLD_MIN}|{THRESHOLD_MAX}|{THRESHOLD_STEP}"
        f"|{GLOBAL_FEATURES}|{SESSION_FEATURE_SUFFIXES}|{ATR_NORMALIZE}"
        f"|{CATEGORICAL_FEATURES}|{sorted(LOOKAHEAD_BLACKLIST)}"
        f"|{TRADE_CONFIG_FEATURES}|{E6_NOISE_PREFIXES}|{E6_NOISE_EXACT}"
        f"|{CROSS_SESSION_FEATURES}|{LEVEL_PROXIMITY_FEATURES}"
        f"|{SESSION_CHRONOLOGICAL_ORDER}|{MIN_SESSION_SAMPLES}|{MAX_EARLY_SESSION_INDEX}"
        f"|split=60/20/20"  # 3-way split: train/val/test
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]

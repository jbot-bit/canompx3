"""ML meta-labeling configuration: features, hyperparameters, constants.

Single source of truth for all ML feature lists and model parameters.
Add new features here → they propagate to training + live prediction.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

# ---------------------------------------------------------------------------
# Global ML kill switch — canonical ON/OFF for the entire ML gate.
# ---------------------------------------------------------------------------
# Policy lives in callers (session_orchestrator, paper_trader), not in
# LiveMLPredictor itself — unit tests need to construct LiveMLPredictor
# directly with synthetic bundles regardless of this flag.
#
# Callers check ML_ENABLED before instantiating LiveMLPredictor AND pass
# require_models=True when they do. That combination is fail-closed:
#   - ML_ENABLED=0 (default): no LiveMLPredictor created, self._ml_predictor=None
#   - ML_ENABLED=1 + models present: LiveMLPredictor loads and gates trades
#   - ML_ENABLED=1 + models missing: RuntimeError at orchestrator startup
#
# Until ML V3 produces validated .joblib bundles, ML_ENABLED must stay unset.
# See docs/audit/ml_v3/2026-04-11-stage-0-verification.md for the V3 sprint
# and docs/runtime/stages/ml-v3-stage-1-fail-closed.md for this guard.
ML_ENABLED: bool = os.environ.get("ML_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")

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
    # overnight_range REMOVED (2026-03-19) — session-dependent look-ahead.
    # Computed from 09:00-17:00 Brisbane (full Asia window). Contaminates ANY
    # session starting inside that window: TOKYO_OPEN (10:00), SINGAPORE_OPEN
    # (11:00), CME_REOPEN (09:00 winter/CST). DST split confirmed: summer-only
    # (clean) shows zero signal; winter-only (contaminated) carries all apparent
    # edge. As a GLOBAL feature used across mixed sessions, it leaks future
    # price action into Asian session predictions.
    # @research-source: sweep look-ahead audit 2026-03-19 (DST split test)
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
    "vwap",  # Session VWAP from trading day start to ORB start (pre-break, no look-ahead)
    "pre_velocity",  # Slope of last 5 closes before session start (pre-break, no look-ahead)
]

# Features to normalize by atr_20 for stationarity
ATR_NORMALIZE: list[str] = [
    "gap_open_points",
    "prev_day_range",
    "orb_size",  # Session-specific, normalized after extraction
    # orb_vwap REMOVED from ATR_NORMALIZE (Mar 2026 code review):
    # VWAP is absolute price, not a delta. Dividing price by ATR produces
    # a non-stationary feature that trends with the underlying. The meaningful
    # signal is (vwap - orb_midpoint) / atr_20, which is already computed as
    # orb_vwap_norm in build_daily_features.py.
    "orb_pre_velocity",  # Pre-session velocity in points → ATR units
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
# @research-source: scripts/tools/ml_audit.py, all active instruments
# @revalidated-for: E1, E2
TRADE_CONFIG_FEATURES: list[str] = [
    "confirm_bars",
    "orb_minutes",
]

# ---------------------------------------------------------------------------
# V2 Methodology: Expert-prior features (EPV fix)
# ---------------------------------------------------------------------------
# 5 features selected by structural mechanism, NOT data-driven scan.
# Avoids scan-on-train bias (Hastie/Tibshirani ESL §7.10).
# Verified present in E6-filtered matrix (25 cols → 5 selected).
# EPV at O5 RR1.0 MNQ: ~1300 × 55% WR / 5 = ~143 (well above 10).
# EPV at O30 RR2.0 MNQ: ~430 × 34% WR / 5 = ~29 (above 10).
#
# @research-source: expert prior selection (not data-driven)
# @revalidated-for: E2
ML_CORE_FEATURES: list[str] = [
    "orb_size_norm",  # ORB size IS the edge (Blueprint §2, cost mechanism)
    "atr_20_pct",  # Vol regime rank (confirmed ATR70_VOL filter)
    "gap_open_points_norm",  # Overnight institutional repositioning (ATR-normalized)
    "orb_pre_velocity_norm",  # Pre-session momentum slope (ATR-normalized)
    "prior_sessions_broken",  # Cross-session flow (#1 importance in prior experiments)
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
    # SESSION-DEPENDENT LOOK-AHEAD (added 2026-03-19)
    # overnight_* computed from 09:00-17:00 Brisbane. Contaminates sessions
    # starting inside that window (TOKYO 10:00, SINGAPORE 11:00, CME_REOPEN
    # 09:00 winter). Valid ONLY for sessions starting AFTER 17:00 Brisbane
    # (LONDON_METALS, US_DATA_830, NYSE_OPEN, etc.) but unsafe as global
    # features in mixed-session models.
    # @research-source: sweep look-ahead audit 2026-03-19 (DST split test)
    "overnight_range",  # 09:00-17:00 range — look-ahead for Asian sessions
    "overnight_high",  # 09:00-17:00 high — look-ahead for Asian sessions
    "overnight_low",  # 09:00-17:00 low — look-ahead for Asian sessions
    "overnight_took_pdh",  # overnight_high > prev_day_high — look-ahead
    "overnight_took_pdl",  # overnight_low < prev_day_low — look-ahead
    "session_asia_high",  # Same window as overnight — look-ahead
    "session_asia_low",  # Same window as overnight — look-ahead
    "took_pdh_before_1000",  # 09:00-10:00 — clean for TOKYO but not global
    "took_pdl_before_1000",  # 09:00-10:00 — clean for TOKYO but not global
    "pre_1000_high",  # 09:00-10:00 high — clean for TOKYO but not global
    "pre_1000_low",  # 09:00-10:00 low — clean for TOKYO but not global
    # AT-BREAK features — valid theory but unknown pre-break (added Mar 4 2026)
    # @research-source: M2.5 audit + trading theory review
    "break_delay_min",  # Minutes from ORB close to break — unknown pre-break
    "break_bar_volume",  # Volume on the break bar — unknown pre-break
    "break_bar_continues",  # Break bar close direction — unknown pre-break
    "break_dir",  # Direction — unknown pre-break (DIR_BOTH) or constant (DIR_LONG/SHORT)
    # Defense-in-depth: currently safe (never enter X) but prevent future refactoring leaks
    "break_ts",  # Exact break timestamp — post-break
    "garch_forecast_vol",  # GARCH model output — statistical artifact, no structural mechanism
    "garch_atr_ratio",  # GARCH/ATR ratio — statistical artifact, no structural mechanism
    "session_london_high",  # London session high — look-ahead for pre-London sessions
    "session_london_low",  # London session low — look-ahead for pre-London sessions
    "session_ny_high",  # NY session high — look-ahead for pre-NY sessions
    "session_ny_low",  # NY session low — look-ahead for pre-NY sessions
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
#   All had <1% importance across all active instruments in E3/E6 experiments.
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

# ---------------------------------------------------------------------------
# Methodology version gate
# ---------------------------------------------------------------------------
# Bump this when methodology changes invalidate existing models.
# predict_live.py checks bundle version matches this — fail-open if mismatch.
# V1 = original (23 features, validated_setups source, no baseline/EPV gates)
# V2 = methodology fix (≤5 features, full universe, baseline+EPV gates)
ML_METHODOLOGY_VERSION: int = 2


def compute_config_hash() -> str:
    """Compute SHA-256 hash of V2 ML config for drift detection.

    Used by BOTH training (meta_label.py) and prediction (predict_live.py)
    to detect when models were trained with different config. Must be called
    from a single source to prevent hash formula divergence.

    V2: Only includes elements that affect V2 training output.
    V1-only elements (CATEGORICAL_FEATURES, TRADE_CONFIG_FEATURES, etc.)
    removed to prevent false-positive hash mismatches.
    """
    config_str = (
        f"v2|{ML_CORE_FEATURES}"
        f"|{RF_PARAMS}|{THRESHOLD_MIN}|{THRESHOLD_MAX}|{THRESHOLD_STEP}"
        f"|{CPCV_N_GROUPS}|{CPCV_K_TEST}|{CPCV_PURGE_DAYS}|{CPCV_EMBARGO_DAYS}"
        f"|{MIN_SESSION_SAMPLES}|{MAX_EARLY_SESSION_INDEX}"
        f"|{ML_METHODOLOGY_VERSION}"
        f"|split=60/20/20"
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]

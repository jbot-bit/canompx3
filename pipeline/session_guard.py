"""Session chronological guard — prevents look-ahead contamination.

PROBLEM: daily_features has ONE row per trading day with ALL sessions' data
populated after market close. Any code that uses a feature from session B
to evaluate session A (where B is later) has look-ahead contamination.

This module provides a single source of truth for which features are safe
to use for a given target session.

USAGE:
    from pipeline.session_guard import get_allowed_features, mask_future_sessions

    # Get list of session-prefixed columns safe for TOKYO_OPEN
    allowed = get_allowed_features("TOKYO_OPEN", df.columns)

    # Zero out / NaN any future-session features in a DataFrame
    df_clean = mask_future_sessions(df, "TOKYO_OPEN")

THREE TIMES THIS BUG HAS HIT US (Mar 19-20 2026):
1. overnight_range (09:00-17:00 window) → look-ahead for TOKYO (10:00)
2. overnight_took_pdh/pdl → look-ahead for TOKYO/SINGAPORE/BRISBANE
3. orb_SINGAPORE_OPEN_break_bar_volume → look-ahead for TOKYO (10:00)

The root cause: daily_features computes everything at once. This guard
ensures downstream code can ONLY see features from sessions that have
ALREADY completed before the target session.
"""

from __future__ import annotations

import re

import pandas as pd

# Canonical session order (Brisbane time).
# Source of truth: trading_app/ml/config.py SESSION_CHRONOLOGICAL_ORDER
# Duplicated here to avoid circular imports (pipeline/ cannot import trading_app/).
_SESSION_ORDER: list[str] = [
    "CME_REOPEN",  # 08:00-09:00 Brisbane
    "TOKYO_OPEN",  # 10:00
    "BRISBANE_1025",  # 10:25
    "SINGAPORE_OPEN",  # 11:00
    "LONDON_METALS",  # 17:00 winter / 18:00 summer
    "EUROPE_FLOW",  # 17:00 summer / 18:00 winter (swaps with LM)
    "US_DATA_830",  # ~23:30-00:30 Brisbane (8:30 AM ET)
    "NYSE_OPEN",  # ~00:30 Brisbane (9:30 AM ET)
    "US_DATA_1000",  # ~01:00 Brisbane (10:00 AM ET)
    "COMEX_SETTLE",  # ~04:30 Brisbane (1:30 PM ET)
    "CME_PRECLOSE",  # ~05:45 Brisbane (2:45 PM CT)
    "NYSE_CLOSE",  # ~07:00 Brisbane (4:00 PM ET)
]

# Features that are ALWAYS safe (computed from prior day or trading day start)
_ALWAYS_SAFE: set[str] = {
    "trading_day",
    "symbol",
    "orb_minutes",
    "bar_count_1m",
    "atr_20",
    "atr_20_pct",
    "atr_vel_ratio",
    "atr_vel_regime",
    "gap_open_points",
    "gap_type",
    "prev_day_high",
    "prev_day_low",
    "prev_day_close",
    "prev_day_range",
    "prev_day_direction",
    "daily_open",  # known at 09:00
    "rsi_14_at_CME_REOPEN",  # computed from prior-day 5m bars, known before any session
    "confirm_bars",  # trade config, not market data
    "entry_model",  # trade config
    "stop_multiplier",  # trade config
}

# Features that are NEVER safe for ANY session (computed from full trading day)
_NEVER_SAFE: set[str] = {
    "daily_high",
    "daily_low",
    "daily_close",
    "day_type",
}

# Overnight/pre-session features with their safe-after session
# These are computed from specific time windows
_WINDOW_FEATURES: dict[str, str] = {
    # Asia window (09:00-17:00 Brisbane) — safe only AFTER 17:00
    "overnight_high": "LONDON_METALS",
    "overnight_low": "LONDON_METALS",
    "overnight_range": "LONDON_METALS",
    "overnight_took_pdh": "LONDON_METALS",
    "overnight_took_pdl": "LONDON_METALS",
    "session_asia_high": "LONDON_METALS",
    "session_asia_low": "LONDON_METALS",
    # Pre-1000 window (09:00-10:00) — safe after TOKYO_OPEN
    "pre_1000_high": "TOKYO_OPEN",
    "pre_1000_low": "TOKYO_OPEN",
    "took_pdh_before_1000": "TOKYO_OPEN",
    "took_pdl_before_1000": "TOKYO_OPEN",
    # London window — safe after NYSE_OPEN (London closes ~midnight Brisbane)
    "session_london_high": "NYSE_OPEN",
    "session_london_low": "NYSE_OPEN",
    # NY window — safe after NYSE_CLOSE
    "session_ny_high": "NYSE_CLOSE",
    "session_ny_low": "NYSE_CLOSE",
}

# Regex to extract session name from column like "orb_TOKYO_OPEN_size"
_SESSION_COL_RE = re.compile(r"^(?:orb|rel_vol)_(" + "|".join(re.escape(s) for s in _SESSION_ORDER) + r")_?")


def _session_index(session: str) -> int:
    """Get the chronological index of a session. Raises ValueError if unknown."""
    try:
        return _SESSION_ORDER.index(session)
    except ValueError:
        raise ValueError(f"Unknown session '{session}'. Valid: {_SESSION_ORDER}") from None


def get_prior_sessions(target_session: str) -> list[str]:
    """Return sessions that complete BEFORE target_session starts."""
    idx = _session_index(target_session)
    return _SESSION_ORDER[:idx]


def is_feature_safe(column: str, target_session: str) -> bool:
    """Check if a single column is safe (no look-ahead) for a target session.

    Returns True if the feature is known before the target session's ORB forms.
    """
    # Always safe
    if column in _ALWAYS_SAFE:
        return True

    # Never safe
    if column in _NEVER_SAFE:
        return False

    # Window-based features
    if column in _WINDOW_FEATURES:
        safe_after = _WINDOW_FEATURES[column]
        return _session_index(target_session) > _session_index(safe_after)

    # Session-prefixed features (orb_TOKYO_OPEN_size, rel_vol_SINGAPORE_OPEN, etc.)
    match = _SESSION_COL_RE.match(column)
    if match:
        feature_session = match.group(1)
        # Feature from session X is safe for session Y if X is BEFORE or SAME as Y
        # Same-session features (orb_TOKYO_OPEN_size for TOKYO) ARE known at trade time
        return _session_index(feature_session) <= _session_index(target_session)

    # Compression features follow the same pattern
    for sess in _SESSION_ORDER:
        if sess in column and ("compression" in column or "orb_" in column):
            return _session_index(sess) <= _session_index(target_session)

    # Unknown column — fail CLOSED (not safe)
    return False


def get_allowed_features(
    target_session: str,
    columns: list[str] | pd.Index,
) -> list[str]:
    """Return only columns that are safe (no look-ahead) for target_session."""
    return [c for c in columns if is_feature_safe(c, target_session)]


def get_blocked_features(
    target_session: str,
    columns: list[str] | pd.Index,
) -> list[str]:
    """Return columns that are NOT safe (look-ahead) for target_session."""
    return [c for c in columns if not is_feature_safe(c, target_session)]


def mask_future_sessions(
    df: pd.DataFrame,
    target_session: str,
    fill_value=float("nan"),
) -> pd.DataFrame:
    """Return a copy of df with future-session features set to fill_value.

    Use this before ANY analysis that evaluates trades at a specific session.
    """
    result = df.copy()
    blocked = get_blocked_features(target_session, df.columns)
    if blocked:
        result[blocked] = fill_value
    return result


def validate_features_for_session(
    feature_names: list[str],
    target_session: str,
) -> tuple[list[str], list[str]]:
    """Split features into safe and blocked for a target session.

    Returns (safe_features, blocked_features).
    Raises no errors — caller decides what to do with blocked features.
    """
    safe = []
    blocked = []
    for f in feature_names:
        if is_feature_safe(f, target_session):
            safe.append(f)
        else:
            blocked.append(f)
    return safe, blocked

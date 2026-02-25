#!/usr/bin/env python3
"""
Build daily_features from bars_1m and bars_5m.

Staged build (each stage is gated):
  1. Trading day assignment (09:00 Brisbane boundary)
  2. ORB ranges (10 dynamic sessions, configurable duration)
  3. Break detection (first 1m close outside ORB)
  4. Session stats (dynamic-window high-low for session range features)
  5. RSI (Wilder's 14-period on 5m closes)
  6. Overnight/pre-session stats, market profile, ATR velocity, compression, GARCH, rel volume

Idempotent: DELETE existing daily_features rows for the date range, then INSERT.

Usage:
    python pipeline/build_daily_features.py --instrument MGC --start 2024-01-01 --end 2024-01-31
    python pipeline/build_daily_features.py --instrument MGC --start 2024-01-01 --end 2024-01-31 --orb-minutes 15
    python pipeline/build_daily_features.py --instrument MGC --start 2024-01-01 --end 2024-01-31 --dry-run
"""

import sys
import argparse
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import numpy as np

# Add project root to path

from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import get_asset_config, list_instruments
from pipeline.init_db import ORB_LABELS
from pipeline.cost_model import get_cost_spec, pnl_points_to_r, CostSpec
from pipeline.dst import (is_us_dst, is_uk_dst, DYNAMIC_ORB_RESOLVERS, get_break_group,
                          DST_AFFECTED_SESSIONS, DST_CLEAN_SESSIONS)
from pipeline.calendar_filters import is_nfp_day, is_opex_day, is_friday, is_monday, is_tuesday, day_of_week

from pipeline.log import get_logger
logger = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

BRISBANE_TZ = ZoneInfo("Australia/Brisbane")
UTC_TZ = ZoneInfo("UTC")

# Trading day starts at 09:00 Brisbane = 23:00 UTC previous day
TRADING_DAY_START_HOUR_LOCAL = 9  # Brisbane hour

# ORB local times (Brisbane) -> UTC hour offsets
# Brisbane = UTC+10, so local HH:00 = UTC (HH-10):00
# For 00:30 Brisbane = 14:30 UTC previous day? No:
#   00:30 Brisbane = 14:30 UTC (same calendar day in UTC terms)
# Actually: Brisbane = UTC+10, so to get UTC subtract 10.
#   09:00 Brisbane = 23:00 UTC (previous calendar day)
#   10:00 Brisbane = 00:00 UTC (same calendar day)
#   11:00 Brisbane = 01:00 UTC
#   18:00 Brisbane = 08:00 UTC
#   23:00 Brisbane = 13:00 UTC
#   00:30 Brisbane = 14:30 UTC (next day Brisbane, same trading day)
#
# ORB times are defined as (hour, minute) in Brisbane local time.
# All sessions are now dynamic — resolved per-day by DYNAMIC_ORB_RESOLVERS.
# ORB_TIMES_LOCAL removed: was empty, replaced by explicit ValueError in _orb_utc_window().

# Session stat windows: FIXED Brisbane-time approximations for computing
# session range features (high/low). These do NOT track actual market opens
# which shift with DST. For DST-aware session times, see pipeline/dst.py.
SESSION_WINDOWS = {
    "asia":   (9, 0, 17, 0),
    "london": (18, 0, 23, 0),
    "ny":     (23, 0, 2, 0),    # crosses midnight
}

# Valid ORB durations in minutes
VALID_ORB_MINUTES = [5, 15, 30]

# FAIL-CLOSED: every ORB label must be classified as DST-affected or DST-clean.
# Prevents silent contamination if a new session is added without DST classification.
_dst_classified = set(DST_AFFECTED_SESSIONS.keys()) | DST_CLEAN_SESSIONS
_unclassified = set(ORB_LABELS) - _dst_classified
if _unclassified:
    raise RuntimeError(
        f"ORB labels not classified in DST_AFFECTED_SESSIONS or DST_CLEAN_SESSIONS: "
        f"{sorted(_unclassified)}. Update pipeline/dst.py before building features."
    )

# =============================================================================
# MODULE 1: TRADING DAY ASSIGNMENT
# =============================================================================

def compute_trading_day(ts_utc: pd.Timestamp) -> date:
    """
    Assign a UTC timestamp to its Brisbane trading day.

    Trading day boundary: 09:00 Brisbane (23:00 UTC previous day).
    A bar at exactly 23:00 UTC belongs to the NEW trading day.
    A bar at 22:59 UTC belongs to the previous trading day.

    Formula: DATE(ts_brisbane - 9 hours)
    Equivalent: subtract 9h from Brisbane time, take the date.
    """
    # Convert to Brisbane
    ts_bris = ts_utc.astimezone(BRISBANE_TZ)
    # Subtract 9 hours and take date
    shifted = ts_bris - timedelta(hours=TRADING_DAY_START_HOUR_LOCAL)
    return shifted.date()

def compute_trading_day_utc_range(trading_day: date) -> tuple[datetime, datetime]:
    """
    Return the [start, end) UTC range for a given trading day.

    trading_day 2024-01-05:
      start = 2024-01-04 23:00:00 UTC (09:00 Brisbane on 2024-01-05)
      end   = 2024-01-05 23:00:00 UTC (09:00 Brisbane on 2024-01-06)
    """
    # 09:00 Brisbane on trading_day = 23:00 UTC on (trading_day - 1)
    start_utc = datetime(
        trading_day.year, trading_day.month, trading_day.day,
        TRADING_DAY_START_HOUR_LOCAL, 0, 0,
        tzinfo=BRISBANE_TZ
    ).astimezone(UTC_TZ)

    end_utc = start_utc + timedelta(hours=24)
    return start_utc, end_utc

def get_trading_days_in_range(con: duckdb.DuckDBPyConnection, symbol: str,
                               start_date: date, end_date: date) -> list[date]:
    """
    Get distinct trading days that have bars_1m data in the date range.

    Uses the trading day formula in SQL:
      CAST((ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours') AS DATE)
    """
    query = """
        SELECT DISTINCT
            CAST((ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours') AS DATE) AS trading_day
        FROM bars_1m
        WHERE symbol = ?
        AND CAST((ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours') AS DATE) >= ?
        AND CAST((ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours') AS DATE) <= ?
        ORDER BY trading_day
    """
    rows = con.execute(query, [symbol, start_date, end_date]).fetchall()
    return [r[0] for r in rows]

def get_bars_for_trading_day(con: duckdb.DuckDBPyConnection, symbol: str,
                              trading_day: date) -> pd.DataFrame:
    """
    Fetch all bars_1m for a single trading day, ordered by ts_utc.

    Returns DataFrame with columns: ts_utc, open, high, low, close, volume, source_symbol
    ts_utc is timezone-aware (UTC).
    """
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)

    query = """
        SELECT ts_utc, open, high, low, close, volume, source_symbol
        FROM bars_1m
        WHERE symbol = ?
        AND ts_utc >= ?::TIMESTAMPTZ
        AND ts_utc < ?::TIMESTAMPTZ
        ORDER BY ts_utc ASC
    """
    df = con.execute(query, [symbol, start_utc.isoformat(), end_utc.isoformat()]).fetchdf()

    if not df.empty:
        # Ensure ts_utc is tz-aware
        df['ts_utc'] = pd.to_datetime(df['ts_utc'], utc=True)

    return df

# =============================================================================
# MODULE 2: ORB RANGES
# =============================================================================

def _orb_utc_window(trading_day: date, orb_label: str,
                     orb_minutes: int) -> tuple[datetime, datetime]:
    """
    Compute the [start, end) UTC window for an ORB on a given trading day.

    The ORB starts at the local Brisbane time and lasts orb_minutes.

    All sessions are dynamic — the Brisbane hour is resolved per-day
    based on DST via pipeline/dst.py DYNAMIC_ORB_RESOLVERS.

    Example: CME_REOPEN ORB with 5 min duration on trading_day 2024-01-05
      local start = 2024-01-05 09:00 Brisbane
      local end   = 2024-01-05 09:05 Brisbane
      UTC start   = 2024-01-04 23:00 UTC
      UTC end     = 2024-01-04 23:05 UTC

    Special case: NYSE_OPEN ORB belongs to the SAME trading day but is
    at 00:30 the NEXT calendar day in Brisbane.
      trading_day 2024-01-05, NYSE_OPEN ORB:
      local start = 2024-01-06 00:30 Brisbane (next calendar day)
      UTC start   = 2024-01-05 14:30 UTC
    """
    if orb_label in DYNAMIC_ORB_RESOLVERS:
        hour, minute = DYNAMIC_ORB_RESOLVERS[orb_label](trading_day)
    else:
        raise ValueError(f"Unknown ORB label '{orb_label}' — not in DYNAMIC_ORB_RESOLVERS")

    # Determine the Brisbane calendar date for this ORB time
    # Trading day 09:00 Brisbane starts at calendar_date = trading_day
    # Times 09:00-23:59 are on the same calendar day
    # Times 00:00-08:59 are on the NEXT calendar day
    if hour < TRADING_DAY_START_HOUR_LOCAL:
        # After midnight Brisbane — next calendar day
        cal_date = trading_day + timedelta(days=1)
    else:
        cal_date = trading_day

    local_start = datetime(
        cal_date.year, cal_date.month, cal_date.day,
        hour, minute, 0,
        tzinfo=BRISBANE_TZ
    )
    local_end = local_start + timedelta(minutes=orb_minutes)

    utc_start = local_start.astimezone(UTC_TZ)
    utc_end = local_end.astimezone(UTC_TZ)

    # Fail-closed: ORB must fall within the trading day's UTC window
    td_start, td_end = compute_trading_day_utc_range(trading_day)
    if not (td_start <= utc_start < td_end):
        raise ValueError(
            f"ORB {orb_label} on {trading_day} resolved to {utc_start} UTC, "
            f"outside trading day window [{td_start}, {td_end})"
        )

    return utc_start, utc_end

def compute_orb_range(bars_df: pd.DataFrame, trading_day: date,
                       orb_label: str, orb_minutes: int) -> dict:
    """
    Compute ORB high/low/size/volume for a single ORB on a single trading day.

    Returns dict with keys: high, low, size, volume (or all None if no bars in window).
    volume = total contracts traded during the ORB window.
    """
    utc_start, utc_end = _orb_utc_window(trading_day, orb_label, orb_minutes)

    # Filter bars within [start, end)
    mask = (bars_df['ts_utc'] >= utc_start) & (bars_df['ts_utc'] < utc_end)
    orb_bars = bars_df[mask]

    if orb_bars.empty:
        return {"high": None, "low": None, "size": None, "volume": None}

    high = float(orb_bars['high'].max())
    low = float(orb_bars['low'].min())
    size = high - low
    volume = int(orb_bars['volume'].sum())

    return {"high": high, "low": low, "size": size, "volume": volume}

# =============================================================================
# MODULE 3: BREAK DETECTION
# =============================================================================

def _break_detection_window(trading_day: date, orb_label: str,
                             orb_minutes: int) -> tuple[datetime, datetime]:
    """
    Return the [start, end) UTC window for break detection.

    Start: end of ORB window.
    End: start of next ORB window in a DIFFERENT break_group (by UTC time),
         or end of trading day if this is the last group of the day.

    Break groups (defined in pipeline/dst.py SESSION_CATALOG) prevent adding
    a nearby session from silently shrinking an existing session's break
    window. Sessions in the same group (e.g., TOKYO_OPEN/SINGAPORE_OPEN in "asia")
    all extend their break windows to the same boundary (e.g., LONDON_METALS "london").

    With dynamic sessions, ORB ordering is determined by actual UTC start
    time (which varies with DST), not by list position.
    """
    orb_start, orb_end = _orb_utc_window(trading_day, orb_label, orb_minutes)
    my_group = get_break_group(orb_label)

    # Find the earliest ORB start in a DIFFERENT group that's after our end
    next_group_start = None
    for label in ORB_LABELS:
        if label == orb_label:
            continue
        other_group = get_break_group(label)
        # Only consider labels in a different group as boundaries
        if my_group is not None and other_group == my_group:
            continue
        other_start, _ = _orb_utc_window(trading_day, label, orb_minutes)
        if other_start >= orb_end:
            if next_group_start is None or other_start < next_group_start:
                next_group_start = other_start

    if next_group_start is not None:
        return orb_end, next_group_start
    else:
        # Last group of the day — window until end of trading day
        _, td_end = compute_trading_day_utc_range(trading_day)
        return orb_end, td_end

def detect_break(bars_df: pd.DataFrame, trading_day: date,
                  orb_label: str, orb_minutes: int,
                  orb_high: float, orb_low: float) -> dict:
    """
    Detect the first 1m bar whose CLOSE breaks outside the ORB range.

    Break long: close > orb_high
    Break short: close < orb_low

    Returns dict with keys:
      break_dir ('long'/'short'/None)
      break_ts (datetime/None)
      break_delay_min (float/None) - minutes from ORB end to first break
      break_bar_continues (bool/None) - break bar closes in break direction
    """
    no_break = {
        "break_dir": None, "break_ts": None,
        "break_delay_min": None, "break_bar_continues": None,
        "break_bar_volume": None,
    }

    if orb_high is None or orb_low is None:
        return no_break

    window_start, window_end = _break_detection_window(
        trading_day, orb_label, orb_minutes
    )
    # window_start = ORB end time (start of break detection window)
    orb_end = window_start

    # Filter bars in break detection window
    mask = (bars_df['ts_utc'] >= window_start) & (bars_df['ts_utc'] < window_end)
    window_bars = bars_df[mask].sort_values('ts_utc')

    for bar in window_bars.itertuples():
        close = float(bar.close)
        bar_open = float(bar.open)
        bar_ts = bar.ts_utc.to_pydatetime()

        if close > orb_high:
            delay = (bar_ts - orb_end).total_seconds() / 60.0
            return {
                "break_dir": "long",
                "break_ts": bar_ts,
                "break_delay_min": delay,
                "break_bar_continues": close > bar_open,  # green candle = continuation
                "break_bar_volume": int(bar.volume),
            }
        elif close < orb_low:
            delay = (bar_ts - orb_end).total_seconds() / 60.0
            return {
                "break_dir": "short",
                "break_ts": bar_ts,
                "break_delay_min": delay,
                "break_bar_continues": close < bar_open,  # red candle = continuation
                "break_bar_volume": int(bar.volume),
            }

    return no_break

def detect_double_break(bars_df: pd.DataFrame, trading_day: date,
                         orb_label: str, orb_minutes: int,
                         orb_high: float | None,
                         orb_low: float | None) -> bool | None:
    """
    Detect if BOTH the ORB high and low were breached during the session.

    A "double break" means price hit both sides of the ORB range after the
    ORB closed. This is a regime signal -- high double-break frequency
    indicates choppy/mean-reverting conditions where single-direction
    breakout strategies degrade.

    Returns True if both boundaries breached, False if only one or none,
    None if ORB data is missing.
    """
    if orb_high is None or orb_low is None:
        return None

    window_start, window_end = _break_detection_window(
        trading_day, orb_label, orb_minutes
    )

    mask = (bars_df['ts_utc'] >= window_start) & (bars_df['ts_utc'] < window_end)
    window_bars = bars_df[mask]

    if window_bars.empty:
        return None

    hit_high = (window_bars['high'] >= orb_high).any()
    hit_low = (window_bars['low'] <= orb_low).any()

    return bool(hit_high and hit_low)

# =============================================================================
# MODULE 4: SESSION STATS
# =============================================================================

def _session_utc_window(trading_day: date, session: str) -> tuple[datetime, datetime]:
    """
    Compute the [start, end) UTC window for a session on a trading day.

    Fixed Brisbane-time windows for session range stats.
    These are approximations — actual market opens shift with DST.
    See pipeline/dst.py for DST-aware times.
    """
    start_h, start_m, end_h, end_m = SESSION_WINDOWS[session]

    # Start date in Brisbane calendar terms
    if start_h < TRADING_DAY_START_HOUR_LOCAL:
        start_cal = trading_day + timedelta(days=1)
    else:
        start_cal = trading_day

    local_start = datetime(
        start_cal.year, start_cal.month, start_cal.day,
        start_h, start_m, 0, tzinfo=BRISBANE_TZ
    )

    # End date — if end_h < start_h, it crosses midnight (next calendar day)
    if end_h < start_h:
        end_cal = start_cal + timedelta(days=1)
    else:
        end_cal = start_cal

    local_end = datetime(
        end_cal.year, end_cal.month, end_cal.day,
        end_h, end_m, 0, tzinfo=BRISBANE_TZ
    )

    return local_start.astimezone(UTC_TZ), local_end.astimezone(UTC_TZ)

def compute_session_stats(bars_df: pd.DataFrame,
                           trading_day: date) -> dict:
    """
    Compute session range stats using fixed Brisbane-time windows.

    These are approximate session ranges for features, NOT DST-aware.
    For actual market open times, see pipeline/dst.py SESSION_CATALOG.

    Returns dict with keys:
      session_asia_high, session_asia_low,
      session_london_high, session_london_low,
      session_ny_high, session_ny_low
    """
    result = {}

    for session in ["asia", "london", "ny"]:
        utc_start, utc_end = _session_utc_window(trading_day, session)

        mask = (bars_df['ts_utc'] >= utc_start) & (bars_df['ts_utc'] < utc_end)
        session_bars = bars_df[mask]

        if session_bars.empty:
            result[f"session_{session}_high"] = None
            result[f"session_{session}_low"] = None
        else:
            result[f"session_{session}_high"] = float(session_bars['high'].max())
            result[f"session_{session}_low"] = float(session_bars['low'].min())

    return result


def compute_overnight_stats(bars_df: pd.DataFrame, trading_day: date) -> dict:
    """
    Compute overnight (Asia session) and pre-TOKYO_OPEN session stats.

    overnight_*: Asia session window (09:00-17:00 Brisbane = first 8h of trading day).
    pre_1000_*:  Bars from trading day start to 10:00 Brisbane (hour before TOKYO_OPEN ORB).

    These are pre-entry for the TOKYO_OPEN session — no look-ahead.
    """
    result: dict = {
        "overnight_high": None, "overnight_low": None, "overnight_range": None,
        "pre_1000_high": None, "pre_1000_low": None,
    }

    if bars_df.empty:
        return result

    # Overnight: Asia session window (SESSION_WINDOWS["asia"] = 09:00-17:00 Brisbane)
    asia_start, asia_end = _session_utc_window(trading_day, "asia")
    asia_mask = (bars_df['ts_utc'] >= asia_start) & (bars_df['ts_utc'] < asia_end)
    asia_bars = bars_df[asia_mask]

    if not asia_bars.empty:
        result["overnight_high"] = float(asia_bars['high'].max())
        result["overnight_low"]  = float(asia_bars['low'].min())
        result["overnight_range"] = round(result["overnight_high"] - result["overnight_low"], 4)

    # Pre-TOKYO_OPEN: all bars from trading day start up to (but not including) 10:00 Brisbane
    # _orb_utc_window(day, "TOKYO_OPEN", 5)[0] = start of the TOKYO_OPEN ORB window = 10:00 Brisbane
    pre_1000_start = _orb_utc_window(trading_day, "TOKYO_OPEN", 5)[0]
    td_start, _ = compute_trading_day_utc_range(trading_day)
    pre_mask = (bars_df['ts_utc'] >= td_start) & (bars_df['ts_utc'] < pre_1000_start)
    pre_bars = bars_df[pre_mask]

    if not pre_bars.empty:
        result["pre_1000_high"] = float(pre_bars['high'].max())
        result["pre_1000_low"]  = float(pre_bars['low'].min())

    return result


def classify_day_type(
    daily_open: float | None,
    daily_high: float | None,
    daily_low: float | None,
    daily_close: float | None,
    atr_20: float | None,
) -> str | None:
    """
    Classify the retrospective day type using daily OHLC and ATR.

    Returns one of:
      'TREND_UP'      — wide range, closed in top 30%
      'TREND_DOWN'    — wide range, closed in bottom 30%
      'REVERSAL_UP'   — opened low, closed high (medium range)
      'REVERSAL_DOWN' — opened high, closed low (medium range)
      'BALANCED'      — medium range, no clear direction
      'NON_TREND'     — tight range (< 50% of ATR)
      None            — insufficient data

    NOTE: This is LOOK-AHEAD relative to intraday entry. For research only.
    Do NOT use as a live trading filter.
    """
    if None in (daily_open, daily_high, daily_low, daily_close, atr_20):
        return None
    if atr_20 <= 0:
        return None

    day_range = daily_high - daily_low
    if day_range <= 0:
        return None

    range_pct = day_range / atr_20
    close_pct = (daily_close - daily_low) / day_range  # 0=closed at low, 1=at high

    if range_pct < 0.5:
        return "NON_TREND"
    if close_pct >= 0.7:
        return "TREND_UP"
    if close_pct <= 0.3:
        return "TREND_DOWN"
    # Medium range — check reversal pattern
    lower_40pct = daily_low + day_range * 0.4
    upper_60pct = daily_low + day_range * 0.6
    if daily_open < lower_40pct and daily_close > upper_60pct:
        return "REVERSAL_UP"
    if daily_open > upper_60pct and daily_close < lower_40pct:
        return "REVERSAL_DOWN"
    return "BALANCED"


def compute_garch_forecast(daily_closes: list[float], min_obs: int = 252) -> float | None:
    """
    Fit GARCH(1,1) on trailing daily close-to-close log returns.
    Returns 1-step-ahead annualized conditional volatility forecast.

    Returns None if:
      - Fewer than min_obs closes
      - All returns are zero (constant prices)
      - Model fails to converge
    """
    if len(daily_closes) < min_obs:
        return None

    closes = np.array(daily_closes, dtype=float)
    log_returns = np.diff(np.log(closes)) * 100  # percentage for numerical stability

    if np.all(log_returns == 0):
        return None

    try:
        from arch import arch_model
        model = arch_model(log_returns, vol="Garch", p=1, q=1, dist="Normal", mean="Zero")
        result = model.fit(disp="off", show_warning=False)
        forecast = result.forecast(horizon=1)
        cond_var = forecast.variance.iloc[-1, 0]
        # Undo percentage scaling, annualize
        daily_vol = (cond_var ** 0.5) / 100
        annual_vol = daily_vol * (252 ** 0.5)
        return round(float(annual_vol), 6)
    except Exception:
        return None


# =============================================================================
# MODULE 5: RSI (Wilder's 14-period on 5m closes)
# =============================================================================

def compute_rsi_at_0900(con: duckdb.DuckDBPyConnection, symbol: str,
                         trading_day: date,
                         bars_5m_ts: np.ndarray | None = None,
                         bars_5m_closes: np.ndarray | None = None) -> float | None:
    """
    Compute RSI-14 (Wilder's smoothing) on 5m closes, evaluated at 09:00 Brisbane.

    We need at least 14 prior 5m bars to compute RSI.
    We take the most recent 200 5m bars ending at or before 09:00 Brisbane (23:00 UTC)
    to ensure enough history for Wilder's smoothing to stabilize.

    Args:
        bars_5m_ts: Pre-loaded sorted timestamps (numpy datetime64). If None, queries DB.
        bars_5m_closes: Pre-loaded close prices matching bars_5m_ts.

    Returns RSI value (0-100) or None if insufficient data.
    """
    # 09:00 Brisbane on trading_day = 23:00 UTC on (trading_day - 1)
    orb_0900_utc = datetime(
        trading_day.year, trading_day.month, trading_day.day,
        TRADING_DAY_START_HOUR_LOCAL, 0, 0,
        tzinfo=BRISBANE_TZ
    ).astimezone(UTC_TZ)

    if bars_5m_ts is not None and bars_5m_closes is not None:
        # Fast path: slice from pre-loaded arrays using binary search
        end_idx = int(np.searchsorted(bars_5m_ts, pd.Timestamp(orb_0900_utc).asm8, side="right"))
        start_idx = max(0, end_idx - 200)
        closes = bars_5m_closes[start_idx:end_idx]

        if len(closes) < 15:
            return None

        return _wilders_rsi(closes, period=14)

    # Slow path: query DB (fallback)
    query = """
        SELECT ts_utc, close
        FROM bars_5m
        WHERE symbol = ?
        AND ts_utc <= ?::TIMESTAMPTZ
        ORDER BY ts_utc DESC
        LIMIT 200
    """
    df = con.execute(query, [symbol, orb_0900_utc.isoformat()]).fetchdf()

    if len(df) < 15:  # Need at least 14+1 bars
        return None

    # Sort ascending for computation
    df = df.sort_values('ts_utc').reset_index(drop=True)
    closes = df['close'].astype(float).values

    # Compute RSI with Wilder's smoothing
    return _wilders_rsi(closes, period=14)

def _wilders_rsi(closes: np.ndarray, period: int = 14) -> float | None:
    """
    Compute RSI using Wilder's smoothing method.

    Wilder's smoothing:
      avg_gain[0] = mean(gains[0:period])
      avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period

    Returns the final RSI value, or None if insufficient data.
    """
    if len(closes) < period + 1:
        return None

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial averages (SMA of first `period` values)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Wilder's smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss < 1e-12:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi, 4)

# =============================================================================
# MODULE 6: OUTCOME AT RR=1.0
# =============================================================================

def compute_outcome(bars_df: pd.DataFrame, trading_day: date,
                     orb_label: str, orb_minutes: int,
                     orb_high: float, orb_low: float,
                     break_dir: str | None,
                     break_ts: datetime | None,
                     cost_spec: CostSpec | None = None) -> dict:
    """
    Determine the outcome at RR=1.0 after an ORB break.

    For a LONG break (close > orb_high):
      Entry = orb_high (breakout level)
      Risk = orb_high - orb_low (ORB size)
      Target = entry + risk = orb_high + (orb_high - orb_low) = 2*orb_high - orb_low
      Stop = orb_low
      Win: price reaches target before stop
      Loss: price reaches stop before target
      Scratch: neither reached by end of trading day

    For a SHORT break (close < orb_low):
      Entry = orb_low
      Risk = orb_high - orb_low
      Target = entry - risk = orb_low - (orb_high - orb_low) = 2*orb_low - orb_high
      Stop = orb_high
      Win/Loss/Scratch same logic

    MAE/MFE: computed in R-multiples using cost_spec if provided.
      MAE = max adverse excursion (worst point against trade, in R)
      MFE = max favorable excursion (best point for trade, in R)

    Returns dict: outcome, mae_r, mfe_r
    """
    result = {"outcome": None, "mae_r": None, "mfe_r": None}

    if break_dir is None or break_ts is None or orb_high is None or orb_low is None:
        return result

    orb_size = orb_high - orb_low
    if orb_size <= 0:
        return result

    if break_dir == "long":
        entry = orb_high
        target = entry + orb_size
        stop = orb_low
    else:  # short
        entry = orb_low
        target = entry - orb_size
        stop = orb_high

    # Get bars after break_ts until end of trading day
    _, td_end = compute_trading_day_utc_range(trading_day)
    # Start from the bar AFTER the break bar (no same-bar execution)
    mask = (bars_df['ts_utc'] > break_ts) & (bars_df['ts_utc'] < td_end)
    post_break = bars_df[mask].sort_values('ts_utc')

    # Track MAE/MFE
    max_adverse_points = 0.0   # worst excursion against us
    max_favorable_points = 0.0  # best excursion for us

    for bar in post_break.itertuples():
        bar_high = float(bar.high)
        bar_low = float(bar.low)

        # Track excursions
        if break_dir == "long":
            favorable = bar_high - entry
            adverse = entry - bar_low
            hit_target = bar_high >= target
            hit_stop = bar_low <= stop
        else:  # short
            favorable = entry - bar_low
            adverse = bar_high - entry
            hit_target = bar_low <= target
            hit_stop = bar_high >= stop

        max_favorable_points = max(max_favorable_points, favorable)
        max_adverse_points = max(max_adverse_points, adverse)

        if hit_target and hit_stop:
            result["outcome"] = "loss"
            break
        elif hit_target:
            result["outcome"] = "win"
            break
        elif hit_stop:
            result["outcome"] = "loss"
            break
    else:
        # Neither target nor stop reached
        result["outcome"] = "scratch"

    # Convert MAE/MFE to R-multiples if cost_spec provided
    if cost_spec is not None:
        result["mae_r"] = round(
            pnl_points_to_r(cost_spec, entry, stop, max_adverse_points), 4
        )
        result["mfe_r"] = round(
            pnl_points_to_r(cost_spec, entry, stop, max_favorable_points), 4
        )

    return result

# =============================================================================
# ORCHESTRATOR: BUILD ONE TRADING DAY
# =============================================================================

def build_features_for_day(con: duckdb.DuckDBPyConnection, symbol: str,
                            trading_day: date,
                            orb_minutes: int,
                            cost_spec: CostSpec | None = None,
                            bars_df: pd.DataFrame | None = None,
                            bars_5m_ts: np.ndarray | None = None,
                            bars_5m_closes: np.ndarray | None = None) -> dict:
    """
    Build all daily_features columns for a single trading day.

    Args:
        bars_df: Pre-loaded 1m bars for this trading day. If None, queries DB (slow path).
        bars_5m_ts: Pre-loaded sorted 5m timestamps for RSI. If None, queries DB.
        bars_5m_closes: Pre-loaded 5m close prices matching bars_5m_ts.

    Returns a dict matching daily_features column names.
    """
    # Use pre-loaded bars if available, otherwise fetch (slow path)
    if bars_df is None:
        bars_df = get_bars_for_trading_day(con, symbol, trading_day)

    row = {
        "trading_day": trading_day,
        "symbol": symbol,
        "orb_minutes": orb_minutes,
        "bar_count_1m": len(bars_df),
    }

    # DST flags for this trading day
    row["us_dst"] = is_us_dst(trading_day)
    row["uk_dst"] = is_uk_dst(trading_day)

    # Calendar skip flags (deterministic from date)
    row["is_nfp_day"] = is_nfp_day(trading_day)
    row["is_opex_day"] = is_opex_day(trading_day)
    row["is_friday"] = is_friday(trading_day)
    row["is_monday"] = is_monday(trading_day)
    row["is_tuesday"] = is_tuesday(trading_day)
    row["day_of_week"] = day_of_week(trading_day)

    # Daily OHLC from all 1m bars
    if not bars_df.empty:
        row["daily_open"] = float(bars_df.iloc[0]["open"])
        row["daily_high"] = float(bars_df["high"].max())
        row["daily_low"] = float(bars_df["low"].min())
        row["daily_close"] = float(bars_df.iloc[-1]["close"])
    else:
        row["daily_open"] = None
        row["daily_high"] = None
        row["daily_low"] = None
        row["daily_close"] = None

    # gap_open_points, atr_20, atr velocity, compression, and rel_vol computed in post-pass
    row["gap_open_points"] = None
    row["atr_20"] = None
    row["atr_vel_ratio"] = None
    row["atr_vel_regime"] = None
    row["garch_forecast_vol"] = None
    row["garch_atr_ratio"] = None
    for _sl in ["CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS"]:
        row[f"orb_{_sl}_compression_z"] = None
        row[f"orb_{_sl}_compression_tier"] = None
    # rel_vol initialised here; computed in post-pass after all days are processed
    for _sl in ORB_LABELS:
        row[f"rel_vol_{_sl}"] = None

    # Market Profile context columns — overnight/pre-session computed below;
    # prev_day_*, sweep labels, gap_type, day_type computed in post-pass
    row["overnight_high"]        = None
    row["overnight_low"]         = None
    row["overnight_range"]       = None
    row["pre_1000_high"]         = None
    row["pre_1000_low"]          = None
    row["prev_day_high"]         = None
    row["prev_day_low"]          = None
    row["prev_day_close"]        = None
    row["prev_day_range"]        = None
    row["prev_day_direction"]    = None
    row["gap_type"]              = None
    row["took_pdh_before_1000"]  = None
    row["took_pdl_before_1000"]  = None
    row["overnight_took_pdh"]    = None
    row["overnight_took_pdl"]    = None
    row["day_type"]              = None

    # Module 4: Session stats
    session_stats = compute_session_stats(bars_df, trading_day)
    row.update(session_stats)

    # Module 4b: Overnight + pre-session stats (pre-entry for 1000 session)
    overnight_stats = compute_overnight_stats(bars_df, trading_day)
    row.update(overnight_stats)

    # Module 5: RSI at 0900
    row["rsi_14_at_0900"] = compute_rsi_at_0900(
        con, symbol, trading_day,
        bars_5m_ts=bars_5m_ts, bars_5m_closes=bars_5m_closes,
    )

    # Modules 2, 3, 6: ORBs, breaks, outcomes
    for label in ORB_LABELS:
        # Module 2: ORB range
        orb = compute_orb_range(bars_df, trading_day, label, orb_minutes)
        row[f"orb_{label}_high"] = orb["high"]
        row[f"orb_{label}_low"] = orb["low"]
        row[f"orb_{label}_size"] = orb["size"]
        row[f"orb_{label}_volume"] = orb["volume"]

        # Module 3: Break detection
        brk = detect_break(
            bars_df, trading_day, label, orb_minutes,
            orb["high"], orb["low"]
        )
        row[f"orb_{label}_break_dir"] = brk["break_dir"]
        row[f"orb_{label}_break_ts"] = brk["break_ts"]
        row[f"orb_{label}_break_delay_min"] = brk["break_delay_min"]
        row[f"orb_{label}_break_bar_continues"] = brk["break_bar_continues"]
        row[f"orb_{label}_break_bar_volume"] = brk["break_bar_volume"]

        # Module 6: Outcome + MAE/MFE
        outcome = compute_outcome(
            bars_df, trading_day, label, orb_minutes,
            orb["high"], orb["low"],
            brk["break_dir"], brk["break_ts"],
            cost_spec=cost_spec,
        )
        row[f"orb_{label}_outcome"] = outcome["outcome"]
        row[f"orb_{label}_mae_r"] = outcome["mae_r"]
        row[f"orb_{label}_mfe_r"] = outcome["mfe_r"]

        # Double-break detection (regime signal)
        row[f"orb_{label}_double_break"] = detect_double_break(
            bars_df, trading_day, label, orb_minutes,
            orb["high"], orb["low"],
        )

    return row

# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_daily_features(con: duckdb.DuckDBPyConnection, symbol: str,
                          start_date: date, end_date: date,
                          orb_minutes: int, dry_run: bool) -> int:
    """
    Build daily_features for all trading days in [start_date, end_date].

    Returns number of rows written.
    """
    # Load cost model (optional — only if instrument is validated)
    try:
        cost_spec = get_cost_spec(symbol)
        logger.info(f"  Cost model: {symbol} (friction={cost_spec.total_friction:.2f})")
    except ValueError:
        cost_spec = None
        logger.info(f"  Cost model: not available for {symbol} (MAE/MFE will be NULL)")

    # Get trading days with data
    trading_days = get_trading_days_in_range(con, symbol, start_date, end_date)
    logger.info(f"  Trading days with data: {len(trading_days)}")

    if not trading_days:
        logger.info("  No trading days found. Nothing to build.")
        return 0

    if dry_run:
        logger.info(f"  DRY RUN: Would build {len(trading_days)} daily_features rows")
        logger.info(f"  Date range: {trading_days[0]} to {trading_days[-1]}")
        return len(trading_days)

    # Bulk-load all bars_1m for the date range (one query instead of ~1500)
    range_start_utc, _ = compute_trading_day_utc_range(trading_days[0])
    _, range_end_utc = compute_trading_day_utc_range(trading_days[-1])
    logger.info(f"  Bulk-loading bars_1m ({range_start_utc.date()} to {range_end_utc.date()})...")

    all_bars_df = con.execute("""
        SELECT ts_utc, open, high, low, close, volume, source_symbol
        FROM bars_1m
        WHERE symbol = ?
        AND ts_utc >= ?::TIMESTAMPTZ
        AND ts_utc < ?::TIMESTAMPTZ
        ORDER BY ts_utc ASC
    """, [symbol, range_start_utc.isoformat(), range_end_utc.isoformat()]).fetchdf()

    if not all_bars_df.empty:
        all_bars_df['ts_utc'] = pd.to_datetime(all_bars_df['ts_utc'], utc=True)

    # Pre-compute sorted timestamps for binary search (O(log n) per day)
    all_ts = all_bars_df['ts_utc'].values if not all_bars_df.empty else np.array([])

    logger.info(f"  Loaded {len(all_bars_df):,} bars for slicing")

    # Bulk-load bars_5m for RSI computation (one query instead of ~1500)
    # RSI needs up to 200 bars BEFORE the first trading day's 09:00, so extend lookback
    rsi_lookback_start = range_start_utc - timedelta(days=10)  # ~200 5m bars ≈ ~3.5 days
    logger.info("  Bulk-loading bars_5m for RSI...")

    all_bars_5m_df = con.execute("""
        SELECT ts_utc, close
        FROM bars_5m
        WHERE symbol = ?
        AND ts_utc >= ?::TIMESTAMPTZ
        AND ts_utc < ?::TIMESTAMPTZ
        ORDER BY ts_utc ASC
    """, [symbol, rsi_lookback_start.isoformat(), range_end_utc.isoformat()]).fetchdf()

    if not all_bars_5m_df.empty:
        all_bars_5m_df['ts_utc'] = pd.to_datetime(all_bars_5m_df['ts_utc'], utc=True)

    bars_5m_ts = all_bars_5m_df['ts_utc'].values if not all_bars_5m_df.empty else np.array([])
    bars_5m_closes = all_bars_5m_df['close'].astype(float).values if not all_bars_5m_df.empty else np.array([])

    logger.info(f"  Loaded {len(all_bars_5m_df):,} 5m bars for RSI")

    # Build features for each trading day
    rows = []
    for i, td in enumerate(trading_days):
        # Slice bars for this trading day using binary search
        td_start, td_end = compute_trading_day_utc_range(td)
        start_idx = int(np.searchsorted(all_ts, pd.Timestamp(td_start).asm8, side="left"))
        end_idx = int(np.searchsorted(all_ts, pd.Timestamp(td_end).asm8, side="left"))
        day_bars = all_bars_df.iloc[start_idx:end_idx]

        row = build_features_for_day(
            con, symbol, td, orb_minutes, cost_spec,
            bars_df=day_bars,
            bars_5m_ts=bars_5m_ts, bars_5m_closes=bars_5m_closes,
        )
        rows.append(row)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(trading_days)} trading days...")

    # Post-pass: compute gap_open_points and ATR(20)
    # Both need previous day data, so computed after all rows are built.
    #
    # True Range = max(H - L, |H - prevClose|, |L - prevClose|)
    # ATR(20) = simple moving average of last 20 True Range values
    # First 20 rows get partial averages (best available), row 0 gets H-L only.
    true_ranges = []
    for i in range(len(rows)):
        prev_close = rows[i - 1].get("daily_close") if i > 0 else None
        today_open = rows[i].get("daily_open")
        today_high = rows[i].get("daily_high")
        today_low = rows[i].get("daily_low")

        # gap_open_points (existing logic)
        if prev_close is not None and today_open is not None:
            rows[i]["gap_open_points"] = round(today_open - prev_close, 4)

        # True Range
        if today_high is not None and today_low is not None:
            hl = today_high - today_low
            if prev_close is not None:
                tr = max(hl, abs(today_high - prev_close), abs(today_low - prev_close))
            else:
                tr = hl  # first row: no prev close, use H-L
            true_ranges.append(tr)
        else:
            true_ranges.append(None)

        # ATR(20) = SMA of last 20 True Range values, prior days only (no look-ahead)
        lookback = [v for v in true_ranges[max(0, i - 20):i] if v is not None]
        if lookback:
            rows[i]["atr_20"] = round(sum(lookback) / len(lookback), 4)
        else:
            rows[i]["atr_20"] = None

        # ATR Velocity: today's ATR_20 vs 5-day prior average.
        # Uses ROWS [i-5 .. i-1] — prior days only, no look-ahead.
        atr_today = rows[i]["atr_20"]
        prior_atrs = [
            rows[j]["atr_20"] for j in range(max(0, i - 5), i)
            if rows[j].get("atr_20") is not None
        ]
        if atr_today is not None and len(prior_atrs) >= 5:
            avg_5d = sum(prior_atrs) / len(prior_atrs)
            if avg_5d > 0:
                vel = atr_today / avg_5d
                rows[i]["atr_vel_ratio"] = round(vel, 4)
                if vel > 1.05:
                    rows[i]["atr_vel_regime"] = "Expanding"
                elif vel < 0.95:
                    rows[i]["atr_vel_regime"] = "Contracting"
                else:
                    rows[i]["atr_vel_regime"] = "Stable"

        # Per-session ORB compression z-score (prior 20 days, no look-ahead).
        # Compression = rolling z-score of (orb_size / atr_20).
        # Requires ≥5 prior days to be meaningful.
        if atr_today is not None and atr_today > 0:
            for sess_label in ["CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS"]:
                size_col = f"orb_{sess_label}_size"
                size_today = rows[i].get(size_col)
                if size_today is None:
                    continue
                ratio_today = size_today / atr_today
                prior_ratios = []
                for j in range(max(0, i - 20), i):
                    s = rows[j].get(size_col)
                    a = rows[j].get("atr_20")
                    if s is not None and a is not None and a > 0:
                        prior_ratios.append(s / a)
                if len(prior_ratios) < 5:
                    continue
                mean_r = sum(prior_ratios) / len(prior_ratios)
                variance = sum((x - mean_r) ** 2 for x in prior_ratios) / len(prior_ratios)
                std_r = variance ** 0.5
                if std_r <= 0:
                    continue
                z = (ratio_today - mean_r) / std_r
                rows[i][f"orb_{sess_label}_compression_z"] = round(z, 4)
                if z < -0.5:
                    rows[i][f"orb_{sess_label}_compression_tier"] = "Compressed"
                elif z > 0.5:
                    rows[i][f"orb_{sess_label}_compression_tier"] = "Expanded"
                else:
                    rows[i][f"orb_{sess_label}_compression_tier"] = "Neutral"

        # GARCH(1,1) forward vol forecast from trailing daily closes.
        # Uses rows[0..i-1] daily_close values — prior days only, no look-ahead.
        prior_closes = [
            rows[j]["daily_close"] for j in range(i)
            if rows[j].get("daily_close") is not None
        ]
        garch_vol = compute_garch_forecast(prior_closes)
        if garch_vol is not None:
            rows[i]["garch_forecast_vol"] = garch_vol
            # Convert annualized vol to implied daily ATR-equivalent points,
            # then ratio against ATR-20 for apples-to-apples comparison.
            # garch_atr_ratio ~1.0 means GARCH agrees with ATR; >1 = GARCH sees more vol.
            last_close = prior_closes[-1] if prior_closes else None
            if atr_today is not None and atr_today > 0 and last_close is not None:
                implied_daily_atr = (garch_vol / (252 ** 0.5)) * last_close
                rows[i]["garch_atr_ratio"] = round(implied_daily_atr / atr_today, 4)

        # Prior day reference levels + gap_type + liquidity sweep labels + day_type.
        # All use rows[i-1] for prior-day data — no look-ahead.
        prev_high  = rows[i - 1].get("daily_high")  if i > 0 else None
        prev_low   = rows[i - 1].get("daily_low")   if i > 0 else None
        prev_close = rows[i - 1].get("daily_close") if i > 0 else None
        prev_open  = rows[i - 1].get("daily_open")  if i > 0 else None

        rows[i]["prev_day_high"]  = prev_high
        rows[i]["prev_day_low"]   = prev_low
        rows[i]["prev_day_close"] = prev_close

        if prev_high is not None and prev_low is not None:
            rows[i]["prev_day_range"] = round(prev_high - prev_low, 4)
        if prev_close is not None and prev_open is not None:
            rows[i]["prev_day_direction"] = "bull" if prev_close >= prev_open else "bear"

        # gap_type: classify gap_open_points relative to prior range
        gap_pts    = rows[i].get("gap_open_points")
        prev_range = rows[i].get("prev_day_range")
        if gap_pts is not None and prev_range is not None and prev_range > 0:
            threshold = 0.1 * prev_range
            if gap_pts > threshold:
                rows[i]["gap_type"] = "gap_up"
            elif gap_pts < -threshold:
                rows[i]["gap_type"] = "gap_down"
            else:
                rows[i]["gap_type"] = "inside"
        elif gap_pts is not None:
            rows[i]["gap_type"] = "none"

        # Liquidity sweep labels — compare pre-session high/low to prior day high/low
        pre_1000_high  = rows[i].get("pre_1000_high")
        pre_1000_low   = rows[i].get("pre_1000_low")
        overnight_high = rows[i].get("overnight_high")
        overnight_low  = rows[i].get("overnight_low")

        if pre_1000_high is not None and prev_high is not None:
            rows[i]["took_pdh_before_1000"] = bool(pre_1000_high > prev_high)
        if pre_1000_low is not None and prev_low is not None:
            rows[i]["took_pdl_before_1000"] = bool(pre_1000_low < prev_low)
        if overnight_high is not None and prev_high is not None:
            rows[i]["overnight_took_pdh"] = bool(overnight_high > prev_high)
        if overnight_low is not None and prev_low is not None:
            rows[i]["overnight_took_pdl"] = bool(overnight_low < prev_low)

        # Retrospective day type (atr_20 already computed in this loop pass)
        rows[i]["day_type"] = classify_day_type(
            rows[i].get("daily_open"),
            rows[i].get("daily_high"),
            rows[i].get("daily_low"),
            rows[i].get("daily_close"),
            rows[i].get("atr_20"),
        )

    # Post-pass: relative volume per session.
    #
    # rel_vol_{label} = break_bar_volume / median(break_bar_volume, last 20 break-days
    #                   for the same session label).
    #
    # Only defined on days where a break occurred (break_bar_volume is not None).
    # Compared against the same session's own prior breaks — controls for the
    # fact that CME_REOPEN gold is structurally thinner than US_DATA_830 gold without needing
    # same-minute cross-day lookups.
    #
    # Minimum lookback: 5 break-days (returns None during warm-up).
    # No look-ahead: only rows[0..i-1] are used when computing rows[i].
    for label in ORB_LABELS:
        bvol_col = f"orb_{label}_break_bar_volume"
        rel_col = f"rel_vol_{label}"
        prior_break_vols: list[int] = []

        for i in range(len(rows)):
            bvol = rows[i].get(bvol_col)
            if bvol is not None:
                # Compute rel_vol BEFORE appending today (no look-ahead)
                if len(prior_break_vols) >= 5:
                    window = prior_break_vols[-20:]
                    median_vol = float(sorted(window)[len(window) // 2])
                    if median_vol > 0:
                        rows[i][rel_col] = round(bvol / median_vol, 4)
                # Append today's break volume for future rows
                prior_break_vols.append(bvol)

    # Convert to DataFrame for bulk insert
    features_df = pd.DataFrame(rows)

    # IDEMPOTENT: Delete existing rows, then insert
    con.execute("BEGIN TRANSACTION")

    try:
        # Delete existing rows in range (scoped to this orb_minutes)
        delete_count = con.execute("""
            SELECT COUNT(*) FROM daily_features
            WHERE symbol = ?
            AND trading_day >= ?
            AND trading_day <= ?
            AND orb_minutes = ?
        """, [symbol, start_date, end_date, orb_minutes]).fetchone()[0]

        con.execute("""
            DELETE FROM daily_features
            WHERE symbol = ?
            AND trading_day >= ?
            AND trading_day <= ?
            AND orb_minutes = ?
        """, [symbol, start_date, end_date, orb_minutes])

        if delete_count > 0:
            logger.info(f"  Deleted {delete_count:,} existing daily_features rows")

        # Insert new rows — use explicit column list for safety
        col_names = ", ".join(features_df.columns)
        con.execute(f"""
            INSERT INTO daily_features ({col_names})
            SELECT {col_names} FROM features_df
        """)

        con.execute("COMMIT")

        logger.info(f"  Inserted {len(rows):,} daily_features rows")
        return len(rows)

    except Exception as e:
        con.execute("ROLLBACK")
        logger.error(f"FATAL: Exception during daily_features build: {e}")
        raise

# =============================================================================
# VERIFICATION
# =============================================================================

def verify_daily_features(con: duckdb.DuckDBPyConnection, symbol: str,
                           start_date: date, end_date: date) -> tuple[bool, list[str]]:
    """
    Verify daily_features integrity after build.

    Checks:
    1. No duplicate (symbol, trading_day, orb_minutes)
    2. bar_count_1m > 0 for all rows
    3. ORB size >= 0 where not NULL
    4. Break direction is valid enum
    5. Outcome is valid enum
    """
    failures = []

    # Check 1: duplicates (PK is symbol, trading_day, orb_minutes)
    dupe_count = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT symbol, trading_day, orb_minutes FROM daily_features
            WHERE symbol = ?
            AND trading_day >= ? AND trading_day <= ?
            GROUP BY symbol, trading_day, orb_minutes
            HAVING COUNT(*) > 1
        )
    """, [symbol, start_date, end_date]).fetchone()[0]

    if dupe_count > 0:
        failures.append(f"Duplicate (symbol, trading_day, orb_minutes): {dupe_count}")

    # Check 2: bar_count_1m > 0
    zero_bars = con.execute("""
        SELECT COUNT(*) FROM daily_features
        WHERE symbol = ?
        AND trading_day >= ? AND trading_day <= ?
        AND (bar_count_1m IS NULL OR bar_count_1m <= 0)
    """, [symbol, start_date, end_date]).fetchone()[0]

    if zero_bars > 0:
        failures.append(f"Rows with zero/null bar_count_1m: {zero_bars}")

    # Check 3: ORB size >= 0
    for label in ORB_LABELS:
        neg_size = con.execute(f"""
            SELECT COUNT(*) FROM daily_features
            WHERE symbol = ?
            AND trading_day >= ? AND trading_day <= ?
            AND orb_{label}_size < 0
        """, [symbol, start_date, end_date]).fetchone()[0]

        if neg_size > 0:
            failures.append(f"Negative ORB size for {label}: {neg_size}")

    # Check 4: Break direction enum
    for label in ORB_LABELS:
        bad_dir = con.execute(f"""
            SELECT COUNT(*) FROM daily_features
            WHERE symbol = ?
            AND trading_day >= ? AND trading_day <= ?
            AND orb_{label}_break_dir IS NOT NULL
            AND orb_{label}_break_dir NOT IN ('long', 'short')
        """, [symbol, start_date, end_date]).fetchone()[0]

        if bad_dir > 0:
            failures.append(f"Invalid break_dir for {label}: {bad_dir}")

    # Check 5: Outcome enum
    for label in ORB_LABELS:
        bad_outcome = con.execute(f"""
            SELECT COUNT(*) FROM daily_features
            WHERE symbol = ?
            AND trading_day >= ? AND trading_day <= ?
            AND orb_{label}_outcome IS NOT NULL
            AND orb_{label}_outcome NOT IN ('win', 'loss', 'scratch')
        """, [symbol, start_date, end_date]).fetchone()[0]

        if bad_outcome > 0:
            failures.append(f"Invalid outcome for {label}: {bad_outcome}")

    return len(failures) == 0, failures

# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build daily_features from bars_1m and bars_5m"
    )
    parser.add_argument(
        "--instrument", type=str, required=True,
        help=f"Instrument ({', '.join(list_instruments())})"
    )
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--orb-minutes", type=int, default=5, choices=VALID_ORB_MINUTES,
        help="ORB duration in minutes (default: 5)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Count only, no DB writes")
    args = parser.parse_args()

    # Validate instrument
    config = get_asset_config(args.instrument)
    symbol = config["symbol"]

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    if args.orb_minutes not in VALID_ORB_MINUTES:
        logger.error(f"FATAL: Invalid --orb-minutes {args.orb_minutes}. Must be one of {VALID_ORB_MINUTES}")
        sys.exit(1)

    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info(f"BUILD DAILY_FEATURES ({symbol})")
    logger.info("=" * 60)
    logger.info(f"Instrument: {symbol}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"ORB duration: {args.orb_minutes} minutes")
    logger.info(f"Database: {GOLD_DB_PATH}")
    logger.info(f"Dry run: {args.dry_run}")

    if not GOLD_DB_PATH.exists():
        logger.error(f"FATAL: Database not found: {GOLD_DB_PATH}")
        sys.exit(1)

    with duckdb.connect(str(GOLD_DB_PATH)) as con:
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)

        # Build
        logger.info("Building daily features...")
        row_count = build_daily_features(
            con, symbol, start_date, end_date, args.orb_minutes, args.dry_run
        )

        # Verify (skip for dry run)
        if not args.dry_run and row_count > 0:
            logger.info("Verifying integrity...")
            ok, failures = verify_daily_features(con, symbol, start_date, end_date)

            if not ok:
                logger.error("FATAL: Integrity verification FAILED:")
                for f in failures:
                    logger.info(f"  - {f}")
                sys.exit(1)

            logger.info("  No duplicates: PASSED [OK]")
            logger.info("  Bar counts: PASSED [OK]")
            logger.info("  ORB sizes: PASSED [OK]")
            logger.info("  Break directions: PASSED [OK]")
            logger.info("  Outcomes: PASSED [OK]")
            logger.info("ALL INTEGRITY CHECKS PASSED [OK]")
        elif args.dry_run:
            logger.info("Integrity check skipped (dry run)")

        elapsed = datetime.now() - start_time

        logger.info("=" * 60)
        logger.info(f"SUMMARY: {row_count:,} daily_features rows "
                    f"{'(would be) ' if args.dry_run else ''}built")
        logger.info(f"ORB duration: {args.orb_minutes}m")
        logger.info(f"Wall time: {elapsed}")
        logger.info("=" * 60)

        sys.exit(0)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build daily_features from bars_1m and bars_5m.

Staged build (each stage is gated):
  1. Trading day assignment (09:00 Brisbane boundary)
  2. ORB ranges (6 ORBs, configurable duration)
  3. Break detection (first 1m close outside ORB)
  4. Session stats (fixed-window high-low for session range features)
  5. RSI (Wilder's 14-period on 5m closes)
  6. Outcome at RR=1.0

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
from pipeline.dst import is_us_dst, is_uk_dst, DYNAMIC_ORB_RESOLVERS, get_break_group

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
ORB_TIMES_LOCAL = {
    "0900": (9, 0),
    "1000": (10, 0),
    "1100": (11, 0),
    "1130": (11, 30),   # HK/SG equity open 9:30 AM HKT
    "1800": (18, 0),
    "2300": (23, 0),
    "0030": (0, 30),
}

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

    For fixed sessions (0900, 1000, etc.), the Brisbane hour is constant.
    For dynamic sessions (US_EQUITY_OPEN, US_DATA_OPEN, LONDON_OPEN),
    the Brisbane hour is resolved per-day based on DST via pipeline/dst.py.

    Example: 0900 ORB with 5 min duration on trading_day 2024-01-05
      local start = 2024-01-05 09:00 Brisbane
      local end   = 2024-01-05 09:05 Brisbane
      UTC start   = 2024-01-04 23:00 UTC
      UTC end     = 2024-01-04 23:05 UTC

    Special case: 0030 ORB belongs to the SAME trading day but is
    at 00:30 the NEXT calendar day in Brisbane.
      trading_day 2024-01-05, 0030 ORB:
      local start = 2024-01-06 00:30 Brisbane (next calendar day)
      UTC start   = 2024-01-05 14:30 UTC
    """
    if orb_label in DYNAMIC_ORB_RESOLVERS:
        hour, minute = DYNAMIC_ORB_RESOLVERS[orb_label](trading_day)
    else:
        hour, minute = ORB_TIMES_LOCAL[orb_label]

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

    return utc_start, utc_end

def compute_orb_range(bars_df: pd.DataFrame, trading_day: date,
                       orb_label: str, orb_minutes: int) -> dict:
    """
    Compute ORB high/low/size for a single ORB on a single trading day.

    Returns dict with keys: high, low, size (or all None if no bars in window).
    """
    utc_start, utc_end = _orb_utc_window(trading_day, orb_label, orb_minutes)

    # Filter bars within [start, end)
    mask = (bars_df['ts_utc'] >= utc_start) & (bars_df['ts_utc'] < utc_end)
    orb_bars = bars_df[mask]

    if orb_bars.empty:
        return {"high": None, "low": None, "size": None}

    high = float(orb_bars['high'].max())
    low = float(orb_bars['low'].min())
    size = high - low

    return {"high": high, "low": low, "size": size}

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
    window. Sessions in the same group (e.g., 1000/1100/1130 in "asia")
    all extend their break windows to the same boundary (e.g., 1800 "london").

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

    Returns dict: break_dir ('long'/'short'/None), break_ts (datetime/None)
    """
    if orb_high is None or orb_low is None:
        return {"break_dir": None, "break_ts": None}

    window_start, window_end = _break_detection_window(
        trading_day, orb_label, orb_minutes
    )

    # Filter bars in break detection window
    mask = (bars_df['ts_utc'] >= window_start) & (bars_df['ts_utc'] < window_end)
    window_bars = bars_df[mask].sort_values('ts_utc')

    for _, bar in window_bars.iterrows():
        close = float(bar['close'])
        if close > orb_high:
            return {
                "break_dir": "long",
                "break_ts": bar['ts_utc'].to_pydatetime(),
            }
        elif close < orb_low:
            return {
                "break_dir": "short",
                "break_ts": bar['ts_utc'].to_pydatetime(),
            }

    return {"break_dir": None, "break_ts": None}

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

# =============================================================================
# MODULE 5: RSI (Wilder's 14-period on 5m closes)
# =============================================================================

def compute_rsi_at_0900(con: duckdb.DuckDBPyConnection, symbol: str,
                         trading_day: date) -> float | None:
    """
    Compute RSI-14 (Wilder's smoothing) on 5m closes, evaluated at 09:00 Brisbane.

    We need at least 14 prior 5m bars to compute RSI.
    We take the most recent 200 5m bars ending at or before 09:00 Brisbane (23:00 UTC)
    to ensure enough history for Wilder's smoothing to stabilize.

    Returns RSI value (0-100) or None if insufficient data.
    """
    # 09:00 Brisbane on trading_day = 23:00 UTC on (trading_day - 1)
    orb_0900_utc = datetime(
        trading_day.year, trading_day.month, trading_day.day,
        TRADING_DAY_START_HOUR_LOCAL, 0, 0,
        tzinfo=BRISBANE_TZ
    ).astimezone(UTC_TZ)

    # Fetch up to 200 bars ending at or before this timestamp
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

    for _, bar in post_break.iterrows():
        bar_high = float(bar['high'])
        bar_low = float(bar['low'])

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
                            cost_spec: CostSpec | None = None) -> dict:
    """
    Build all daily_features columns for a single trading day.

    Returns a dict matching daily_features column names.
    """
    # Fetch all 1m bars for this trading day
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

    # gap_open_points and atr_20 computed in orchestrator (needs previous day's data)
    row["gap_open_points"] = None
    row["atr_20"] = None

    # Module 4: Session stats
    session_stats = compute_session_stats(bars_df, trading_day)
    row.update(session_stats)

    # Module 5: RSI at 0900
    row["rsi_14_at_0900"] = compute_rsi_at_0900(con, symbol, trading_day)

    # Modules 2, 3, 6: ORBs, breaks, outcomes
    for label in ORB_LABELS:
        # Module 2: ORB range
        orb = compute_orb_range(bars_df, trading_day, label, orb_minutes)
        row[f"orb_{label}_high"] = orb["high"]
        row[f"orb_{label}_low"] = orb["low"]
        row[f"orb_{label}_size"] = orb["size"]

        # Module 3: Break detection
        brk = detect_break(
            bars_df, trading_day, label, orb_minutes,
            orb["high"], orb["low"]
        )
        row[f"orb_{label}_break_dir"] = brk["break_dir"]
        row[f"orb_{label}_break_ts"] = brk["break_ts"]

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

    # Build features for each trading day
    rows = []
    for i, td in enumerate(trading_days):
        row = build_features_for_day(con, symbol, td, orb_minutes, cost_spec)
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

        # ATR(20) = SMA of last 20 True Range values (skip Nones)
        lookback = [v for v in true_ranges[max(0, i - 20):i] if v is not None]
        if lookback:
            rows[i]["atr_20"] = round(sum(lookback) / len(lookback), 4)
        else:
            rows[i]["atr_20"] = None

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

    print("=" * 60)
    logger.info(f"BUILD DAILY_FEATURES ({symbol})")
    print("=" * 60)
    print()
    logger.info(f"Instrument: {symbol}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"ORB duration: {args.orb_minutes} minutes")
    logger.info(f"Database: {GOLD_DB_PATH}")
    logger.info(f"Dry run: {args.dry_run}")
    print()

    if not GOLD_DB_PATH.exists():
        logger.error(f"FATAL: Database not found: {GOLD_DB_PATH}")
        sys.exit(1)

    with duckdb.connect(str(GOLD_DB_PATH)) as con:
        # Build
        logger.info("Building daily features...")
        row_count = build_daily_features(
            con, symbol, start_date, end_date, args.orb_minutes, args.dry_run
        )
        print()

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
            print()
            logger.info("ALL INTEGRITY CHECKS PASSED [OK]")
        elif args.dry_run:
            logger.info("Integrity check skipped (dry run)")
        print()

        elapsed = datetime.now() - start_time

        print("=" * 60)
        logger.info(f"SUMMARY: {row_count:,} daily_features rows "
                    f"{'(would be) ' if args.dry_run else ''}built")
        logger.info(f"ORB duration: {args.orb_minutes}m")
        logger.info(f"Wall time: {elapsed}")
        print("=" * 60)

        sys.exit(0)

if __name__ == "__main__":
    main()

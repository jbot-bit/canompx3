#!/usr/bin/env python3
"""
Alternative Exit Methods for IB-Aligned Trades.

Tests 6 exit strategies on 0900/1000 IB-aligned trades:
  1. Fixed target (control -- stored pnl_r)
  2. 7h hold (existing exploit -- hold with stop only for 7 hours)
  3. VWAP trail (exit when price crosses running VWAP after being in profit)
  4. Session close (exit at session end / various cutoffs)
  5. ATR trail (trail stop at 1.5x ATR(20) from highest close since entry)
  6. Chandelier exit (trail stop at 1.5x ATR(20) from highest HIGH since entry)

For opposed trades, all strategies exit at IB break (same as exploit).
Trail exits only act on aligned trades.

Read-only. No DB writes.

Usage:
    python scripts/analyze_trend_exits.py --db-path C:/db/gold.db
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, to_r_multiple
from scripts._alt_strategy_utils import compute_strategy_metrics

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_ORB_SIZE = 4.0
RR_TARGET = 2.0
CONFIRM_BARS = 2
HOLD_HOURS = 7
ATR_MULT = 1.5
SESSIONS = ["0900", "1000"]

SESSION_UTC = {"0900": 23, "1000": 0}
MARKET_OPEN_UTC_HOUR = 23

SESSION_IB_CONFIG = {
    "0900": ("session", 120),
    "1000": ("mktopen", 120),
}

# Session close cutoffs to test (hours from entry)
SESSION_CLOSE_HOURS = [4, 8, 12, 24]

SPEC = get_cost_spec("MGC")


# ---------------------------------------------------------------------------
# IB computation (same as analyze_trend_holding.py)
# ---------------------------------------------------------------------------

def compute_ib(ts, highs, lows, anchor_utc_hour, duration_minutes):
    """Compute IB from numpy arrays."""
    hours = np.array([t.hour for t in ts])
    minutes = np.array([t.minute for t in ts])
    anchor_idx = np.flatnonzero((hours == anchor_utc_hour) & (minutes == 0))
    if len(anchor_idx) == 0:
        return None
    ib_start = ts[anchor_idx[0]]
    ib_end = ib_start + timedelta(minutes=duration_minutes)
    ib_mask = (ts >= ib_start) & (ts < ib_end)
    if ib_mask.sum() < max(10, duration_minutes // 12):
        return None
    return {
        "ib_high": float(highs[ib_mask].max()),
        "ib_low": float(lows[ib_mask].min()),
        "ib_end": ib_end,
    }


def find_ib_break(ts, highs, lows, ib):
    """Find first IB break after ib_end. Returns (direction, break_ts, break_idx)."""
    post_idx = np.flatnonzero(ts >= ib["ib_end"])
    for i in post_idx:
        bh = highs[i] > ib["ib_high"]
        bl = lows[i] < ib["ib_low"]
        if bh and bl:
            return None, ts[i], i
        if bh:
            return "long", ts[i], i
        if bl:
            return "short", ts[i], i
    return None, None, None


# ---------------------------------------------------------------------------
# VWAP computation
# ---------------------------------------------------------------------------

def compute_running_vwap(prices, volumes):
    """Compute running VWAP from arrays of typical price and volume.

    Returns array of VWAP values (same length as input).
    """
    cum_pv = np.cumsum(prices * volumes)
    cum_v = np.cumsum(volumes)
    # Avoid division by zero
    mask = cum_v > 0
    vwap = np.full_like(prices, np.nan)
    vwap[mask] = cum_pv[mask] / cum_v[mask]
    return vwap


# ---------------------------------------------------------------------------
# Exit simulations
# ---------------------------------------------------------------------------

def _exit_pnl_r(entry_price, stop_price, exit_price, is_long):
    """Compute pnl_r from exit price."""
    pnl_pts = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    return to_r_multiple(SPEC, entry_price, stop_price, pnl_pts)


def sim_fixed_target(ts, highs, lows, closes, entry_idx, entry_price,
                     stop_price, target_price, is_long, cutoff_ts):
    """Plain fixed-target strategy (control). Returns (pnl_r, exit_reason)."""
    for i in range(entry_idx, len(ts)):
        h, lo = highs[i], lows[i]
        # Stop (priority)
        if is_long and lo <= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "stop"
        if not is_long and h >= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "stop"
        # Target
        if is_long and h >= target_price:
            return _exit_pnl_r(entry_price, stop_price, target_price, is_long), "target"
        if not is_long and lo <= target_price:
            return _exit_pnl_r(entry_price, stop_price, target_price, is_long), "target"
        # Time
        if ts[i] >= cutoff_ts:
            return _exit_pnl_r(entry_price, stop_price, closes[i], is_long), "time"
    return _exit_pnl_r(entry_price, stop_price, closes[-1], is_long), "eod"


def sim_hold_7h(ts, highs, lows, closes, entry_idx, entry_price,
                stop_price, is_long, cutoff_ts):
    """Hold with stop only for 7 hours. No target. Returns (pnl_r, exit_reason)."""
    for i in range(entry_idx, len(ts)):
        h, lo = highs[i], lows[i]
        # Stop always active
        if is_long and lo <= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "stop"
        if not is_long and h >= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "stop"
        # Time exit
        if ts[i] >= cutoff_ts:
            return _exit_pnl_r(entry_price, stop_price, closes[i], is_long), "time_7h"
    return _exit_pnl_r(entry_price, stop_price, closes[-1], is_long), "eod"


def sim_vwap_trail(ts, highs, lows, closes, volumes, vwap, entry_idx,
                   entry_price, stop_price, is_long, session_end_ts):
    """VWAP trail exit.

    After being in profit, exit when price crosses VWAP.
    For longs: exit when low crosses below VWAP after high was above entry.
    For shorts: exit when high crosses above VWAP after low was below entry.
    Stop is always active. Session end is the hard cutoff.

    Returns (pnl_r, exit_reason).
    """
    was_in_profit = False

    for i in range(entry_idx, len(ts)):
        h, lo, c = highs[i], lows[i], closes[i]
        v = vwap[i]

        # Stop always active
        if is_long and lo <= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "stop"
        if not is_long and h >= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "stop"

        # Check if we've been in profit
        if not was_in_profit:
            if is_long and h > entry_price:
                was_in_profit = True
            elif not is_long and lo < entry_price:
                was_in_profit = True

        # VWAP trail: only after being in profit and VWAP is valid
        if was_in_profit and not np.isnan(v):
            if is_long and lo <= v:
                # Price crossed below VWAP
                exit_price = min(c, v)  # Conservative: exit at VWAP or close
                return _exit_pnl_r(entry_price, stop_price, exit_price, is_long), "vwap_trail"
            if not is_long and h >= v:
                # Price crossed above VWAP
                exit_price = max(c, v)
                return _exit_pnl_r(entry_price, stop_price, exit_price, is_long), "vwap_trail"

        # Session end
        if ts[i] >= session_end_ts:
            return _exit_pnl_r(entry_price, stop_price, c, is_long), "session_close"

    return _exit_pnl_r(entry_price, stop_price, closes[-1], is_long), "eod"


def sim_session_close(ts, highs, lows, closes, entry_idx, entry_price,
                      stop_price, is_long, cutoff_ts):
    """Hold until session close with stop. Returns (pnl_r, exit_reason)."""
    for i in range(entry_idx, len(ts)):
        h, lo = highs[i], lows[i]
        # Stop always active
        if is_long and lo <= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "stop"
        if not is_long and h >= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "stop"
        # Session close
        if ts[i] >= cutoff_ts:
            return _exit_pnl_r(entry_price, stop_price, closes[i], is_long), "session_close"
    return _exit_pnl_r(entry_price, stop_price, closes[-1], is_long), "eod"


def sim_atr_trail(ts, highs, lows, closes, entry_idx, entry_price,
                  stop_price, is_long, atr_value, session_end_ts):
    """ATR trailing stop anchored to highest CLOSE (longs) or lowest CLOSE (shorts).

    Trail distance = ATR_MULT * atr_value.
    Stop ratchets but never loosens.
    Original stop is active until trail stop tightens past it.

    Returns (pnl_r, exit_reason).
    """
    trail_dist = ATR_MULT * atr_value
    if is_long:
        best_close = closes[entry_idx]
        trail_stop = best_close - trail_dist
        effective_stop = max(stop_price, trail_stop)
    else:
        best_close = closes[entry_idx]
        trail_stop = best_close + trail_dist
        effective_stop = min(stop_price, trail_stop)

    for i in range(entry_idx, len(ts)):
        h, lo, c = highs[i], lows[i], closes[i]

        # Update trailing stop BEFORE checking (uses prior bar's close for trail)
        if i > entry_idx:
            if is_long:
                if c > best_close:
                    best_close = c
                new_trail = best_close - trail_dist
                effective_stop = max(effective_stop, new_trail)
            else:
                if c < best_close:
                    best_close = c
                new_trail = best_close + trail_dist
                effective_stop = min(effective_stop, new_trail)

        # Check stop
        if is_long and lo <= effective_stop:
            return _exit_pnl_r(entry_price, stop_price, effective_stop, is_long), "atr_trail"
        if not is_long and h >= effective_stop:
            return _exit_pnl_r(entry_price, stop_price, effective_stop, is_long), "atr_trail"

        # Session end
        if ts[i] >= session_end_ts:
            return _exit_pnl_r(entry_price, stop_price, c, is_long), "session_close"

    return _exit_pnl_r(entry_price, stop_price, closes[-1], is_long), "eod"


def sim_chandelier(ts, highs, lows, closes, entry_idx, entry_price,
                   stop_price, is_long, atr_value, session_end_ts):
    """Chandelier exit: ATR trail anchored to highest HIGH (longs) or lowest LOW (shorts).

    Trail distance = ATR_MULT * atr_value.
    Stop ratchets but never loosens.
    Original stop is active until trail stop tightens past it.

    Returns (pnl_r, exit_reason).
    """
    trail_dist = ATR_MULT * atr_value
    if is_long:
        best_high = highs[entry_idx]
        trail_stop = best_high - trail_dist
        effective_stop = max(stop_price, trail_stop)
    else:
        best_low = lows[entry_idx]
        trail_stop = best_low + trail_dist
        effective_stop = min(stop_price, trail_stop)

    for i in range(entry_idx, len(ts)):
        h, lo, c = highs[i], lows[i], closes[i]

        # Update trailing stop BEFORE checking (uses current bar's extreme)
        if is_long:
            if h > best_high:
                best_high = h
            new_trail = best_high - trail_dist
            effective_stop = max(effective_stop, new_trail)
        else:
            if lo < best_low:
                best_low = lo
            new_trail = best_low + trail_dist
            effective_stop = min(effective_stop, new_trail)

        # Check stop
        if is_long and lo <= effective_stop:
            return _exit_pnl_r(entry_price, stop_price, effective_stop, is_long), "chandelier"
        if not is_long and h >= effective_stop:
            return _exit_pnl_r(entry_price, stop_price, effective_stop, is_long), "chandelier"

        # Session end
        if ts[i] >= session_end_ts:
            return _exit_pnl_r(entry_price, stop_price, c, is_long), "session_close"

    return _exit_pnl_r(entry_price, stop_price, closes[-1], is_long), "eod"


# ---------------------------------------------------------------------------
# Wrapper: run all exits with IB gating
# ---------------------------------------------------------------------------

def simulate_all_exits(ts, highs, lows, closes, volumes, vwap,
                       entry_idx, entry_price, stop_price, target_price,
                       is_long, orb_dir, ib_dir, ib_break_idx,
                       cutoff_7h, session_end_ts, atr_value):
    """Run all 6 exit strategies for one trade.

    For limbo phase (before IB break): fixed target active.
    On opposed IB break: all strategies exit at market.
    On aligned IB break: each strategy takes over.
    On no IB break: keep fixed target.

    Returns dict of {method_name: (pnl_r, exit_reason)}.
    """
    results = {}

    # 1. Fixed target (control) -- always uses stored logic
    results["fixed"] = sim_fixed_target(
        ts, highs, lows, closes, entry_idx, entry_price,
        stop_price, target_price, is_long, session_end_ts)

    # Determine alignment
    if ib_dir is None:
        alignment = "no_break"
    elif ib_dir == orb_dir:
        alignment = "aligned"
    else:
        alignment = "opposed"

    # For opposed trades: all trail/hold methods exit at IB break
    if alignment == "opposed":
        # Find the close price at IB break bar
        if ib_break_idx is not None and ib_break_idx < len(closes):
            kill_price = closes[ib_break_idx]
        else:
            kill_price = closes[-1]
        kill_r = _exit_pnl_r(entry_price, stop_price, kill_price, is_long)

        # But check if stopped out BEFORE IB break
        stopped_before = _check_stop_before(
            ts, highs, lows, entry_idx, entry_price, stop_price,
            is_long, ib_break_idx)
        if stopped_before is not None:
            kill_r = _exit_pnl_r(entry_price, stop_price, stop_price, is_long)
            reason = "stop"
        else:
            reason = "opposed_kill"

        results["hold_7h"] = (kill_r, reason)
        results["vwap_trail"] = (kill_r, reason)
        for h in SESSION_CLOSE_HOURS:
            results[f"close_{h}h"] = (kill_r, reason)
        results["atr_trail"] = (kill_r, reason)
        results["chandelier"] = (kill_r, reason)
        return results, alignment

    # For no_break: keep fixed target for all methods
    if alignment == "no_break":
        fixed_r, fixed_reason = results["fixed"]
        results["hold_7h"] = (fixed_r, fixed_reason)
        results["vwap_trail"] = (fixed_r, fixed_reason)
        for h in SESSION_CLOSE_HOURS:
            results[f"close_{h}h"] = (fixed_r, fixed_reason)
        results["atr_trail"] = (fixed_r, fixed_reason)
        results["chandelier"] = (fixed_r, fixed_reason)
        return results, alignment

    # --- Aligned: run each exit method from IB break point ---
    # First check if trade resolved BEFORE IB break (limbo phase)
    limbo_resolved = _check_limbo_resolution(
        ts, highs, lows, closes, entry_idx, entry_price,
        stop_price, target_price, is_long, ib_break_idx)

    if limbo_resolved is not None:
        # Trade already exited before IB break -- all methods get same result
        for method in ["hold_7h", "vwap_trail", "atr_trail", "chandelier"]:
            results[method] = limbo_resolved
        for h in SESSION_CLOSE_HOURS:
            results[f"close_{h}h"] = limbo_resolved
        return results, alignment

    # 2. Hold 7h
    results["hold_7h"] = sim_hold_7h(
        ts, highs, lows, closes, entry_idx, entry_price,
        stop_price, is_long, cutoff_7h)

    # 3. VWAP trail
    results["vwap_trail"] = sim_vwap_trail(
        ts, highs, lows, closes, volumes, vwap, entry_idx,
        entry_price, stop_price, is_long, session_end_ts)

    # 4. Session close at various cutoffs
    for h in SESSION_CLOSE_HOURS:
        cutoff = ts[entry_idx] + timedelta(hours=h)
        results[f"close_{h}h"] = sim_session_close(
            ts, highs, lows, closes, entry_idx, entry_price,
            stop_price, is_long, cutoff)

    # 5. ATR trail
    if atr_value is not None and atr_value > 0:
        results["atr_trail"] = sim_atr_trail(
            ts, highs, lows, closes, entry_idx, entry_price,
            stop_price, is_long, atr_value, session_end_ts)
    else:
        results["atr_trail"] = results["fixed"]  # Fallback

    # 6. Chandelier
    if atr_value is not None and atr_value > 0:
        results["chandelier"] = sim_chandelier(
            ts, highs, lows, closes, entry_idx, entry_price,
            stop_price, is_long, atr_value, session_end_ts)
    else:
        results["chandelier"] = results["fixed"]  # Fallback

    return results, alignment


def _check_stop_before(ts, highs, lows, entry_idx, entry_price,
                       stop_price, is_long, ib_break_idx):
    """Check if original stop was hit before IB break. Returns bar index or None."""
    if ib_break_idx is None:
        return None
    for i in range(entry_idx, min(ib_break_idx, len(ts))):
        if is_long and lows[i] <= stop_price:
            return i
        if not is_long and highs[i] >= stop_price:
            return i
    return None


def _check_limbo_resolution(ts, highs, lows, closes, entry_idx, entry_price,
                            stop_price, target_price, is_long, ib_break_idx):
    """Check if trade resolved (stop or target) before IB break.

    Returns (pnl_r, exit_reason) or None.
    """
    if ib_break_idx is None:
        return None
    for i in range(entry_idx, min(ib_break_idx, len(ts))):
        h, lo = highs[i], lows[i]
        # Stop
        if is_long and lo <= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "limbo_stop"
        if not is_long and h >= stop_price:
            return _exit_pnl_r(entry_price, stop_price, stop_price, is_long), "limbo_stop"
        # Target
        if is_long and h >= target_price:
            return _exit_pnl_r(entry_price, stop_price, target_price, is_long), "limbo_target"
        if not is_long and lo <= target_price:
            return _exit_pnl_r(entry_price, stop_price, target_price, is_long), "limbo_target"
    return None


# ---------------------------------------------------------------------------
# Process session
# ---------------------------------------------------------------------------

def process_session(db_path, session_label, start, end):
    """Load trades + bars, run all exit simulations, return DataFrame."""
    session_utc_hour = SESSION_UTC[session_label]
    anchor, duration = SESSION_IB_CONFIG[session_label]
    anchor_hour = session_utc_hour if anchor == "session" else MARKET_OPEN_UTC_HOUR

    con = duckdb.connect(str(db_path), read_only=True)

    # Load outcomes
    df = con.execute(f"""
        SELECT o.trading_day, o.entry_ts, o.entry_price, o.stop_price,
               o.target_price, o.pnl_r,
               d.orb_{session_label}_break_dir,
               d.atr_20
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC' AND o.orb_minutes = 5
          AND o.orb_label = ? AND o.entry_model = 'E1'
          AND o.rr_target = ? AND o.confirm_bars = ?
          AND o.entry_ts IS NOT NULL AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL AND d.orb_{session_label}_size >= ?
          AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day
    """, [session_label, RR_TARGET, CONFIRM_BARS, MIN_ORB_SIZE, start, end]).fetchdf()

    if df.empty:
        con.close()
        return pd.DataFrame()

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    # Bulk-load bars (including volume for VWAP)
    unique_days = sorted(df["trading_day"].unique())
    bars_cache = {}
    for td in unique_days:
        s, e = compute_trading_day_utc_range(td)
        b = con.execute(
            "SELECT ts_utc, open, high, low, close, volume FROM bars_1m "
            "WHERE symbol='MGC' AND ts_utc>=? AND ts_utc<? ORDER BY ts_utc",
            [s, e],
        ).fetchdf()
        if not b.empty:
            b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True)
            ts_raw = b["ts_utc"].values.astype("datetime64[ms]")
            ts_py = np.array([pd.Timestamp(t).to_pydatetime().replace(tzinfo=None)
                              for t in ts_raw])

            opens = b["open"].values.astype(np.float64)
            h_arr = b["high"].values.astype(np.float64)
            l_arr = b["low"].values.astype(np.float64)
            c_arr = b["close"].values.astype(np.float64)
            v_arr = b["volume"].values.astype(np.float64)

            # Compute typical price for VWAP: (H+L+C)/3
            typical = (h_arr + l_arr + c_arr) / 3.0

            # Find session start (23:00 UTC) for VWAP anchor
            # VWAP resets at market open (23:00 UTC)
            session_start_mask = np.array([t.hour == MARKET_OPEN_UTC_HOUR and t.minute == 0
                                           for t in ts_py])
            session_start_indices = np.flatnonzero(session_start_mask)

            # Compute VWAP from session start
            vwap_arr = np.full_like(c_arr, np.nan)
            if len(session_start_indices) > 0:
                ss_idx = session_start_indices[0]
                session_typical = typical[ss_idx:]
                session_vol = v_arr[ss_idx:]
                cum_pv = np.cumsum(session_typical * session_vol)
                cum_v = np.cumsum(session_vol)
                mask_v = cum_v > 0
                vwap_session = np.full_like(session_typical, np.nan)
                vwap_session[mask_v] = cum_pv[mask_v] / cum_v[mask_v]
                vwap_arr[ss_idx:] = vwap_session

            bars_cache[td] = (ts_py, ts_raw, h_arr, l_arr, c_arr, v_arr, vwap_arr)

    con.close()

    # Process each trade
    results = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        if td not in bars_cache:
            continue
        ts_py, ts_raw, h_arr, l_arr, c_arr, v_arr, vwap_arr = bars_cache[td]

        entry_ts_aware = row["entry_ts"].to_pydatetime()
        entry_ts = entry_ts_aware.replace(tzinfo=None)
        entry_p = float(row["entry_price"])
        stop_p = float(row["stop_price"])
        target_p = float(row["target_price"])
        orb_dir = row[f"orb_{session_label}_break_dir"]
        is_long = orb_dir == "long"
        stored_pnl = float(row["pnl_r"])
        atr_20 = float(row["atr_20"]) if pd.notna(row["atr_20"]) else None

        entry_idx = int(np.searchsorted(ts_raw, np.datetime64(entry_ts_aware, "ms")))
        if entry_idx >= len(ts_py):
            continue

        cutoff_7h = entry_ts + timedelta(hours=HOLD_HOURS)
        # Session end: 23:00 UTC next day for 0900, 24h from session start for others
        # 0900 session starts at 23:00 UTC, so session end = 23:00 UTC (next day)
        # 1000 session starts at 00:00 UTC, so session end = 23:00 UTC (same day)
        if session_label == "0900":
            # Session runs 23:00 UTC -> 23:00 UTC next day
            session_end = entry_ts.replace(hour=23, minute=0, second=0) + timedelta(hours=24)
        else:
            # 1000 session: session end at 23:00 UTC same calendar day
            session_end = entry_ts.replace(hour=23, minute=0, second=0)
            if session_end <= entry_ts:
                session_end += timedelta(hours=24)

        # Compute IB
        ib = compute_ib(ts_py, h_arr, l_arr, anchor_hour, duration)
        if ib is None:
            continue
        ib_dir, ib_break_ts, ib_break_idx = find_ib_break(ts_py, h_arr, l_arr, ib)

        # Run all exits
        exit_results, alignment = simulate_all_exits(
            ts_py, h_arr, l_arr, c_arr, v_arr, vwap_arr,
            entry_idx, entry_p, stop_p, target_p,
            is_long, orb_dir, ib_dir, ib_break_idx,
            cutoff_7h, session_end, atr_20)

        year = str(td.year) if hasattr(td, "year") else str(td)[:4]

        record = {
            "td": td,
            "year": year,
            "alignment": alignment,
            "stored_pnl": stored_pnl,
        }
        for method, (pnl_r, reason) in exit_results.items():
            record[f"{method}_pnl"] = pnl_r
            record[f"{method}_reason"] = reason

        results.append(record)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

EXIT_METHODS = [
    ("Fixed RR (control)", "fixed_pnl"),
    ("Hold 7h", "hold_7h_pnl"),
    ("VWAP Trail", "vwap_trail_pnl"),
    ("Close 4h", "close_4h_pnl"),
    ("Close 8h", "close_8h_pnl"),
    ("Close 12h", "close_12h_pnl"),
    ("Close 24h", "close_24h_pnl"),
    ("ATR Trail 1.5x", "atr_trail_pnl"),
    ("Chandelier 1.5x", "chandelier_pnl"),
]


def fmt_row(label, m):
    """Format one metrics row."""
    return (f"  {label:22s} {m['n']:>5d} {m['wr']:>7.1%} {m['expr']:>+8.4f} "
            f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>+8.1f}")


def print_comparison_table(pdf, title, methods=None):
    """Print a comparison table of all exit methods."""
    if methods is None:
        methods = EXIT_METHODS
    if len(pdf) == 0:
        print(f"  No trades for {title}")
        return

    print(f"\n  {title} (N={len(pdf)})")
    print(f"  {'Strategy':22s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} "
          f"{'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
    print(f"  {'-'*22} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, col in methods:
        if col not in pdf.columns:
            continue
        m = compute_strategy_metrics(pdf[col].values)
        if m:
            print(fmt_row(label, m))


def print_report(pdf, session_label):
    """Print full report for one session."""
    if len(pdf) == 0:
        print(f"  No trades for {session_label}")
        return

    n = len(pdf)
    anchor, dur = SESSION_IB_CONFIG[session_label]
    ib_name = f"{anchor}_{dur}m"

    print(f"\n{'=' * 95}")
    print(f"{session_label} SESSION  |  IB: {ib_name}  |  N={n}")
    print(f"E1 CB{CONFIRM_BARS} RR{RR_TARGET} G{int(MIN_ORB_SIZE)}+  |  "
          f"ATR mult={ATR_MULT}  |  Hold={HOLD_HOURS}h")
    print(f"{'=' * 95}")

    # Alignment distribution
    for a in ["aligned", "opposed", "no_break"]:
        na = len(pdf[pdf["alignment"] == a])
        print(f"  {a:10s}: {na:>4d} ({na/n*100:.0f}%)", end="  ")
    print()

    # --- ALL TRADES ---
    print_comparison_table(pdf, "ALL TRADES")

    # --- ALIGNED ONLY ---
    aligned = pdf[pdf["alignment"] == "aligned"]
    if len(aligned) >= 5:
        print_comparison_table(aligned, "ALIGNED ONLY")

    # --- OPPOSED ONLY ---
    opposed = pdf[pdf["alignment"] == "opposed"]
    if len(opposed) >= 5:
        # Opposed only has fixed and the kill -- show subset
        opposed_methods = [
            ("Fixed RR (control)", "fixed_pnl"),
            ("Hold 7h (kill)", "hold_7h_pnl"),
        ]
        print_comparison_table(opposed, "OPPOSED ONLY", methods=opposed_methods)

    # --- Exit reason distribution for trail methods ---
    print(f"\n  Exit reason distribution (aligned trades only, N={len(aligned)}):")
    for label, col in EXIT_METHODS:
        reason_col = col.replace("_pnl", "_reason")
        if reason_col in aligned.columns and len(aligned) > 0:
            reasons = aligned[reason_col].value_counts()
            parts = [f"{r}={c}" for r, c in reasons.items()]
            print(f"    {label:22s}: {', '.join(parts)}")

    # --- Yearly Sharpe comparison ---
    print(f"\n  Yearly Sharpe (per-trade, all trades):")
    # Select subset of methods to show yearly
    yearly_methods = [
        ("Fixed", "fixed_pnl"),
        ("7h Hold", "hold_7h_pnl"),
        ("VWAP", "vwap_trail_pnl"),
        ("Close8h", "close_8h_pnl"),
        ("ATR Trl", "atr_trail_pnl"),
        ("Chand", "chandelier_pnl"),
    ]
    header_labels = [l for l, _ in yearly_methods]
    print(f"  {'Year':5s} {'N':>4s} " + " ".join(f"{l:>8s}" for l in header_labels))
    print(f"  {'-'*5} {'-'*4} " + " ".join(f"{'-'*8}" for _ in header_labels))

    for year in sorted(pdf["year"].unique()):
        ydf = pdf[pdf["year"] == year]
        if len(ydf) < 3:
            continue
        row_parts = [f"  {year:5s} {len(ydf):>4d}"]
        for _, col in yearly_methods:
            if col in ydf.columns:
                m = compute_strategy_metrics(ydf[col].values)
                if m:
                    row_parts.append(f"{m['sharpe']:>8.3f}")
                else:
                    row_parts.append(f"{'N/A':>8s}")
            else:
                row_parts.append(f"{'N/A':>8s}")
        print(" ".join(row_parts))

    # Total row
    row_parts = [f"  {'TOTAL':5s} {len(pdf):>4d}"]
    for _, col in yearly_methods:
        if col in pdf.columns:
            m = compute_strategy_metrics(pdf[col].values)
            if m:
                row_parts.append(f"{m['sharpe']:>8.3f}")
            else:
                row_parts.append(f"{'N/A':>8s}")
        else:
            row_parts.append(f"{'N/A':>8s}")
    print(" ".join(row_parts))

    # --- Best method per alignment ---
    print(f"\n  Best exit method (by ExpR):")
    for subset_name, subset in [("All", pdf), ("Aligned", aligned)]:
        if len(subset) < 5:
            continue
        best_label = None
        best_expr = -999
        for label, col in EXIT_METHODS:
            if col not in subset.columns:
                continue
            m = compute_strategy_metrics(subset[col].values)
            if m and m["expr"] > best_expr:
                best_expr = m["expr"]
                best_label = label
        if best_label:
            print(f"    {subset_name:10s}: {best_label} (ExpR={best_expr:+.4f})")


def run_integrity_checks(pdf, session_label):
    """Run basic integrity checks."""
    print(f"\n  Integrity checks ({session_label}):")
    ok = total = 0

    # Check 1: all trades have results for all methods
    total += 1
    required_cols = [col for _, col in EXIT_METHODS]
    missing = [c for c in required_cols if c not in pdf.columns]
    if not missing:
        null_counts = pdf[required_cols].isnull().sum()
        if null_counts.sum() == 0:
            print(f"    [PASS] all methods have results for all trades")
            ok += 1
        else:
            bad = null_counts[null_counts > 0]
            print(f"    [WARN] null results: {dict(bad)}")
    else:
        print(f"    [FAIL] missing columns: {missing}")

    # Check 2: opposed trades -- all non-fixed methods should have same pnl
    total += 1
    opposed = pdf[pdf["alignment"] == "opposed"]
    if len(opposed) > 0:
        # hold_7h and atr_trail should be identical for opposed
        ref = opposed["hold_7h_pnl"].values
        atr = opposed["atr_trail_pnl"].values
        if np.allclose(ref, atr, equal_nan=True):
            print(f"    [PASS] opposed trades: all trail methods identical ({len(opposed)} trades)")
            ok += 1
        else:
            diff_count = (~np.isclose(ref, atr, equal_nan=True)).sum()
            print(f"    [WARN] opposed trades: {diff_count} differ between hold_7h and atr_trail")
    else:
        print(f"    [SKIP] no opposed trades")
        ok += 1

    # Check 3: stored_pnl should be close to fixed_pnl for most trades
    total += 1
    if "stored_pnl" in pdf.columns and "fixed_pnl" in pdf.columns:
        stored = pdf["stored_pnl"].values
        fixed = pdf["fixed_pnl"].values
        # They may differ because session_end cutoff differs from stored outcome
        # but correlation should be very high
        if len(stored) > 10:
            corr = np.corrcoef(stored, fixed)[0, 1]
            if corr > 0.85:
                print(f"    [PASS] stored vs fixed pnl correlation: {corr:.4f}")
                ok += 1
            else:
                print(f"    [WARN] stored vs fixed pnl correlation: {corr:.4f} (expected >0.85)")
        else:
            print(f"    [SKIP] too few trades for correlation check")
            ok += 1

    print(f"    {ok}/{total} passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path, start, end):
    print("Alternative Exit Methods for IB-Aligned Trades")
    print(f"Date range: {start} to {end}")
    print(f"Cost model: MGC ($10/pt, ${SPEC.total_friction:.2f} RT friction)")
    print(f"Params: E1 CB{CONFIRM_BARS} RR{RR_TARGET} G{int(MIN_ORB_SIZE)}+")
    print(f"Trail params: ATR mult={ATR_MULT}, Hold={HOLD_HOURS}h")
    print(f"Session close cutoffs: {SESSION_CLOSE_HOURS}h")
    print(f"IB configs: {dict((s, f'{a}_{d}m') for s,(a,d) in SESSION_IB_CONFIG.items())}")

    for session in SESSIONS:
        print(f"\nProcessing {session} session...")
        t0 = time.time()
        pdf = process_session(db_path, session, start, end)
        elapsed = time.time() - t0
        print(f"  {len(pdf)} trades in {elapsed:.1f}s")
        print_report(pdf, session)
        run_integrity_checks(pdf, session)

    print(f"\n{'#' * 95}")
    print("VERDICT GUIDE")
    print(f"{'#' * 95}")
    print("Key questions:")
    print("  1. Does VWAP trail beat fixed target on aligned trades?")
    print("     -> If yes, trend continuation is real and VWAP captures it")
    print("  2. Does ATR trail beat 7h hold?")
    print("     -> If yes, adaptive trailing is better than time-based exit")
    print("  3. Which session close cutoff is optimal?")
    print("     -> Longer hold = more trend capture but more reversion risk")
    print("  4. Does Chandelier (high-anchored) beat ATR trail (close-anchored)?")
    print("     -> If yes, swing highs are better anchors than closing prices")
    print("  5. Are results consistent across years?")
    print("     -> Check yearly Sharpe -- must work across regimes")
    print("")
    print("Decision rules:")
    print("  - Best method must beat Fixed RR on ExpR AND Sharpe for aligned trades")
    print("  - Must be positive in 80%+ of years")
    print("  - MaxDD must not be materially worse than Fixed RR")
    print("  - If no trail method wins: fixed target is the correct exit")


def main():
    parser = argparse.ArgumentParser(
        description="Alternative Exit Methods for IB-Aligned Trades")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"ERROR: database not found at {args.db_path}")
        print(f"Copy master to working location first:")
        print(f'  cp "C:\\canodrive\\canompx3\\gold.db" "C:\\db\\gold.db"')
        sys.exit(1)

    run(args.db_path, args.start, args.end)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive MCL (Micro Crude Oil) strategy scan.

Tests EVERY major strategy type to determine if ANY approach works on MCL,
not just ORB breakout. MCL has 47-80% double break rates = mean-reverting
instrument, so strategies that EXPLOIT mean-reversion are prioritized.

Strategy types tested:
  1. ORB Fade -- enter OPPOSITE direction of ORB break
  2. Double-Break Fade -- enter direction of SECOND break (exploits high DB rate)
  3. Range Compression -- small ORB days, wait for delayed breakout
  4. Session Fade -- fade Asian session direction at 1800 London open
  5. VWAP Reversion -- fade deviations > 1 ATR from VWAP
  6. Time-of-Day -- scan hourly directional bias, test mechanical entry

Usage:
    python scripts/analyze_mcl_comprehensive.py
    python scripts/analyze_mcl_comprehensive.py --db-path C:/db/gold.db
"""

import argparse
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(line_buffering=True)

from pipeline.cost_model import get_cost_spec, CostSpec
from pipeline.build_daily_features import compute_trading_day_utc_range
from research._alt_strategy_utils import compute_strategy_metrics

BRISBANE_TZ = ZoneInfo("Australia/Brisbane")
UTC_TZ = ZoneInfo("UTC")

INSTRUMENT = "MCL"
SPEC = get_cost_spec(INSTRUMENT)  # $100/point, $5.24 friction, tick=0.01

ORB_LABELS = ["0900", "1000", "1100", "1800", "2300", "0030"]
MCL_START = date(2021, 7, 11)
MCL_END = date(2026, 2, 10)

# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def load_mcl_features(db_path: Path, orb_minutes: int = 5) -> pd.DataFrame:
    """Load daily_features for MCL."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT *
            FROM daily_features
            WHERE symbol = ?
              AND orb_minutes = ?
              AND trading_day BETWEEN ? AND ?
            ORDER BY trading_day
        """, [INSTRUMENT, orb_minutes, MCL_START, MCL_END]).fetchdf()
    finally:
        con.close()
    return df

def load_mcl_bars_for_day(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 1m bars for one MCL trading day."""
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = ?
              AND ts_utc >= ?::TIMESTAMPTZ AND ts_utc < ?::TIMESTAMPTZ
            ORDER BY ts_utc
        """, [INSTRUMENT, start_utc.isoformat(), end_utc.isoformat()]).fetchdf()
    finally:
        con.close()
    if not df.empty:
        df['ts_utc'] = pd.to_datetime(df['ts_utc'], utc=True)
    return df

def load_mcl_outcomes(db_path: Path) -> pd.DataFrame:
    """Load orb_outcomes for MCL."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT *
            FROM orb_outcomes
            WHERE symbol = ?
            ORDER BY trading_day, orb_label
        """, [INSTRUMENT]).fetchdf()
    finally:
        con.close()
    return df

def load_all_mcl_bars(db_path: Path) -> pd.DataFrame:
    """Load ALL MCL 1m bars (used for VWAP and time-of-day strategies)."""
    start_utc, _ = compute_trading_day_utc_range(MCL_START)
    _, end_utc = compute_trading_day_utc_range(MCL_END)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = ?
              AND ts_utc >= ?::TIMESTAMPTZ AND ts_utc < ?::TIMESTAMPTZ
            ORDER BY ts_utc
        """, [INSTRUMENT, start_utc.isoformat(), end_utc.isoformat()]).fetchdf()
    finally:
        con.close()
    if not df.empty:
        df['ts_utc'] = pd.to_datetime(df['ts_utc'], utc=True)
    return df

# ============================================================================
# SHARED HELPERS
# ============================================================================

def resolve_bar_outcome(bars: pd.DataFrame, entry_price: float,
                        stop_price: float, target_price: float,
                        direction: str, start_idx: int) -> dict | None:
    """Scan bars for stop/target hit. Ambiguous bar = LOSS (Gate C)."""
    is_long = direction == "long"
    for i in range(start_idx, len(bars)):
        bar = bars.iloc[i]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if is_long:
            stop_hit = bar_low <= stop_price
            target_hit = bar_high >= target_price
        else:
            stop_hit = bar_high >= stop_price
            target_hit = bar_low <= target_price

        if stop_hit and target_hit:
            pnl = stop_price - entry_price if is_long else entry_price - stop_price
            return {"outcome": "loss", "pnl_points": pnl, "exit_idx": i}
        if stop_hit:
            pnl = stop_price - entry_price if is_long else entry_price - stop_price
            return {"outcome": "loss", "pnl_points": pnl, "exit_idx": i}
        if target_hit:
            pnl = target_price - entry_price if is_long else entry_price - target_price
            return {"outcome": "win", "pnl_points": pnl, "exit_idx": i}
    return None

def pnl_to_r_gross(entry: float, stop: float, pnl_points: float) -> float:
    """Convert pnl_points to R-multiple WITHOUT friction (gross)."""
    risk_pts = abs(entry - stop)
    if risk_pts <= 0:
        return 0.0
    return pnl_points / risk_pts

def pnl_to_r_net(entry: float, stop: float, pnl_points: float) -> float:
    """Convert pnl_points to R-multiple WITH friction (net)."""
    risk_dollars = abs(entry - stop) * SPEC.point_value + SPEC.total_friction
    if risk_dollars <= 0:
        return 0.0
    pnl_dollars = pnl_points * SPEC.point_value - SPEC.total_friction
    return pnl_dollars / risk_dollars

def print_metrics(label: str, gross_pnls: np.ndarray, net_pnls: np.ndarray):
    """Print a formatted metrics line for one strategy variant."""
    g = compute_strategy_metrics(gross_pnls)
    n = compute_strategy_metrics(net_pnls)
    if g is None or n is None:
        print(f"  {label:40s}  N=0  (no trades)")
        return
    years = (MCL_END - MCL_START).days / 365.25
    tpy = g["n"] / years if years > 0 else 0
    g_sha = g["sharpe"] * np.sqrt(tpy) if tpy > 0 else 0.0
    n_sha = n["sharpe"] * np.sqrt(tpy) if tpy > 0 else 0.0
    print(f"  {label:40s}  N={g['n']:5d}  "
          f"WR={g['wr']:.1%}  "
          f"GrossExpR={g['expr']:+.3f}  GrossSh={g_sha:+.2f}  "
          f"NetExpR={n['expr']:+.3f}  NetSh={n_sha:+.2f}  "
          f"MaxDD={n['maxdd']:.1f}R  TotalR={n['total']:.1f}")

def orb_utc_hour(orb_label: str) -> tuple[int, int]:
    """Return (hour, minute) in UTC for an ORB label."""
    # Brisbane = UTC+10
    local_times = {
        "0900": (9, 0), "1000": (10, 0), "1100": (11, 0),
        "1800": (18, 0), "2300": (23, 0), "0030": (0, 30),
    }
    lh, lm = local_times[orb_label]
    utc_h = (lh - 10) % 24
    return utc_h, lm

# ============================================================================
# STRATEGY 1: ORB FADE
# ============================================================================

def run_orb_fade(db_path: Path, features: pd.DataFrame):
    """
    ORB Fade: enter OPPOSITE direction of ORB break after N confirm bars.
    Stop = beyond ORB extreme + buffer. Target based on RR.
    """
    print("\n" + "=" * 80)
    print("STRATEGY 1: ORB FADE (enter opposite of ORB break)")
    print("=" * 80)

    confirm_bars_list = [1, 2, 3]
    rr_targets = [1.0, 1.5, 2.0]
    buffer_mult = 0.5  # buffer = 0.5 * orb_size beyond ORB extreme

    results_summary = []

    for orb_label in ORB_LABELS:
        h_col = f"orb_{orb_label}_high"
        l_col = f"orb_{orb_label}_low"
        s_col = f"orb_{orb_label}_size"
        d_col = f"orb_{orb_label}_break_dir"
        t_col = f"orb_{orb_label}_break_ts"

        # Filter days with valid breaks
        mask = (
            features[d_col].notna() &
            features[d_col].isin(["long", "short"]) &
            features[h_col].notna() &
            features[l_col].notna() &
            features[s_col].notna() &
            (features[s_col] > 0)
        )
        valid_days = features[mask].copy()
        if valid_days.empty:
            continue

        print(f"\n  ORB {orb_label} -- {len(valid_days)} break-days")

        for cb in confirm_bars_list:
            for rr in rr_targets:
                gross_pnls = []
                net_pnls = []

                for _, row in valid_days.iterrows():
                    td = row["trading_day"]
                    if isinstance(td, str):
                        td = date.fromisoformat(td)
                    elif hasattr(td, 'date'):
                        td = td.date() if callable(td.date) else td.date

                    orb_high = float(row[h_col])
                    orb_low = float(row[l_col])
                    orb_size = float(row[s_col])
                    break_dir = row[d_col]
                    break_ts = row[t_col]

                    if pd.isna(break_ts):
                        continue

                    # Risk floor check
                    if orb_size < SPEC.tick_size * SPEC.min_ticks_floor:
                        continue

                    # Fade direction: opposite of break
                    if break_dir == "long":
                        direction = "short"
                    elif break_dir == "short":
                        direction = "long"
                    else:
                        continue

                    # Load bars for this day
                    bars = load_mcl_bars_for_day(db_path, td)
                    if bars.empty:
                        continue

                    # Convert break_ts for comparison
                    if isinstance(break_ts, str):
                        break_ts = pd.Timestamp(break_ts, tz="UTC")
                    elif not hasattr(break_ts, 'tzinfo') or break_ts.tzinfo is None:
                        break_ts = pd.Timestamp(break_ts).tz_localize("UTC")
                    else:
                        break_ts = pd.Timestamp(break_ts).tz_convert("UTC")

                    # Find bar index after break + cb confirm bars
                    post_break = bars[bars['ts_utc'] > break_ts]
                    if len(post_break) < cb + 1:
                        continue

                    entry_idx_in_bars = post_break.index[cb]  # cb bars after break
                    entry_bar = bars.loc[entry_idx_in_bars]
                    entry_price = float(entry_bar["open"])

                    buffer = buffer_mult * orb_size

                    if direction == "short":
                        stop_price = orb_high + buffer
                        risk_pts = stop_price - entry_price
                        target_price = entry_price - rr * risk_pts
                    else:
                        stop_price = orb_low - buffer
                        risk_pts = entry_price - stop_price
                        target_price = entry_price + rr * risk_pts

                    if risk_pts <= 0:
                        continue

                    # Risk floor on computed stop
                    if risk_pts < SPEC.tick_size * SPEC.min_ticks_floor:
                        continue

                    scan_start = bars.index.get_loc(entry_idx_in_bars) + 1
                    if scan_start >= len(bars):
                        continue

                    result = resolve_bar_outcome(
                        bars, entry_price, stop_price, target_price,
                        direction, scan_start
                    )
                    if result is None:
                        # EOD exit: mark-to-market at last bar close
                        last_close = float(bars.iloc[-1]["close"])
                        if direction == "long":
                            pnl_pts = last_close - entry_price
                        else:
                            pnl_pts = entry_price - last_close
                    else:
                        pnl_pts = result["pnl_points"]

                    gross_pnls.append(pnl_to_r_gross(entry_price, stop_price, pnl_pts))
                    net_pnls.append(pnl_to_r_net(entry_price, stop_price, pnl_pts))

                label = f"ORB_{orb_label}_CB{cb}_RR{rr}"
                gross_arr = np.array(gross_pnls) if gross_pnls else np.array([])
                net_arr = np.array(net_pnls) if net_pnls else np.array([])
                print_metrics(label, gross_arr, net_arr)

                if net_pnls:
                    m = compute_strategy_metrics(np.array(net_pnls))
                    if m:
                        results_summary.append({
                            "strategy": f"ORB_FADE_{label}",
                            **m
                        })

    return results_summary

# ============================================================================
# STRATEGY 2: DOUBLE-BREAK FADE
# ============================================================================

def run_double_break_fade(db_path: Path, features: pd.DataFrame):
    """
    Double-Break Fade: on days with double_break=True, enter in direction
    of the SECOND break (opposite of break_dir, since break_dir = first break).
    Stop = beyond first break extreme. Target = RR * risk.
    """
    print("\n" + "=" * 80)
    print("STRATEGY 2: DOUBLE-BREAK FADE (enter on second break)")
    print("=" * 80)

    rr_targets = [1.0, 1.5, 2.0, 2.5]
    results_summary = []

    for orb_label in ORB_LABELS:
        h_col = f"orb_{orb_label}_high"
        l_col = f"orb_{orb_label}_low"
        s_col = f"orb_{orb_label}_size"
        d_col = f"orb_{orb_label}_break_dir"
        t_col = f"orb_{orb_label}_break_ts"
        db_col = f"orb_{orb_label}_double_break"

        # Only double-break days
        if db_col not in features.columns:
            print(f"  ORB {orb_label} -- no double_break column, skipping")
            continue

        mask = (
            features[db_col] == True &
            features[d_col].notna() &
            features[d_col].isin(["long", "short"]) &
            features[h_col].notna() &
            features[l_col].notna() &
            features[s_col].notna() &
            (features[s_col] > 0)
        )
        valid_days = features[mask].copy()
        if valid_days.empty:
            print(f"  ORB {orb_label} -- 0 double-break days, skipping")
            continue

        print(f"\n  ORB {orb_label} -- {len(valid_days)} double-break days")

        for rr in rr_targets:
            gross_pnls = []
            net_pnls = []

            for _, row in valid_days.iterrows():
                td = row["trading_day"]
                if isinstance(td, str):
                    td = date.fromisoformat(td)
                elif hasattr(td, 'date'):
                    td = td.date() if callable(td.date) else td.date

                orb_high = float(row[h_col])
                orb_low = float(row[l_col])
                orb_size = float(row[s_col])
                break_dir = row[d_col]  # first break direction
                break_ts = row[t_col]

                if pd.isna(break_ts):
                    continue

                # Risk floor
                if orb_size < SPEC.tick_size * SPEC.min_ticks_floor:
                    continue

                # Second break direction is OPPOSITE of first break
                if break_dir == "long":
                    direction = "short"
                    entry_price = orb_low  # crosses below ORB low
                    stop_price = orb_high + orb_size * 0.25  # above first break extreme + buffer
                else:
                    direction = "long"
                    entry_price = orb_high  # crosses above ORB high
                    stop_price = orb_low - orb_size * 0.25  # below first break extreme + buffer

                risk_pts = abs(entry_price - stop_price)
                if risk_pts <= 0:
                    continue
                if risk_pts < SPEC.tick_size * SPEC.min_ticks_floor:
                    continue

                if direction == "long":
                    target_price = entry_price + rr * risk_pts
                else:
                    target_price = entry_price - rr * risk_pts

                # Load bars, find second break crossing
                bars = load_mcl_bars_for_day(db_path, td)
                if bars.empty:
                    continue

                if isinstance(break_ts, str):
                    break_ts = pd.Timestamp(break_ts, tz="UTC")
                elif not hasattr(break_ts, 'tzinfo') or break_ts.tzinfo is None:
                    break_ts = pd.Timestamp(break_ts).tz_localize("UTC")
                else:
                    break_ts = pd.Timestamp(break_ts).tz_convert("UTC")

                # Scan for second break crossing
                post_break = bars[bars['ts_utc'] > break_ts]
                entry_bar_idx = None
                fakeout_extreme = None

                for idx in post_break.index:
                    bar = bars.loc[idx]
                    bar_h = float(bar["high"])
                    bar_l = float(bar["low"])

                    # Track fakeout extreme (worst price during first break)
                    if break_dir == "long":
                        if fakeout_extreme is None or bar_h > fakeout_extreme:
                            fakeout_extreme = bar_h
                        # Second break: close below orb_low
                        if float(bar["close"]) < orb_low:
                            entry_bar_idx = idx
                            break
                    else:
                        if fakeout_extreme is None or bar_l < fakeout_extreme:
                            fakeout_extreme = bar_l
                        # Second break: close above orb_high
                        if float(bar["close"]) > orb_high:
                            entry_bar_idx = idx
                            break

                if entry_bar_idx is None:
                    continue

                # Use fakeout extreme as stop instead if more conservative
                if fakeout_extreme is not None:
                    if break_dir == "long" and fakeout_extreme > stop_price:
                        stop_price = fakeout_extreme + SPEC.tick_size
                    elif break_dir == "short" and fakeout_extreme < stop_price:
                        stop_price = fakeout_extreme - SPEC.tick_size

                risk_pts = abs(entry_price - stop_price)
                if risk_pts <= 0 or risk_pts < SPEC.tick_size * SPEC.min_ticks_floor:
                    continue

                if direction == "long":
                    target_price = entry_price + rr * risk_pts
                else:
                    target_price = entry_price - rr * risk_pts

                scan_start = bars.index.get_loc(entry_bar_idx) + 1
                if scan_start >= len(bars):
                    continue

                result = resolve_bar_outcome(
                    bars, entry_price, stop_price, target_price,
                    direction, scan_start
                )
                if result is None:
                    last_close = float(bars.iloc[-1]["close"])
                    pnl_pts = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
                else:
                    pnl_pts = result["pnl_points"]

                gross_pnls.append(pnl_to_r_gross(entry_price, stop_price, pnl_pts))
                net_pnls.append(pnl_to_r_net(entry_price, stop_price, pnl_pts))

            label = f"ORB_{orb_label}_RR{rr}"
            gross_arr = np.array(gross_pnls) if gross_pnls else np.array([])
            net_arr = np.array(net_pnls) if net_pnls else np.array([])
            print_metrics(label, gross_arr, net_arr)

            if net_pnls:
                m = compute_strategy_metrics(np.array(net_pnls))
                if m:
                    results_summary.append({
                        "strategy": f"DB_FADE_{label}",
                        **m
                    })

    return results_summary

# ============================================================================
# STRATEGY 3: RANGE COMPRESSION (small ORB delayed breakout)
# ============================================================================

def run_range_compression(db_path: Path, features: pd.DataFrame):
    """
    Range Compression: on days with tiny ORBs (below median), wait for
    delayed breakout after 30/60/120 minutes of compression. Small ORBs
    = coiled spring hypothesis.
    """
    print("\n" + "=" * 80)
    print("STRATEGY 3: RANGE COMPRESSION (small ORB -> delayed breakout)")
    print("=" * 80)

    wait_minutes_list = [30, 60, 120]
    rr_targets = [1.0, 1.5, 2.0]
    results_summary = []

    for orb_label in ORB_LABELS:
        h_col = f"orb_{orb_label}_high"
        l_col = f"orb_{orb_label}_low"
        s_col = f"orb_{orb_label}_size"

        # Need valid ORBs
        mask = (
            features[h_col].notna() &
            features[l_col].notna() &
            features[s_col].notna() &
            (features[s_col] > 0)
        )
        valid = features[mask].copy()
        if valid.empty:
            continue

        # Compute median ORB size
        median_size = valid[s_col].median()
        # Filter to SMALL ORBs only (below median)
        small_orb_days = valid[valid[s_col] <= median_size].copy()
        if small_orb_days.empty:
            continue

        print(f"\n  ORB {orb_label} -- median size={median_size:.4f}, "
              f"{len(small_orb_days)} small-ORB days")

        for wait_min in wait_minutes_list:
            for rr in rr_targets:
                gross_pnls = []
                net_pnls = []

                for _, row in small_orb_days.iterrows():
                    td = row["trading_day"]
                    if isinstance(td, str):
                        td = date.fromisoformat(td)
                    elif hasattr(td, 'date'):
                        td = td.date() if callable(td.date) else td.date

                    orb_high = float(row[h_col])
                    orb_low = float(row[l_col])
                    orb_size = float(row[s_col])

                    if orb_size < SPEC.tick_size * SPEC.min_ticks_floor:
                        continue

                    bars = load_mcl_bars_for_day(db_path, td)
                    if bars.empty:
                        continue

                    # Find the ORB end time
                    utc_h, utc_m = orb_utc_hour(orb_label)
                    # ORB is first 5 minutes, so ORB end = orb_start + 5 min
                    # We want to wait wait_min minutes AFTER the ORB
                    # Compression window end = orb_start + 5 + wait_min
                    orb_start_mask = bars['ts_utc'].apply(
                        lambda t: t.hour == utc_h and t.minute == utc_m
                    )
                    orb_start_bars = bars[orb_start_mask]
                    if orb_start_bars.empty:
                        continue

                    orb_start_ts = orb_start_bars.iloc[0]['ts_utc']
                    compression_end = orb_start_ts + pd.Timedelta(minutes=5 + wait_min)

                    # Get bars in compression window (after ORB, before compression_end)
                    comp_mask = (
                        (bars['ts_utc'] > orb_start_ts + pd.Timedelta(minutes=5)) &
                        (bars['ts_utc'] <= compression_end)
                    )
                    comp_bars = bars[comp_mask]

                    # Track compression range
                    if comp_bars.empty:
                        continue
                    comp_high = comp_bars['high'].max()
                    comp_low = comp_bars['low'].min()

                    # Extended range = union of ORB and compression period
                    range_high = max(orb_high, float(comp_high))
                    range_low = min(orb_low, float(comp_low))

                    # Now scan AFTER compression window for breakout
                    post_comp = bars[bars['ts_utc'] > compression_end]
                    if post_comp.empty:
                        continue

                    # Find first breakout bar
                    entry_bar_idx = None
                    direction = None
                    for idx in post_comp.index:
                        bar = bars.loc[idx]
                        if float(bar["close"]) > range_high:
                            direction = "long"
                            entry_bar_idx = idx
                            break
                        elif float(bar["close"]) < range_low:
                            direction = "short"
                            entry_bar_idx = idx
                            break

                    if entry_bar_idx is None or direction is None:
                        continue

                    entry_price = float(bars.loc[entry_bar_idx, "close"])
                    range_size = range_high - range_low
                    if range_size <= 0:
                        continue

                    if direction == "long":
                        stop_price = range_low
                        risk_pts = entry_price - stop_price
                    else:
                        stop_price = range_high
                        risk_pts = stop_price - entry_price

                    if risk_pts <= 0 or risk_pts < SPEC.tick_size * SPEC.min_ticks_floor:
                        continue

                    if direction == "long":
                        target_price = entry_price + rr * risk_pts
                    else:
                        target_price = entry_price - rr * risk_pts

                    scan_start = bars.index.get_loc(entry_bar_idx) + 1
                    if scan_start >= len(bars):
                        continue

                    result = resolve_bar_outcome(
                        bars, entry_price, stop_price, target_price,
                        direction, scan_start
                    )
                    if result is None:
                        last_close = float(bars.iloc[-1]["close"])
                        pnl_pts = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
                    else:
                        pnl_pts = result["pnl_points"]

                    gross_pnls.append(pnl_to_r_gross(entry_price, stop_price, pnl_pts))
                    net_pnls.append(pnl_to_r_net(entry_price, stop_price, pnl_pts))

                label = f"ORB_{orb_label}_WAIT{wait_min}_RR{rr}"
                gross_arr = np.array(gross_pnls) if gross_pnls else np.array([])
                net_arr = np.array(net_pnls) if net_pnls else np.array([])
                print_metrics(label, gross_arr, net_arr)

                if net_pnls:
                    m = compute_strategy_metrics(np.array(net_pnls))
                    if m:
                        results_summary.append({
                            "strategy": f"COMPRESSION_{label}",
                            **m
                        })

    return results_summary

# ============================================================================
# STRATEGY 4: SESSION FADE
# ============================================================================

def run_session_fade(db_path: Path, features: pd.DataFrame):
    """
    Session Fade: at 1800 session (London open), fade the Asian session
    (0900-1000) direction. If Asia went long, go short at 1800.
    Stop = session high/low + buffer. Target = 1.0/1.5/2.0 x risk.
    """
    print("\n" + "=" * 80)
    print("STRATEGY 4: SESSION FADE (fade Asian direction at 1800)")
    print("=" * 80)

    rr_targets = [1.0, 1.5, 2.0]
    buffer_mult = 0.3  # buffer beyond session extreme as fraction of range
    results_summary = []

    # Need 0900 break_dir to know Asian session direction,
    # and 1800 ORB data for entry timing
    h_0900 = "orb_0900_high"
    l_0900 = "orb_0900_low"
    d_0900 = "orb_0900_break_dir"

    # Also use session stats if available
    asia_h = "session_asia_high"
    asia_l = "session_asia_low"

    # We need the 1800 bar for entry
    has_session_cols = asia_h in features.columns and asia_l in features.columns
    has_0900_dir = d_0900 in features.columns

    if not has_0900_dir:
        print("  No 0900 break_dir column available, skipping session fade")
        return results_summary

    # Filter: need valid 0900 break direction and session data
    cols_needed = [d_0900, h_0900, l_0900]
    if has_session_cols:
        cols_needed.extend([asia_h, asia_l])

    mask = features[d_0900].notna() & features[d_0900].isin(["long", "short"])
    if has_session_cols:
        mask = mask & features[asia_h].notna() & features[asia_l].notna()

    valid_days = features[mask].copy()
    print(f"  {len(valid_days)} days with valid Asian session direction")

    for rr in rr_targets:
        gross_pnls = []
        net_pnls = []

        for _, row in valid_days.iterrows():
            td = row["trading_day"]
            if isinstance(td, str):
                td = date.fromisoformat(td)
            elif hasattr(td, 'date'):
                td = td.date() if callable(td.date) else td.date

            asia_dir = row[d_0900]  # use 0900 break as proxy for Asian direction

            if has_session_cols and not pd.isna(row[asia_h]) and not pd.isna(row[asia_l]):
                sess_high = float(row[asia_h])
                sess_low = float(row[asia_l])
                sess_range = sess_high - sess_low
            else:
                # Fallback: use 0900 ORB as proxy
                sess_high = float(row[h_0900])
                sess_low = float(row[l_0900])
                sess_range = sess_high - sess_low

            if sess_range <= 0:
                continue

            # Load bars for this day
            bars = load_mcl_bars_for_day(db_path, td)
            if bars.empty:
                continue

            # Find 1800 Brisbane = 08:00 UTC bar
            entry_mask = bars['ts_utc'].apply(lambda t: t.hour == 8 and t.minute == 0)
            entry_bars = bars[entry_mask]
            if entry_bars.empty:
                continue

            entry_bar_idx = entry_bars.index[0]
            entry_price = float(bars.loc[entry_bar_idx, "open"])

            buffer = buffer_mult * sess_range

            # Fade: opposite of Asian direction
            if asia_dir == "long":
                direction = "short"
                stop_price = sess_high + buffer
                risk_pts = stop_price - entry_price
            else:
                direction = "long"
                stop_price = sess_low - buffer
                risk_pts = entry_price - stop_price

            if risk_pts <= 0 or risk_pts < SPEC.tick_size * SPEC.min_ticks_floor:
                continue

            if direction == "long":
                target_price = entry_price + rr * risk_pts
            else:
                target_price = entry_price - rr * risk_pts

            scan_start = bars.index.get_loc(entry_bar_idx) + 1
            if scan_start >= len(bars):
                continue

            result = resolve_bar_outcome(
                bars, entry_price, stop_price, target_price,
                direction, scan_start
            )
            if result is None:
                last_close = float(bars.iloc[-1]["close"])
                pnl_pts = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
            else:
                pnl_pts = result["pnl_points"]

            gross_pnls.append(pnl_to_r_gross(entry_price, stop_price, pnl_pts))
            net_pnls.append(pnl_to_r_net(entry_price, stop_price, pnl_pts))

        label = f"FADE_ASIA_AT_1800_RR{rr}"
        gross_arr = np.array(gross_pnls) if gross_pnls else np.array([])
        net_arr = np.array(net_pnls) if net_pnls else np.array([])
        print_metrics(label, gross_arr, net_arr)

        if net_pnls:
            m = compute_strategy_metrics(np.array(net_pnls))
            if m:
                results_summary.append({
                    "strategy": f"SESSION_FADE_{label}",
                    **m
                })

    return results_summary

# ============================================================================
# STRATEGY 5: VWAP REVERSION
# ============================================================================

def run_vwap_reversion(db_path: Path, features: pd.DataFrame):
    """
    VWAP Reversion: when price deviates > threshold * ATR from VWAP,
    fade back to VWAP. Entry on deviation, stop = 2x deviation, target = VWAP.
    """
    print("\n" + "=" * 80)
    print("STRATEGY 5: VWAP REVERSION (fade extreme deviations from VWAP)")
    print("=" * 80)

    atr_thresholds = [0.5, 1.0, 1.5]
    stop_mult_list = [1.5, 2.0, 2.5]  # stop = entry deviation * mult
    results_summary = []

    # We need atr_20 from features
    if 'atr_20' not in features.columns:
        print("  No atr_20 column in features, skipping VWAP reversion")
        return results_summary

    # Get unique trading days with valid ATR
    atr_data = features[features['atr_20'].notna()][['trading_day', 'atr_20']].drop_duplicates('trading_day')
    atr_dict = dict(zip(atr_data['trading_day'], atr_data['atr_20']))

    for atr_thresh in atr_thresholds:
        for stop_mult in stop_mult_list:
            gross_pnls = []
            net_pnls = []
            days_processed = 0

            for td_raw, atr_val in atr_dict.items():
                if isinstance(td_raw, str):
                    td = date.fromisoformat(td_raw)
                elif hasattr(td_raw, 'date'):
                    td = td_raw.date() if callable(td_raw.date) else td_raw.date
                else:
                    td = td_raw

                atr = float(atr_val)
                if atr <= 0:
                    continue

                bars = load_mcl_bars_for_day(db_path, td)
                if bars.empty or len(bars) < 60:
                    continue

                days_processed += 1

                # Compute running VWAP for the day
                typical_price = (bars['high'] + bars['low'] + bars['close']) / 3.0
                cum_tp_vol = (typical_price * bars['volume']).cumsum()
                cum_vol = bars['volume'].cumsum()
                vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

                # Skip first 30 minutes to let VWAP stabilize
                if len(bars) < 30:
                    continue

                deviation = bars['close'] - vwap
                deviation_atr = deviation / atr

                # Scan for entry signals
                in_trade = False
                for i in range(30, len(bars) - 1):
                    if in_trade:
                        continue

                    dev = float(deviation_atr.iloc[i])
                    if abs(dev) < atr_thresh:
                        continue

                    # Signal: deviation exceeds threshold
                    close_price = float(bars.iloc[i]['close'])
                    vwap_price = float(vwap.iloc[i])
                    dev_points = abs(close_price - vwap_price)

                    if dev > 0:
                        # Price above VWAP -> go short
                        direction = "short"
                        entry_price = close_price
                        stop_price = entry_price + dev_points * (stop_mult - 1.0)
                        target_price = vwap_price
                    else:
                        # Price below VWAP -> go long
                        direction = "long"
                        entry_price = close_price
                        stop_price = entry_price - dev_points * (stop_mult - 1.0)
                        target_price = vwap_price

                    risk_pts = abs(entry_price - stop_price)
                    if risk_pts <= 0 or risk_pts < SPEC.tick_size * SPEC.min_ticks_floor:
                        continue

                    scan_start = i + 1
                    result = resolve_bar_outcome(
                        bars, entry_price, stop_price, target_price,
                        direction, scan_start
                    )
                    if result is None:
                        last_close = float(bars.iloc[-1]["close"])
                        pnl_pts = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
                    else:
                        pnl_pts = result["pnl_points"]

                    gross_pnls.append(pnl_to_r_gross(entry_price, stop_price, pnl_pts))
                    net_pnls.append(pnl_to_r_net(entry_price, stop_price, pnl_pts))
                    in_trade = True  # one trade per day max

            label = f"ATR_THRESH{atr_thresh}_STOP{stop_mult}x"
            gross_arr = np.array(gross_pnls) if gross_pnls else np.array([])
            net_arr = np.array(net_pnls) if net_pnls else np.array([])
            print_metrics(label, gross_arr, net_arr)

            if net_pnls:
                m = compute_strategy_metrics(np.array(net_pnls))
                if m:
                    results_summary.append({
                        "strategy": f"VWAP_REV_{label}",
                        **m
                    })

    return results_summary

# ============================================================================
# STRATEGY 6: TIME-OF-DAY DIRECTIONAL BIAS
# ============================================================================

def run_time_of_day(db_path: Path):
    """
    Time-of-Day: scan which UTC hours have directional bias.
    Compute mean return per hour, then test mechanical strategies
    on hours with significant bias.
    """
    print("\n" + "=" * 80)
    print("STRATEGY 6: TIME-OF-DAY (hourly directional bias scan)")
    print("=" * 80)

    results_summary = []

    # Load all bars
    print("  Loading all MCL 1m bars...")
    all_bars = load_all_mcl_bars(db_path)
    if all_bars.empty:
        print("  No MCL bars found, skipping time-of-day analysis")
        return results_summary

    print(f"  Loaded {len(all_bars):,} bars")

    # Compute per-bar returns
    all_bars = all_bars.sort_values('ts_utc').reset_index(drop=True)
    all_bars['return_pts'] = all_bars['close'] - all_bars['open']
    all_bars['utc_hour'] = all_bars['ts_utc'].dt.hour

    # Assign trading days
    all_bars['trading_day'] = all_bars['ts_utc'].apply(
        lambda t: (t - pd.Timedelta(hours=23)).date()
        if t.hour >= 23 else t.date()
    )
    # Simpler: use the Brisbane boundary
    # 23:00 UTC = 09:00 Brisbane = new trading day
    all_bars['trading_day'] = all_bars['ts_utc'].apply(
        lambda t: t.date() if t.hour >= 23 else (t - pd.Timedelta(hours=0)).date()
    )
    # Actually just group by UTC hour for hourly returns
    # Compute 1-hour bar returns for each UTC hour
    all_bars['date'] = all_bars['ts_utc'].dt.date

    # Group: per day per hour, compute open-to-close return
    hourly = all_bars.groupby(['date', 'utc_hour']).agg(
        open_price=('open', 'first'),
        close_price=('close', 'last'),
        high=('high', 'max'),
        low=('low', 'min'),
        n_bars=('close', 'count'),
    ).reset_index()

    hourly['hour_return_pts'] = hourly['close_price'] - hourly['open_price']

    # Phase 1: Scan for directional bias per UTC hour
    print("\n  UTC Hour | Mean Ret(pts) | Std(pts) | t-stat | N days | Direction")
    print("  " + "-" * 72)

    bias_hours = []
    for hour in range(24):
        h_data = hourly[hourly['utc_hour'] == hour]
        if len(h_data) < 50:
            continue
        rets = h_data['hour_return_pts'].values
        mean_ret = np.mean(rets)
        std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9
        t_stat = mean_ret / (std_ret / np.sqrt(len(rets)))
        direction = "LONG" if mean_ret > 0 else "SHORT"
        significant = "*" if abs(t_stat) > 2.0 else " "
        print(f"  {hour:02d}:00 UTC | {mean_ret:+.5f}      | {std_ret:.5f}  | "
              f"{t_stat:+.2f}   | {len(rets):5d}  | {direction} {significant}")
        if abs(t_stat) > 1.5:  # looser threshold for further testing
            bias_hours.append((hour, direction.lower(), mean_ret, t_stat))

    # Phase 2: Test mechanical entry on biased hours
    if not bias_hours:
        print("\n  No hours with significant directional bias (|t| > 1.5)")
        return results_summary

    print(f"\n  Testing mechanical strategies on {len(bias_hours)} biased hours...")

    hold_periods = [1, 2, 3]  # hold for 1, 2, 3 hours after entry

    for hour, direction, _, t_stat in bias_hours:
        for hold_hours in hold_periods:
            gross_pnls = []
            net_pnls = []

            h_data = hourly[hourly['utc_hour'] == hour]

            for _, h_row in h_data.iterrows():
                d = h_row['date']
                entry_price = float(h_row['open_price'])

                # Find exit: N hours later
                exit_hour = (hour + hold_hours) % 24
                exit_data = hourly[
                    (hourly['date'] == d) &
                    (hourly['utc_hour'] == exit_hour)
                ]
                if exit_data.empty:
                    continue

                exit_price = float(exit_data.iloc[0]['close_price'])

                if direction == "long":
                    pnl_pts = exit_price - entry_price
                else:
                    pnl_pts = entry_price - exit_price

                # Use fixed stop = typical hourly range (from hourly stats)
                day_hourly = hourly[hourly['date'] == d]
                if day_hourly.empty:
                    continue
                avg_range = float((day_hourly['high'] - day_hourly['low']).median())
                if avg_range <= 0:
                    avg_range = 0.10  # fallback

                stop_distance = avg_range * 1.5
                if direction == "long":
                    stop_price = entry_price - stop_distance
                else:
                    stop_price = entry_price + stop_distance

                # Clamp: if actual loss exceeds stop, cap at stop loss
                if pnl_pts < -stop_distance:
                    pnl_pts = -stop_distance

                gross_pnls.append(pnl_to_r_gross(entry_price, stop_price, pnl_pts))
                net_pnls.append(pnl_to_r_net(entry_price, stop_price, pnl_pts))

            label = f"H{hour:02d}_{direction.upper()}_HOLD{hold_hours}h"
            gross_arr = np.array(gross_pnls) if gross_pnls else np.array([])
            net_arr = np.array(net_pnls) if net_pnls else np.array([])
            print_metrics(label, gross_arr, net_arr)

            if net_pnls:
                m = compute_strategy_metrics(np.array(net_pnls))
                if m:
                    results_summary.append({
                        "strategy": f"TOD_{label}",
                        **m
                    })

    return results_summary

# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary_table(all_results: list[dict]):
    """Print final comparison table of all strategies tested."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: ALL STRATEGY TYPES COMPARED (net of costs)")
    print("=" * 80)

    if not all_results:
        print("  No strategies produced trades.")
        return

    # Sort by ExpR descending
    all_results.sort(key=lambda x: x.get("expr", -999), reverse=True)

    years = (MCL_END - MCL_START).days / 365.25

    print(f"\n  {'Strategy':<50s} {'N':>5s} {'WR':>6s} {'ExpR':>8s} "
          f"{'ShANN':>7s} {'MaxDD':>7s} {'TotalR':>8s}")
    print("  " + "-" * 93)

    for r in all_results:
        tpy = r["n"] / years if years > 0 else 0
        sha = r["sharpe"] * np.sqrt(tpy) if tpy > 0 else 0.0
        verdict = ""
        if r["expr"] > 0 and sha > 0.3:
            verdict = " <-- INVESTIGATE"
        elif r["expr"] > 0:
            verdict = " <-- marginal"
        print(f"  {r['strategy']:<50s} {r['n']:5d} {r['wr']:5.1%} "
              f"{r['expr']:+7.3f} {sha:+6.2f} {r['maxdd']:6.1f}R "
              f"{r['total']:+7.1f}{verdict}")

    # Count positive ExpR
    positive = [r for r in all_results if r.get("expr", 0) > 0]
    print(f"\n  {len(positive)}/{len(all_results)} strategies have positive net ExpR")

    if not positive:
        print("\n  VERDICT: NO strategy type shows edge on MCL after costs.")
        print("  MCL is a NO-GO instrument. Do not revisit without new data/model.")
    else:
        best = positive[0]
        tpy = best["n"] / years if years > 0 else 0
        sha = best["sharpe"] * np.sqrt(tpy) if tpy > 0 else 0.0
        print(f"\n  Best strategy: {best['strategy']}")
        print(f"  ExpR={best['expr']:+.3f}, ShANN={sha:+.2f}, N={best['n']}, "
              f"MaxDD={best['maxdd']:.1f}R")
        if sha < 0.5:
            print("  WARNING: Best Sharpe_ann < 0.5 -- likely not tradeable.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive MCL strategy scan -- all strategy types"
    )
    parser.add_argument("--db-path", type=str, default="C:/db/gold.db",
                        help="Path to DuckDB database (default: C:/db/gold.db)")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"FATAL: Database not found at {db_path}")
        sys.exit(1)

    print(f"MCL Comprehensive Strategy Scan")
    print(f"Database: {db_path}")
    print(f"Instrument: {INSTRUMENT}")
    print(f"Cost model: ${SPEC.point_value}/point, ${SPEC.total_friction:.2f} RT friction, "
          f"tick={SPEC.tick_size}")
    print(f"Date range: {MCL_START} to {MCL_END}")

    # Quick data check
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        bar_count = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = ?", [INSTRUMENT]
        ).fetchone()[0]
        feat_count = con.execute(
            "SELECT COUNT(*) FROM daily_features WHERE symbol = ?", [INSTRUMENT]
        ).fetchone()[0]
        try:
            outcome_count = con.execute(
                "SELECT COUNT(*) FROM orb_outcomes WHERE symbol = ?", [INSTRUMENT]
            ).fetchone()[0]
        except Exception:
            outcome_count = 0
    finally:
        con.close()

    print(f"\nData check: bars_1m={bar_count:,}, daily_features={feat_count:,}, "
          f"orb_outcomes={outcome_count:,}")

    if bar_count == 0:
        print("FATAL: No MCL bars in database. Run pipeline first.")
        sys.exit(1)

    # Load features once
    print("\nLoading daily features...")
    features = load_mcl_features(db_path)
    print(f"  Loaded {len(features)} feature rows")

    if features.empty:
        print("FATAL: No MCL daily_features. Run build_daily_features first.")
        sys.exit(1)

    all_results = []

    # Strategy 1: ORB Fade
    t0 = time.time()
    try:
        results = run_orb_fade(db_path, features)
        all_results.extend(results)
    except Exception as e:
        print(f"\n  *** ORB FADE CRASHED: {e}")
        traceback.print_exc()
    print(f"\n  [ORB Fade completed in {time.time() - t0:.1f}s]")

    # Strategy 2: Double-Break Fade
    t0 = time.time()
    try:
        results = run_double_break_fade(db_path, features)
        all_results.extend(results)
    except Exception as e:
        print(f"\n  *** DOUBLE-BREAK FADE CRASHED: {e}")
        traceback.print_exc()
    print(f"\n  [Double-Break Fade completed in {time.time() - t0:.1f}s]")

    # Strategy 3: Range Compression
    t0 = time.time()
    try:
        results = run_range_compression(db_path, features)
        all_results.extend(results)
    except Exception as e:
        print(f"\n  *** RANGE COMPRESSION CRASHED: {e}")
        traceback.print_exc()
    print(f"\n  [Range Compression completed in {time.time() - t0:.1f}s]")

    # Strategy 4: Session Fade
    t0 = time.time()
    try:
        results = run_session_fade(db_path, features)
        all_results.extend(results)
    except Exception as e:
        print(f"\n  *** SESSION FADE CRASHED: {e}")
        traceback.print_exc()
    print(f"\n  [Session Fade completed in {time.time() - t0:.1f}s]")

    # Strategy 5: VWAP Reversion
    t0 = time.time()
    try:
        results = run_vwap_reversion(db_path, features)
        all_results.extend(results)
    except Exception as e:
        print(f"\n  *** VWAP REVERSION CRASHED: {e}")
        traceback.print_exc()
    print(f"\n  [VWAP Reversion completed in {time.time() - t0:.1f}s]")

    # Strategy 6: Time-of-Day
    t0 = time.time()
    try:
        results = run_time_of_day(db_path)
        all_results.extend(results)
    except Exception as e:
        print(f"\n  *** TIME-OF-DAY CRASHED: {e}")
        traceback.print_exc()
    print(f"\n  [Time-of-Day completed in {time.time() - t0:.1f}s]")

    # Final summary
    print_summary_table(all_results)

if __name__ == "__main__":
    main()

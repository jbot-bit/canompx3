#!/usr/bin/env python3
"""Non-ORB Strategy Research — V3: Mechanism-Driven (Logic-First).

V1/V2 used blind parameter grids. V3 starts from WHY something should work:

Key insight from V1/V2 data:
  - These markets TREND intraday (VWAP reversion = anti-edge, p=1.0)
  - ORB breakout momentum is REAL (fading it = uniformly negative)
  - PDH/PDL sweep was the ONLY signal with consistent directional bias (MNQ p=0.053)

LOGIC: If markets trend, find entries WITH the trend at good prices.
Each strategy has a STRUCTURAL stop (defined by price pattern, not ATR grid).

Strategies:
  1. PDH/PDL Sweep Rejection — Stop at sweep extreme. Mechanically-defined R.
     WHY: Stops cluster at visible levels. Sweep absorbs them, then price reverses.
     WHEN: Low-liquidity sessions (easier to push through for stop hunt).

  2. Session Trend Pullback — Enter WITH trend at 50% retracement.
     WHY: Session establishes direction in first 15-30 min. Pullbacks are opportunities.
     WHEN: Strong initial move (>0.3 ATR) then orderly pullback (30-70% retrace).

  3. VWAP Bounce WITH Trend — Buy at VWAP support in established uptrend.
     WHY: VWAP is institutional benchmark. In a trend, it acts as support, not a target.
     WHEN: Trend established, price pulls back to VWAP, then bounces.

Exits (per strategy, not a grid):
  - R-based: 1R, 2R, 3R targets where R = entry-to-structural-stop distance
  - Trailing: Trail from peak by 1R
  - Time: Session-end safety fallback

Usage:
    python research/research_v3_mechanism.py --db-path gold.db
    python research/research_v3_mechanism.py --db-path gold.db --instrument MNQ
    python research/research_v3_mechanism.py --db-path gold.db --strategy sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import COST_SPECS, CostSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INSTRUMENTS = ["MES", "MNQ", "M2K"]  # Only equity micros (MGC trends too hard)
MIN_TRADES = 30
FDR_ALPHA = 0.05

SESSION_BLOCKS = {
    "asia": {"start_utc_h": 0, "end_utc_h": 8},
    "london": {"start_utc_h": 8, "end_utc_h": 13},
    "us": {"start_utc_h": 13, "end_utc_h": 19},
}

# US market open times (UTC) for session trend detection
US_OPEN = {
    "winter": {"h": 14, "m": 30},  # 9:30 ET = 14:30 UTC (EST)
    "summer": {"h": 13, "m": 30},  # 9:30 ET = 13:30 UTC (EDT)
}
LONDON_OPEN = {
    "winter": {"h": 8, "m": 0},  # 8:00 UTC (GMT)
    "summer": {"h": 7, "m": 0},  # 7:00 UTC (BST)
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class MechanicalEntry:
    """Trade entry with a STRUCTURAL stop defined by the price pattern."""

    direction: int  # +1 long, -1 short
    entry_price: float
    stop_price: float  # Defined by the mechanism, not an ATR multiple
    remaining_bars: pd.DataFrame
    atr: float
    trading_day: object
    metadata: dict = field(default_factory=dict)

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop_price)


@dataclass
class StrategyResult:
    strategy: str
    instrument: str
    variant: str
    exit_type: str
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    sharpe: float = 0.0
    max_dd_r: float = 0.0
    p_value: float = 1.0
    yearly_results: dict = field(default_factory=dict)
    pnl_series: list = field(default_factory=list)
    trade_dates: list = field(default_factory=list)
    avg_risk_pts: float = 0.0
    median_risk_pts: float = 0.0


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------
def compute_max_dd(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return round(float(np.max(dd)), 4) if len(dd) > 0 else 0.0


def yearly_breakdown(dates: list, pnls: list[float]) -> dict:
    if not dates or not pnls:
        return {}
    df = pd.DataFrame({"date": dates, "pnl_r": pnls})
    df["year"] = pd.to_datetime(df["date"]).dt.year
    result = {}
    for year, grp in df.groupby("year"):
        n = len(grp)
        if n < 5:
            continue
        result[int(year)] = {
            "n": n,
            "avg_r": round(float(grp["pnl_r"].mean()), 4),
            "total_r": round(float(grp["pnl_r"].sum()), 2),
            "wr": round(float((grp["pnl_r"] > 0).mean()), 4),
        }
    return result


def to_r_multiple(pnl_points: float, risk_points: float, spec: CostSpec) -> float:
    """Convert PnL in points to R-multiple with friction properly applied.

    R = (pnl_pts * point_value - friction) / (risk_pts * point_value + friction)
    Friction INCREASES risk (denominator) and REDUCES reward (numerator).
    """
    if risk_points <= 0:
        return 0.0
    pnl_dollars = pnl_points * spec.point_value - spec.total_friction
    risk_dollars = risk_points * spec.point_value + spec.total_friction
    if risk_dollars <= 0:
        return 0.0
    return pnl_dollars / risk_dollars


def finalize_result(
    strategy: str,
    instrument: str,
    variant: str,
    exit_type: str,
    r_multiples: list[float],
    trade_dates: list,
    risk_points_list: list[float],
) -> StrategyResult:
    n = len(r_multiples)
    if n < 1:
        return StrategyResult(strategy=strategy, instrument=instrument, variant=variant, exit_type=exit_type)
    arr = np.array(r_multiples)
    wins = int(np.sum(arr > 0))
    avg_r = float(np.mean(arr))
    total_r = float(np.sum(arr))
    std_r = float(np.std(arr, ddof=1)) if n > 1 else 1.0
    sharpe = avg_r / std_r if std_r > 0 else 0.0
    if n >= 2 and std_r > 0:
        t_stat, p_val = stats.ttest_1samp(arr, 0.0)
        p_val = float(p_val / 2) if t_stat > 0 else 1.0 - float(p_val / 2)
    else:
        p_val = 1.0
    return StrategyResult(
        strategy=strategy,
        instrument=instrument,
        variant=variant,
        exit_type=exit_type,
        n_trades=n,
        n_wins=wins,
        win_rate=round(wins / n, 4),
        avg_r=round(avg_r, 4),
        total_r=round(total_r, 2),
        sharpe=round(sharpe, 4),
        max_dd_r=compute_max_dd(r_multiples),
        p_value=round(p_val, 6),
        yearly_results=yearly_breakdown(trade_dates, r_multiples),
        pnl_series=r_multiples,
        trade_dates=trade_dates,
        avg_risk_pts=round(float(np.mean(risk_points_list)), 4) if risk_points_list else 0.0,
        median_risk_pts=round(float(np.median(risk_points_list)), 4) if risk_points_list else 0.0,
    )


def bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[tuple[int, float, bool]]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj_p = min(prev_adj, p * n / rank, 1.0)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p
    return [(i, adjusted[i], adjusted[i] < alpha) for i in range(n)]


# ---------------------------------------------------------------------------
# Exit Simulators (Mechanism-Based)
# ---------------------------------------------------------------------------
def simulate_r_target(bars_after, direction, entry_price, stop_price, target_r):
    """Exit at fixed R-multiple target. Stop is structural (from entry signal)."""
    risk = abs(entry_price - stop_price)
    if risk <= 0 or len(bars_after) == 0:
        return 0.0

    target_dist = risk * target_r
    if direction == 1:
        target_price = entry_price + target_dist
        sl = stop_price
    else:
        target_price = entry_price - target_dist
        sl = stop_price

    for i in range(min(240, len(bars_after))):
        bar = bars_after.iloc[i]
        if direction == 1:
            if float(bar["low"]) <= sl:
                return sl - entry_price  # stopped out
            if float(bar["high"]) >= target_price:
                return target_dist  # target hit
        else:
            if float(bar["high"]) >= sl:
                return entry_price - sl  # stopped out (negative)
            if float(bar["low"]) <= target_price:
                return target_dist  # target hit

    # Time exit — whatever we have at the end
    exit_price = float(bars_after.iloc[min(239, len(bars_after) - 1)]["close"])
    return (exit_price - entry_price) * direction


def simulate_trailing_r(bars_after, direction, entry_price, stop_price, trail_r=1.0):
    """Trail from peak by trail_r * risk_distance. Initial stop is structural."""
    risk = abs(entry_price - stop_price)
    if risk <= 0 or len(bars_after) == 0:
        return 0.0

    trail_dist = risk * trail_r
    best_price = entry_price
    initial_sl = stop_price

    for i in range(min(240, len(bars_after))):
        bar = bars_after.iloc[i]
        if direction == 1:
            best_price = max(best_price, float(bar["high"]))
            # Trailing stop moves up, but never below initial stop
            trail_stop = max(initial_sl, best_price - trail_dist)
            if float(bar["low"]) <= trail_stop:
                return trail_stop - entry_price
        else:
            best_price = min(best_price, float(bar["low"]))
            trail_stop = min(initial_sl, best_price + trail_dist)
            if float(bar["high"]) >= trail_stop:
                return entry_price - trail_stop

    exit_price = float(bars_after.iloc[min(239, len(bars_after) - 1)]["close"])
    return (exit_price - entry_price) * direction


def run_mechanical_exits(
    entries: list[MechanicalEntry],
    strategy: str,
    instrument: str,
    variant: str,
    spec: CostSpec,
) -> list[StrategyResult]:
    """Run 4 logical exits for a set of mechanical entries."""
    exit_configs = [
        ("R1", lambda b, d, ep, sp: simulate_r_target(b, d, ep, sp, 1.0)),
        ("R2", lambda b, d, ep, sp: simulate_r_target(b, d, ep, sp, 2.0)),
        ("R3", lambda b, d, ep, sp: simulate_r_target(b, d, ep, sp, 3.0)),
        ("TR1", lambda b, d, ep, sp: simulate_trailing_r(b, d, ep, sp, 1.0)),
    ]
    results = []
    for exit_label, exit_fn in exit_configs:
        r_multiples = []
        trade_dates = []
        risk_pts_list = []
        for e in entries:
            pnl_pts = exit_fn(e.remaining_bars, e.direction, e.entry_price, e.stop_price)
            r_mult = to_r_multiple(pnl_pts, e.risk_points, spec)
            r_multiples.append(r_mult)
            trade_dates.append(e.trading_day)
            risk_pts_list.append(e.risk_points)
        results.append(
            finalize_result(
                strategy, instrument, f"{variant}_{exit_label}", exit_label, r_multiples, trade_dates, risk_pts_list
            )
        )
    return results


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_instrument_bars(con, instrument):
    print(f"    Loading 1m bars for {instrument}...", end=" ", flush=True)
    t0 = time.time()
    df = con.execute(
        """
        SELECT ts_utc, open, high, low, close, volume,
               CAST((ts_utc AT TIME ZONE 'Australia/Brisbane'
                     - INTERVAL '9 hours') AS DATE) AS trading_day
        FROM bars_1m WHERE symbol = ? ORDER BY ts_utc
    """,
        [instrument],
    ).fetchdf()
    # Ensure .dt.hour extracts UTC hours, not Brisbane hours
    if df["ts_utc"].dt.tz is not None:
        df["ts_utc"] = df["ts_utc"].dt.tz_convert("UTC")
    print(f"{len(df):,} bars in {time.time() - t0:.1f}s")
    return df


def load_daily_features(con, instrument):
    return con.execute(
        """
        SELECT trading_day, atr_20, us_dst, uk_dst,
               daily_open, daily_close, daily_high, daily_low,
               prev_day_high, prev_day_low, prev_day_close
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5 ORDER BY trading_day
    """,
        [instrument],
    ).fetchdf()


def get_session_bars(day_bars, block_name):
    block = SESSION_BLOCKS[block_name]
    hours = day_bars["ts_utc"].dt.hour
    s, e = block["start_utc_h"], block["end_utc_h"]
    if s < e:
        return day_bars[(hours >= s) & (hours < e)]
    return day_bars[(hours >= s) | (hours < e)]


# =============================================================================
# STRATEGY 1: PDH/PDL Sweep Rejection (Mechanism-Refined)
#
# MECHANISM: Stops cluster above PDH / below PDL. Smart money pushes through
# to trigger those stops (liquidity grab), then reverses. The sweep extreme
# is the invalidation point (structural stop).
#
# REFINEMENTS vs V2:
#   - Stop is at the ACTUAL sweep extreme (not ATR-based)
#   - Filter: minimum penetration (must breach by >= min_penetration * ATR)
#   - Filter: fast rejection (close back within max_rej_bars)
#   - R-based exits (risk = entry to sweep extreme)
# =============================================================================
def find_sweep_entries_v3(daily_df, bars_by_day, instrument, block_name, max_rej_bars=5, min_penetration_atr=0.05):
    """Find PDH/PDL sweeps with fast rejection and structural stop."""
    entries = []
    for _, row in daily_df.iterrows():
        pdh = row.get("prev_day_high")
        pdl = row.get("prev_day_low")
        atr = row.get("atr_20")
        trading_day = row["trading_day"]

        if pd.isna(pdh) or pd.isna(pdl) or pd.isna(atr) or atr <= 0:
            continue
        if pdh <= pdl:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None:
            continue

        session_bars = get_session_bars(day_bars, block_name)
        if len(session_bars) < 30:
            continue

        bars_arr = session_bars[["ts_utc", "open", "high", "low", "close"]].values
        min_pen = min_penetration_atr * atr
        trade_taken = False

        for i in range(len(bars_arr) - max_rej_bars - 5):
            if trade_taken:
                break
            _, _, h, l, c = bars_arr[i]

            # --- PDH sweep (short setup) ---
            if h > pdh + min_pen:
                sweep_high = h
                # Track the maximum sweep extent
                for k in range(1, min(max_rej_bars + 1, len(bars_arr) - i)):
                    _, _, hk, _, ck = bars_arr[i + k]
                    sweep_high = max(sweep_high, hk)
                    if ck < pdh:  # Rejection confirmed — close back below PDH
                        entry_price = float(ck)
                        stop_price = float(sweep_high) + 0.01 * atr  # Just above sweep extreme
                        risk = stop_price - entry_price
                        # Skip if risk is too large (>2 ATR = not a tight rejection)
                        if risk > 2.0 * atr or risk < 0.01 * atr:
                            break
                        remaining = session_bars.iloc[i + k + 1 :]
                        if len(remaining) >= 5:
                            entries.append(
                                MechanicalEntry(
                                    direction=-1,
                                    entry_price=entry_price,
                                    stop_price=stop_price,
                                    remaining_bars=remaining,
                                    atr=float(atr),
                                    trading_day=trading_day,
                                    metadata={"sweep_type": "pdh", "rej_bars": k, "penetration": float(h - pdh) / atr},
                                )
                            )
                            trade_taken = True
                        break

            # --- PDL sweep (long setup) ---
            if l < pdl - min_pen and not trade_taken:
                sweep_low = l
                for k in range(1, min(max_rej_bars + 1, len(bars_arr) - i)):
                    _, _, _, lk, ck = bars_arr[i + k]
                    sweep_low = min(sweep_low, lk)
                    if ck > pdl:  # Rejection confirmed — close back above PDL
                        entry_price = float(ck)
                        stop_price = float(sweep_low) - 0.01 * atr
                        risk = entry_price - stop_price
                        if risk > 2.0 * atr or risk < 0.01 * atr:
                            break
                        remaining = session_bars.iloc[i + k + 1 :]
                        if len(remaining) >= 5:
                            entries.append(
                                MechanicalEntry(
                                    direction=1,
                                    entry_price=entry_price,
                                    stop_price=stop_price,
                                    remaining_bars=remaining,
                                    atr=float(atr),
                                    trading_day=trading_day,
                                    metadata={"sweep_type": "pdl", "rej_bars": k, "penetration": float(pdl - l) / atr},
                                )
                            )
                            trade_taken = True
                        break

    return entries


# =============================================================================
# STRATEGY 2: Session Trend Pullback (50% Retracement Entry)
#
# MECHANISM: Session establishes direction in first 15-30 min. Pullbacks to
# 50% of the initial move are high-probability re-entry points because:
#   1. Profit-taking causes temporary reversal
#   2. New participants enter at "discount" relative to the trend
#   3. The session open level acts as support/resistance
#
# STRUCTURAL STOP: Below the pullback low (for longs) / above pullback high
# =============================================================================
def find_pullback_entries(
    daily_df,
    bars_by_day,
    instrument,
    session_open_cfg,
    drive_minutes=15,
    min_move_atr=0.3,
    retrace_pct=0.50,
    max_wait_bars=60,
):
    """Find trend pullback entries at 50% retracement of initial session move."""
    entries = []

    for _, row in daily_df.iterrows():
        atr = row.get("atr_20")
        trading_day = row["trading_day"]
        is_us_dst = row.get("us_dst", False)
        is_uk_dst = row.get("uk_dst", False)

        if pd.isna(atr) or atr <= 0:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None:
            continue

        # Get session open time
        cfg = session_open_cfg
        if cfg["dst_flag"] == "us_dst":
            is_dst = is_us_dst
        else:
            is_dst = is_uk_dst

        if is_dst:
            start_h, start_m = cfg["summer_h"], cfg["summer_m"]
        else:
            start_h, start_m = cfg["winter_h"], cfg["winter_m"]

        # Find bars in the relevant window
        hours = day_bars["ts_utc"].dt.hour
        minutes_utc = hours * 60 + day_bars["ts_utc"].dt.minute
        session_start_min = start_h * 60 + start_m
        # Need drive_minutes + max_wait_bars + 240 bars of exit room
        window_end_min = session_start_min + drive_minutes + max_wait_bars + 250

        if window_end_min > 24 * 60:
            mask = (minutes_utc >= session_start_min) | (minutes_utc < window_end_min - 24 * 60)
        else:
            mask = (minutes_utc >= session_start_min) & (minutes_utc < window_end_min)

        window_bars = day_bars[mask].sort_values("ts_utc")
        if len(window_bars) < drive_minutes + max_wait_bars + 10:
            continue

        # Phase 1: Measure initial drive
        drive_bars = window_bars.iloc[:drive_minutes]
        drive_open = float(drive_bars.iloc[0]["open"])
        drive_high = float(drive_bars["high"].max())
        drive_low = float(drive_bars["low"].min())
        drive_close = float(drive_bars.iloc[-1]["close"])

        drive_move = drive_close - drive_open
        if abs(drive_move) < min_move_atr * atr:
            continue

        direction = 1 if drive_move > 0 else -1

        if direction == 1:
            swing_high = drive_high
            retrace_level = drive_open + drive_move * (1 - retrace_pct)
        else:
            swing_low = drive_low
            retrace_level = drive_open + drive_move * (1 - retrace_pct)

        # Phase 2: Wait for pullback to retrace level
        post_drive = window_bars.iloc[drive_minutes:]
        pullback_extreme = None
        entry_found = False

        for j in range(min(max_wait_bars, len(post_drive))):
            bar = post_drive.iloc[j]
            price = float(bar["close"])

            if direction == 1:
                # Track pullback low
                bar_low = float(bar["low"])
                if pullback_extreme is None or bar_low < pullback_extreme:
                    pullback_extreme = bar_low

                # Check if price pulled back enough
                if price <= retrace_level and pullback_extreme is not None:
                    # Wait for bounce: next bar closes above retrace level
                    if j + 1 < len(post_drive):
                        next_bar = post_drive.iloc[j + 1]
                        if float(next_bar["close"]) > retrace_level:
                            entry_price = float(next_bar["close"])
                            # Stop below pullback low
                            stop_price = pullback_extreme - 0.01 * atr
                            risk = entry_price - stop_price
                            if 0.05 * atr < risk < 2.0 * atr:
                                remaining = post_drive.iloc[j + 2 :]
                                if len(remaining) >= 5:
                                    entries.append(
                                        MechanicalEntry(
                                            direction=1,
                                            entry_price=entry_price,
                                            stop_price=stop_price,
                                            remaining_bars=remaining,
                                            atr=float(atr),
                                            trading_day=trading_day,
                                            metadata={
                                                "drive_move": drive_move / atr,
                                                "retrace_depth": (drive_close - pullback_extreme) / abs(drive_move)
                                                if abs(drive_move) > 0
                                                else 0,
                                            },
                                        )
                                    )
                                    entry_found = True
                            break
            else:
                bar_high = float(bar["high"])
                if pullback_extreme is None or bar_high > pullback_extreme:
                    pullback_extreme = bar_high

                if price >= retrace_level and pullback_extreme is not None:
                    if j + 1 < len(post_drive):
                        next_bar = post_drive.iloc[j + 1]
                        if float(next_bar["close"]) < retrace_level:
                            entry_price = float(next_bar["close"])
                            stop_price = pullback_extreme + 0.01 * atr
                            risk = stop_price - entry_price
                            if 0.05 * atr < risk < 2.0 * atr:
                                remaining = post_drive.iloc[j + 2 :]
                                if len(remaining) >= 5:
                                    entries.append(
                                        MechanicalEntry(
                                            direction=-1,
                                            entry_price=entry_price,
                                            stop_price=stop_price,
                                            remaining_bars=remaining,
                                            atr=float(atr),
                                            trading_day=trading_day,
                                            metadata={
                                                "drive_move": drive_move / atr,
                                                "retrace_depth": (pullback_extreme - drive_close) / abs(drive_move)
                                                if abs(drive_move) > 0
                                                else 0,
                                            },
                                        )
                                    )
                                    entry_found = True
                            break

            if entry_found:
                break

    return entries


# =============================================================================
# STRATEGY 3: VWAP Bounce WITH Trend
#
# MECHANISM: VWAP is the volume-weighted average price — institutional benchmark.
# V2 showed fading AWAY from VWAP loses (markets trend). This flips the logic:
#   - In an uptrend (price above VWAP), VWAP acts as SUPPORT
#   - Buy when price pulls back to touch VWAP and bounces
#   - Stop below the pullback low (structural)
#
# This goes WITH the trend, using VWAP as a support level, not a target.
# =============================================================================
def find_vwap_bounce_entries(daily_df, bars_by_day, instrument, block_name, trend_minutes=30, min_trend_atr=0.2):
    """Find VWAP bounce entries that go WITH the established trend."""
    entries = []

    for _, row in daily_df.iterrows():
        atr = row.get("atr_20")
        trading_day = row["trading_day"]
        if pd.isna(atr) or atr <= 0:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None:
            continue

        session_bars = get_session_bars(day_bars, block_name)
        if len(session_bars) < trend_minutes + 60:
            continue

        vol_bars = session_bars[session_bars["volume"] > 0].copy()
        if len(vol_bars) < trend_minutes + 30:
            continue

        # Compute running VWAP
        prices = vol_bars["close"].values.astype(float)
        volumes = vol_bars["volume"].values.astype(float)
        cum_pv = np.cumsum(prices * volumes)
        cum_vol = np.cumsum(volumes)
        vwap = cum_pv / np.where(cum_vol > 0, cum_vol, 1)

        # Phase 1: Establish trend from first N minutes
        trend_close = prices[trend_minutes - 1]
        trend_open = prices[0]
        trend_move = trend_close - trend_open
        if abs(trend_move) < min_trend_atr * atr:
            continue

        direction = 1 if trend_move > 0 else -1
        trade_taken = False

        # Phase 2: After trend established, find VWAP touch + bounce
        pullback_extreme = None
        touched_vwap = False

        for j in range(trend_minutes, len(vol_bars) - 5):
            if trade_taken:
                break

            bar_close = prices[j]
            bar_high = float(vol_bars.iloc[j]["high"])
            bar_low = float(vol_bars.iloc[j]["low"])
            current_vwap = vwap[j]

            if direction == 1:
                # Uptrend: look for pullback to VWAP
                if pullback_extreme is None or bar_low < pullback_extreme:
                    pullback_extreme = bar_low

                # Touch VWAP (bar low reaches VWAP or closes within 0.1 ATR of VWAP)
                if bar_low <= current_vwap + 0.1 * atr:
                    touched_vwap = True

                # Bounce: after touching VWAP, close back above VWAP
                if touched_vwap and bar_close > current_vwap + 0.05 * atr:
                    entry_price = bar_close
                    stop_price = min(pullback_extreme, current_vwap) - 0.1 * atr
                    risk = entry_price - stop_price
                    if 0.05 * atr < risk < 2.0 * atr:
                        remaining = vol_bars.iloc[j + 1 :]
                        if len(remaining) >= 5:
                            entries.append(
                                MechanicalEntry(
                                    direction=1,
                                    entry_price=float(entry_price),
                                    stop_price=float(stop_price),
                                    remaining_bars=remaining,
                                    atr=float(atr),
                                    trading_day=trading_day,
                                    metadata={
                                        "trend_strength": trend_move / atr,
                                        "vwap_distance": (entry_price - current_vwap) / atr,
                                    },
                                )
                            )
                            trade_taken = True
            else:
                # Downtrend: look for pullback up to VWAP
                if pullback_extreme is None or bar_high > pullback_extreme:
                    pullback_extreme = bar_high

                if bar_high >= current_vwap - 0.1 * atr:
                    touched_vwap = True

                if touched_vwap and bar_close < current_vwap - 0.05 * atr:
                    entry_price = bar_close
                    stop_price = max(pullback_extreme, current_vwap) + 0.1 * atr
                    risk = stop_price - entry_price
                    if 0.05 * atr < risk < 2.0 * atr:
                        remaining = vol_bars.iloc[j + 1 :]
                        if len(remaining) >= 5:
                            entries.append(
                                MechanicalEntry(
                                    direction=-1,
                                    entry_price=float(entry_price),
                                    stop_price=float(stop_price),
                                    remaining_bars=remaining,
                                    atr=float(atr),
                                    trading_day=trading_day,
                                    metadata={
                                        "trend_strength": trend_move / atr,
                                        "vwap_distance": (current_vwap - entry_price) / atr,
                                    },
                                )
                            )
                            trade_taken = True

    return entries


# =============================================================================
# Main Runner
# =============================================================================
SESSION_OPEN_CONFIGS = {
    "us_open": {"winter_h": 14, "winter_m": 30, "summer_h": 13, "summer_m": 30, "dst_flag": "us_dst"},
    "london_open": {"winter_h": 8, "winter_m": 0, "summer_h": 7, "summer_m": 0, "dst_flag": "uk_dst"},
}


def run_all(db_path, strategy_filter=None, instrument_filter=None):
    results = []
    con = duckdb.connect(db_path, read_only=True)

    instruments = INSTRUMENTS
    if instrument_filter:
        instruments = [instrument_filter]

    try:
        for instrument in instruments:
            spec = COST_SPECS.get(instrument)
            if spec is None:
                print(f"  No cost model for {instrument}, skipping")
                continue

            print(f"\n{'=' * 70}")
            print(f"  INSTRUMENT: {instrument}")
            print(f"{'=' * 70}")

            bars_df = load_instrument_bars(con, instrument)
            if len(bars_df) < 10000:
                print(f"    Insufficient bars ({len(bars_df)}), skipping")
                continue

            daily_df = load_daily_features(con, instrument)
            if len(daily_df) < 200:
                print(f"    Insufficient daily features ({len(daily_df)}), skipping")
                continue

            bars_by_day = {day: grp for day, grp in bars_df.groupby("trading_day")}

            # =========================================================
            # 1. PDH/PDL Sweep (Mechanism-Based)
            # =========================================================
            if not strategy_filter or strategy_filter == "sweep":
                print(f"\n  --- PDH/PDL Sweep Rejection ({instrument}) ---")
                for block in SESSION_BLOCKS:
                    for max_rej in [3, 5, 10]:
                        entries = find_sweep_entries_v3(daily_df, bars_by_day, instrument, block, max_rej_bars=max_rej)
                        if not entries:
                            continue
                        variant = f"{block}_rej{max_rej}"
                        for r in run_mechanical_exits(entries, "sweep_rejection", instrument, variant, spec):
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {r.variant:40s} "
                                    f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                    f"AvgR={r.avg_r:+.4f} p={r.p_value:.4f} "
                                    f"risk={r.avg_risk_pts:.2f}pts"
                                )

            # =========================================================
            # 2. Session Trend Pullback
            # =========================================================
            if not strategy_filter or strategy_filter == "pullback":
                print(f"\n  --- Session Trend Pullback ({instrument}) ---")
                for sess_name, sess_cfg in SESSION_OPEN_CONFIGS.items():
                    for drive_min in [15, 30]:
                        for min_move in [0.05, 0.08]:
                            entries = find_pullback_entries(
                                daily_df,
                                bars_by_day,
                                instrument,
                                sess_cfg,
                                drive_minutes=drive_min,
                                min_move_atr=min_move,
                            )
                            if not entries:
                                continue
                            variant = f"{sess_name}_drv{drive_min}_mv{min_move}"
                            for r in run_mechanical_exits(entries, "trend_pullback", instrument, variant, spec):
                                results.append(r)
                                if r.n_trades >= MIN_TRADES:
                                    sig = "*" if r.p_value < 0.05 else " "
                                    print(
                                        f"    {sig} {r.variant:40s} "
                                        f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                        f"AvgR={r.avg_r:+.4f} p={r.p_value:.4f} "
                                        f"risk={r.avg_risk_pts:.2f}pts"
                                    )

            # =========================================================
            # 3. VWAP Bounce WITH Trend
            # =========================================================
            if not strategy_filter or strategy_filter == "vwap":
                print(f"\n  --- VWAP Bounce WITH Trend ({instrument}) ---")
                for block in SESSION_BLOCKS:
                    for trend_min in [20, 30]:
                        for min_trend in [0.03, 0.06]:
                            entries = find_vwap_bounce_entries(
                                daily_df,
                                bars_by_day,
                                instrument,
                                block,
                                trend_minutes=trend_min,
                                min_trend_atr=min_trend,
                            )
                            if not entries:
                                continue
                            variant = f"{block}_trnd{trend_min}_mv{min_trend}"
                            for r in run_mechanical_exits(entries, "vwap_bounce", instrument, variant, spec):
                                results.append(r)
                                if r.n_trades >= MIN_TRADES:
                                    sig = "*" if r.p_value < 0.05 else " "
                                    print(
                                        f"    {sig} {r.variant:40s} "
                                        f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                        f"AvgR={r.avg_r:+.4f} p={r.p_value:.4f} "
                                        f"risk={r.avg_risk_pts:.2f}pts"
                                    )

            del bars_df, bars_by_day, daily_df

        # =================================================================
        # FDR CORRECTION
        # =================================================================
        print("\n" + "=" * 70)
        print("BH FDR CORRECTION (V3 Mechanism-Based)")
        print("=" * 70)

        valid_results = [r for r in results if r.n_trades >= MIN_TRADES]
        fdr_results = []
        if valid_results:
            p_values = [r.p_value for r in valid_results]
            fdr_results = bh_fdr(p_values, alpha=FDR_ALPHA)
            n_tested = len(valid_results)
            n_sig = sum(1 for _, _, sig in fdr_results if sig)
            print(f"\nTests with N >= {MIN_TRADES}: {n_tested}")
            print(f"FDR-significant at alpha={FDR_ALPHA}: {n_sig}")

            if n_sig > 0:
                print(f"\n*** FDR SURVIVORS ***")
                print(
                    f"{'Strategy':<20s} {'Inst':<6s} {'Variant':<42s} "
                    f"{'Exit':>4s} {'N':>5s} {'WR':>7s} {'AvgR':>8s} "
                    f"{'Sharpe':>7s} {'raw_p':>8s} {'adj_p':>8s}"
                )
                print("-" * 140)
                for idx, adj_p, sig in fdr_results:
                    if sig:
                        r = valid_results[idx]
                        print(
                            f"{r.strategy:<20s} {r.instrument:<6s} "
                            f"{r.variant:<42s} {r.exit_type:>4s} "
                            f"{r.n_trades:5d} {r.win_rate:7.2%} "
                            f"{r.avg_r:+8.4f} {r.sharpe:7.3f} "
                            f"{r.p_value:8.4f} {adj_p:8.4f}"
                        )
                        # Year by year
                        for yr, yd in sorted(r.yearly_results.items()):
                            print(f"         {yr}: N={yd['n']:3d} AvgR={yd['avg_r']:+.4f} WR={yd['wr']:.2%}")
            else:
                print("\nNo strategies survived FDR correction.")

            # Top 15 by raw p-value
            print(f"\n  TOP 15 BY RAW P-VALUE:")
            sorted_by_p = sorted(enumerate(valid_results), key=lambda x: x[1].p_value)
            for rank, (idx, r) in enumerate(sorted_by_p[:15]):
                adj_p = fdr_results[idx][1]
                print(
                    f"    {rank + 1:2d}. {r.strategy:18s} {r.instrument:6s} "
                    f"{r.variant:40s} N={r.n_trades:4d} AvgR={r.avg_r:+.4f} "
                    f"raw_p={r.p_value:.4f} adj_p={adj_p:.4f}"
                )

            # Summary per strategy
            print(f"\n  STRATEGY SUMMARY:")
            for strat in ["sweep_rejection", "trend_pullback", "vwap_bounce"]:
                strat_valid = [r for r in valid_results if r.strategy == strat]
                if not strat_valid:
                    continue
                n_pos = sum(1 for r in strat_valid if r.avg_r > 0)
                avg_r = np.mean([r.avg_r for r in strat_valid])
                best = min(strat_valid, key=lambda r: r.p_value)
                n_fdr = sum(
                    1 for i, (_, _, sig) in enumerate(fdr_results) if sig and valid_results[i].strategy == strat
                )
                print(f"\n    {strat} ({len(strat_valid)} valid, {n_pos} positive, {n_fdr} FDR):")
                print(f"      Average AvgR: {avg_r:+.4f}")
                print(
                    f"      Best: {best.instrument} {best.variant} "
                    f"N={best.n_trades} AvgR={best.avg_r:+.4f} p={best.p_value:.4f}"
                )

                # Best per exit type
                for exit_type in ["R1", "R2", "R3", "TR1"]:
                    et_results = [r for r in strat_valid if r.exit_type == exit_type]
                    if not et_results:
                        continue
                    et_best = min(et_results, key=lambda r: r.p_value)
                    et_pos = sum(1 for r in et_results if r.avg_r > 0)
                    print(
                        f"        {exit_type}: {et_pos}/{len(et_results)} positive, "
                        f"best p={et_best.p_value:.4f} ({et_best.variant})"
                    )
        else:
            print(f"\nNo strategies had N >= {MIN_TRADES}")

        # =================================================================
        # HONEST SUMMARY
        # =================================================================
        print("\n" + "=" * 70)
        print("HONEST SUMMARY - V3 Mechanism-Based Strategies")
        print("=" * 70)
        print("\nPhilosophy: Structural stops from price patterns.")
        print("R-multiples: risk = entry to structural invalidation point.")
        print("Friction applied: increases risk, reduces reward.\n")

        survived = []
        if fdr_results:
            survived = [valid_results[i] for i, (_, _, sig) in enumerate(fdr_results) if sig]

        print("SURVIVED SCRUTINY:")
        if survived:
            for r in survived:
                print(
                    f"  + {r.strategy} | {r.instrument} | {r.variant} | "
                    f"N={r.n_trades}, AvgR={r.avg_r:+.4f}, p={r.p_value:.4f}"
                )
        else:
            print("  None.")

        print("\nDID NOT SURVIVE:")
        for strat in ["sweep_rejection", "trend_pullback", "vwap_bounce"]:
            strat_valid = [r for r in valid_results if r.strategy == strat]
            if not strat_valid:
                print(f"  - {strat}: no valid tests (insufficient triggers)")
                continue
            best = min(strat_valid, key=lambda r: r.p_value)
            avg_r = np.mean([r.avg_r for r in strat_valid])
            n_pos = sum(1 for r in strat_valid if r.avg_r > 0)
            if not any(r in survived for r in strat_valid):
                print(f"  - {strat}: {n_pos}/{len(strat_valid)} positive, avg={avg_r:+.4f}, best p={best.p_value:.4f}")

        print("\nKEY DIAGNOSTICS:")
        # Risk profile
        all_risks = []
        for r in valid_results:
            if r.avg_risk_pts > 0:
                all_risks.append((r.strategy, r.instrument, r.avg_risk_pts, r.median_risk_pts))
        if all_risks:
            print(f"  Average structural risk: {np.mean([x[2] for x in all_risks]):.2f} pts")
            print(f"  Median structural risk: {np.median([x[3] for x in all_risks]):.2f} pts")

    finally:
        con.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Non-ORB V3 Mechanism-Based Strategy Research")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "gold.db")
    parser.add_argument("--strategy", type=str, default=None, choices=["sweep", "pullback", "vwap"])
    parser.add_argument("--instrument", type=str, default=None, choices=INSTRUMENTS + ["MGC"])
    args = parser.parse_args()

    print("=" * 70)
    print("Non-ORB Strategy Research - V3: MECHANISM-DRIVEN")
    print("=" * 70)
    print(f"Database: {args.db_path}")
    print(f"Strategy: {args.strategy or 'ALL'}")
    print(f"Instrument: {args.instrument or 'ALL (MES, MNQ, M2K)'}")
    print(f"Date: {date.today()}")
    print()
    print("LOGIC:")
    print("  These markets TREND intraday (proven by V1/V2).")
    print("  V3 finds entries WITH the trend at structurally-defined levels.")
    print("  Stops come from the MECHANISM, not parameter grids.")
    print("  Exits: R1, R2, R3 targets + trailing stop (4 per signal, not 13).")
    print()

    t0 = time.time()
    results = run_all(str(args.db_path), args.strategy, args.instrument)
    elapsed = time.time() - t0

    total = len(results)
    valid = sum(1 for r in results if r.n_trades >= MIN_TRADES)
    print(f"\nTotal tests: {total}, Valid (N>={MIN_TRADES}): {valid}")
    print(f"Runtime: {elapsed:.0f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Non-ORB Strategy Research — V2: Multi-Exit & Alternative Structures.

V1 applied ORB-style fixed stop/target exits to mean-reversion strategies,
which is wrong — tight stops systematically exit before reversion completes.

V2 fixes this with:
  - Multiple exit styles: time exit, trailing stop, ATR-based SL/TP
  - ATR-normalized PnL (not stop-based R-multiples)
  - Prior-day level sweep rejection (order flow, not ORB)
  - Opening drive fade (fade initial directional move)
  - Each strategy tests ALL exit styles to find the right reward structure

Strategies:
  1. VWAP Reversion (signal-based entry)
  2. Failed Breakout Fade (ORB re-entry)
  3. PDH/PDL Sweep Rejection (order flow)
  4. Opening Drive Fade (session-open overreaction)

Exit Styles:
  - Time exit: hold 30/60/90/120 min, exit at market
  - Trailing stop: trail 0.5/1.0/1.5 ATR from peak, max 120 min
  - Wide SL/TP: SL=1.0-2.0 ATR, TP=1.0-4.0 ATR (various R:R ratios)

Usage:
    python research/research_non_orb_v2.py --db-path gold.db
    python research/research_non_orb_v2.py --db-path gold.db --instrument MES
    python research/research_non_orb_v2.py --db-path gold.db --archetype vwap
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
ALL_INSTRUMENTS = ["MGC", "MES", "MNQ", "M2K", "MCL", "M6E", "SIL"]
MIN_TRADES = 30
FDR_ALPHA = 0.05

# Exit configurations: each strategy tests ALL of these
EXIT_CONFIGS = [
    # --- Pure time exits (no risk management) ---
    {"mode": "time", "hold": 30, "label": "T30"},
    {"mode": "time", "hold": 60, "label": "T60"},
    {"mode": "time", "hold": 90, "label": "T90"},
    {"mode": "time", "hold": 120, "label": "T120"},
    # --- Trailing stops (lock in profit, trail from peak) ---
    {"mode": "trail", "trail_atr": 0.5, "max_hold": 120, "label": "TR0.5"},
    {"mode": "trail", "trail_atr": 1.0, "max_hold": 120, "label": "TR1.0"},
    {"mode": "trail", "trail_atr": 1.5, "max_hold": 120, "label": "TR1.5"},
    # --- ATR-based SL/TP (wider than ORB style, various R:R) ---
    {"mode": "sl_tp", "sl_atr": 1.0, "tp_atr": 1.0, "max_hold": 240, "label": "SL1-TP1"},
    {"mode": "sl_tp", "sl_atr": 1.0, "tp_atr": 2.0, "max_hold": 240, "label": "SL1-TP2"},
    {"mode": "sl_tp", "sl_atr": 1.5, "tp_atr": 1.5, "max_hold": 240, "label": "SL1.5-TP1.5"},
    {"mode": "sl_tp", "sl_atr": 1.5, "tp_atr": 3.0, "max_hold": 240, "label": "SL1.5-TP3"},
    {"mode": "sl_tp", "sl_atr": 2.0, "tp_atr": 2.0, "max_hold": 240, "label": "SL2-TP2"},
    {"mode": "sl_tp", "sl_atr": 2.0, "tp_atr": 4.0, "max_hold": 240, "label": "SL2-TP4"},
]

FADE_ORB_LABELS = ["1000", "1800", "0030", "US_EQUITY_OPEN"]

SESSION_BLOCKS = {
    "asia": {"start_utc_h": 0, "end_utc_h": 8},
    "london": {"start_utc_h": 8, "end_utc_h": 13},
    "us": {"start_utc_h": 13, "end_utc_h": 19},
}

DRIVE_SESSIONS = {
    "us_open": {
        "winter_start_h": 14,
        "winter_start_m": 30,
        "summer_start_h": 13,
        "summer_start_m": 30,
        "dst_flag": "us_dst",
    },
    "london_open": {
        "winter_start_h": 8,
        "winter_start_m": 0,
        "summer_start_h": 7,
        "summer_start_m": 0,
        "dst_flag": "uk_dst",
    },
}

MAX_EXIT_BARS = 240


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class StrategyResult:
    archetype: str
    instrument: str
    variant: str
    params: dict
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    avg_pnl_r: float = 0.0
    total_pnl_r: float = 0.0
    sharpe: float = 0.0
    max_dd_r: float = 0.0
    p_value: float = 1.0
    yearly_results: dict = field(default_factory=dict)
    pnl_series: list = field(default_factory=list)
    trade_dates: list = field(default_factory=list)

    @property
    def classification(self) -> str:
        if self.n_trades < 30:
            return "INVALID"
        elif self.n_trades < 100:
            return "REGIME"
        elif self.n_trades < 200:
            return "PRELIMINARY"
        return "CORE"


@dataclass
class EntrySignal:
    """A detected trade entry, independent of exit style."""

    direction: int
    entry_price: float
    remaining_bars: pd.DataFrame
    atr: float
    trading_day: object


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


def atr_normalize(pnl_points: float, atr: float, spec: CostSpec) -> float:
    """PnL in ATR units after friction."""
    if atr <= 0:
        return 0.0
    return (pnl_points - spec.friction_in_points) / atr


def finalize_result(
    archetype: str,
    instrument: str,
    variant: str,
    params: dict,
    trades: list[float],
    trade_dates: list,
) -> StrategyResult:
    n = len(trades)
    if n < 1:
        return StrategyResult(archetype=archetype, instrument=instrument, variant=variant, params=params)
    arr = np.array(trades)
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
        archetype=archetype,
        instrument=instrument,
        variant=variant,
        params=params,
        n_trades=n,
        n_wins=wins,
        win_rate=round(wins / n, 4),
        avg_pnl_r=round(avg_r, 4),
        total_pnl_r=round(total_r, 2),
        sharpe=round(sharpe, 4),
        max_dd_r=compute_max_dd(trades),
        p_value=round(p_val, 6),
        yearly_results=yearly_breakdown(trade_dates, trades),
        pnl_series=trades,
        trade_dates=trade_dates,
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


def compute_orb_correlation(con, result, instrument):
    if not result.trade_dates or not result.pnl_series:
        return None
    orb_df = con.execute(
        """
        SELECT o.trading_day, o.pnl_r FROM orb_outcomes o
        WHERE o.symbol = ? AND o.orb_label = '1000'
          AND o.entry_model = 'E0' AND o.confirm_bars = 1
          AND o.rr_target = 1.0 AND o.orb_minutes = 5
          AND o.outcome IN ('WIN', 'LOSS')
        ORDER BY o.trading_day
    """,
        [instrument],
    ).fetchdf()
    if orb_df.empty:
        orb_df = con.execute(
            """
            SELECT o.trading_day, o.pnl_r FROM orb_outcomes o
            WHERE o.symbol = ? AND o.orb_label = '0900'
              AND o.entry_model = 'E1' AND o.confirm_bars = 1
              AND o.rr_target = 1.0 AND o.orb_minutes = 5
              AND o.outcome IN ('WIN', 'LOSS')
            ORDER BY o.trading_day
        """,
            [instrument],
        ).fetchdf()
    if orb_df.empty:
        return None
    strat_df = pd.DataFrame({"trading_day": result.trade_dates, "strat_pnl": result.pnl_series})
    strat_df["trading_day"] = pd.to_datetime(strat_df["trading_day"])
    orb_df["trading_day"] = pd.to_datetime(orb_df["trading_day"])
    merged = strat_df.merge(orb_df, on="trading_day", how="inner")
    if len(merged) < 20:
        return None
    corr = merged["strat_pnl"].corr(merged["pnl_r"])
    return round(float(corr), 4) if not pd.isna(corr) else None


# ---------------------------------------------------------------------------
# Exit Simulators
# ---------------------------------------------------------------------------
def simulate_time_exit(bars_after, direction, entry_price, hold_minutes):
    """Hold for N bars, exit at bar close."""
    if len(bars_after) == 0:
        return 0.0
    exit_idx = min(hold_minutes - 1, len(bars_after) - 1)
    return (float(bars_after.iloc[exit_idx]["close"]) - entry_price) * direction


def simulate_trailing_stop(bars_after, direction, entry_price, atr, trail_atr, max_hold):
    """Trail from peak favorable price by trail_atr * ATR."""
    if len(bars_after) == 0 or atr <= 0:
        return 0.0
    trail_dist = trail_atr * atr
    best_price = entry_price
    n_bars = min(max_hold, len(bars_after))
    for i in range(n_bars):
        bar = bars_after.iloc[i]
        if direction == 1:
            best_price = max(best_price, float(bar["high"]))
            stop = best_price - trail_dist
            if float(bar["low"]) <= stop:
                return (stop - entry_price) * direction
        else:
            best_price = min(best_price, float(bar["low"]))
            stop = best_price + trail_dist
            if float(bar["high"]) >= stop:
                return (stop - entry_price) * direction
    return (float(bars_after.iloc[n_bars - 1]["close"]) - entry_price) * direction


def simulate_wide_sl_tp(bars_after, direction, entry_price, atr, sl_atr, tp_atr, max_hold=240):
    """ATR-based stop loss and take profit."""
    if len(bars_after) == 0 or atr <= 0:
        return 0.0
    sl_dist = sl_atr * atr
    tp_dist = tp_atr * atr
    if direction == 1:
        stop = entry_price - sl_dist
        target = entry_price + tp_dist
    else:
        stop = entry_price + sl_dist
        target = entry_price - tp_dist
    n_bars = min(max_hold, len(bars_after))
    for i in range(n_bars):
        bar = bars_after.iloc[i]
        if direction == 1:
            if float(bar["low"]) <= stop:
                return -sl_dist
            if float(bar["high"]) >= target:
                return tp_dist
        else:
            if float(bar["high"]) >= stop:
                return -sl_dist
            if float(bar["low"]) <= target:
                return tp_dist
    return (float(bars_after.iloc[n_bars - 1]["close"]) - entry_price) * direction


def apply_exit(bars_after, direction, entry_price, atr, exit_config):
    """Dispatch to the right exit simulator."""
    mode = exit_config["mode"]
    if mode == "time":
        return simulate_time_exit(bars_after, direction, entry_price, exit_config["hold"])
    elif mode == "trail":
        return simulate_trailing_stop(
            bars_after, direction, entry_price, atr, exit_config["trail_atr"], exit_config["max_hold"]
        )
    elif mode == "sl_tp":
        return simulate_wide_sl_tp(
            bars_after,
            direction,
            entry_price,
            atr,
            exit_config["sl_atr"],
            exit_config["tp_atr"],
            exit_config.get("max_hold", 240),
        )
    return 0.0


# ---------------------------------------------------------------------------
# Multi-Exit Runner
# ---------------------------------------------------------------------------
def run_all_exits(
    entries: list[EntrySignal],
    archetype: str,
    instrument: str,
    base_variant: str,
    base_params: dict,
    spec: CostSpec,
) -> list[StrategyResult]:
    """Given entry signals, simulate ALL exit configs."""
    results = []
    for ec in EXIT_CONFIGS:
        trades, trade_dates = [], []
        for e in entries:
            pnl_pts = apply_exit(e.remaining_bars, e.direction, e.entry_price, e.atr, ec)
            pnl_r = atr_normalize(pnl_pts, e.atr, spec)
            trades.append(pnl_r)
            trade_dates.append(e.trading_day)
        variant = f"{base_variant}_{ec['label']}"
        params = {**base_params, "exit": ec}
        results.append(finalize_result(archetype, instrument, variant, params, trades, trade_dates))
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
    print(f"{len(df):,} bars in {time.time() - t0:.1f}s")
    return df


def load_daily_features(con, instrument):
    cols = [
        "trading_day",
        "atr_20",
        "us_dst",
        "uk_dst",
        "daily_open",
        "daily_close",
        "daily_high",
        "daily_low",
        "prev_day_high",
        "prev_day_low",
        "prev_day_close",
    ]
    for label in FADE_ORB_LABELS:
        cols.extend(
            [
                f"orb_{label}_high",
                f"orb_{label}_low",
                f"orb_{label}_size",
                f"orb_{label}_break_dir",
                f"orb_{label}_break_ts",
            ]
        )
    col_str = ", ".join(cols)
    return con.execute(
        f"""
        SELECT {col_str} FROM daily_features
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
# STRATEGY 1: VWAP Reversion
# =============================================================================
def find_vwap_entries(daily_df, bars_by_day, instrument, block_name, sigma_mult):
    min_bars = 60
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
        if len(session_bars) < min_bars + 30:
            continue

        vol_bars = session_bars[session_bars["volume"] > 0].copy()
        if len(vol_bars) < min_bars:
            continue

        cum_pv = (vol_bars["close"].values * vol_bars["volume"].values).cumsum()
        cum_vol = vol_bars["volume"].values.cumsum()
        vwap = cum_pv / cum_vol
        deviation = vol_bars["close"].values - vwap
        dev_std = pd.Series(deviation).expanding(min_periods=min_bars).std().values

        for j in range(min_bars, len(vol_bars)):
            if dev_std[j] <= 0 or np.isnan(dev_std[j]):
                continue
            if abs(deviation[j]) / dev_std[j] < sigma_mult:
                continue

            entry_price = float(vol_bars.iloc[j]["close"])
            entry_ts = vol_bars.iloc[j]["ts_utc"]
            direction = -1 if entry_price > vwap[j] else 1

            remaining = session_bars[session_bars["ts_utc"] > entry_ts]
            if len(remaining) < 5:
                continue

            entries.append(
                EntrySignal(
                    direction=direction,
                    entry_price=entry_price,
                    remaining_bars=remaining,
                    atr=float(atr),
                    trading_day=trading_day,
                )
            )
            break
    return entries


# =============================================================================
# STRATEGY 2: Failed Breakout Fade
# =============================================================================
def find_fade_entries(daily_df, bars_by_day, instrument, orb_label, n_return_bars):
    h_col = f"orb_{orb_label}_high"
    l_col = f"orb_{orb_label}_low"
    dir_col = f"orb_{orb_label}_break_dir"
    ts_col = f"orb_{orb_label}_break_ts"

    entries = []
    for _, row in daily_df.iterrows():
        break_dir = row.get(dir_col)
        break_ts = row.get(ts_col)
        orb_high = row.get(h_col)
        orb_low = row.get(l_col)
        atr = row.get("atr_20")
        trading_day = row["trading_day"]

        if pd.isna(break_dir) or pd.isna(break_ts):
            continue
        if pd.isna(orb_high) or pd.isna(orb_low) or orb_high <= orb_low:
            continue
        if pd.isna(atr) or atr <= 0:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None:
            continue

        post_break = day_bars[day_bars["ts_utc"] > break_ts]
        if len(post_break) < n_return_bars + 5:
            continue

        scan = post_break.iloc[:n_return_bars]
        entry_idx = None

        if break_dir == "long":
            for j in range(len(scan)):
                if float(scan.iloc[j]["close"]) < orb_high:
                    entry_idx = j
                    break
        elif break_dir == "short":
            for j in range(len(scan)):
                if float(scan.iloc[j]["close"]) > orb_low:
                    entry_idx = j
                    break

        if entry_idx is None:
            continue

        entry_price = float(scan.iloc[entry_idx]["close"])
        direction = -1 if break_dir == "long" else 1

        remaining = post_break.iloc[entry_idx + 1 :]
        if len(remaining) < 5:
            continue

        entries.append(
            EntrySignal(
                direction=direction,
                entry_price=entry_price,
                remaining_bars=remaining,
                atr=float(atr),
                trading_day=trading_day,
            )
        )
    return entries


# =============================================================================
# STRATEGY 3: PDH/PDL Sweep Rejection
# =============================================================================
def find_sweep_entries(daily_df, bars_by_day, instrument, block_name, n_sweep_bars):
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
        trade_taken = False

        for i in range(len(bars_arr) - n_sweep_bars - 5):
            ts, o, h, l, c = bars_arr[i]

            if h > pdh and not trade_taken:
                for k in range(1, min(n_sweep_bars + 1, len(bars_arr) - i)):
                    rej_ts, _, _, _, rej_c = bars_arr[i + k]
                    if rej_c < pdh:
                        remaining = session_bars.iloc[i + k + 1 :]
                        if len(remaining) >= 5:
                            entries.append(
                                EntrySignal(
                                    direction=-1,
                                    entry_price=float(rej_c),
                                    remaining_bars=remaining,
                                    atr=float(atr),
                                    trading_day=trading_day,
                                )
                            )
                            trade_taken = True
                        break

            if l < pdl and not trade_taken:
                for k in range(1, min(n_sweep_bars + 1, len(bars_arr) - i)):
                    rej_ts, _, _, _, rej_c = bars_arr[i + k]
                    if rej_c > pdl:
                        remaining = session_bars.iloc[i + k + 1 :]
                        if len(remaining) >= 5:
                            entries.append(
                                EntrySignal(
                                    direction=1,
                                    entry_price=float(rej_c),
                                    remaining_bars=remaining,
                                    atr=float(atr),
                                    trading_day=trading_day,
                                )
                            )
                            trade_taken = True
                        break

            if trade_taken:
                break
    return entries


# =============================================================================
# STRATEGY 4: Opening Drive Fade
# =============================================================================
def find_drive_entries(daily_df, bars_by_day, instrument, session_name, drive_minutes, threshold_atr):
    sess_cfg = DRIVE_SESSIONS[session_name]
    dst_col = sess_cfg["dst_flag"]
    entries = []

    for _, row in daily_df.iterrows():
        atr = row.get("atr_20")
        trading_day = row["trading_day"]
        is_dst = row.get(dst_col, False)
        if pd.isna(atr) or atr <= 0:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None:
            continue

        if is_dst:
            start_h = sess_cfg["summer_start_h"]
            start_m = sess_cfg["summer_start_m"]
        else:
            start_h = sess_cfg["winter_start_h"]
            start_m = sess_cfg["winter_start_m"]

        hours = day_bars["ts_utc"].dt.hour
        minutes = day_bars["ts_utc"].dt.minute
        bar_minutes_utc = hours * 60 + minutes
        session_start_min = start_h * 60 + start_m

        window_end_min = session_start_min + drive_minutes + MAX_EXIT_BARS + 30
        if window_end_min > 24 * 60:
            mask = (bar_minutes_utc >= session_start_min) | (bar_minutes_utc < window_end_min - 24 * 60)
        else:
            mask = (bar_minutes_utc >= session_start_min) & (bar_minutes_utc < window_end_min)

        window_bars = day_bars[mask].sort_values("ts_utc")
        if len(window_bars) < drive_minutes + 10:
            continue

        drive_bars = window_bars.iloc[:drive_minutes]
        drive_open = float(drive_bars.iloc[0]["open"])
        drive_close = float(drive_bars.iloc[-1]["close"])
        drive_move = drive_close - drive_open

        if abs(drive_move) < threshold_atr * atr:
            continue

        entry_price = drive_close
        direction = -1 if drive_move > 0 else 1

        remaining = window_bars.iloc[drive_minutes:]
        if len(remaining) < 5:
            continue

        entries.append(
            EntrySignal(
                direction=direction,
                entry_price=entry_price,
                remaining_bars=remaining,
                atr=float(atr),
                trading_day=trading_day,
            )
        )
    return entries


# =============================================================================
# Main Runner
# =============================================================================
def run_all(db_path, archetype_filter=None, instrument_filter=None):
    results = []
    con = duckdb.connect(db_path, read_only=True)

    instruments = ALL_INSTRUMENTS
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

            # =============================================================
            # 1. VWAP Reversion
            # =============================================================
            if not archetype_filter or archetype_filter == "vwap":
                print(f"\n  --- VWAP Reversion ({instrument}) ---")
                for block in SESSION_BLOCKS:
                    for sigma in [1.5, 2.0, 2.5, 3.0]:
                        entries = find_vwap_entries(daily_df, bars_by_day, instrument, block, sigma)
                        if not entries:
                            continue
                        base_var = f"{block}_sig{sigma}"
                        base_p = {"block": block, "sigma_mult": sigma}
                        for r in run_all_exits(entries, "vwap_reversion", instrument, base_var, base_p, spec):
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {r.variant:45s} "
                                    f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                    f"ExpR={r.avg_pnl_r:+.4f} p={r.p_value:.4f}"
                                )

            # =============================================================
            # 2. Failed Breakout Fade
            # =============================================================
            if not archetype_filter or archetype_filter == "fade":
                print(f"\n  --- Failed Breakout Fade ({instrument}) ---")
                for label in FADE_ORB_LABELS:
                    for n_ret in [10, 20]:
                        entries = find_fade_entries(daily_df, bars_by_day, instrument, label, n_ret)
                        if not entries:
                            continue
                        base_var = f"{label}_N{n_ret}"
                        base_p = {"orb_label": label, "n_return_bars": n_ret}
                        for r in run_all_exits(entries, "fade_breakout", instrument, base_var, base_p, spec):
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {r.variant:45s} "
                                    f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                    f"ExpR={r.avg_pnl_r:+.4f} p={r.p_value:.4f}"
                                )

            # =============================================================
            # 3. PDH/PDL Sweep Rejection
            # =============================================================
            if not archetype_filter or archetype_filter == "sweep":
                print(f"\n  --- PDH/PDL Sweep Rejection ({instrument}) ---")
                for block in SESSION_BLOCKS:
                    for n_sweep in [5, 10]:
                        entries = find_sweep_entries(daily_df, bars_by_day, instrument, block, n_sweep)
                        if not entries:
                            continue
                        base_var = f"{block}_sw{n_sweep}"
                        base_p = {"block": block, "n_sweep_bars": n_sweep}
                        for r in run_all_exits(entries, "pdh_pdl_sweep", instrument, base_var, base_p, spec):
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {r.variant:45s} "
                                    f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                    f"ExpR={r.avg_pnl_r:+.4f} p={r.p_value:.4f}"
                                )

            # =============================================================
            # 4. Opening Drive Fade
            # =============================================================
            if not archetype_filter or archetype_filter == "drive":
                print(f"\n  --- Opening Drive Fade ({instrument}) ---")
                for sess in DRIVE_SESSIONS:
                    for drv_min in [10, 15]:
                        for threshold in [0.3, 0.5]:
                            entries = find_drive_entries(daily_df, bars_by_day, instrument, sess, drv_min, threshold)
                            if not entries:
                                continue
                            base_var = f"{sess}_drv{drv_min}_thr{threshold}"
                            base_p = {"session": sess, "drive_minutes": drv_min, "threshold_atr": threshold}
                            for r in run_all_exits(entries, "opening_drive_fade", instrument, base_var, base_p, spec):
                                results.append(r)
                                if r.n_trades >= MIN_TRADES:
                                    sig = "*" if r.p_value < 0.05 else " "
                                    print(
                                        f"    {sig} {r.variant:45s} "
                                        f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                        f"ExpR={r.avg_pnl_r:+.4f} p={r.p_value:.4f}"
                                    )

            del bars_df, bars_by_day, daily_df

        # =================================================================
        # FDR CORRECTION
        # =================================================================
        print("\n" + "=" * 70)
        print("BH FDR CORRECTION (V2 Multi-Exit)")
        print("=" * 70)

        valid_results = [r for r in results if r.n_trades >= MIN_TRADES]
        fdr_results = []
        if valid_results:
            p_values = [r.p_value for r in valid_results]
            fdr_results = bh_fdr(p_values, alpha=FDR_ALPHA)
            n_tested = len(valid_results)
            n_sig = sum(1 for _, _, sig in fdr_results if sig)
            n_time = sum(1 for r in valid_results if r.params.get("exit", {}).get("mode") == "time")
            n_trail = sum(1 for r in valid_results if r.params.get("exit", {}).get("mode") == "trail")
            n_sltp = sum(1 for r in valid_results if r.params.get("exit", {}).get("mode") == "sl_tp")
            print(f"\nTests with N >= {MIN_TRADES}: {n_tested}")
            print(f"  Time exits: {n_time}  Trailing: {n_trail}  SL/TP: {n_sltp}")
            print(f"FDR-significant at alpha={FDR_ALPHA}: {n_sig}")

            if n_sig > 0:
                print(
                    f"\n{'Archetype':<22s} {'Inst':<6s} {'Variant':<45s} "
                    f"{'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>7s} "
                    f"{'raw_p':>8s} {'adj_p':>8s}"
                )
                print("-" * 130)
                for idx, adj_p, sig in fdr_results:
                    if sig:
                        r = valid_results[idx]
                        print(
                            f"{r.archetype:<22s} {r.instrument:<6s} "
                            f"{r.variant:<45s} "
                            f"{r.n_trades:5d} {r.win_rate:7.2%} "
                            f"{r.avg_pnl_r:+8.4f} {r.sharpe:7.3f} "
                            f"{r.p_value:8.4f} {adj_p:8.4f}"
                        )
            else:
                print("\nNo strategies survived FDR correction.")

            # Top 15 by raw p-value
            print(f"\n  TOP 15 BY RAW P-VALUE:")
            sorted_by_p = sorted(enumerate(valid_results), key=lambda x: x[1].p_value)
            for rank, (idx, r) in enumerate(sorted_by_p[:15]):
                adj_p = fdr_results[idx][1]
                exit_lbl = r.params.get("exit", {}).get("label", "?")
                print(
                    f"    {rank + 1:2d}. {r.archetype:22s} {r.instrument:6s} "
                    f"{r.variant:42s} N={r.n_trades:4d} ExpR={r.avg_pnl_r:+.4f} "
                    f"raw_p={r.p_value:.4f} adj_p={adj_p:.4f} [{exit_lbl}]"
                )

            # Best exit mode per archetype
            print(f"\n  BEST EXIT MODE PER ARCHETYPE:")
            for arch in ["vwap_reversion", "fade_breakout", "pdh_pdl_sweep", "opening_drive_fade"]:
                arch_valid = [r for r in valid_results if r.archetype == arch]
                if not arch_valid:
                    continue
                by_mode = {}
                for r in arch_valid:
                    mode = r.params.get("exit", {}).get("mode", "?")
                    by_mode.setdefault(mode, []).append(r)
                print(f"\n    {arch}:")
                for mode in ["time", "trail", "sl_tp"]:
                    mr = by_mode.get(mode, [])
                    if mr:
                        avg_r = np.mean([r.avg_pnl_r for r in mr])
                        n_pos = sum(1 for r in mr if r.avg_pnl_r > 0)
                        best = min(mr, key=lambda r: r.p_value)
                        print(
                            f"      {mode:6s}: {len(mr):3d} tests, "
                            f"{n_pos:3d} pos, avg={avg_r:+.4f}, "
                            f"best p={best.p_value:.4f} ({best.variant})"
                        )
        else:
            print(f"\nNo strategies had N >= {MIN_TRADES}")

        # =================================================================
        # ORB CORRELATION
        # =================================================================
        print("\n" + "=" * 70)
        print("ORB CORRELATION ANALYSIS")
        print("=" * 70)
        for r in valid_results:
            if r.p_value < 0.10:
                corr = compute_orb_correlation(con, r, r.instrument)
                if corr is not None:
                    tag = "UNCORRELATED" if abs(corr) < 0.3 else "CORRELATED"
                    print(f"  {r.archetype:22s} {r.instrument:6s} {r.variant:42s} corr={corr:+.3f} [{tag}]")

        # =================================================================
        # HONEST SUMMARY
        # =================================================================
        print("\n" + "=" * 70)
        print("HONEST SUMMARY - V2 Multi-Exit Strategies")
        print("=" * 70)
        print("\nExpR = ATR-normalized PnL after friction.")
        print("+0.05 = +5% of ATR per trade.\n")

        survived = []
        if fdr_results:
            survived = [valid_results[i] for i, (_, _, sig) in enumerate(fdr_results) if sig]

        print("SURVIVED SCRUTINY:")
        if survived:
            for r in survived:
                print(
                    f"  - {r.archetype} | {r.instrument} | {r.variant} | "
                    f"N={r.n_trades}, ExpR={r.avg_pnl_r:+.4f}, p={r.p_value:.4f}"
                )
        else:
            print("  None.")

        for arch in ["vwap_reversion", "fade_breakout", "pdh_pdl_sweep", "opening_drive_fade"]:
            arch_results = [r for r in valid_results if r.archetype == arch]
            if arch_results:
                best = min(arch_results, key=lambda r: r.p_value)
                avg_exp = np.mean([r.avg_pnl_r for r in arch_results])
                n_pos = sum(1 for r in arch_results if r.avg_pnl_r > 0)
                print(f"\n  {arch} ({len(arch_results)} valid, {n_pos} positive):")
                print(f"    Average ExpR: {avg_exp:+.4f}")
                print(
                    f"    Best: {best.instrument} {best.variant} "
                    f"N={best.n_trades} ExpR={best.avg_pnl_r:+.4f} p={best.p_value:.4f}"
                )

        print("\nEXIT MODE VERDICT:")
        for arch in ["vwap_reversion", "fade_breakout", "pdh_pdl_sweep", "opening_drive_fade"]:
            arch_valid = [r for r in valid_results if r.archetype == arch]
            if not arch_valid:
                continue
            best = min(arch_valid, key=lambda r: r.p_value)
            exit_lbl = best.params.get("exit", {}).get("label", "?")
            print(f"  {arch:24s} -> best exit: {exit_lbl:12s} (p={best.p_value:.4f}, ExpR={best.avg_pnl_r:+.4f})")

        print("\nCAVEATS:")
        print("  - ATR normalization is NOT a true R-multiple")
        print("  - Time exit has no risk management (informational only)")
        print("  - DST may shift session boundaries +/-1 hour")
        print("  - MNQ/SIL: only 2 years of data")
        print("  - All results are IN-SAMPLE")
        print("  - BH FDR applied across ALL tests including exit variations")

    finally:
        con.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Non-ORB V2 Multi-Exit Strategy Research")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "gold.db")
    parser.add_argument("--archetype", type=str, default=None, choices=["vwap", "fade", "sweep", "drive"])
    parser.add_argument("--instrument", type=str, default=None, choices=ALL_INSTRUMENTS)
    args = parser.parse_args()

    n_t = sum(1 for e in EXIT_CONFIGS if e["mode"] == "time")
    n_tr = sum(1 for e in EXIT_CONFIGS if e["mode"] == "trail")
    n_sl = sum(1 for e in EXIT_CONFIGS if e["mode"] == "sl_tp")
    print("Non-ORB Strategy Research - V2: Multi-Exit Strategies")
    print(f"Database: {args.db_path}")
    print(f"Archetype: {args.archetype or 'ALL'}")
    print(f"Instrument: {args.instrument or 'ALL'}")
    print(f"Exit configs: {len(EXIT_CONFIGS)} ({n_t} time, {n_tr} trail, {n_sl} SL/TP)")
    print(f"Date: {date.today()}")
    print()

    t0 = time.time()
    results = run_all(str(args.db_path), args.archetype, args.instrument)
    elapsed = time.time() - t0

    total = len(results)
    valid = sum(1 for r in results if r.n_trades >= MIN_TRADES)
    print(f"\nTotal tests: {total}, Valid (N>={MIN_TRADES}): {valid}")
    print(f"Runtime: {elapsed:.0f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()

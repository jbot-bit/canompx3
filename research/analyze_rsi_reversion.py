#!/usr/bin/env python3
"""
RSI Extreme Mean Reversion strategy analysis.

Trades reversals when intraday RSI on 5-minute bars hits extreme levels
(oversold < 20, overbought > 80).

Entry logic:
  1. Compute RSI(period) on 5-minute bars in real-time throughout the day
  2. When RSI < 20: BUY signal (oversold)
  3. When RSI > 80: SELL signal (overbought)
  4. Entry = next 5m bar's open after signal
  5. Stop = lowest low (for long) or highest high (for short) of last N bars
  6. Target = entry +/- RR * risk_distance
  7. Time filter: only trade during active sessions (avoid dead hours)

Gates:
  B. Risk floor: swing stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS
  F. Only one open trade per direction at a time (skip overlapping signals)

Grid: 3 RR targets x 3 RSI periods (5, 7, 10) x 2 time filters = 18 combos
Walk-forward: 12-month training windows, monthly steps, OOS from 2018-01-01
"""

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.cost_model import get_cost_spec, to_r_multiple
from pipeline.paths import GOLD_DB_PATH
from research._alt_strategy_utils import (
    compute_strategy_metrics,
    compute_walk_forward_windows,
    save_results,
)

SPEC = get_cost_spec("MGC")

# Grid dimensions
RR_TARGETS = [1.0, 1.5, 2.0]
RSI_PERIODS = [5, 7, 10]
RSI_OVERSOLD = 20.0
RSI_OVERBOUGHT = 80.0
SWING_LOOKBACK = 10  # bars for swing stop

# Time filters (UTC hours)
TIME_FILTERS = {
    "active": (0, 23),     # Nearly all hours (exclude 23:00 UTC = dead hour)
    "core": (8, 18),       # London + NY core hours only
}

REGIME_BOUNDARY = date(2025, 1, 1)

def load_bars_5m_for_day(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 5-minute bars for one trading day (09:00 Brisbane boundary)."""
    from pipeline.build_daily_features import compute_trading_day_utc_range
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = 'MGC'
              AND ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc
        """, [start_utc, end_utc]).fetchdf()
    finally:
        con.close()
    return df

def compute_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute Wilder's RSI on an array of close prices.

    Returns array of same length, with NaN for first `period` elements.
    """
    n = len(closes)
    rsi = np.full(n, np.nan)
    if n <= period:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed: simple average of first `period` changes
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's smoothing for remaining bars
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

def compute_rsi_outcomes_for_day(
    bars_5m: pd.DataFrame,
    rsi_period: int,
    time_filter_name: str,
) -> list[dict]:
    """Find RSI extreme signals and compute outcomes for one day.

    Returns list of outcome dicts for all (signal, rr) combos found.
    """
    if len(bars_5m) < rsi_period + 5:
        return []

    closes = bars_5m["close"].values
    rsi_values = compute_rsi(closes, rsi_period)

    # Apply time filter
    time_range = TIME_FILTERS[time_filter_name]
    ts_col = bars_5m["ts_utc"]
    if ts_col.dt.tz is not None:
        hours = ts_col.dt.tz_convert("UTC").dt.hour
    else:
        hours = ts_col.dt.hour

    outcomes = []
    last_long_exit = -1  # Gate F: track last exit to avoid overlapping trades
    last_short_exit = -1

    for i in range(rsi_period + 1, len(bars_5m) - 1):
        rsi_val = rsi_values[i]
        if np.isnan(rsi_val):
            continue

        hour = hours.iloc[i]
        if not (time_range[0] <= hour < time_range[1]):
            continue

        # Determine signal
        if rsi_val < RSI_OVERSOLD:
            direction = "long"
            if i <= last_long_exit:
                continue  # Gate F: overlapping
        elif rsi_val > RSI_OVERBOUGHT:
            direction = "short"
            if i <= last_short_exit:
                continue  # Gate F: overlapping
        else:
            continue

        # Entry = next bar's open
        entry_bar_idx = i + 1
        entry_price = bars_5m.iloc[entry_bar_idx]["open"]

        # Stop = swing extreme over lookback
        lookback_start = max(0, i - SWING_LOOKBACK + 1)
        lookback_bars = bars_5m.iloc[lookback_start:i + 1]

        if direction == "long":
            stop_price = lookback_bars["low"].min()
        else:
            stop_price = lookback_bars["high"].max()

        risk_points = abs(entry_price - stop_price)

        # Gate B: Risk floor
        if risk_points < SPEC.min_risk_floor_points:
            continue

        for rr in RR_TARGETS:
            reward = rr * risk_points
            if direction == "long":
                target_price = entry_price + reward
            else:
                target_price = entry_price - reward

            # Resolve outcome scanning bars after entry
            result = _resolve_5m_outcome(
                bars_5m, entry_price, stop_price, target_price,
                direction, entry_bar_idx + 1,
            )

            if result is None:
                # EOD
                last_close = bars_5m.iloc[-1]["close"]
                if direction == "long":
                    pnl_points = last_close - entry_price
                else:
                    pnl_points = entry_price - last_close
                pnl_r = to_r_multiple(SPEC, entry_price, stop_price, pnl_points)
                outcome_type = "eod"
                exit_idx = len(bars_5m) - 1
            else:
                pnl_r = to_r_multiple(
                    SPEC, entry_price, stop_price, result["pnl_points"]
                )
                outcome_type = result["outcome"]
                exit_idx = result["exit_bar_idx"]

            # Update Gate F tracking (use max exit across RR targets)
            if direction == "long":
                last_long_exit = max(last_long_exit, exit_idx)
            else:
                last_short_exit = max(last_short_exit, exit_idx)

            outcomes.append({
                "rsi_period": rsi_period,
                "rsi_value": rsi_val,
                "time_filter": time_filter_name,
                "direction": direction,
                "rr_target": rr,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_points": risk_points,
                "pnl_r": pnl_r,
                "outcome": outcome_type,
            })

    return outcomes

def _resolve_5m_outcome(
    bars: pd.DataFrame,
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: str,
    start_idx: int,
) -> dict | None:
    """Resolve outcome on 5m bars (same logic as resolve_bar_outcome)."""
    is_long = direction == "long"

    for i in range(start_idx, len(bars)):
        bar = bars.iloc[i]
        bar_high = bar["high"]
        bar_low = bar["low"]

        if is_long:
            stop_hit = bar_low <= stop_price
            target_hit = bar_high >= target_price
        else:
            stop_hit = bar_high >= stop_price
            target_hit = bar_low <= target_price

        # Gate C: ambiguous -> LOSS
        if stop_hit and target_hit:
            pnl_points = stop_price - entry_price if is_long else entry_price - stop_price
            return {"outcome": "loss", "exit_price": stop_price,
                    "exit_bar_idx": i, "pnl_points": pnl_points}

        if stop_hit:
            pnl_points = stop_price - entry_price if is_long else entry_price - stop_price
            return {"outcome": "loss", "exit_price": stop_price,
                    "exit_bar_idx": i, "pnl_points": pnl_points}

        if target_hit:
            pnl_points = target_price - entry_price if is_long else entry_price - target_price
            return {"outcome": "win", "exit_price": target_price,
                    "exit_bar_idx": i, "pnl_points": pnl_points}

    return None

def compute_all_rsi_outcomes(
    db_path: Path,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Compute RSI outcomes for all days in date range.

    Loads 5m bars day by day and computes signals for all grid combos.
    """
    # Get list of trading days from daily_features
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        days = con.execute("""
            SELECT DISTINCT trading_day
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 5
              AND trading_day BETWEEN ? AND ?
            ORDER BY trading_day
        """, [start, end]).fetchdf()
    finally:
        con.close()

    if days.empty:
        return pd.DataFrame()

    all_outcomes = []
    trading_days = days["trading_day"].tolist()
    total = len(trading_days)

    for idx, td in enumerate(trading_days):
        if idx % 200 == 0:
            print(f"    Processing day {idx+1}/{total}...")

        if hasattr(td, "date") and callable(td.date):
            td_date = td.date()
        elif isinstance(td, str):
            td_date = date.fromisoformat(td)
        else:
            td_date = td

        bars_5m = load_bars_5m_for_day(db_path, td_date)
        if bars_5m.empty:
            continue

        for rsi_period in RSI_PERIODS:
            for time_filter_name in TIME_FILTERS:
                day_outcomes = compute_rsi_outcomes_for_day(
                    bars_5m, rsi_period, time_filter_name
                )
                for o in day_outcomes:
                    o["trading_day"] = str(td_date)
                all_outcomes.extend(day_outcomes)

    if not all_outcomes:
        return pd.DataFrame()
    return pd.DataFrame(all_outcomes)

def run_walk_forward(
    db_path: Path,
    train_months: int = 12,
    test_start: date = date(2018, 1, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward analysis for RSI reversion strategy."""
    full_start = date(2016, 1, 1)

    print("  Computing all RSI reversion outcomes (this may take a while)...")
    outcomes_df = compute_all_rsi_outcomes(db_path, full_start, test_end)
    if outcomes_df.empty:
        print("  No outcomes found")
        return {"windows": [], "combined_oos": None, "regime_split": None}

    outcomes_df["trading_day_date"] = pd.to_datetime(outcomes_df["trading_day"]).dt.date
    n_days = outcomes_df["trading_day_date"].nunique()
    n_signals = len(outcomes_df)
    print(f"  {n_signals} total outcomes across {n_days} days")

    windows = compute_walk_forward_windows(test_start, test_end, train_months)
    window_results = []
    oos_all_pnls = []
    oos_all_dates = []

    for w in windows:
        train_mask = (
            (outcomes_df["trading_day_date"] >= w["train_start"])
            & (outcomes_df["trading_day_date"] <= w["train_end"])
        )
        test_mask = (
            (outcomes_df["trading_day_date"] >= w["test_start"])
            & (outcomes_df["trading_day_date"] <= w["test_end"])
        )

        train_data = outcomes_df[train_mask]
        test_data = outcomes_df[test_mask]

        if train_data.empty:
            continue

        # Find best (rsi_period, time_filter, rr) on training data
        best_combo = None
        best_sharpe = -999.0

        for rsi_period in RSI_PERIODS:
            rp_train = train_data[train_data["rsi_period"] == rsi_period]
            for tf_name in TIME_FILTERS:
                tf_train = rp_train[rp_train["time_filter"] == tf_name]
                for rr in RR_TARGETS:
                    rr_train = tf_train[tf_train["rr_target"] == rr]
                    if len(rr_train) < 30:
                        continue
                    stats = compute_strategy_metrics(rr_train["pnl_r"].values)
                    if stats and stats["sharpe"] > best_sharpe:
                        best_sharpe = stats["sharpe"]
                        best_combo = (rsi_period, tf_name, rr)

        if best_combo is None:
            continue

        rsi_period, tf_name, rr = best_combo

        # Apply to OOS
        oos = test_data[
            (test_data["rsi_period"] == rsi_period)
            & (test_data["time_filter"] == tf_name)
            & (test_data["rr_target"] == rr)
        ]
        if oos.empty:
            continue

        oos_pnls = oos["pnl_r"].values
        oos_stats = compute_strategy_metrics(oos_pnls)
        oos_all_pnls.extend(oos_pnls)
        oos_all_dates.extend(oos["trading_day_date"].values)

        window_results.append({
            "test_start": str(w["test_start"]),
            "test_end": str(w["test_end"]),
            "train_n": len(train_data[
                (train_data["rsi_period"] == rsi_period)
                & (train_data["time_filter"] == tf_name)
                & (train_data["rr_target"] == rr)
            ]),
            "selected": f"RSI{rsi_period}_{tf_name}_RR{rr}",
            "train_sharpe": best_sharpe,
            "oos_stats": oos_stats,
        })

    combined_oos = None
    regime_split = None
    if oos_all_pnls:
        all_pnls = np.array(oos_all_pnls)
        all_dates = np.array(oos_all_dates)
        combined_oos = compute_strategy_metrics(all_pnls)

        # Regime split
        low_vol_mask = all_dates < np.datetime64(REGIME_BOUNDARY)
        high_vol_mask = ~low_vol_mask
        low_vol = compute_strategy_metrics(all_pnls[low_vol_mask]) if low_vol_mask.sum() > 0 else None
        high_vol = compute_strategy_metrics(all_pnls[high_vol_mask]) if high_vol_mask.sum() > 0 else None
        regime_split = {"low_vol_2018_2024": low_vol, "high_vol_2025_2026": high_vol}

    return {
        "train_months": train_months,
        "windows": window_results,
        "combined_oos": combined_oos,
        "regime_split": regime_split,
    }

def run_full_period_analysis(
    db_path: Path,
    start: date = date(2017, 1, 1),
    end: date = date(2026, 2, 1),
) -> dict:
    """Run full-period grid search."""
    outcomes_df = compute_all_rsi_outcomes(db_path, start, end)

    if outcomes_df.empty:
        return {"grid": []}

    grid_results = []
    for rsi_period in RSI_PERIODS:
        rp_data = outcomes_df[outcomes_df["rsi_period"] == rsi_period]
        for tf_name in TIME_FILTERS:
            tf_data = rp_data[rp_data["time_filter"] == tf_name]
            for rr in RR_TARGETS:
                rr_data = tf_data[tf_data["rr_target"] == rr]
                if rr_data.empty:
                    continue
                stats = compute_strategy_metrics(rr_data["pnl_r"].values)
                if stats:
                    grid_results.append({
                        "rsi_period": rsi_period,
                        "time_filter": tf_name,
                        "rr_target": rr,
                        **stats,
                    })

    return {"grid": grid_results}

def _print_go_no_go(combined_oos: dict | None, regime_split: dict | None) -> None:
    """Print GO/NO-GO evaluation."""
    print()
    print("--- GO/NO-GO EVALUATION ---")
    if combined_oos is None:
        print("  NO-GO: No OOS data")
        return

    c = combined_oos
    checks = {
        "Combined OOS ExpR > 0": c["expr"] > 0,
        "Combined OOS N > 100": c["n"] > 100,
        "Combined OOS Sharpe > 0.05": c["sharpe"] > 0.05,
    }

    if regime_split and regime_split.get("low_vol_2018_2024"):
        lv = regime_split["low_vol_2018_2024"]
        checks["Low-vol (2018-2024) ExpR > 0"] = lv["expr"] > 0
    else:
        checks["Low-vol (2018-2024) ExpR > 0"] = False

    all_pass = all(checks.values())
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    verdict = "GO" if all_pass else "NO-GO"
    print(f"\n  VERDICT: {verdict}")

def main():
    parser = argparse.ArgumentParser(description="RSI Extreme Mean Reversion analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--full-period-only", action="store_true",
                        help="Skip walk-forward, just run full-period grid search")
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("RSI EXTREME MEAN REVERSION STRATEGY ANALYSIS")
    print(sep)
    print()
    print("Entry: Buy when RSI < 20 (oversold), Sell when RSI > 80 (overbought)")
    print("Stop: Swing low/high over last 10 bars")
    print(f"Grid: {len(RSI_PERIODS)} RSI periods x {len(TIME_FILTERS)} time filters x "
          f"{len(RR_TARGETS)} RR = {len(RSI_PERIODS) * len(TIME_FILTERS) * len(RR_TARGETS)} combos")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print(f"Gate F: No overlapping trades per direction")
    print()

    if args.full_period_only:
        print("--- FULL PERIOD GRID SEARCH ---")
        result = run_full_period_analysis(args.db_path)

        if result["grid"]:
            print(f"  Grid results ({len(result['grid'])} combos):")
            sorted_grid = sorted(result["grid"], key=lambda x: x["sharpe"], reverse=True)
            for g in sorted_grid[:15]:
                print(f"    RSI{g['rsi_period']} {g['time_filter']} RR{g['rr_target']}: "
                      f"N={g['n']}, WR={g['wr']:.0%}, ExpR={g['expr']:+.3f}, "
                      f"Sharpe={g['sharpe']:.3f}, MaxDD={g['maxdd']:+.1f}R")
        else:
            print("  No outcomes found")

        if args.output:
            save_results(result, args.output)

    else:
        print("--- WALK-FORWARD ANALYSIS ---")
        result = run_walk_forward(args.db_path, args.train_months)

        if result["windows"]:
            for w in result["windows"]:
                oos = w["oos_stats"]
                if oos:
                    print(f"  {w['test_start']} to {w['test_end']}: "
                          f"Selected {w['selected']}, "
                          f"OOS N={oos['n']}, WR={oos['wr']:.0%}, "
                          f"ExpR={oos['expr']:+.3f}, Sharpe={oos['sharpe']:.3f}")

            if result["combined_oos"]:
                c = result["combined_oos"]
                print(f"\n  COMBINED OOS: N={c['n']}, WR={c['wr']:.0%}, "
                      f"ExpR={c['expr']:+.3f}, Sharpe={c['sharpe']:.3f}, "
                      f"MaxDD={c['maxdd']:+.1f}R, Total={c['total']:+.1f}R")

            if result["regime_split"]:
                rs = result["regime_split"]
                print("\n  REGIME SPLIT:")
                for label, stats in rs.items():
                    if stats:
                        print(f"    {label}: N={stats['n']}, WR={stats['wr']:.0%}, "
                              f"ExpR={stats['expr']:+.3f}, Sharpe={stats['sharpe']:.3f}, "
                              f"MaxDD={stats['maxdd']:+.1f}R")
                    else:
                        print(f"    {label}: No data")

            _print_go_no_go(result["combined_oos"], result["regime_split"])
        else:
            print("  No qualifying windows")
            _print_go_no_go(None, None)

        if args.output:
            save_results(result, args.output)

    print()
    print(sep)
    print("DONE")
    print(sep)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Concretum Bands (Dynamic Volatility Breakout) strategy analysis.

Instead of a fixed-time ORB window, uses volatility to define a dynamic noise
zone around the daily open. Adapts to the day's regime automatically.

Entry logic:
  1. Compute sigma = stddev of (daily_close - daily_open) over last 14 days (shifted 1)
  2. Upper band = daily_open + (band_mult * sigma)
  3. Lower band = daily_open - (band_mult * sigma)
  4. Entry = first 1m bar that closes outside the band
  5. Direction = long if breaks upper, short if breaks lower
  6. Stop = opposite band (or entry -/+ stop_mult * sigma)
  7. Target = entry + RR * risk_distance

Gates:
  B. Risk floor: stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS

Grid: 4 band multipliers x 2 stop multipliers x 4 RR targets x 2 time filters = 64 combos
Walk-forward: 12-month training windows, monthly steps
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec, to_r_multiple
from pipeline.paths import GOLD_DB_PATH
from scripts._alt_strategy_utils import (
    annualize_sharpe,
    compute_strategy_metrics,
    compute_walk_forward_windows,
    load_bars_for_day,
    load_daily_features,
    save_results,
)

SPEC = get_cost_spec("MGC")

# Grid dimensions
BAND_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]  # sigma
STOP_MULTIPLIERS = [0.5, 1.0]             # sigma
RR_TARGETS = [1.0, 1.5, 2.0, 2.5]
TIME_FILTERS = {
    "first_2h": 120,  # Only look for entry in first 2 hours after open
    "all_day": 1440,  # Full trading day
}
SIGMA_LOOKBACK = 14  # Days for sigma calc


def prepare_sigma_data(features: pd.DataFrame) -> pd.DataFrame:
    """Add sigma (stddev of daily_close - daily_open) to features."""
    df = features.copy()
    df = df.sort_values("trading_day")

    # Close-open range per day
    df["co_range"] = (df["daily_close"] - df["daily_open"]).abs()

    # Rolling sigma, shifted by 1 (no lookahead)
    df["sigma"] = df["co_range"].rolling(window=SIGMA_LOOKBACK, min_periods=SIGMA_LOOKBACK).std().shift(1)

    return df


def compute_concretum_outcomes(
    db_path: Path,
    start: date,
    end: date,
) -> list[dict]:
    """Compute all concretum band trade outcomes."""
    features = load_daily_features(db_path, start, end)
    if features.empty:
        return []

    sigma_data = prepare_sigma_data(features)
    eligible = sigma_data.dropna(subset=["sigma", "daily_open"])
    eligible = eligible[eligible["sigma"] > 0]
    total = len(eligible)
    all_outcomes = []

    print(f"    Processing {total} eligible days...")
    for idx, (_, row) in enumerate(eligible.iterrows()):
        if idx % 200 == 0:
            print(f"    Day {idx+1}/{total}...")

        td = row["trading_day"]
        if hasattr(td, "date") and callable(td.date):
            td = td.date()
        elif isinstance(td, str):
            td = date.fromisoformat(td)

        daily_open = row["daily_open"]
        sigma = row["sigma"]

        bars = load_bars_for_day(db_path, td)
        if bars.empty or len(bars) < 60:
            continue

        # Get UTC hours for time filtering
        ts_col = bars["ts_utc"]
        if ts_col.dt.tz is not None:
            ts_utc = ts_col.dt.tz_convert("UTC")
        else:
            ts_utc = ts_col

        # Find the first bar (session open)
        first_bar_time = ts_utc.iloc[0]

        for band_mult in BAND_MULTIPLIERS:
            upper_band = daily_open + band_mult * sigma
            lower_band = daily_open - band_mult * sigma

            for tf_name, tf_minutes in TIME_FILTERS.items():
                # Find first bar that closes outside band within time filter
                entry_signal = None

                for i in range(1, len(bars)):
                    # Time filter: minutes since session open
                    bar_time = ts_utc.iloc[i]
                    minutes_elapsed = (bar_time - first_bar_time).total_seconds() / 60.0
                    if minutes_elapsed > tf_minutes:
                        break

                    bar_close = bars.iloc[i]["close"]

                    if bar_close > upper_band:
                        entry_signal = {
                            "direction": "long",
                            "entry_bar_idx": i + 1,
                            "entry_price": bars.iloc[i + 1]["open"] if i + 1 < len(bars) else bar_close,
                        }
                        break
                    elif bar_close < lower_band:
                        entry_signal = {
                            "direction": "short",
                            "entry_bar_idx": i + 1,
                            "entry_price": bars.iloc[i + 1]["open"] if i + 1 < len(bars) else bar_close,
                        }
                        break

                if entry_signal is None:
                    continue

                if entry_signal["entry_bar_idx"] >= len(bars):
                    continue

                entry_price = entry_signal["entry_price"]
                direction = entry_signal["direction"]
                entry_bar_idx = entry_signal["entry_bar_idx"]

                for stop_mult in STOP_MULTIPLIERS:
                    stop_distance = stop_mult * sigma
                    if stop_distance < SPEC.min_risk_floor_points:
                        continue

                    if direction == "long":
                        stop_price = entry_price - stop_distance
                    else:
                        stop_price = entry_price + stop_distance

                    for rr in RR_TARGETS:
                        reward = rr * stop_distance
                        if direction == "long":
                            target_price = entry_price + reward
                        else:
                            target_price = entry_price - reward

                        result = _resolve_1m(
                            bars, entry_price, stop_price, target_price,
                            direction, entry_bar_idx
                        )

                        if result is None:
                            last_close = bars.iloc[-1]["close"]
                            pnl_pts = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
                            pnl_r = to_r_multiple(SPEC, entry_price, stop_price, pnl_pts)
                            outcome_type = "eod"
                        else:
                            pnl_r = to_r_multiple(SPEC, entry_price, stop_price, result["pnl_points"])
                            outcome_type = result["outcome"]

                        all_outcomes.append({
                            "trading_day": str(td),
                            "direction": direction,
                            "band_mult": band_mult,
                            "stop_multiplier": stop_mult,
                            "rr_target": rr,
                            "time_filter": tf_name,
                            "sigma": sigma,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "pnl_r": pnl_r,
                            "outcome": outcome_type,
                        })

    return all_outcomes


def _resolve_1m(bars, entry, stop, target, direction, start_idx):
    """Resolve outcome on 1m bars."""
    is_long = direction == "long"
    for i in range(start_idx, len(bars)):
        bar = bars.iloc[i]
        if is_long:
            stop_hit = bar["low"] <= stop
            target_hit = bar["high"] >= target
        else:
            stop_hit = bar["high"] >= stop
            target_hit = bar["low"] <= target

        if stop_hit and target_hit:
            pnl = stop - entry if is_long else entry - stop
            return {"outcome": "loss", "pnl_points": pnl, "exit_bar_idx": i}
        if stop_hit:
            pnl = stop - entry if is_long else entry - stop
            return {"outcome": "loss", "pnl_points": pnl, "exit_bar_idx": i}
        if target_hit:
            pnl = target - entry if is_long else entry - target
            return {"outcome": "win", "pnl_points": pnl, "exit_bar_idx": i}
    return None


def run_walk_forward(
    db_path: Path,
    train_months: int = 12,
    test_start: date = date(2024, 8, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward analysis for concretum bands strategy."""
    # Only load data needed for training + OOS
    from scripts._alt_strategy_utils import _add_months
    full_start = _add_months(test_start, -(train_months + 2))

    print("  Computing all concretum band outcomes...")
    all_outcomes = compute_concretum_outcomes(db_path, full_start, test_end)
    if not all_outcomes:
        print("  No outcomes found")
        return {"windows": [], "combined_oos": None, "regime_split": None}

    outcomes_df = pd.DataFrame(all_outcomes)
    outcomes_df["trading_day_date"] = pd.to_datetime(outcomes_df["trading_day"]).dt.date
    n_days = outcomes_df["trading_day_date"].nunique()
    print(f"  {len(outcomes_df)} total outcomes across {n_days} days")

    windows = compute_walk_forward_windows(test_start, test_end, train_months)
    window_results = []
    oos_all_pnls = []
    oos_all_dates = []

    regime_boundary = date(2025, 1, 1)

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

        best_combo = None
        best_sharpe = -999.0

        for bm in BAND_MULTIPLIERS:
            bm_train = train_data[train_data["band_mult"] == bm]
            for tf_name in TIME_FILTERS:
                tf_train = bm_train[bm_train["time_filter"] == tf_name]
                for sm in STOP_MULTIPLIERS:
                    sm_train = tf_train[tf_train["stop_multiplier"] == sm]
                    for rr in RR_TARGETS:
                        rr_train = sm_train[sm_train["rr_target"] == rr]
                        if len(rr_train) < 20:
                            continue
                        stats = compute_strategy_metrics(rr_train["pnl_r"].values)
                        if stats and stats["sharpe"] > best_sharpe:
                            best_sharpe = stats["sharpe"]
                            best_combo = (bm, tf_name, sm, rr)

        if best_combo is None:
            continue

        bm, tf_name, sm, rr = best_combo
        oos = test_data[
            (test_data["band_mult"] == bm)
            & (test_data["time_filter"] == tf_name)
            & (test_data["stop_multiplier"] == sm)
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
            "selected": f"BM{bm}_{tf_name}_SM{sm}_RR{rr}",
            "train_sharpe": best_sharpe,
            "oos_stats": oos_stats,
        })

    combined_oos = None
    regime_split = None
    if oos_all_pnls:
        all_pnls = np.array(oos_all_pnls)
        all_dates = np.array(oos_all_dates)
        combined_oos = compute_strategy_metrics(all_pnls)
        oos_years = (test_end - test_start).days / 365.25
        annualize_sharpe(combined_oos, oos_years)

        low_vol_mask = all_dates < np.datetime64(regime_boundary)
        high_vol_mask = ~low_vol_mask
        low_vol = compute_strategy_metrics(all_pnls[low_vol_mask]) if low_vol_mask.sum() > 0 else None
        high_vol = compute_strategy_metrics(all_pnls[high_vol_mask]) if high_vol_mask.sum() > 0 else None
        regime_split = {"low_vol_pre2025": low_vol, "high_vol_2025_2026": high_vol}

    return {
        "train_months": train_months,
        "windows": window_results,
        "combined_oos": combined_oos,
        "regime_split": regime_split,
    }


def main():
    parser = argparse.ArgumentParser(description="Concretum Bands strategy analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="OOS start date (YYYY-MM-DD), default 2024-08-01")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="OOS end date (YYYY-MM-DD), default 2026-02-01")
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("CONCRETUM BANDS (DYNAMIC VOLATILITY BREAKOUT) STRATEGY ANALYSIS")
    print(sep)
    print()
    print("Entry: Breakout from dynamic sigma-based bands around daily open")
    print(f"Grid: {len(BAND_MULTIPLIERS)} band mults x {len(STOP_MULTIPLIERS)} stop mults x "
          f"{len(RR_TARGETS)} RR x {len(TIME_FILTERS)} time filters = "
          f"{len(BAND_MULTIPLIERS) * len(STOP_MULTIPLIERS) * len(RR_TARGETS) * len(TIME_FILTERS)} combos")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print()

    print("--- WALK-FORWARD ANALYSIS ---")
    wf_kwargs = {"train_months": args.train_months}
    if args.start:
        wf_kwargs["test_start"] = args.start
    if args.end:
        wf_kwargs["test_end"] = args.end
    result = run_walk_forward(args.db_path, **wf_kwargs)

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
            sha = c.get("sharpe_ann")
            sha_str = f", ShANN={sha:.3f}" if sha is not None else ""
            print(f"\n  COMBINED OOS: N={c['n']}, WR={c['wr']:.0%}, "
                  f"ExpR={c['expr']:+.3f}, Sharpe={c['sharpe']:.3f}{sha_str}, "
                  f"MaxDD={c['maxdd']:+.1f}R, Total={c['total']:+.1f}R")

        if result["regime_split"]:
            rs = result["regime_split"]
            print("\n  REGIME SPLIT:")
            for label, stats in rs.items():
                if stats:
                    print(f"    {label}: N={stats['n']}, WR={stats['wr']:.0%}, "
                          f"ExpR={stats['expr']:+.3f}, Sharpe={stats['sharpe']:.3f}")
                else:
                    print(f"    {label}: No data")
    else:
        print("  No qualifying windows")

    if args.output:
        save_results(result, args.output)

    print()
    print(sep)
    print("DONE")
    print(sep)


if __name__ == "__main__":
    main()

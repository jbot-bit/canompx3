#!/usr/bin/env python3
"""
Gap Fade (Mean Reversion) strategy analysis.

Trades the fill of overnight gaps. When the market gaps open away from
yesterday's close, we bet it returns to the previous close.

Entry logic:
  1. Compute gap = daily_open - previous_daily_close (from daily_features)
  2. Compute ATR_20 = rolling 20-day mean of (daily_high - daily_low)
  3. If abs(gap) >= threshold * ATR_20: trade is eligible
  4. Direction = opposite of gap (gap up -> short, gap down -> long)
  5. Entry = daily_open (09:00 Brisbane bar open)
  6. Target = previous day's daily_close
  7. Stop = entry +/- stop_mult * ATR_20

Gates:
  B. Risk floor: ATR-based stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS

Grid: gap threshold (0.2-0.4 * ATR) x stop multiplier (0.5-1.5 * ATR) x day-of-week
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
    compute_strategy_metrics,
    compute_walk_forward_windows,
    load_bars_for_day,
    load_daily_features,
    resolve_bar_outcome,
    save_results,
)

SPEC = get_cost_spec("MGC")

# Grid dimensions
GAP_THRESHOLDS = [0.2, 0.3, 0.4]  # fraction of ATR_20
STOP_MULTIPLIERS = [0.5, 1.0, 1.5]  # fraction of ATR_20


def prepare_gap_data(features: pd.DataFrame) -> pd.DataFrame:
    """Add ATR_20 and gap metrics to features dataframe.

    Uses daily_features canonical columns: daily_open, daily_high, daily_low,
    daily_close, gap_open_points. Does NOT duplicate build_daily_features logic.
    """
    df = features.copy()

    # True range per day (simplified: daily_high - daily_low)
    df["true_range"] = df["daily_high"] - df["daily_low"]

    # Rolling 20-day mean of true range, shifted by 1 to avoid lookahead
    # ATR must use only PRIOR days' data (today's high/low unknown at open)
    df = df.sort_values("trading_day")
    df["atr_20"] = df["true_range"].rolling(window=20, min_periods=20).mean().shift(1)

    # Previous day's close
    df["prev_close"] = df["daily_close"].shift(1)

    # Gap in ATR units
    df["gap_atr"] = np.where(
        df["atr_20"] > 0,
        df["gap_open_points"].abs() / df["atr_20"],
        np.nan,
    )

    # Day of week (0=Monday, 4=Friday)
    df["dow"] = pd.to_datetime(df["trading_day"]).dt.dayofweek

    return df


def compute_gap_fade_outcomes(
    db_path: Path,
    gap_data: pd.DataFrame,
) -> list[dict]:
    """Compute trade outcomes for all eligible gap fade days.

    For each (gap_threshold, stop_multiplier) combo, determine entry/stop/target
    and resolve via bar scanning.
    """
    outcomes = []

    # Filter to days with valid ATR and gap data
    eligible = gap_data.dropna(subset=["atr_20", "gap_open_points", "prev_close", "daily_open"])
    eligible = eligible[eligible["atr_20"] > 0]

    for _, row in eligible.iterrows():
        trading_day = row["trading_day"]
        if isinstance(trading_day, str):
            trading_day = date.fromisoformat(trading_day)
        elif hasattr(trading_day, "date") and callable(trading_day.date):
            trading_day = trading_day.date()

        gap = row["gap_open_points"]
        atr = row["atr_20"]
        gap_atr_ratio = abs(gap) / atr if atr > 0 else 0
        daily_open = row["daily_open"]
        prev_close = row["prev_close"]
        dow = row["dow"]

        if pd.isna(gap) or abs(gap) < 0.01:
            continue

        # Direction: fade the gap
        if gap > 0:
            direction = "short"  # Gap up -> short to fill
        else:
            direction = "long"   # Gap down -> long to fill

        entry_price = daily_open
        target_price = prev_close

        # For each (gap_threshold, stop_multiplier) combo
        for gap_thresh in GAP_THRESHOLDS:
            # Eligibility check: gap must be large enough
            if gap_atr_ratio < gap_thresh:
                continue

            for stop_mult in STOP_MULTIPLIERS:
                stop_distance = stop_mult * atr

                # Gate B: Risk floor check
                if stop_distance < SPEC.min_risk_floor_points:
                    continue

                if direction == "long":
                    stop_price = entry_price - stop_distance
                else:
                    stop_price = entry_price + stop_distance

                # Sanity: target must be on the right side of entry
                if direction == "long" and target_price <= entry_price:
                    continue
                if direction == "short" and target_price >= entry_price:
                    continue

                # Load bars and resolve
                bars = load_bars_for_day(db_path, trading_day)
                if bars.empty:
                    continue

                result = resolve_bar_outcome(
                    bars, entry_price, stop_price, target_price,
                    direction, 0,  # Start from first bar (entry at open)
                )

                if result is None:
                    # EOD: use last bar close
                    last_close = bars.iloc[-1]["close"]
                    if direction == "long":
                        pnl_points = last_close - entry_price
                    else:
                        pnl_points = entry_price - last_close
                    pnl_r = to_r_multiple(SPEC, entry_price, stop_price, pnl_points)
                    outcome_type = "eod"
                else:
                    pnl_r = to_r_multiple(
                        SPEC, entry_price, stop_price, result["pnl_points"]
                    )
                    outcome_type = result["outcome"]

                outcomes.append({
                    "trading_day": str(trading_day),
                    "gap_points": gap,
                    "gap_atr_ratio": gap_atr_ratio,
                    "atr_20": atr,
                    "gap_threshold": gap_thresh,
                    "stop_multiplier": stop_mult,
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "stop_distance": stop_distance,
                    "pnl_r": pnl_r,
                    "outcome": outcome_type,
                    "dow": dow,
                })

    return outcomes


def run_walk_forward(
    db_path: Path,
    train_months: int = 12,
    test_start: date = date(2023, 1, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward analysis for gap fade strategy."""
    # Load all features once
    full_start = date(2016, 1, 1)
    features = load_daily_features(db_path, full_start, test_end)
    gap_data = prepare_gap_data(features)

    print("  Computing all gap fade outcomes...")
    all_outcomes = compute_gap_fade_outcomes(db_path, gap_data)
    if not all_outcomes:
        print("  No gap fade outcomes found")
        return {"windows": [], "combined_oos": None}

    outcomes_df = pd.DataFrame(all_outcomes)
    outcomes_df["trading_day_date"] = pd.to_datetime(outcomes_df["trading_day"]).dt.date
    print(f"  {len(outcomes_df)} total outcomes across {outcomes_df['trading_day_date'].nunique()} days")

    windows = compute_walk_forward_windows(test_start, test_end, train_months)
    window_results = []
    oos_all_pnls = []

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

        # Find best (gap_threshold, stop_multiplier) on training data
        best_combo = None
        best_sharpe = -999.0

        for gap_thresh in GAP_THRESHOLDS:
            gt_train = train_data[train_data["gap_threshold"] == gap_thresh]
            for stop_mult in STOP_MULTIPLIERS:
                sm_train = gt_train[gt_train["stop_multiplier"] == stop_mult]
                if len(sm_train) < 15:
                    continue
                stats = compute_strategy_metrics(sm_train["pnl_r"].values)
                if stats and stats["sharpe"] > best_sharpe:
                    best_sharpe = stats["sharpe"]
                    best_combo = (gap_thresh, stop_mult)

        if best_combo is None:
            continue

        gap_thresh, stop_mult = best_combo

        # Apply to OOS
        oos = test_data[
            (test_data["gap_threshold"] == gap_thresh)
            & (test_data["stop_multiplier"] == stop_mult)
        ]
        if oos.empty:
            continue

        oos_pnls = oos["pnl_r"].values
        oos_stats = compute_strategy_metrics(oos_pnls)
        oos_all_pnls.extend(oos_pnls)

        window_results.append({
            "test_start": str(w["test_start"]),
            "test_end": str(w["test_end"]),
            "train_n": len(train_data[
                (train_data["gap_threshold"] == gap_thresh)
                & (train_data["stop_multiplier"] == stop_mult)
            ]),
            "selected": f"GT{gap_thresh}_SM{stop_mult}",
            "train_sharpe": best_sharpe,
            "oos_stats": oos_stats,
        })

    combined_oos = None
    if oos_all_pnls:
        combined_oos = compute_strategy_metrics(np.array(oos_all_pnls))

    return {
        "train_months": train_months,
        "windows": window_results,
        "combined_oos": combined_oos,
    }


def run_full_period_analysis(
    db_path: Path,
    start: date = date(2022, 1, 1),
    end: date = date(2026, 2, 1),
) -> dict:
    """Run full-period grid search + day-of-week analysis."""
    features = load_daily_features(db_path, start, end)
    gap_data = prepare_gap_data(features)
    outcomes = compute_gap_fade_outcomes(db_path, gap_data)

    if not outcomes:
        return {"grid": [], "dow_analysis": []}

    outcomes_df = pd.DataFrame(outcomes)

    # Grid results
    grid_results = []
    for gap_thresh in GAP_THRESHOLDS:
        gt_data = outcomes_df[outcomes_df["gap_threshold"] == gap_thresh]
        for stop_mult in STOP_MULTIPLIERS:
            sm_data = gt_data[gt_data["stop_multiplier"] == stop_mult]
            if sm_data.empty:
                continue
            stats = compute_strategy_metrics(sm_data["pnl_r"].values)
            if stats:
                grid_results.append({
                    "gap_threshold": gap_thresh,
                    "stop_multiplier": stop_mult,
                    **stats,
                })

    # Day-of-week analysis (best combo)
    dow_analysis = []
    if grid_results:
        best = max(grid_results, key=lambda x: x["sharpe"])
        best_data = outcomes_df[
            (outcomes_df["gap_threshold"] == best["gap_threshold"])
            & (outcomes_df["stop_multiplier"] == best["stop_multiplier"])
        ]
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for dow in range(5):
            dow_data = best_data[best_data["dow"] == dow]
            if dow_data.empty:
                continue
            stats = compute_strategy_metrics(dow_data["pnl_r"].values)
            if stats:
                dow_analysis.append({"day": dow_names[dow], "dow": dow, **stats})

    return {"grid": grid_results, "dow_analysis": dow_analysis}


def main():
    parser = argparse.ArgumentParser(description="Gap Fade strategy analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--full-period-only", action="store_true",
                        help="Skip walk-forward, just run full-period grid search")
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("GAP FADE (MEAN REVERSION) STRATEGY ANALYSIS")
    print(sep)
    print()
    print("Entry: Fade overnight gap at market open, target previous close")
    print(f"Grid: {len(GAP_THRESHOLDS)} gap thresholds x {len(STOP_MULTIPLIERS)} stop multipliers")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print()

    if args.full_period_only:
        print("--- FULL PERIOD GRID SEARCH ---")
        result = run_full_period_analysis(args.db_path)

        if result["grid"]:
            print(f"  Grid results ({len(result['grid'])} combos):")
            sorted_grid = sorted(result["grid"], key=lambda x: x["sharpe"], reverse=True)
            for g in sorted_grid:
                print(f"    GT{g['gap_threshold']} SM{g['stop_multiplier']}: "
                      f"N={g['n']}, WR={g['wr']:.0%}, ExpR={g['expr']:+.3f}, "
                      f"Sharpe={g['sharpe']:.3f}, MaxDD={g['maxdd']:+.1f}R")
        else:
            print("  No outcomes found")

        print()
        if result["dow_analysis"]:
            print("  Day-of-week (best combo):")
            for d in result["dow_analysis"]:
                print(f"    {d['day']}: N={d['n']}, WR={d['wr']:.0%}, ExpR={d['expr']:+.3f}")

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
                print(f"  COMBINED OOS: N={c['n']}, WR={c['wr']:.0%}, "
                      f"ExpR={c['expr']:+.3f}, Sharpe={c['sharpe']:.3f}, "
                      f"MaxDD={c['maxdd']:+.1f}R, Total={c['total']:+.1f}R")
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

#!/usr/bin/env python3
"""
Prior Day High/Low Fade strategy analysis.

Trades reversals at yesterday's high and low levels. When price reaches
yesterday's high, we SHORT; when it reaches yesterday's low, we go LONG.

Entry logic:
  1. Compute prev_day_high and prev_day_low from daily_features (shifted by 1)
  2. Compute ATR_20 = rolling 20-day mean of (daily_high - daily_low)
  3. Gap check: if daily_open already beyond the level, skip that direction
     (price gapped past the level, limit order would have filled at open)
  4. Scan bars_1m: find first touch of prev_day_high or prev_day_low
  5. Entry = the level (limit order)
  6. Stop = level +/- stop_mult * ATR_20
  7. Target = entry +/- RR * risk_distance

Gates:
  B. Risk floor: stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS
  D. Gap guard: skip direction if open already beyond level

Grid: 4 RR targets x 3 stop multipliers x 2 directions = 24 combos per day
Walk-forward: 12-month training windows, monthly steps, OOS from 2018-01-01
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec, to_r_multiple
from pipeline.paths import GOLD_DB_PATH
from research._alt_strategy_utils import (
    compute_strategy_metrics,
    compute_walk_forward_windows,
    load_bars_for_day,
    load_daily_features,
    resolve_bar_outcome,
    save_results,
)

SPEC = get_cost_spec("MGC")

# Grid dimensions
RR_TARGETS = [1.0, 1.5, 2.0, 2.5]
STOP_MULTIPLIERS = [0.5, 1.0, 1.5]  # fraction of ATR_20

REGIME_BOUNDARY = date(2025, 1, 1)


def prepare_prior_day_data(features: pd.DataFrame) -> pd.DataFrame:
    """Add previous day H/L and ATR_20 to features."""
    df = features.copy()
    df = df.sort_values("trading_day")

    # Previous day's high/low
    df["prev_day_high"] = df["daily_high"].shift(1)
    df["prev_day_low"] = df["daily_low"].shift(1)

    # ATR_20 (shifted by 1 to avoid lookahead)
    df["true_range"] = df["daily_high"] - df["daily_low"]
    df["atr_20"] = df["true_range"].rolling(window=20, min_periods=20).mean().shift(1)

    return df


def compute_prior_day_hl_outcomes(
    db_path: Path,
    prep_data: pd.DataFrame,
) -> list[dict]:
    """Compute trade outcomes for all prior-day H/L fade setups.

    For each day, checks both SHORT at prev_day_high and LONG at prev_day_low.
    Returns list of outcome dicts across all (day, direction, rr, stop_mult) combos.
    """
    outcomes = []

    eligible = prep_data.dropna(
        subset=["prev_day_high", "prev_day_low", "atr_20", "daily_open"]
    )
    eligible = eligible[eligible["atr_20"] > 0]

    total = len(eligible)
    for idx, (_, row) in enumerate(eligible.iterrows()):
        if idx % 200 == 0:
            print(f"    Processing day {idx+1}/{total}...")

        trading_day = row["trading_day"]
        if isinstance(trading_day, str):
            trading_day = date.fromisoformat(trading_day)
        elif hasattr(trading_day, "date") and callable(trading_day.date):
            trading_day = trading_day.date()

        prev_high = row["prev_day_high"]
        prev_low = row["prev_day_low"]
        atr = row["atr_20"]
        daily_open = row["daily_open"]

        bars = None  # Lazy-load only if needed

        # Try both directions
        for direction, level in [("short", prev_high), ("long", prev_low)]:
            # Gate D: Gap guard
            if direction == "short" and daily_open > level:
                continue  # Already gapped above prev high
            if direction == "long" and daily_open < level:
                continue  # Already gapped below prev low

            for stop_mult in STOP_MULTIPLIERS:
                stop_distance = stop_mult * atr

                # Gate B: Risk floor
                if stop_distance < SPEC.min_risk_floor_points:
                    continue

                if direction == "short":
                    stop_price = level + stop_distance
                else:
                    stop_price = level - stop_distance

                for rr in RR_TARGETS:
                    reward_distance = rr * stop_distance
                    if direction == "short":
                        target_price = level - reward_distance
                    else:
                        target_price = level + reward_distance

                    # Lazy-load bars
                    if bars is None:
                        bars = load_bars_for_day(db_path, trading_day)
                        if bars.empty:
                            break  # No bars -> skip this day entirely

                    # Find first bar that touches the level
                    entry_bar_idx = _find_touch_bar(bars, level, direction)
                    if entry_bar_idx is None:
                        continue  # Price never reached the level

                    # Resolve outcome starting from bar AFTER entry
                    result = resolve_bar_outcome(
                        bars, level, stop_price, target_price,
                        direction, entry_bar_idx + 1,
                    )

                    if result is None:
                        last_close = bars.iloc[-1]["close"]
                        if direction == "long":
                            pnl_points = last_close - level
                        else:
                            pnl_points = level - last_close
                        pnl_r = to_r_multiple(SPEC, level, stop_price, pnl_points)
                        outcome_type = "eod"
                    else:
                        pnl_r = to_r_multiple(
                            SPEC, level, stop_price, result["pnl_points"]
                        )
                        outcome_type = result["outcome"]

                    outcomes.append({
                        "trading_day": str(trading_day),
                        "direction": direction,
                        "level": level,
                        "atr_20": atr,
                        "stop_multiplier": stop_mult,
                        "rr_target": rr,
                        "entry_price": level,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "stop_distance": stop_distance,
                        "pnl_r": pnl_r,
                        "outcome": outcome_type,
                    })

                if bars is not None and bars.empty:
                    break  # No bars for this day
            if bars is not None and bars.empty:
                break

    return outcomes


def _find_touch_bar(
    bars: pd.DataFrame, level: float, direction: str
) -> int | None:
    """Find first bar index where price touches the level.

    For SHORT at prev_high: bar high >= level (price reached up to the level).
    For LONG at prev_low: bar low <= level (price reached down to the level).
    """
    if direction == "short":
        for i in range(len(bars)):
            if bars.iloc[i]["high"] >= level:
                return i
    else:
        for i in range(len(bars)):
            if bars.iloc[i]["low"] <= level:
                return i
    return None


def run_walk_forward(
    db_path: Path,
    train_months: int = 12,
    test_start: date = date(2018, 1, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward analysis for prior day H/L fade."""
    full_start = date(2016, 1, 1)
    features = load_daily_features(db_path, full_start, test_end)
    prep_data = prepare_prior_day_data(features)

    print("  Computing all prior-day H/L outcomes...")
    all_outcomes = compute_prior_day_hl_outcomes(db_path, prep_data)
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

        # Find best (stop_mult, rr, direction) on training data
        best_combo = None
        best_sharpe = -999.0

        for stop_mult in STOP_MULTIPLIERS:
            sm_train = train_data[train_data["stop_multiplier"] == stop_mult]
            for rr in RR_TARGETS:
                rr_train = sm_train[sm_train["rr_target"] == rr]
                # Try combined directions and each direction separately
                for dir_filter in [None, "long", "short"]:
                    if dir_filter:
                        subset = rr_train[rr_train["direction"] == dir_filter]
                    else:
                        subset = rr_train
                    if len(subset) < 20:
                        continue
                    stats = compute_strategy_metrics(subset["pnl_r"].values)
                    if stats and stats["sharpe"] > best_sharpe:
                        best_sharpe = stats["sharpe"]
                        best_combo = (stop_mult, rr, dir_filter)

        if best_combo is None:
            continue

        stop_mult, rr, dir_filter = best_combo

        # Apply to OOS
        oos = test_data[
            (test_data["stop_multiplier"] == stop_mult)
            & (test_data["rr_target"] == rr)
        ]
        if dir_filter:
            oos = oos[oos["direction"] == dir_filter]
        if oos.empty:
            continue

        oos_pnls = oos["pnl_r"].values
        oos_stats = compute_strategy_metrics(oos_pnls)
        oos_all_pnls.extend(oos_pnls)
        oos_all_dates.extend(oos["trading_day_date"].values)

        dir_label = dir_filter or "both"
        window_results.append({
            "test_start": str(w["test_start"]),
            "test_end": str(w["test_end"]),
            "train_n": len(train_data[
                (train_data["stop_multiplier"] == stop_mult)
                & (train_data["rr_target"] == rr)
            ]),
            "selected": f"SM{stop_mult}_RR{rr}_{dir_label}",
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
    features = load_daily_features(db_path, start, end)
    prep_data = prepare_prior_day_data(features)
    outcomes = compute_prior_day_hl_outcomes(db_path, prep_data)

    if not outcomes:
        return {"grid": []}

    outcomes_df = pd.DataFrame(outcomes)
    grid_results = []

    for stop_mult in STOP_MULTIPLIERS:
        sm_data = outcomes_df[outcomes_df["stop_multiplier"] == stop_mult]
        for rr in RR_TARGETS:
            rr_data = sm_data[sm_data["rr_target"] == rr]
            for dir_filter in [None, "long", "short"]:
                if dir_filter:
                    subset = rr_data[rr_data["direction"] == dir_filter]
                else:
                    subset = rr_data
                if subset.empty:
                    continue
                stats = compute_strategy_metrics(subset["pnl_r"].values)
                if stats:
                    dir_label = dir_filter or "both"
                    grid_results.append({
                        "stop_multiplier": stop_mult,
                        "rr_target": rr,
                        "direction": dir_label,
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

    # Regime check: must show edge in low-vol period
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
    parser = argparse.ArgumentParser(description="Prior Day H/L Fade analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--full-period-only", action="store_true",
                        help="Skip walk-forward, just run full-period grid search")
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("PRIOR DAY HIGH/LOW FADE STRATEGY ANALYSIS")
    print(sep)
    print()
    print("Entry: Fade at previous day's high (SHORT) or low (LONG)")
    print("Stop: ATR_20-based distance beyond the level")
    print(f"Grid: {len(RR_TARGETS)} RR x {len(STOP_MULTIPLIERS)} stop mults x 3 dirs = "
          f"{len(RR_TARGETS) * len(STOP_MULTIPLIERS) * 3} combos")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print(f"Gate D: Skip if open already beyond level")
    print()

    if args.full_period_only:
        print("--- FULL PERIOD GRID SEARCH ---")
        result = run_full_period_analysis(args.db_path)

        if result["grid"]:
            print(f"  Grid results ({len(result['grid'])} combos):")
            sorted_grid = sorted(result["grid"], key=lambda x: x["sharpe"], reverse=True)
            for g in sorted_grid[:15]:
                print(f"    SM{g['stop_multiplier']} RR{g['rr_target']} {g['direction']}: "
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

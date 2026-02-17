#!/usr/bin/env python3
"""
Double Break (Fakeout) strategy analysis.

Trades the reversal after a failed ORB breakout. When price breaks one side of
the ORB then reverses to break the opposite side (a "double break"), we enter
in the reversal direction.

Entry logic:
  1. Day must have double_break = True for the ORB session
  2. First break direction = orb_{label}_break_dir (this is the FAILED direction)
  3. Entry direction = OPPOSITE of first break
  4. Scan bars_1m after first break to find when price crosses opposite ORB level
  5. Entry price = opposite ORB level (limit order)
  6. Stop = fakeout extreme (max adverse move during first break attempt)
  7. Target = entry +/- RR * risk

Gates:
  A. Entry MUST occur AFTER break_ts (no same-bar fill)
  B. Risk floor: fakeout stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS

Grid: ORB label x RR target (1.0-3.0) x ORB size filter (G2-G6)
Walk-forward: 12-month training windows, monthly steps
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
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

BRISBANE_TZ = ZoneInfo("Australia/Brisbane")
UTC_TZ = ZoneInfo("UTC")

SPEC = get_cost_spec("MGC")

# Grid dimensions
ORB_LABELS = ["0900", "1000", "1800"]
RR_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0]
SIZE_FILTERS = {"G2": 2.0, "G3": 3.0, "G4": 4.0, "G5": 5.0, "G6": 6.0}


def find_double_break_entry(
    bars: pd.DataFrame,
    break_ts: pd.Timestamp,
    break_dir: str,
    orb_high: float,
    orb_low: float,
) -> dict | None:
    """Find the double-break reversal entry point.

    Scans bars after break_ts to find when price crosses the opposite ORB level.
    Also computes the fakeout extreme (worst price during first breakout attempt).

    Returns dict with entry_price, stop_price, direction, entry_bar_idx,
    fakeout_extreme, or None if no reversal found.
    """
    # Ensure break_ts is timezone-aware UTC for comparison
    if break_ts.tzinfo is None:
        break_ts_utc = break_ts.tz_localize("UTC")
    else:
        break_ts_utc = break_ts.astimezone(UTC_TZ)

    # Determine reversal direction
    # NOTE: daily_features stores break_dir as "long"/"short", not "bull"/"bear"
    if break_dir == "long":
        # First break was up (long), it failed -> we go SHORT
        # Entry when price crosses below orb_low
        entry_price = orb_low
        direction = "short"
    elif break_dir == "short":
        # First break was down (short), it failed -> we go LONG
        # Entry when price crosses above orb_high
        entry_price = orb_high
        direction = "long"
    else:
        return None

    # Track fakeout extreme between break_ts and reversal entry
    fakeout_extreme = None
    entry_bar_idx = None

    for i in range(len(bars)):
        bar = bars.iloc[i]
        bar_ts = bar["ts_utc"]
        # Ensure bar_ts is comparable
        if hasattr(bar_ts, "astimezone"):
            bar_ts_utc = bar_ts.astimezone(UTC_TZ)
        elif hasattr(bar_ts, "tz_localize"):
            bar_ts_utc = bar_ts.tz_localize("UTC")
        else:
            bar_ts_utc = bar_ts

        # Gate A: Only consider bars AFTER break_ts (strict >)
        if bar_ts_utc <= break_ts_utc:
            continue

        # Track fakeout extreme (before entry is found)
        if entry_bar_idx is None:
            if break_dir == "long":
                # Fakeout extreme = highest high during failed long break
                # Check for reversal FIRST -- if this bar triggers entry,
                # don't include its high in fakeout extreme (Bug 3 fix)
                if bar["low"] <= entry_price:
                    entry_bar_idx = i
                else:
                    # Only update fakeout extreme for non-entry bars
                    if fakeout_extreme is None or bar["high"] > fakeout_extreme:
                        fakeout_extreme = bar["high"]
            else:
                # Fakeout extreme = lowest low during failed short break
                if bar["high"] >= entry_price:
                    entry_bar_idx = i
                else:
                    if fakeout_extreme is None or bar["low"] < fakeout_extreme:
                        fakeout_extreme = bar["low"]

    if entry_bar_idx is None:
        return None

    # If entry happened on the first bar after break (no prior fakeout bars),
    # use the entry bar's extreme as fakeout extreme
    if fakeout_extreme is None:
        entry_bar = bars.iloc[entry_bar_idx]
        if break_dir == "long":
            fakeout_extreme = entry_bar["high"]
        else:
            fakeout_extreme = entry_bar["low"]

    # Compute stop from fakeout extreme
    if direction == "short":
        stop_price = fakeout_extreme  # Stop above the failed bull high
    else:
        stop_price = fakeout_extreme  # Stop below the failed bear low

    # Gate B: Risk floor check
    risk_points = abs(entry_price - stop_price)
    if risk_points < SPEC.min_risk_floor_points:
        return None

    return {
        "entry_price": entry_price,
        "stop_price": stop_price,
        "direction": direction,
        "entry_bar_idx": entry_bar_idx,
        "fakeout_extreme": fakeout_extreme,
        "risk_points": risk_points,
    }


def compute_double_break_outcomes(
    db_path: Path,
    features: pd.DataFrame,
    orb_label: str,
) -> list[dict]:
    """Compute trade outcomes for all double-break days in features.

    Returns list of outcome dicts, one per (day, rr) where entry was found.
    """
    outcomes = []
    col_db = f"orb_{orb_label}_double_break"
    col_dir = f"orb_{orb_label}_break_dir"
    col_high = f"orb_{orb_label}_high"
    col_low = f"orb_{orb_label}_low"
    col_size = f"orb_{orb_label}_size"
    col_break_ts = f"orb_{orb_label}_break_ts"

    # Filter to double-break days only
    db_days = features[features[col_db] == True].copy()  # noqa: E712

    for _, row in db_days.iterrows():
        trading_day = row["trading_day"]
        if isinstance(trading_day, str):
            trading_day = date.fromisoformat(trading_day)
        elif hasattr(trading_day, "date"):
            trading_day = trading_day.date() if callable(trading_day.date) else trading_day

        break_dir = row[col_dir]
        orb_high = row[col_high]
        orb_low = row[col_low]
        orb_size = row[col_size]
        break_ts = row[col_break_ts]

        if pd.isna(break_dir) or pd.isna(orb_high) or pd.isna(orb_low):
            continue
        if pd.isna(break_ts):
            continue

        # Load 1m bars for this day
        bars = load_bars_for_day(db_path, trading_day)
        if bars.empty:
            continue

        # Find entry point
        entry_info = find_double_break_entry(
            bars, break_ts, break_dir, orb_high, orb_low
        )
        if entry_info is None:
            continue

        # Compute outcome for each RR target
        for rr in RR_TARGETS:
            risk = entry_info["risk_points"]
            reward = rr * risk

            if entry_info["direction"] == "long":
                target_price = entry_info["entry_price"] + reward
            else:
                target_price = entry_info["entry_price"] - reward

            result = resolve_bar_outcome(
                bars,
                entry_info["entry_price"],
                entry_info["stop_price"],
                target_price,
                entry_info["direction"],
                entry_info["entry_bar_idx"] + 1,  # Start AFTER entry bar
            )

            if result is None:
                # No resolution: treat as open at end of day -> use last bar close
                last_close = bars.iloc[-1]["close"]
                if entry_info["direction"] == "long":
                    pnl_points = last_close - entry_info["entry_price"]
                else:
                    pnl_points = entry_info["entry_price"] - last_close
                pnl_r = to_r_multiple(
                    SPEC, entry_info["entry_price"],
                    entry_info["stop_price"], pnl_points
                )
                outcome_type = "eod"
            else:
                pnl_r = to_r_multiple(
                    SPEC, entry_info["entry_price"],
                    entry_info["stop_price"], result["pnl_points"]
                )
                outcome_type = result["outcome"]

            outcomes.append({
                "trading_day": str(trading_day),
                "orb_label": orb_label,
                "orb_size": orb_size,
                "rr_target": rr,
                "direction": entry_info["direction"],
                "entry_price": entry_info["entry_price"],
                "stop_price": entry_info["stop_price"],
                "target_price": target_price,
                "risk_points": entry_info["risk_points"],
                "pnl_r": pnl_r,
                "outcome": outcome_type,
            })

    return outcomes


def run_walk_forward(
    db_path: Path,
    orb_label: str,
    train_months: int = 12,
    test_start: date = date(2023, 1, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward analysis for double-break strategy on one ORB session.

    For each window:
      1. Compute outcomes on training period
      2. Find best (size_filter, rr) combo by Sharpe
      3. Apply that combo to test period (OOS)
    """
    # Load all features once
    full_start = date(2016, 1, 1)
    features = load_daily_features(db_path, full_start, test_end)

    # Pre-compute all outcomes (expensive but avoids reloading bars per window)
    print(f"  Computing all double-break outcomes for {orb_label}...")
    all_outcomes = compute_double_break_outcomes(db_path, features, orb_label)
    if not all_outcomes:
        print(f"  No double-break outcomes found for {orb_label}")
        return {"orb_label": orb_label, "windows": [], "combined_oos": None}

    outcomes_df = pd.DataFrame(all_outcomes)
    outcomes_df["trading_day"] = pd.to_datetime(outcomes_df["trading_day"]).dt.date
    print(f"  {len(outcomes_df)} total outcomes across {outcomes_df['trading_day'].nunique()} days")

    windows = compute_walk_forward_windows(test_start, test_end, train_months)
    window_results = []
    oos_all_pnls = []

    for w in windows:
        train_mask = (
            (outcomes_df["trading_day"] >= w["train_start"])
            & (outcomes_df["trading_day"] <= w["train_end"])
        )
        test_mask = (
            (outcomes_df["trading_day"] >= w["test_start"])
            & (outcomes_df["trading_day"] <= w["test_end"])
        )

        train_data = outcomes_df[train_mask]
        test_data = outcomes_df[test_mask]

        if train_data.empty:
            continue

        # Find best (size_filter, rr) on training data
        best_combo = None
        best_sharpe = -999.0

        for filt_name, filt_thresh in SIZE_FILTERS.items():
            filt_train = train_data[train_data["orb_size"] >= filt_thresh]
            for rr in RR_TARGETS:
                rr_train = filt_train[filt_train["rr_target"] == rr]
                if len(rr_train) < 20:
                    continue
                stats = compute_strategy_metrics(rr_train["pnl_r"].values)
                if stats and stats["sharpe"] > best_sharpe:
                    best_sharpe = stats["sharpe"]
                    best_combo = (filt_name, filt_thresh, rr)

        if best_combo is None:
            continue

        filt_name, filt_thresh, rr = best_combo

        # Apply to OOS
        oos = test_data[
            (test_data["orb_size"] >= filt_thresh)
            & (test_data["rr_target"] == rr)
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
                (train_data["orb_size"] >= filt_thresh)
                & (train_data["rr_target"] == rr)
            ]),
            "selected": f"{filt_name}_RR{rr}",
            "train_sharpe": best_sharpe,
            "oos_stats": oos_stats,
        })

    combined_oos = None
    if oos_all_pnls:
        combined_oos = compute_strategy_metrics(np.array(oos_all_pnls))

    return {
        "orb_label": orb_label,
        "train_months": train_months,
        "windows": window_results,
        "combined_oos": combined_oos,
    }


def run_full_period_analysis(
    db_path: Path,
    orb_label: str,
    start: date = date(2022, 1, 1),
    end: date = date(2026, 2, 1),
) -> dict:
    """Run full-period (non-walk-forward) analysis for grid search."""
    features = load_daily_features(db_path, start, end)
    outcomes = compute_double_break_outcomes(db_path, features, orb_label)

    if not outcomes:
        return {"orb_label": orb_label, "grid": []}

    outcomes_df = pd.DataFrame(outcomes)
    grid_results = []

    for filt_name, filt_thresh in SIZE_FILTERS.items():
        filt_data = outcomes_df[outcomes_df["orb_size"] >= filt_thresh]
        for rr in RR_TARGETS:
            rr_data = filt_data[filt_data["rr_target"] == rr]
            if rr_data.empty:
                continue
            stats = compute_strategy_metrics(rr_data["pnl_r"].values)
            if stats:
                grid_results.append({
                    "filter": filt_name,
                    "rr_target": rr,
                    **stats,
                })

    return {"orb_label": orb_label, "grid": grid_results}


def main():
    parser = argparse.ArgumentParser(description="Double Break strategy analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--orb-labels", nargs="+", default=ORB_LABELS)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--full-period-only", action="store_true",
                        help="Skip walk-forward, just run full-period grid search")
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("DOUBLE BREAK (FAKEOUT) STRATEGY ANALYSIS")
    print(sep)
    print()
    print("Entry: Reversal after failed ORB breakout (double break)")
    print("Stop: Fakeout extreme (max adverse during first break)")
    print(f"Grid: {len(ORB_LABELS)} sessions x {len(RR_TARGETS)} RR x {len(SIZE_FILTERS)} filters")
    print(f"Gate A: No same-bar fill (entry after break_ts)")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print()

    all_results = {}

    for orb_label in args.orb_labels:
        print(f"--- ORB SESSION: {orb_label} ---")

        if args.full_period_only:
            result = run_full_period_analysis(args.db_path, orb_label)
            all_results[orb_label] = result

            if result["grid"]:
                print(f"  Grid results ({len(result['grid'])} combos):")
                # Sort by Sharpe
                sorted_grid = sorted(result["grid"], key=lambda x: x["sharpe"], reverse=True)
                for g in sorted_grid[:10]:
                    print(f"    {g['filter']} RR{g['rr_target']}: N={g['n']}, WR={g['wr']:.0%}, "
                          f"ExpR={g['expr']:+.3f}, Sharpe={g['sharpe']:.3f}, MaxDD={g['maxdd']:+.1f}R")
            else:
                print("  No outcomes found")
            print()
        else:
            result = run_walk_forward(
                args.db_path, orb_label, args.train_months
            )
            all_results[orb_label] = result

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
            print()

    if args.output:
        save_results(all_results, args.output)

    print(sep)
    print("DONE")
    print(sep)


if __name__ == "__main__":
    main()

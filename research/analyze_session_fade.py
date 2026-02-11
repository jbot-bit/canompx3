#!/usr/bin/env python3
"""
Session Range Fade strategy analysis (Asia range -> London fade).

When London session pushes price to the Asia session extremes, we fade
the move expecting a reversal back into the range.

Entry logic:
  1. After Asia session ends (07:00 UTC / 17:00 Brisbane), record
     session_asia_high and session_asia_low from daily_features
  2. During London session (08:00-13:00 UTC), scan bars_1m:
     - If price reaches session_asia_high: SHORT (fade the push)
     - If price reaches session_asia_low: LONG (fade the push)
  3. Stop = level +/- stop_mult * asia_range beyond the level
  4. Target = entry +/- RR * risk_distance
  5. Filter: Asia range must be >= min_range threshold (analogous to ORB G2/G4)

Gates:
  B. Risk floor: stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS
  E. Only scan London session bars (08:00-13:00 UTC)

Grid: 3 RR targets x 3 stop multipliers x 3 range filters = 27 combos
Walk-forward: 12-month training windows, monthly steps, OOS from 2018-01-01
"""

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

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
RR_TARGETS = [1.0, 1.5, 2.0]
STOP_MULTIPLIERS = [0.5, 1.0, 1.5]  # fraction of asia_range
RANGE_FILTERS = {"R2": 2.0, "R4": 4.0, "R6": 6.0}  # min asia range in points

# London session: 08:00-13:00 UTC
LONDON_START_HOUR = 8
LONDON_END_HOUR = 13

REGIME_BOUNDARY = date(2025, 1, 1)


def compute_session_fade_outcomes(
    db_path: Path,
    features: pd.DataFrame,
) -> list[dict]:
    """Compute trade outcomes for all Asia range fade setups during London.

    Returns list of outcome dicts across all (day, direction, rr, stop_mult, range_filter) combos.
    """
    outcomes = []

    # Need asia session columns
    eligible = features.dropna(subset=["session_asia_high", "session_asia_low"])
    eligible = eligible.copy()
    eligible["asia_range"] = eligible["session_asia_high"] - eligible["session_asia_low"]
    eligible = eligible[eligible["asia_range"] > 0]

    total = len(eligible)
    for idx, (_, row) in enumerate(eligible.iterrows()):
        if idx % 200 == 0:
            print(f"    Processing day {idx+1}/{total}...")

        trading_day = row["trading_day"]
        if isinstance(trading_day, str):
            trading_day = date.fromisoformat(trading_day)
        elif hasattr(trading_day, "date") and callable(trading_day.date):
            trading_day = trading_day.date()

        asia_high = row["session_asia_high"]
        asia_low = row["session_asia_low"]
        asia_range = asia_high - asia_low

        # Load bars and filter to London session only
        bars = load_bars_for_day(db_path, trading_day)
        if bars.empty:
            continue

        london_bars = _filter_london_bars(bars)
        if london_bars.empty:
            continue

        # Try both directions
        for direction, level in [("short", asia_high), ("long", asia_low)]:
            # Find first London bar that touches the level
            entry_bar_idx_london = _find_touch_bar(london_bars, level, direction)
            if entry_bar_idx_london is None:
                continue

            # Map London bar index back to full bars index for outcome resolution
            london_bar_ts = london_bars.iloc[entry_bar_idx_london]["ts_utc"]
            entry_bar_idx = _find_bar_by_ts(bars, london_bar_ts)
            if entry_bar_idx is None:
                continue

            for range_name, range_min in RANGE_FILTERS.items():
                if asia_range < range_min:
                    continue

                for stop_mult in STOP_MULTIPLIERS:
                    stop_distance = stop_mult * asia_range

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

                        # Resolve outcome using ALL bars after entry (not just London)
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
                            "asia_range": asia_range,
                            "range_filter": range_name,
                            "stop_multiplier": stop_mult,
                            "rr_target": rr,
                            "entry_price": level,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "stop_distance": stop_distance,
                            "pnl_r": pnl_r,
                            "outcome": outcome_type,
                        })

    return outcomes


def _filter_london_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter bars to London session (08:00-13:00 UTC)."""
    if bars.empty:
        return bars

    ts = bars["ts_utc"]
    # Extract UTC hour - handle both tz-aware and tz-naive timestamps
    if hasattr(ts.iloc[0], "hour"):
        # tz-aware: convert to UTC first
        if ts.dt.tz is not None:
            ts_utc = ts.dt.tz_convert("UTC")
        else:
            ts_utc = ts
        hours = ts_utc.dt.hour
    else:
        hours = pd.to_datetime(ts).dt.hour

    mask = (hours >= LONDON_START_HOUR) & (hours < LONDON_END_HOUR)
    return bars[mask].reset_index(drop=True)


def _find_touch_bar(
    bars: pd.DataFrame, level: float, direction: str
) -> int | None:
    """Find first bar index where price touches the level."""
    if direction == "short":
        for i in range(len(bars)):
            if bars.iloc[i]["high"] >= level:
                return i
    else:
        for i in range(len(bars)):
            if bars.iloc[i]["low"] <= level:
                return i
    return None


def _find_bar_by_ts(bars: pd.DataFrame, ts) -> int | None:
    """Find bar index matching a timestamp."""
    for i in range(len(bars)):
        if bars.iloc[i]["ts_utc"] == ts:
            return i
    return None


def run_walk_forward(
    db_path: Path,
    train_months: int = 12,
    test_start: date = date(2018, 1, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward analysis for session fade strategy."""
    full_start = date(2016, 1, 1)
    features = load_daily_features(db_path, full_start, test_end)

    print("  Computing all session fade outcomes...")
    all_outcomes = compute_session_fade_outcomes(db_path, features)
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

        # Find best (range_filter, stop_mult, rr, direction) on training data
        best_combo = None
        best_sharpe = -999.0

        for range_name in RANGE_FILTERS:
            rf_train = train_data[train_data["range_filter"] == range_name]
            for stop_mult in STOP_MULTIPLIERS:
                sm_train = rf_train[rf_train["stop_multiplier"] == stop_mult]
                for rr in RR_TARGETS:
                    rr_train = sm_train[sm_train["rr_target"] == rr]
                    for dir_filter in [None, "long", "short"]:
                        if dir_filter:
                            subset = rr_train[rr_train["direction"] == dir_filter]
                        else:
                            subset = rr_train
                        if len(subset) < 15:
                            continue
                        stats = compute_strategy_metrics(subset["pnl_r"].values)
                        if stats and stats["sharpe"] > best_sharpe:
                            best_sharpe = stats["sharpe"]
                            best_combo = (range_name, stop_mult, rr, dir_filter)

        if best_combo is None:
            continue

        range_name, stop_mult, rr, dir_filter = best_combo

        # Apply to OOS
        oos = test_data[
            (test_data["range_filter"] == range_name)
            & (test_data["stop_multiplier"] == stop_mult)
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
                (train_data["range_filter"] == range_name)
                & (train_data["stop_multiplier"] == stop_mult)
                & (train_data["rr_target"] == rr)
            ]),
            "selected": f"{range_name}_SM{stop_mult}_RR{rr}_{dir_label}",
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
    outcomes = compute_session_fade_outcomes(db_path, features)

    if not outcomes:
        return {"grid": []}

    outcomes_df = pd.DataFrame(outcomes)
    grid_results = []

    for range_name in RANGE_FILTERS:
        rf_data = outcomes_df[outcomes_df["range_filter"] == range_name]
        for stop_mult in STOP_MULTIPLIERS:
            sm_data = rf_data[rf_data["stop_multiplier"] == stop_mult]
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
                            "range_filter": range_name,
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
    parser = argparse.ArgumentParser(description="Session Range Fade analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--full-period-only", action="store_true",
                        help="Skip walk-forward, just run full-period grid search")
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("SESSION RANGE FADE STRATEGY ANALYSIS (ASIA -> LONDON)")
    print(sep)
    print()
    print("Entry: Fade Asia H/L during London session (SHORT at asia_high, LONG at asia_low)")
    print("Stop: Asia-range-based distance beyond the level")
    print(f"Grid: {len(RR_TARGETS)} RR x {len(STOP_MULTIPLIERS)} stop mults x "
          f"{len(RANGE_FILTERS)} range filters x 3 dirs = "
          f"{len(RR_TARGETS) * len(STOP_MULTIPLIERS) * len(RANGE_FILTERS) * 3} combos")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print(f"Gate E: Entry only during London session (08:00-13:00 UTC)")
    print()

    if args.full_period_only:
        print("--- FULL PERIOD GRID SEARCH ---")
        result = run_full_period_analysis(args.db_path)

        if result["grid"]:
            print(f"  Grid results ({len(result['grid'])} combos):")
            sorted_grid = sorted(result["grid"], key=lambda x: x["sharpe"], reverse=True)
            for g in sorted_grid[:15]:
                print(f"    {g['range_filter']} SM{g['stop_multiplier']} RR{g['rr_target']} "
                      f"{g['direction']}: N={g['n']}, WR={g['wr']:.0%}, "
                      f"ExpR={g['expr']:+.3f}, Sharpe={g['sharpe']:.3f}, "
                      f"MaxDD={g['maxdd']:+.1f}R")
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

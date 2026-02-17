#!/usr/bin/env python3
"""
Value Area (Volume Profile) strategy analysis.

Computes daily Market Profile from 1m bars, then trades reversion to POC
and breakout from Value Area boundaries.

Entry logic - REVERSION:
  1. Compute prior day's volume profile: POC, VAH, VAL (70% of volume)
  2. If today's open > prior VAH and price crosses back below VAH: SHORT to POC
  3. If today's open < prior VAL and price crosses back above VAL: LONG to POC
  4. Stop: entry +/- stop_mult * ATR_20

Entry logic - BREAKOUT:
  1. Price breaks out of prior day's Value Area with volume > 1.5x session avg
  2. Enter in direction of break
  3. Target: entry +/- RR * risk_distance
  4. Stop: entry -/+ stop_mult * ATR_20

Gates:
  B. Risk floor: stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS

Grid: 3 stop multipliers x 3 RR targets x 2 bin sizes x 2 time filters x 2 modes = 72 combos
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
from research._alt_strategy_utils import (
    annualize_sharpe,
    compute_strategy_metrics,
    compute_walk_forward_windows,
    load_bars_for_day,
    load_daily_features,
    save_results,
)

SPEC = get_cost_spec("MGC")

# Grid dimensions
STOP_MULTIPLIERS = [0.5, 1.0, 1.5]  # fraction of ATR_20
RR_TARGETS = [1.0, 1.5, 2.0]        # for breakout mode
BIN_SIZES = [0.1, 0.5]              # price bin width in points
TIME_FILTERS = {
    "ny_only": (23, 7),   # 09:00-17:00 Brisbane = 23:00-07:00 UTC
    "all": (0, 23),
}
VALUE_AREA_PCT = 0.70  # 70% of volume
VOLUME_BREAKOUT_MULT = 1.5  # Volume must exceed 1.5x average


def compute_volume_profile(bars_1m: pd.DataFrame, bin_size: float) -> dict | None:
    """Compute volume profile from 1m bars.

    Returns dict with poc, vah, val, total_volume or None if insufficient data.
    """
    if bars_1m.empty or len(bars_1m) < 60:
        return None

    # Typical price per bar
    tp = (bars_1m["high"].values + bars_1m["low"].values + bars_1m["close"].values) / 3.0
    vol = bars_1m["volume"].values.astype(float)

    if vol.sum() == 0:
        return None

    # Create price bins
    price_min = tp.min()
    price_max = tp.max()
    if price_max - price_min < bin_size:
        return None

    bins = np.arange(price_min, price_max + bin_size, bin_size)
    if len(bins) < 3:
        return None

    # Assign each bar's volume to its price bin
    bin_indices = np.digitize(tp, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    volume_per_bin = np.zeros(len(bins) - 1)
    for i in range(len(tp)):
        volume_per_bin[bin_indices[i]] += vol[i]

    total_vol = volume_per_bin.sum()
    if total_vol == 0:
        return None

    # POC = bin with highest volume
    poc_idx = np.argmax(volume_per_bin)
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0

    # Value Area: expand from POC until 70% of total volume captured
    va_vol = volume_per_bin[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx

    while va_vol / total_vol < VALUE_AREA_PCT:
        expand_lo = volume_per_bin[lo_idx - 1] if lo_idx > 0 else 0
        expand_hi = volume_per_bin[hi_idx + 1] if hi_idx < len(volume_per_bin) - 2 else 0

        if expand_lo == 0 and expand_hi == 0:
            break

        if expand_lo >= expand_hi and lo_idx > 0:
            lo_idx -= 1
            va_vol += volume_per_bin[lo_idx]
        elif hi_idx < len(volume_per_bin) - 2:
            hi_idx += 1
            va_vol += volume_per_bin[hi_idx]
        elif lo_idx > 0:
            lo_idx -= 1
            va_vol += volume_per_bin[lo_idx]
        else:
            break

    val = bins[lo_idx]  # Value Area Low
    vah = bins[hi_idx + 1]  # Value Area High

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "total_volume": total_vol,
        "avg_bar_volume": total_vol / len(bars_1m),
    }


def find_reversion_signals(
    bars_1m: pd.DataFrame,
    prior_profile: dict,
    time_filter: tuple,
) -> list[dict]:
    """Find reversion signals: price opens outside VA and crosses back in."""
    if bars_1m.empty or prior_profile is None:
        return []

    vah = prior_profile["vah"]
    val = prior_profile["val"]
    poc = prior_profile["poc"]

    signals = []

    # Get UTC hours for time filtering
    ts_col = bars_1m["ts_utc"]
    if ts_col.dt.tz is not None:
        hours = ts_col.dt.tz_convert("UTC").dt.hour.values
    else:
        hours = ts_col.dt.hour.values

    first_open = bars_1m.iloc[0]["open"]

    # Check if opened outside VA
    if first_open > vah:
        # Look for cross back below VAH -> SHORT to POC
        for i in range(1, len(bars_1m)):
            hour = hours[i]
            tf_start, tf_end = time_filter
            if tf_start < tf_end:
                if not (tf_start <= hour < tf_end):
                    continue
            else:  # wraps midnight
                if not (hour >= tf_start or hour < tf_end):
                    continue

            if bars_1m.iloc[i]["close"] < vah:
                entry_price = bars_1m.iloc[i]["close"]
                signals.append({
                    "mode": "reversion",
                    "direction": "short",
                    "entry_bar_idx": i + 1,
                    "entry_price": entry_price,
                    "target_price": poc,
                })
                break

    elif first_open < val:
        # Look for cross back above VAL -> LONG to POC
        for i in range(1, len(bars_1m)):
            hour = hours[i]
            tf_start, tf_end = time_filter
            if tf_start < tf_end:
                if not (tf_start <= hour < tf_end):
                    continue
            else:
                if not (hour >= tf_start or hour < tf_end):
                    continue

            if bars_1m.iloc[i]["close"] > val:
                entry_price = bars_1m.iloc[i]["close"]
                signals.append({
                    "mode": "reversion",
                    "direction": "long",
                    "entry_bar_idx": i + 1,
                    "entry_price": entry_price,
                    "target_price": poc,
                })
                break

    return signals


def find_breakout_signals(
    bars_1m: pd.DataFrame,
    prior_profile: dict,
    time_filter: tuple,
) -> list[dict]:
    """Find breakout signals: price breaks VA boundary with high volume."""
    if bars_1m.empty or prior_profile is None:
        return []

    vah = prior_profile["vah"]
    val = prior_profile["val"]
    avg_vol = prior_profile["avg_bar_volume"]

    signals = []
    had_long = False
    had_short = False

    ts_col = bars_1m["ts_utc"]
    if ts_col.dt.tz is not None:
        hours = ts_col.dt.tz_convert("UTC").dt.hour.values
    else:
        hours = ts_col.dt.hour.values

    # Use rolling 5-bar volume to smooth
    vol_arr = bars_1m["volume"].values.astype(float)

    for i in range(5, len(bars_1m) - 1):
        hour = hours[i]
        tf_start, tf_end = time_filter
        if tf_start < tf_end:
            if not (tf_start <= hour < tf_end):
                continue
        else:
            if not (hour >= tf_start or hour < tf_end):
                continue

        bar_close = bars_1m.iloc[i]["close"]
        recent_vol = vol_arr[i-4:i+1].mean()

        # Breakout above VAH with volume
        if bar_close > vah and recent_vol > VOLUME_BREAKOUT_MULT * avg_vol and not had_long:
            entry_price = bars_1m.iloc[i + 1]["open"]
            signals.append({
                "mode": "breakout",
                "direction": "long",
                "entry_bar_idx": i + 1,
                "entry_price": entry_price,
                "target_price": None,  # Set by RR grid
            })
            had_long = True

        # Breakout below VAL with volume
        if bar_close < val and recent_vol > VOLUME_BREAKOUT_MULT * avg_vol and not had_short:
            entry_price = bars_1m.iloc[i + 1]["open"]
            signals.append({
                "mode": "breakout",
                "direction": "short",
                "entry_bar_idx": i + 1,
                "entry_price": entry_price,
                "target_price": None,
            })
            had_short = True

        if had_long and had_short:
            break

    return signals


def compute_value_area_outcomes(
    db_path: Path,
    start: date,
    end: date,
) -> list[dict]:
    """Compute all value area trade outcomes."""
    features = load_daily_features(db_path, start, end)
    if features.empty:
        return []

    # ATR_20 (shifted by 1)
    features = features.sort_values("trading_day")
    features["true_range"] = features["daily_high"] - features["daily_low"]
    features["atr_20"] = features["true_range"].rolling(20, min_periods=20).mean().shift(1)

    eligible = features.dropna(subset=["atr_20"])
    total = len(eligible)
    all_outcomes = []

    # Pre-compute volume profiles for each day
    profiles = {}  # trading_day -> {bin_size -> profile}
    trading_days = []

    for _, row in eligible.iterrows():
        td = row["trading_day"]
        if hasattr(td, "date") and callable(td.date):
            td = td.date()
        elif isinstance(td, str):
            td = date.fromisoformat(td)
        trading_days.append((td, row["atr_20"]))

    print(f"    Computing volume profiles for {len(trading_days)} days...")
    for idx, (td, _) in enumerate(trading_days):
        if idx % 200 == 0:
            print(f"    Day {idx+1}/{len(trading_days)}...")

        bars = load_bars_for_day(db_path, td)
        if bars.empty:
            continue

        profiles[td] = {"bars": bars}
        for bin_size in BIN_SIZES:
            profiles[td][bin_size] = compute_volume_profile(bars, bin_size)

    print(f"    Resolving trade outcomes...")
    for idx in range(1, len(trading_days)):
        td, atr = trading_days[idx]
        prev_td = trading_days[idx - 1][0]

        if td not in profiles or prev_td not in profiles:
            continue

        bars = profiles[td]["bars"]

        if idx % 200 == 0:
            print(f"    Processing {idx+1}/{len(trading_days)}...")

        for bin_size in BIN_SIZES:
            prior_prof = profiles[prev_td].get(bin_size)
            if prior_prof is None:
                continue

            for tf_name, tf_range in TIME_FILTERS.items():
                # Reversion signals
                rev_signals = find_reversion_signals(bars, prior_prof, tf_range)
                for sig in rev_signals:
                    entry_price = sig["entry_price"]
                    target_price = sig["target_price"]
                    direction = sig["direction"]
                    entry_bar_idx = sig["entry_bar_idx"]

                    if entry_bar_idx >= len(bars):
                        continue

                    for stop_mult in STOP_MULTIPLIERS:
                        stop_distance = stop_mult * atr
                        if stop_distance < SPEC.min_risk_floor_points:
                            continue

                        if direction == "long":
                            stop_price = entry_price - stop_distance
                        else:
                            stop_price = entry_price + stop_distance

                        # Sanity: target must be on right side
                        if direction == "long" and target_price <= entry_price:
                            continue
                        if direction == "short" and target_price >= entry_price:
                            continue

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
                            "mode": "reversion",
                            "direction": direction,
                            "bin_size": bin_size,
                            "time_filter": tf_name,
                            "stop_multiplier": stop_mult,
                            "rr_target": 0,  # reversion targets POC directly
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "pnl_r": pnl_r,
                            "outcome": outcome_type,
                        })

                # Breakout signals
                bk_signals = find_breakout_signals(bars, prior_prof, tf_range)
                for sig in bk_signals:
                    entry_price = sig["entry_price"]
                    direction = sig["direction"]
                    entry_bar_idx = sig["entry_bar_idx"]

                    if entry_bar_idx >= len(bars):
                        continue

                    for stop_mult in STOP_MULTIPLIERS:
                        stop_distance = stop_mult * atr
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
                                "mode": "breakout",
                                "direction": direction,
                                "bin_size": bin_size,
                                "time_filter": tf_name,
                                "stop_multiplier": stop_mult,
                                "rr_target": rr,
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

        # Gate C: ambiguous bar = LOSS
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
    """Run walk-forward analysis for value area strategy."""
    # Only load data needed for training + OOS
    from research._alt_strategy_utils import _add_months
    full_start = _add_months(test_start, -(train_months + 2))

    print("  Computing all value area outcomes...")
    all_outcomes = compute_value_area_outcomes(db_path, full_start, test_end)
    if not all_outcomes:
        print("  No outcomes found")
        return {"windows": [], "combined_oos": None}

    outcomes_df = pd.DataFrame(all_outcomes)
    outcomes_df["trading_day_date"] = pd.to_datetime(outcomes_df["trading_day"]).dt.date
    n_days = outcomes_df["trading_day_date"].nunique()
    print(f"  {len(outcomes_df)} total outcomes across {n_days} days")

    # Split by mode for separate analysis
    results = {}
    for mode in ["reversion", "breakout"]:
        mode_df = outcomes_df[outcomes_df["mode"] == mode]
        if mode_df.empty:
            results[mode] = {"windows": [], "combined_oos": None}
            continue

        windows = compute_walk_forward_windows(test_start, test_end, train_months)
        window_results = []
        oos_all_pnls = []

        for w in windows:
            train_mask = (
                (mode_df["trading_day_date"] >= w["train_start"])
                & (mode_df["trading_day_date"] <= w["train_end"])
            )
            test_mask = (
                (mode_df["trading_day_date"] >= w["test_start"])
                & (mode_df["trading_day_date"] <= w["test_end"])
            )

            train_data = mode_df[train_mask]
            test_data = mode_df[test_mask]

            if train_data.empty:
                continue

            best_combo = None
            best_sharpe = -999.0

            for bs in BIN_SIZES:
                bs_train = train_data[train_data["bin_size"] == bs]
                for tf_name in TIME_FILTERS:
                    tf_train = bs_train[bs_train["time_filter"] == tf_name]
                    for sm in STOP_MULTIPLIERS:
                        sm_train = tf_train[tf_train["stop_multiplier"] == sm]
                        if mode == "breakout":
                            for rr in RR_TARGETS:
                                rr_train = sm_train[sm_train["rr_target"] == rr]
                                if len(rr_train) < 15:
                                    continue
                                stats = compute_strategy_metrics(rr_train["pnl_r"].values)
                                if stats and stats["sharpe"] > best_sharpe:
                                    best_sharpe = stats["sharpe"]
                                    best_combo = (bs, tf_name, sm, rr)
                        else:
                            if len(sm_train) < 15:
                                continue
                            stats = compute_strategy_metrics(sm_train["pnl_r"].values)
                            if stats and stats["sharpe"] > best_sharpe:
                                best_sharpe = stats["sharpe"]
                                best_combo = (bs, tf_name, sm, 0)

            if best_combo is None:
                continue

            bs, tf_name, sm, rr = best_combo
            oos_filter = (
                (test_data["bin_size"] == bs)
                & (test_data["time_filter"] == tf_name)
                & (test_data["stop_multiplier"] == sm)
            )
            if mode == "breakout":
                oos_filter = oos_filter & (test_data["rr_target"] == rr)

            oos = test_data[oos_filter]
            if oos.empty:
                continue

            oos_pnls = oos["pnl_r"].values
            oos_stats = compute_strategy_metrics(oos_pnls)
            oos_all_pnls.extend(oos_pnls)

            window_results.append({
                "test_start": str(w["test_start"]),
                "test_end": str(w["test_end"]),
                "selected": f"BS{bs}_{tf_name}_SM{sm}_RR{rr}",
                "train_sharpe": best_sharpe,
                "oos_stats": oos_stats,
            })

        combined_oos = None
        if oos_all_pnls:
            combined_oos = compute_strategy_metrics(np.array(oos_all_pnls))
            oos_years = (test_end - test_start).days / 365.25
            annualize_sharpe(combined_oos, oos_years)

        results[mode] = {
            "windows": window_results,
            "combined_oos": combined_oos,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Value Area (Volume Profile) strategy analysis")
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
    print("VALUE AREA (VOLUME PROFILE) STRATEGY ANALYSIS")
    print(sep)
    print()
    print("Mode 1: REVERSION -- fade outside Value Area, target POC")
    print("Mode 2: BREAKOUT -- volume-confirmed break of Value Area boundary")
    print(f"Grid: {len(BIN_SIZES)} bin sizes x {len(TIME_FILTERS)} time filters x "
          f"{len(STOP_MULTIPLIERS)} stop mults x {len(RR_TARGETS)} RR (breakout)")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print()

    print("--- WALK-FORWARD ANALYSIS ---")
    wf_kwargs = {"train_months": args.train_months}
    if args.start:
        wf_kwargs["test_start"] = args.start
    if args.end:
        wf_kwargs["test_end"] = args.end
    results = run_walk_forward(args.db_path, **wf_kwargs)

    for mode in ["reversion", "breakout"]:
        print(f"\n  === {mode.upper()} MODE ===")
        r = results.get(mode, {})
        windows = r.get("windows", [])

        if windows:
            for w in windows:
                oos = w.get("oos_stats")
                if oos:
                    print(f"    {w['test_start']} to {w['test_end']}: "
                          f"Selected {w['selected']}, "
                          f"OOS N={oos['n']}, WR={oos['wr']:.0%}, "
                          f"ExpR={oos['expr']:+.3f}, Sharpe={oos['sharpe']:.3f}")

            combined = r.get("combined_oos")
            if combined:
                sha = combined.get("sharpe_ann")
                sha_str = f", ShANN={sha:.3f}" if sha is not None else ""
                print(f"\n    COMBINED OOS: N={combined['n']}, WR={combined['wr']:.0%}, "
                      f"ExpR={combined['expr']:+.3f}, Sharpe={combined['sharpe']:.3f}{sha_str}, "
                      f"MaxDD={combined['maxdd']:+.1f}R, Total={combined['total']:+.1f}R")
        else:
            print("    No qualifying windows")

    if args.output:
        save_results(results, args.output)

    print()
    print(sep)
    print("DONE")
    print(sep)


if __name__ == "__main__":
    main()

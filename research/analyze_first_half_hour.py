#!/usr/bin/env python3
"""
First Half-Hour Momentum Continuation strategy analysis.

Academic research shows the first 30 minutes of a session predicts the
remainder of the day. Strong initial moves tend to continue.

Entry logic:
  1. Compute return over first 30 minutes of session (0900 Brisbane = 23:00 UTC)
  2. If first_30m_return > threshold * ATR_20: go LONG (momentum continuation)
  3. If first_30m_return < -threshold * ATR_20: go SHORT
  4. Entry = close of the 30-minute mark (bars_5m at 23:30 UTC for 0900)
  5. Stop = session low (for longs) or session high (for shorts) during first 30 min
  6. Target = entry + RR * risk_distance
  7. Alternative stop: ATR-based (stop_mult * ATR_20)

Sources:
  - "Intraday Momentum: The First Half-Hour Return Predicts the Last Half-Hour Return"
  - "The night effect of intraday trading: Chinese gold futures" (ScienceDirect)
  - "Intraday time-series momentum: global evidence" (Reading University)

Gates:
  B. Risk floor: stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS

Grid: 3 return thresholds x 3 RR targets x 2 stop types x 2 sessions = 36 combos
Walk-forward: 12-month training windows, monthly steps, OOS from 2018-01-01
"""

import argparse
import sys
from datetime import date, timedelta
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
RETURN_THRESHOLDS = [0.1, 0.2, 0.3]  # fraction of ATR_20
RR_TARGETS = [1.0, 1.5, 2.0]
STOP_TYPES = ["swing", "atr"]  # swing = 30min H/L, atr = 1.0*ATR_20
# Sessions to test: 0900 Brisbane (23:00 UTC) and 1000 Brisbane (00:00 UTC)
SESSIONS = {
    "0900": 23,  # UTC hour of session start
    "1000": 0,   # UTC hour of session start
}
HALF_HOUR_BARS = 6  # 6 x 5-minute bars = 30 minutes

REGIME_BOUNDARY = date(2025, 1, 1)

def load_bars_5m_for_day(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 5-minute bars for one trading day."""
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

def compute_first_half_hour_outcomes(
    db_path: Path, start: date, end: date
) -> pd.DataFrame:
    """Compute first-half-hour momentum outcomes for all days."""
    # Get ATR from daily_features
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        features = con.execute("""
            SELECT trading_day, daily_high, daily_low, daily_open
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 5
              AND trading_day BETWEEN ? AND ?
            ORDER BY trading_day
        """, [start, end]).fetchdf()
    finally:
        con.close()

    if features.empty:
        return pd.DataFrame()

    features["true_range"] = features["daily_high"] - features["daily_low"]
    features["atr_20"] = features["true_range"].rolling(20, min_periods=20).mean().shift(1)

    all_outcomes = []
    eligible = features.dropna(subset=["atr_20"])
    eligible = eligible[eligible["atr_20"] > 0]
    total = len(eligible)

    for idx, (_, row) in enumerate(eligible.iterrows()):
        if idx % 200 == 0:
            print(f"    Processing day {idx+1}/{total}...")

        td = row["trading_day"]
        if hasattr(td, "date") and callable(td.date):
            td_date = td.date()
        elif isinstance(td, str):
            td_date = date.fromisoformat(td)
        else:
            td_date = td

        atr = row["atr_20"]

        bars_5m = load_bars_5m_for_day(db_path, td_date)
        if bars_5m.empty or len(bars_5m) < HALF_HOUR_BARS + 10:
            continue

        # Get UTC hours
        ts_col = bars_5m["ts_utc"]
        if ts_col.dt.tz is not None:
            hours = ts_col.dt.tz_convert("UTC").dt.hour.values
        else:
            hours = ts_col.dt.hour.values

        for session_name, session_start_hour in SESSIONS.items():
            # Find the first 30 minutes of this session
            session_mask = hours == session_start_hour
            session_indices = np.where(session_mask)[0]

            if len(session_indices) == 0:
                continue

            first_bar_idx = session_indices[0]
            # Need at least HALF_HOUR_BARS bars from session start
            end_bar_idx = first_bar_idx + HALF_HOUR_BARS
            if end_bar_idx >= len(bars_5m):
                continue

            # First 30-min return
            session_open = bars_5m.iloc[first_bar_idx]["open"]
            half_hour_close = bars_5m.iloc[end_bar_idx - 1]["close"]
            half_hour_return = half_hour_close - session_open

            # First 30-min high/low (for swing stop)
            first_half = bars_5m.iloc[first_bar_idx:end_bar_idx]
            half_hour_high = first_half["high"].max()
            half_hour_low = first_half["low"].min()

            return_atr = abs(half_hour_return) / atr if atr > 0 else 0

            for ret_thresh in RETURN_THRESHOLDS:
                if return_atr < ret_thresh:
                    continue

                # Direction follows momentum
                if half_hour_return > 0:
                    direction = "long"
                else:
                    direction = "short"

                entry_price = half_hour_close

                for stop_type in STOP_TYPES:
                    if stop_type == "swing":
                        if direction == "long":
                            stop_price = half_hour_low
                        else:
                            stop_price = half_hour_high
                    else:  # atr
                        if direction == "long":
                            stop_price = entry_price - 1.0 * atr
                        else:
                            stop_price = entry_price + 1.0 * atr

                    risk_points = abs(entry_price - stop_price)
                    if risk_points < SPEC.min_risk_floor_points:
                        continue

                    for rr in RR_TARGETS:
                        reward = rr * risk_points
                        if direction == "long":
                            target_price = entry_price + reward
                        else:
                            target_price = entry_price - reward

                        # Resolve from bar after the 30-min mark
                        result = _resolve_5m(
                            bars_5m, entry_price, stop_price, target_price,
                            direction, end_bar_idx,
                        )

                        if result is None:
                            last_close = bars_5m.iloc[-1]["close"]
                            pnl_pts = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
                            pnl_r = to_r_multiple(SPEC, entry_price, stop_price, pnl_pts)
                            outcome_type = "eod"
                        else:
                            pnl_r = to_r_multiple(SPEC, entry_price, stop_price, result["pnl_points"])
                            outcome_type = result["outcome"]

                        all_outcomes.append({
                            "trading_day": str(td_date),
                            "session": session_name,
                            "direction": direction,
                            "return_threshold": ret_thresh,
                            "return_atr": return_atr,
                            "half_hour_return": half_hour_return,
                            "stop_type": stop_type,
                            "rr_target": rr,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "risk_points": risk_points,
                            "pnl_r": pnl_r,
                            "outcome": outcome_type,
                        })

    if not all_outcomes:
        return pd.DataFrame()
    return pd.DataFrame(all_outcomes)

def _resolve_5m(bars, entry, stop, target, direction, start_idx):
    """Resolve outcome on 5m bars."""
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
    test_start: date = date(2018, 1, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward for first-half-hour momentum strategy."""
    full_start = date(2016, 1, 1)

    print("  Computing all first-half-hour outcomes...")
    outcomes_df = compute_first_half_hour_outcomes(db_path, full_start, test_end)
    if outcomes_df.empty:
        print("  No outcomes found")
        return {"windows": [], "combined_oos": None, "regime_split": None}

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

        best_combo = None
        best_sharpe = -999.0

        for session_name in SESSIONS:
            s_train = train_data[train_data["session"] == session_name]
            for ret_thresh in RETURN_THRESHOLDS:
                rt_train = s_train[s_train["return_threshold"] == ret_thresh]
                for stop_type in STOP_TYPES:
                    st_train = rt_train[rt_train["stop_type"] == stop_type]
                    for rr in RR_TARGETS:
                        rr_train = st_train[st_train["rr_target"] == rr]
                        if len(rr_train) < 20:
                            continue
                        stats = compute_strategy_metrics(rr_train["pnl_r"].values)
                        if stats and stats["sharpe"] > best_sharpe:
                            best_sharpe = stats["sharpe"]
                            best_combo = (session_name, ret_thresh, stop_type, rr)

        if best_combo is None:
            continue

        session_name, ret_thresh, stop_type, rr = best_combo
        oos = test_data[
            (test_data["session"] == session_name)
            & (test_data["return_threshold"] == ret_thresh)
            & (test_data["stop_type"] == stop_type)
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
            "selected": f"{session_name}_RT{ret_thresh}_{stop_type}_RR{rr}",
            "train_sharpe": best_sharpe,
            "oos_stats": oos_stats,
        })

    combined_oos = None
    regime_split = None
    if oos_all_pnls:
        all_pnls = np.array(oos_all_pnls)
        all_dates = np.array(oos_all_dates)
        combined_oos = compute_strategy_metrics(all_pnls)

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

def _print_go_no_go(combined: dict | None, regime_split: dict | None) -> None:
    print()
    print("--- GO/NO-GO EVALUATION ---")
    if combined is None:
        print("  NO-GO: No OOS data")
        return

    c = combined
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
    parser = argparse.ArgumentParser(description="First Half-Hour Momentum analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("FIRST HALF-HOUR MOMENTUM CONTINUATION STRATEGY")
    print(sep)
    print()
    print("Entry: Continue first 30-minute momentum (academic-backed)")
    print(f"Sessions: {list(SESSIONS.keys())}")
    print(f"Grid: {len(SESSIONS)} sessions x {len(RETURN_THRESHOLDS)} thresholds x "
          f"{len(STOP_TYPES)} stop types x {len(RR_TARGETS)} RR = "
          f"{len(SESSIONS) * len(RETURN_THRESHOLDS) * len(STOP_TYPES) * len(RR_TARGETS)} combos")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print()

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
                          f"ExpR={stats['expr']:+.3f}, Sharpe={stats['sharpe']:.3f}")
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

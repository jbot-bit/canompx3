#!/usr/bin/env python3
"""
VWAP Momentum Pullback strategy analysis.

Trades pullbacks to VWAP in trending markets. When price is consistently
above/below VWAP, a pullback to VWAP offers a low-risk trend continuation entry.

Entry logic:
  1. Compute cumulative VWAP from session open (23:00 UTC prior day) using bars_1m
  2. Identify trend: price > VWAP for at least N consecutive 5m bars = BULLISH
  3. Wait for pullback: price touches VWAP (within 0.2 * ATR_20)
  4. Entry = next bar open after pullback touch
  5. Direction = same as trend (LONG if bullish, SHORT if bearish)
  6. Stop = VWAP - stop_mult * ATR_20 (for longs)
  7. Target = entry + RR * risk_distance

Gates:
  B. Risk floor: stop distance >= min_risk_floor_points (1.0 pt for MGC)
  C. Ambiguous bar (stop + target on same bar) = LOSS
  G. Trend must be established (N consecutive bars on same side of VWAP)
  H. Only one trade per direction per session (no re-entry after stop)

Grid: 3 RR targets x 2 stop multipliers x 3 trend lengths x 2 time filters = 36 combos
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
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec, to_r_multiple
from pipeline.paths import GOLD_DB_PATH
from research._alt_strategy_utils import (
    annualize_sharpe,
    compute_strategy_metrics,
    compute_walk_forward_windows,
    save_results,
)

SPEC = get_cost_spec("MGC")

# Grid dimensions
RR_TARGETS = [1.0, 1.5, 2.0]
STOP_MULTIPLIERS = [0.5, 1.0]  # fraction of ATR_20
TREND_LENGTHS = [6, 12, 20]  # consecutive 5m bars above/below VWAP
TIME_FILTERS = {
    "active": (0, 23),   # Nearly all hours
    "core": (8, 18),     # London + NY
}
VWAP_PROXIMITY = 0.2  # Touch VWAP within this fraction of ATR

REGIME_BOUNDARY = date(2025, 1, 1)


def load_bars_1m_for_day(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 1-minute bars for one trading day."""
    from pipeline.build_daily_features import compute_trading_day_utc_range
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = 'MGC'
              AND ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc
        """, [start_utc, end_utc]).fetchdf()
    finally:
        con.close()
    return df


def compute_vwap(bars_1m: pd.DataFrame) -> np.ndarray:
    """Compute cumulative VWAP from bars.

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3
    """
    tp = (bars_1m["high"].values + bars_1m["low"].values + bars_1m["close"].values) / 3.0
    vol = bars_1m["volume"].values.astype(float)

    # Handle zero volume bars
    vol = np.where(vol > 0, vol, 1.0)

    cum_tp_vol = np.cumsum(tp * vol)
    cum_vol = np.cumsum(vol)
    vwap = cum_tp_vol / cum_vol
    return vwap


def resample_to_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m bars to 5m bars with VWAP."""
    if bars_1m.empty:
        return pd.DataFrame()

    df = bars_1m.copy()
    # Compute VWAP on 1m bars
    vwap_1m = compute_vwap(df)
    df["vwap"] = vwap_1m

    # Create 5m bucket using UTC epoch seconds
    ts = df["ts_utc"]
    if ts.dt.tz is not None:
        # Convert to UTC first, then get epoch
        ts_utc = ts.dt.tz_convert("UTC")
        epoch = (ts_utc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    else:
        epoch = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    df["bucket"] = (epoch // 300) * 300

    grouped = df.groupby("bucket").agg(
        ts_utc=("ts_utc", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        vwap=("vwap", "last"),  # VWAP at end of 5m bar (cumulative)
    ).reset_index(drop=True)

    return grouped


def find_vwap_pullback_signals(
    bars_5m: pd.DataFrame, atr: float, trend_length: int, time_filter: tuple
) -> list[dict]:
    """Find VWAP pullback signals in 5m bars.

    Returns list of signal dicts with direction, entry_bar_idx, entry_price, etc.
    """
    if bars_5m.empty or "vwap" not in bars_5m.columns:
        return []

    closes = bars_5m["close"].values
    vwap = bars_5m["vwap"].values
    signals = []

    # Track consecutive bars above/below VWAP
    above_count = 0
    below_count = 0
    had_long_trade = False
    had_short_trade = False

    # Get hours for time filter
    ts_col = bars_5m["ts_utc"]
    if ts_col.dt.tz is not None:
        hours = ts_col.dt.tz_convert("UTC").dt.hour.values
    else:
        hours = ts_col.dt.hour.values

    proximity = VWAP_PROXIMITY * atr if atr > 0 else 0.5

    for i in range(1, len(bars_5m) - 1):
        hour = hours[i]
        if not (time_filter[0] <= hour < time_filter[1]):
            continue

        # Track trend
        if closes[i] > vwap[i]:
            above_count += 1
            below_count = 0
        elif closes[i] < vwap[i]:
            below_count += 1
            above_count = 0
        else:
            above_count = 0
            below_count = 0

        # Check for pullback after established trend
        # LONG: was above VWAP for trend_length bars, now close is near VWAP
        if above_count >= trend_length:
            bar_low = bars_5m.iloc[i]["low"]
            if abs(bar_low - vwap[i]) <= proximity:
                if not had_long_trade:
                    entry_bar_idx = i + 1
                    entry_price = bars_5m.iloc[entry_bar_idx]["open"]
                    signals.append({
                        "direction": "long",
                        "entry_bar_idx": entry_bar_idx,
                        "entry_price": entry_price,
                        "vwap_at_entry": vwap[i],
                    })
                    had_long_trade = True
                    above_count = 0  # Reset after signal

        # SHORT: was below VWAP for trend_length bars, now close is near VWAP
        if below_count >= trend_length:
            bar_high = bars_5m.iloc[i]["high"]
            if abs(bar_high - vwap[i]) <= proximity:
                if not had_short_trade:
                    entry_bar_idx = i + 1
                    entry_price = bars_5m.iloc[entry_bar_idx]["open"]
                    signals.append({
                        "direction": "short",
                        "entry_bar_idx": entry_bar_idx,
                        "entry_price": entry_price,
                        "vwap_at_entry": vwap[i],
                    })
                    had_short_trade = True
                    below_count = 0

    return signals


def compute_vwap_outcomes(db_path: Path, start: date, end: date) -> pd.DataFrame:
    """Compute VWAP pullback outcomes for all days."""
    # Get trading days and ATR from daily_features
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        features = con.execute("""
            SELECT trading_day, daily_high, daily_low
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 5
              AND trading_day BETWEEN ? AND ?
            ORDER BY trading_day
        """, [start, end]).fetchdf()
    finally:
        con.close()

    if features.empty:
        return pd.DataFrame()

    # Compute ATR_20 (shifted by 1)
    features["true_range"] = features["daily_high"] - features["daily_low"]
    features["atr_20"] = features["true_range"].rolling(20, min_periods=20).mean().shift(1)

    all_outcomes = []
    eligible = features.dropna(subset=["atr_20"])
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

        # Load 1m bars and resample to 5m with VWAP
        bars_1m = load_bars_1m_for_day(db_path, td_date)
        if bars_1m.empty or len(bars_1m) < 60:
            continue

        bars_5m = resample_to_5m(bars_1m)
        if bars_5m.empty or len(bars_5m) < 30:
            continue

        for trend_length in TREND_LENGTHS:
            for tf_name, tf_range in TIME_FILTERS.items():
                signals = find_vwap_pullback_signals(bars_5m, atr, trend_length, tf_range)

                for sig in signals:
                    entry_price = sig["entry_price"]
                    direction = sig["direction"]
                    entry_bar_idx = sig["entry_bar_idx"]

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

                            # Resolve on 5m bars
                            result = _resolve_5m(
                                bars_5m, entry_price, stop_price, target_price,
                                direction, entry_bar_idx + 1
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
                                "direction": direction,
                                "trend_length": trend_length,
                                "time_filter": tf_name,
                                "stop_multiplier": stop_mult,
                                "rr_target": rr,
                                "entry_price": entry_price,
                                "stop_price": stop_price,
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
    test_start: date = date(2024, 8, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward analysis for VWAP pullback strategy."""
    # Only load data needed for training + OOS (train_months before test_start)
    from research._alt_strategy_utils import _add_months
    full_start = _add_months(test_start, -(train_months + 2))  # +2 months buffer for ATR warmup

    print("  Computing all VWAP pullback outcomes...")
    outcomes_df = compute_vwap_outcomes(db_path, full_start, test_end)
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

        for tl in TREND_LENGTHS:
            tl_train = train_data[train_data["trend_length"] == tl]
            for tf_name in TIME_FILTERS:
                tf_train = tl_train[tl_train["time_filter"] == tf_name]
                for sm in STOP_MULTIPLIERS:
                    sm_train = tf_train[tf_train["stop_multiplier"] == sm]
                    for rr in RR_TARGETS:
                        rr_train = sm_train[sm_train["rr_target"] == rr]
                        if len(rr_train) < 20:
                            continue
                        stats = compute_strategy_metrics(rr_train["pnl_r"].values)
                        if stats and stats["sharpe"] > best_sharpe:
                            best_sharpe = stats["sharpe"]
                            best_combo = (tl, tf_name, sm, rr)

        if best_combo is None:
            continue

        tl, tf_name, sm, rr = best_combo
        oos = test_data[
            (test_data["trend_length"] == tl)
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
            "selected": f"TL{tl}_{tf_name}_SM{sm}_RR{rr}",
            "train_sharpe": best_sharpe,
            "oos_stats": oos_stats,
        })

    combined_oos = None
    regime_split = None
    if oos_all_pnls:
        all_pnls = np.array(oos_all_pnls)
        all_dates = np.array(oos_all_dates)
        combined_oos = compute_strategy_metrics(all_pnls)
        # Annualize: OOS period length in years
        oos_years = (test_end - test_start).days / 365.25
        annualize_sharpe(combined_oos, oos_years)

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
    parser = argparse.ArgumentParser(description="VWAP Momentum Pullback analysis")
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
    print("VWAP MOMENTUM PULLBACK STRATEGY ANALYSIS")
    print(sep)
    print()
    print("Entry: Buy/sell pullbacks to VWAP after established trend")
    print(f"Grid: {len(TREND_LENGTHS)} trend lengths x {len(TIME_FILTERS)} time filters x "
          f"{len(STOP_MULTIPLIERS)} stop mults x {len(RR_TARGETS)} RR = "
          f"{len(TREND_LENGTHS) * len(TIME_FILTERS) * len(STOP_MULTIPLIERS) * len(RR_TARGETS)} combos")
    print(f"Gate B: Risk floor >= {SPEC.min_risk_floor_points} points")
    print(f"Gate C: Ambiguous bar = LOSS")
    print(f"Gate G: Trend established ({TREND_LENGTHS} consecutive bars)")
    print(f"Gate H: One trade per direction per session")
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
            sha = c.get('sharpe_ann')
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

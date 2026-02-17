#!/usr/bin/env python3
"""
ADX Trend Filter overlay analysis for ORB breakout strategies.

Tests whether adding an ADX_14 > threshold filter to existing ORB breakout
trades improves OOS performance. This is an OVERLAY filter, not a new strategy.

Logic:
  1. Compute ADX(14) on 5-minute bars at the time of each ORB break
  2. If ADX > threshold: ALLOW the trade (trending market)
  3. If ADX <= threshold: SKIP the trade (choppy market)
  4. Compare filtered vs unfiltered ORB breakout performance

ADX is direction-agnostic (measures trend STRENGTH, not direction).
High ADX = strong trend in either direction = good for breakout strategies.
Low ADX = choppy/range-bound = breakouts more likely to fail.

Grid: ADX threshold (20, 25, 30) x ORB label (0900, 1000) x size filter (G2, G4)
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
    compute_strategy_metrics,
    compute_walk_forward_windows,
    save_results,
)

SPEC = get_cost_spec("MGC")

# Grid dimensions
ADX_THRESHOLDS = [20, 25, 30, 35]
ORB_LABELS = ["0900", "1000"]
SIZE_FILTERS = {"G2": 2.0, "G4": 4.0}
RR_TARGETS = [1.5, 2.0, 2.5]
ENTRY_MODELS = ["E1", "E2"]

REGIME_BOUNDARY = date(2025, 1, 1)


def compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Compute Wilder's ADX from high/low/close arrays.

    Returns array of same length with NaN for first ~2*period elements.
    """
    n = len(highs)
    adx = np.full(n, np.nan)
    if n < 2 * period + 1:
        return adx

    # True Range
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Wilder's smoothing for ATR, +DI, -DI
    atr = np.zeros(n)
    smooth_plus = np.zeros(n)
    smooth_minus = np.zeros(n)

    # Seed
    atr[period] = tr[1:period + 1].mean()
    smooth_plus[period] = plus_dm[1:period + 1].mean()
    smooth_minus[period] = minus_dm[1:period + 1].mean()

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        smooth_plus[i] = (smooth_plus[i - 1] * (period - 1) + plus_dm[i]) / period
        smooth_minus[i] = (smooth_minus[i - 1] * (period - 1) + minus_dm[i]) / period

    # +DI, -DI, DX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if atr[i] > 0:
            plus_di = 100 * smooth_plus[i] / atr[i]
            minus_di = 100 * smooth_minus[i] / atr[i]
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di - minus_di) / di_sum

    # ADX = Wilder's smoothed DX
    # Seed ADX at 2*period
    seed_start = period
    seed_end = 2 * period
    valid_dx = dx[seed_start:seed_end]
    valid_dx = valid_dx[~np.isnan(valid_dx)]
    if len(valid_dx) == 0:
        return adx

    adx[2 * period] = valid_dx.mean()
    for i in range(2 * period + 1, n):
        if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


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


def get_adx_at_break_time(
    bars_5m: pd.DataFrame, break_ts: pd.Timestamp, period: int = 14
) -> float | None:
    """Compute ADX_14 value at the time of ORB break.

    Uses all 5m bars up to and including the break bar.
    Requires 2*period+1 bars of warmup (typically loaded from prev day + current).
    """
    if bars_5m.empty or len(bars_5m) < 2 * period + 1:
        return None

    ts_col = bars_5m["ts_utc"]

    # Handle timezone for comparison
    if break_ts.tzinfo is None:
        break_ts_cmp = break_ts
    else:
        if ts_col.dt.tz is not None:
            break_ts_cmp = break_ts.tz_convert(ts_col.dt.tz)
        else:
            break_ts_cmp = break_ts.tz_localize(None)

    mask = ts_col <= break_ts_cmp
    if mask.sum() < 2 * period + 1:
        return None

    subset = bars_5m[mask]
    adx_values = compute_adx(
        subset["high"].values, subset["low"].values, subset["close"].values, period
    )

    last_adx = adx_values[-1]
    return float(last_adx) if not np.isnan(last_adx) else None


def load_bars_5m_with_warmup(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 5m bars for current + previous trading day (for ADX warmup).

    ADX(14) needs ~29 bars of warmup. A single session might only have 3-4 bars
    before the ORB break, so we include the previous day's bars.
    """
    from pipeline.build_daily_features import compute_trading_day_utc_range
    from datetime import timedelta

    # Get range for prev trading day + current
    # Go back 2 calendar days to be safe (handles weekends)
    prev_day = trading_day - timedelta(days=3)
    start_utc, _ = compute_trading_day_utc_range(prev_day)
    _, end_utc = compute_trading_day_utc_range(trading_day)

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


def load_orb_outcomes_with_adx(db_path: Path, start: date, end: date) -> pd.DataFrame:
    """Load ORB outcomes and compute ADX at break time for each.

    Uses bars from prev+current day for ADX warmup.
    Joins ORB size from daily_features for size filtering.
    Returns DataFrame with orb_outcomes columns plus adx_at_break and orb_size.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        features = con.execute("""
            SELECT trading_day,
                   orb_0900_break_ts, orb_0900_size, orb_0900_break_dir,
                   orb_1000_break_ts, orb_1000_size, orb_1000_break_dir
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 5
              AND trading_day BETWEEN ? AND ?
            ORDER BY trading_day
        """, [start, end]).fetchdf()

        outcomes = con.execute("""
            SELECT trading_day, orb_label, entry_model, rr_target,
                   confirm_bars, pnl_r, outcome
            FROM orb_outcomes
            WHERE symbol = 'MGC'
              AND trading_day BETWEEN ? AND ?
        """, [start, end]).fetchdf()
    finally:
        con.close()

    if features.empty or outcomes.empty:
        return pd.DataFrame()

    adx_map = {}
    total = len(features)

    for idx, (_, row) in enumerate(features.iterrows()):
        if idx % 200 == 0:
            print(f"    Computing ADX: day {idx+1}/{total}...")

        td = row["trading_day"]
        if hasattr(td, "date") and callable(td.date):
            td_date = td.date()
        elif isinstance(td, str):
            td_date = date.fromisoformat(td)
        else:
            td_date = td

        # Load with warmup (prev day's bars for ADX to converge)
        bars_5m = load_bars_5m_with_warmup(db_path, td_date)
        if bars_5m.empty:
            continue

        for orb_label in ORB_LABELS:
            break_ts = row[f"orb_{orb_label}_break_ts"]
            if pd.isna(break_ts):
                continue

            adx_val = get_adx_at_break_time(bars_5m, break_ts)
            if adx_val is not None:
                adx_map[(str(td_date), orb_label)] = adx_val

    print(f"    ADX computed for {len(adx_map)} (day, session) pairs")

    outcomes["trading_day_str"] = outcomes["trading_day"].astype(str).str[:10]
    outcomes["adx_at_break"] = outcomes.apply(
        lambda r: adx_map.get((r["trading_day_str"], r["orb_label"])), axis=1
    )

    # Join ORB size from daily_features for size filtering
    features["trading_day_str"] = features["trading_day"].astype(str).str[:10]
    size_map = {}
    for _, row in features.iterrows():
        td_str = row["trading_day_str"]
        for orb_label in ORB_LABELS:
            sz = row[f"orb_{orb_label}_size"]
            if not pd.isna(sz):
                size_map[(td_str, orb_label)] = sz

    outcomes["orb_size"] = outcomes.apply(
        lambda r: size_map.get((r["trading_day_str"], r["orb_label"])), axis=1
    )

    return outcomes


def run_walk_forward(
    db_path: Path,
    train_months: int = 12,
    test_start: date = date(2018, 1, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run walk-forward comparing ADX-filtered vs unfiltered ORB performance."""
    full_start = date(2016, 1, 1)

    print("  Loading outcomes and computing ADX at break times...")
    outcomes_df = load_orb_outcomes_with_adx(db_path, full_start, test_end)
    if outcomes_df.empty:
        print("  No outcomes found")
        return {"windows": [], "combined_oos": None, "regime_split": None}

    outcomes_df["trading_day_date"] = pd.to_datetime(outcomes_df["trading_day"]).dt.date
    valid = outcomes_df.dropna(subset=["adx_at_break"])
    print(f"  {len(valid)} outcomes with ADX values ({len(outcomes_df) - len(valid)} missing ADX)")

    windows = compute_walk_forward_windows(test_start, test_end, train_months)
    window_results = []
    oos_filtered_pnls = []
    oos_unfiltered_pnls = []
    oos_filtered_dates = []

    for w in windows:
        train_mask = (
            (valid["trading_day_date"] >= w["train_start"])
            & (valid["trading_day_date"] <= w["train_end"])
        )
        test_mask = (
            (valid["trading_day_date"] >= w["test_start"])
            & (valid["trading_day_date"] <= w["test_end"])
        )

        train_data = valid[train_mask]
        test_data = valid[test_mask]

        if train_data.empty:
            continue

        # Find best (orb_label, entry_model, rr, size_filter, adx_threshold) on training
        best_combo = None
        best_sharpe = -999.0

        for orb_label in ORB_LABELS:
            ol_train = train_data[train_data["orb_label"] == orb_label]
            for em in ENTRY_MODELS:
                em_train = ol_train[ol_train["entry_model"] == em]
                for rr in RR_TARGETS:
                    rr_train = em_train[em_train["rr_target"] == rr]
                    # Use CB1 only (avoid overlap between CB levels)
                    cb_train = rr_train[rr_train["confirm_bars"] == 1]
                    if cb_train.empty:
                        continue

                    for sf_name, sf_min in SIZE_FILTERS.items():
                        sf_train = cb_train[cb_train["orb_size"] >= sf_min]
                        if len(sf_train) < 10:
                            continue

                        # Test: size filter ALONE (no ADX) as baseline
                        sf_stats = compute_strategy_metrics(sf_train["pnl_r"].values)
                        if sf_stats and sf_stats["sharpe"] > best_sharpe:
                            best_sharpe = sf_stats["sharpe"]
                            best_combo = (orb_label, em, rr, sf_name, None)  # None = no ADX

                        # Test: size filter + ADX overlay
                        for adx_thresh in ADX_THRESHOLDS:
                            filtered = sf_train[sf_train["adx_at_break"] >= adx_thresh]
                            if len(filtered) < 10:
                                continue
                            stats = compute_strategy_metrics(filtered["pnl_r"].values)
                            if stats and stats["sharpe"] > best_sharpe:
                                best_sharpe = stats["sharpe"]
                                best_combo = (orb_label, em, rr, sf_name, adx_thresh)

        if best_combo is None:
            continue

        orb_label, em, rr, sf_name, adx_thresh = best_combo

        # Apply to OOS: first size filter, then optional ADX
        oos_base = test_data[
            (test_data["orb_label"] == orb_label)
            & (test_data["entry_model"] == em)
            & (test_data["rr_target"] == rr)
            & (test_data["confirm_bars"] == 1)
        ]
        sf_min = SIZE_FILTERS[sf_name]
        oos_sized = oos_base[oos_base["orb_size"] >= sf_min]

        if adx_thresh is not None:
            oos_final = oos_sized[oos_sized["adx_at_break"] >= adx_thresh]
            combo_label = f"{orb_label}_{em}_RR{rr}_{sf_name}_ADX{adx_thresh}"
        else:
            oos_final = oos_sized
            combo_label = f"{orb_label}_{em}_RR{rr}_{sf_name}_noADX"

        if oos_final.empty:
            continue

        # Drop NaN pnl_r before computing metrics
        final_pnls = oos_final["pnl_r"].dropna().values
        sized_pnls = oos_sized["pnl_r"].dropna().values

        oos_f_stats = compute_strategy_metrics(final_pnls)
        oos_sized_stats = compute_strategy_metrics(sized_pnls) if len(sized_pnls) > 0 else None

        oos_filtered_pnls.extend(final_pnls)
        oos_unfiltered_pnls.extend(sized_pnls)
        final_valid = oos_final.dropna(subset=["pnl_r"])
        oos_filtered_dates.extend(final_valid["trading_day_date"].values)

        window_results.append({
            "test_start": str(w["test_start"]),
            "test_end": str(w["test_end"]),
            "selected": combo_label,
            "train_sharpe": best_sharpe,
            "oos_final": oos_f_stats,
            "oos_size_only": oos_sized_stats,
            "adx_applied": adx_thresh is not None,
            "filter_rate": 1 - len(oos_final) / len(oos_sized) if len(oos_sized) > 0 else 0,
        })

    combined_filtered = None
    combined_unfiltered = None
    regime_split = None

    if oos_filtered_pnls:
        f_pnls = np.array(oos_filtered_pnls)
        u_pnls = np.array(oos_unfiltered_pnls)
        f_dates = np.array(oos_filtered_dates)
        combined_filtered = compute_strategy_metrics(f_pnls)
        combined_unfiltered = compute_strategy_metrics(u_pnls)

        low_vol_mask = f_dates < np.datetime64(REGIME_BOUNDARY)
        high_vol_mask = ~low_vol_mask
        low_vol = compute_strategy_metrics(f_pnls[low_vol_mask]) if low_vol_mask.sum() > 0 else None
        high_vol = compute_strategy_metrics(f_pnls[high_vol_mask]) if high_vol_mask.sum() > 0 else None
        regime_split = {"low_vol_2018_2024": low_vol, "high_vol_2025_2026": high_vol}

    return {
        "train_months": train_months,
        "windows": window_results,
        "combined_filtered": combined_filtered,
        "combined_unfiltered": combined_unfiltered,
        "regime_split": regime_split,
    }


def _print_go_no_go(combined: dict | None, regime_split: dict | None) -> None:
    """Print GO/NO-GO evaluation."""
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
    parser = argparse.ArgumentParser(description="ADX Trend Filter overlay analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("ADX TREND FILTER OVERLAY ANALYSIS")
    print(sep)
    print()
    print("Tests: Does ADX > threshold improve G2/G4-filtered ORB breakout OOS performance?")
    n_combos = (len(ORB_LABELS) * len(ENTRY_MODELS) * len(RR_TARGETS) * len(SIZE_FILTERS)
                * (1 + len(ADX_THRESHOLDS)))  # +1 for size-only baseline
    print(f"Grid: {len(ORB_LABELS)} ORBs x {len(ENTRY_MODELS)} EMs x {len(RR_TARGETS)} RR x "
          f"{len(SIZE_FILTERS)} size filters x (1 baseline + {len(ADX_THRESHOLDS)} ADX) = {n_combos} combos")
    print(f"Size filters: {list(SIZE_FILTERS.keys())}, ADX thresholds: {ADX_THRESHOLDS}")
    print()

    print("--- WALK-FORWARD ANALYSIS ---")
    result = run_walk_forward(args.db_path, args.train_months)

    if result["windows"]:
        adx_wins = 0
        size_wins = 0
        for w in result["windows"]:
            f = w["oos_final"]
            u = w["oos_size_only"]
            if f:
                u_expr = f"{u['expr']:+.3f}" if u else "N/A"
                adx_tag = " [+ADX]" if w["adx_applied"] else " [size only]"
                print(f"  {w['test_start']} to {w['test_end']}: {w['selected']}{adx_tag}, "
                      f"N={f['n']}, ExpR={f['expr']:+.3f} vs size-only={u_expr}")
                if w["adx_applied"]:
                    adx_wins += 1
                else:
                    size_wins += 1

        print(f"\n  Walk-forward selected ADX in {adx_wins}/{adx_wins+size_wins} windows "
              f"({size_wins} windows chose size-only)")

        if result["combined_filtered"]:
            cf = result["combined_filtered"]
            cu = result["combined_unfiltered"]
            print(f"\n  COMBINED (ADX FILTERED): N={cf['n']}, WR={cf['wr']:.0%}, "
                  f"ExpR={cf['expr']:+.3f}, Sharpe={cf['sharpe']:.3f}, "
                  f"MaxDD={cf['maxdd']:+.1f}R, Total={cf['total']:+.1f}R")
            if cu:
                print(f"  COMBINED (UNFILTERED):   N={cu['n']}, WR={cu['wr']:.0%}, "
                      f"ExpR={cu['expr']:+.3f}, Sharpe={cu['sharpe']:.3f}, "
                      f"MaxDD={cu['maxdd']:+.1f}R, Total={cu['total']:+.1f}R")
                print(f"  ADX UPLIFT: ExpR {cf['expr'] - cu['expr']:+.3f}, "
                      f"Sharpe {cf['sharpe'] - cu['sharpe']:+.3f}")

        if result["regime_split"]:
            rs = result["regime_split"]
            print("\n  REGIME SPLIT (filtered):")
            for label, stats in rs.items():
                if stats:
                    print(f"    {label}: N={stats['n']}, WR={stats['wr']:.0%}, "
                          f"ExpR={stats['expr']:+.3f}, Sharpe={stats['sharpe']:.3f}")
                else:
                    print(f"    {label}: No data")

        _print_go_no_go(result["combined_filtered"], result["regime_split"])
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

#!/usr/bin/env python3
"""
Range Expansion analysis for Gold (MGC).

Question: When daily range expands past X% of ATR by time T, what happens next?

For each trading day:
  1. Compute cumulative intraday range at 30-min intervals
  2. When range exceeds 50%/75%/100% of ATR_20, record the time and direction
  3. After threshold hit: measure continuation vs reversal for remaining session
  4. Define continuation: close > threshold price (for upside expansion)
  5. Compute PnL if entering at expansion point with 1R risk

Read-only: does NOT write to the database.
"""

import sys
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec
from research._alt_strategy_utils import compute_strategy_metrics, annualize_sharpe

DB_PATH = Path(r"C:\db\gold.db")
SPEC = get_cost_spec("MGC")

# Screen on recent data first; expand to full dataset only if signal found
SCREEN_START = "2024-01-01"

# Range expansion thresholds (fraction of ATR_20)
THRESHOLDS = [0.50, 0.75, 1.00, 1.25, 1.50]


def load_daily_data(db_path: Path) -> pd.DataFrame:
    """Load daily_features with ATR_20 for MGC."""
    print("Loading daily features...", end=" ", flush=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT trading_day, atr_20, daily_open, daily_high, daily_low, daily_close
            FROM daily_features
            WHERE symbol = 'MGC'
              AND orb_minutes = 5
              AND atr_20 IS NOT NULL
              AND trading_day >= ?
            ORDER BY trading_day
        """, [SCREEN_START]).fetchdf()
    finally:
        con.close()
    print(f"{len(df):,} days loaded.")
    return df


def load_1m_bars_bulk(db_path: Path) -> pd.DataFrame:
    """Load 1m bars for MGC (screened date range)."""
    print("Loading 1m bars...", end=" ", flush=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = 'MGC'
              AND ts_utc >= ?::TIMESTAMPTZ
            ORDER BY ts_utc
        """, [f"{SCREEN_START} 00:00:00+00"]).fetchdf()
    finally:
        con.close()
    print(f"{len(df):,} bars loaded.")
    return df


def assign_trading_day(ts_utc: pd.Series) -> pd.Series:
    """Assign trading day: 09:00 Brisbane (23:00 UTC prev day) boundary."""
    if ts_utc.dt.tz is not None:
        utc_ts = ts_utc.dt.tz_convert("UTC")
    else:
        utc_ts = ts_utc.dt.tz_localize("UTC")

    dates = utc_ts.dt.date
    hours = utc_ts.dt.hour
    trading_days = pd.Series(dates, index=ts_utc.index)

    mask_next = hours >= 23
    trading_days[mask_next] = trading_days[mask_next].apply(
        lambda d: d + timedelta(days=1)
    )
    return trading_days


def analyze_range_expansion(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """For each day, track when range hits ATR thresholds and what happens after."""
    print("\nAnalyzing range expansion patterns...")

    if bars["ts_utc"].dt.tz is not None:
        bars["ts_utc"] = bars["ts_utc"].dt.tz_convert("UTC")
    else:
        bars["ts_utc"] = bars["ts_utc"].dt.tz_localize("UTC")

    bars["trading_day"] = assign_trading_day(bars["ts_utc"])

    # Pre-group bars by trading day for O(1) lookup
    print("  Pre-grouping bars by trading day...", flush=True)
    grouped = {td: grp.sort_values("ts_utc") for td, grp in bars.groupby("trading_day")}

    daily["trading_day_date"] = pd.to_datetime(daily["trading_day"]).dt.date
    atr_lookup = dict(zip(daily["trading_day_date"], daily["atr_20"]))

    results = []
    trading_days = sorted(grouped.keys())
    total_days = len(trading_days)

    for i, td in enumerate(trading_days):
        if i % 100 == 0:
            print(f"  Processing day {i+1}/{total_days}...", flush=True)

        atr = atr_lookup.get(td)
        if atr is None or atr <= 0:
            continue

        day_bars = grouped[td]
        if len(day_bars) < 60:
            continue

        # Vectorized cumulative range
        cum_high = day_bars["high"].cummax().values
        cum_low = day_bars["low"].cummin().values
        cum_range = cum_high - cum_low
        close_vals = day_bars["close"].values
        ts_vals = day_bars["ts_utc"].values

        for threshold in THRESHOLDS:
            target_range = threshold * atr

            exceed_mask = cum_range >= target_range
            if not exceed_mask.any():
                continue

            first_idx = np.argmax(exceed_mask)
            trigger_close = close_vals[first_idx]
            trigger_ts = ts_vals[first_idx]

            bar_high = cum_high[first_idx]
            bar_low = cum_low[first_idx]
            range_mid = (bar_high + bar_low) / 2
            direction = "up" if trigger_close > range_mid else "down"

            if first_idx + 6 > len(day_bars):
                continue

            session_close = close_vals[-1]

            if direction == "up":
                continuation = session_close > trigger_close
                move_after = session_close - trigger_close
            else:
                continuation = session_close < trigger_close
                move_after = trigger_close - session_close

            risk_pts = max(target_range * 0.5, SPEC.tick_size * SPEC.min_ticks_floor)
            pnl_dollars = move_after * SPEC.point_value - SPEC.total_friction
            risk_dollars = risk_pts * SPEC.point_value + SPEC.total_friction
            pnl_r = pnl_dollars / risk_dollars

            day_start_ts = ts_vals[0]
            hours_in = (pd.Timestamp(trigger_ts) - pd.Timestamp(day_start_ts)).total_seconds() / 3600

            results.append({
                "trading_day": td,
                "threshold": threshold,
                "direction": direction,
                "trigger_hour_utc": pd.Timestamp(trigger_ts).hour,
                "hours_in_session": hours_in,
                "range_at_trigger": cum_range[first_idx],
                "atr_20": atr,
                "trigger_close": trigger_close,
                "session_close": session_close,
                "move_after_pts": move_after,
                "continuation": continuation,
                "pnl_r": pnl_r,
                "remaining_bars": len(day_bars) - first_idx - 1,
                "year": td.year if hasattr(td, "year") else pd.Timestamp(td).year,
            })

    results_df = pd.DataFrame(results)
    print(f"  {len(results_df):,} expansion events found.")
    return results_df


def print_threshold_summary(data: pd.DataFrame) -> None:
    """Print summary for each expansion threshold."""
    print("\n" + "=" * 110)
    print("RANGE EXPANSION SUMMARY BY THRESHOLD")
    print("=" * 110)

    header = f"{'Thresh':>7} {'N':>7} {'Cont%':>7} {'Mean_Move':>10} {'Med_Move':>10} {'N_up':>7} {'N_dn':>7} {'Mean_Hr':>8}"
    print(header)
    print("-" * 110)

    for thresh in THRESHOLDS:
        subset = data[data["threshold"] == thresh]
        if len(subset) == 0:
            continue

        n = len(subset)
        cont_pct = subset["continuation"].sum() / n * 100
        mean_move = subset["move_after_pts"].mean()
        med_move = subset["move_after_pts"].median()
        n_up = (subset["direction"] == "up").sum()
        n_dn = (subset["direction"] == "down").sum()
        mean_hr = subset["hours_in_session"].mean()

        print(f"{thresh:>7.0%} {n:>7,} {cont_pct:>6.1f}% {mean_move:>+10.3f} {med_move:>+10.3f} {n_up:>7,} {n_dn:>7,} {mean_hr:>8.1f}")


def print_pnl_if_trading(data: pd.DataFrame) -> None:
    """Print ExpR for continuation trade at each threshold."""
    print("\n" + "=" * 110)
    print("EXPECTED R: ENTER AT EXPANSION POINT IN EXPANSION DIRECTION")
    print(f"  Cost model: {SPEC.instrument}, friction={SPEC.total_friction:.2f} $/RT")
    print("=" * 110)

    header = f"{'Thresh':>7} {'Dir':>5} {'N':>7} {'WR%':>6} {'ExpR':>8} {'Sharpe':>8} {'MaxDD_R':>8} {'Total_R':>9}"
    print(header)
    print("-" * 110)

    for thresh in THRESHOLDS:
        for direction in ["up", "down", "all"]:
            if direction == "all":
                subset = data[data["threshold"] == thresh]
            else:
                subset = data[(data["threshold"] == thresh) & (data["direction"] == direction)]

            if len(subset) < 30:
                continue

            pnl_r = subset["pnl_r"].values
            stats = compute_strategy_metrics(pnl_r)
            if stats is None:
                continue

            n_years = max((subset["year"].max() - subset["year"].min()), 1)
            stats = annualize_sharpe(stats, n_years)

            sig = "*" if len(subset) >= 100 else ""
            print(f"{thresh:>7.0%} {direction:>5} {stats['n']:>7,}{sig} {stats['wr']*100:>5.1f}% {stats['expr']:>+8.4f} {stats['sharpe']:>+8.4f} {stats['maxdd']:>+8.2f} {stats['total']:>+9.2f}")


def print_reversal_analysis(data: pd.DataFrame) -> None:
    """After large expansion, does the market reverse?"""
    print("\n" + "=" * 110)
    print("REVERSAL ANALYSIS: FADE THE EXPANSION")
    print("  (Enter OPPOSITE of expansion direction at trigger point)")
    print("=" * 110)

    header = f"{'Thresh':>7} {'N':>7} {'Rev%':>7} {'MeanRevPts':>12} {'Fade_WR':>8} {'Fade_ExpR':>10}"
    print(header)
    print("-" * 110)

    for thresh in THRESHOLDS:
        subset = data[data["threshold"] == thresh]
        if len(subset) < 30:
            continue

        n = len(subset)
        reversal_pct = (~subset["continuation"]).sum() / n * 100

        fade_pnl = -subset["pnl_r"].values
        fade_stats = compute_strategy_metrics(fade_pnl)
        if fade_stats is None:
            continue

        mean_rev = -subset["move_after_pts"].mean()

        print(f"{thresh:>7.0%} {n:>7,} {reversal_pct:>6.1f}% {mean_rev:>+12.3f} {fade_stats['wr']*100:>7.1f}% {fade_stats['expr']:>+10.4f}")


def print_timing_analysis(data: pd.DataFrame) -> None:
    """When does expansion happen? Early vs late in session."""
    print("\n" + "=" * 110)
    print("TIMING: WHEN DO RANGE EXPANSIONS HAPPEN?")
    print("=" * 110)

    for thresh in THRESHOLDS:
        subset = data[data["threshold"] == thresh]
        if len(subset) < 30:
            continue

        print(f"\n  Threshold {thresh:.0%} (N={len(subset):,}):")
        print(f"    Mean hours into session: {subset['hours_in_session'].mean():.1f}")
        print(f"    Median hours: {subset['hours_in_session'].median():.1f}")

        early = subset[subset["hours_in_session"] <= 4]
        late = subset[subset["hours_in_session"] > 4]

        if len(early) >= 30:
            cont_e = early["continuation"].sum() / len(early) * 100
            print(f"    Early (<4h): N={len(early):,}, Cont%={cont_e:.1f}%, MeanMove={early['move_after_pts'].mean():+.3f}")
        if len(late) >= 30:
            cont_l = late["continuation"].sum() / len(late) * 100
            print(f"    Late (>4h):  N={len(late):,}, Cont%={cont_l:.1f}%, MeanMove={late['move_after_pts'].mean():+.3f}")


def print_year_breakdown(data: pd.DataFrame) -> None:
    """Year-by-year consistency check."""
    print("\n" + "=" * 110)
    print("YEAR-BY-YEAR: CONTINUATION RATE BY THRESHOLD")
    print("=" * 110)

    years = sorted(data["year"].unique())
    header = f"{'Thresh':>7}" + "".join(f"{y:>10}" for y in years)
    print(header)
    print("-" * (7 + 10 * len(years)))

    for thresh in THRESHOLDS:
        row = f"{thresh:>7.0%}"
        for y in years:
            subset = data[(data["threshold"] == thresh) & (data["year"] == y)]
            if len(subset) >= 10:
                cont = subset["continuation"].sum() / len(subset) * 100
                row += f"  {cont:5.1f}%({len(subset):>3})"
            else:
                row += f"{'---':>10}"
        print(row)


def main():
    print("=" * 110)
    print(f"GOLD (MGC) RANGE EXPANSION ANALYSIS  [screening: {SCREEN_START}+]")
    print("=" * 110)
    print()

    daily = load_daily_data(DB_PATH)
    bars = load_1m_bars_bulk(DB_PATH)
    events = analyze_range_expansion(bars, daily)

    if events.empty:
        print("No range expansion events found!")
        return

    print_threshold_summary(events)
    print_pnl_if_trading(events)
    print_reversal_analysis(events)
    print_timing_analysis(events)
    print_year_breakdown(events)

    print("\n" + "=" * 110)
    print("NOTES")
    print("=" * 110)
    print(f"  - Screening period: {SCREEN_START}+ (expand to full 10yr only if signal found)")
    print("  - Min N=30 for any conclusion (marked with * if N >= 100)")
    print("  - Risk model: 0.5 * threshold_range as stop distance")
    print(f"  - Friction: ${SPEC.total_friction:.2f}/RT ({SPEC.friction_in_points:.2f} pts)")

    print("\n[Done]")


if __name__ == "__main__":
    main()

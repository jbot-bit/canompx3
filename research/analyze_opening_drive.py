#!/usr/bin/env python3
"""
Opening Drive analysis for Gold (MGC).

Question: Does the first 15/30/60 min direction predict the session close?

For each major session start (Brisbane time -> UTC):
  - 09:00 Brisbane = 23:00 UTC previous day
  - 18:00 Brisbane = 08:00 UTC
  - 23:00 Brisbane = 13:00 UTC

For each session:
  1. Compute direction of first 15/30/60 min (close vs open of that window)
  2. Compute session close vs session open
  3. If opening drive direction matches session close direction = "alignment"
  4. Report alignment rate and conditional PnL (buy at window close if up, sell if down)

NOT the same as ORB: this measures initial impulse direction, not a box breakout.

Read-only: does NOT write to the database.
"""

import sys
from datetime import datetime, timedelta, timezone
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

# Session definitions: name -> (start_hour_utc, duration_hours)
# These are fixed UTC hours; DST nuances are small for gold
SESSIONS = {
    "Asia_0900": {"start_utc": 23, "duration_h": 8, "description": "09:00 Brisbane (23:00 UTC prev day)"},
    "London_1800": {"start_utc": 8, "duration_h": 5, "description": "18:00 Brisbane (08:00 UTC)"},
    "NY_2300": {"start_utc": 13, "duration_h": 8, "description": "23:00 Brisbane (13:00 UTC)"},
}

# Drive windows in minutes
DRIVE_WINDOWS = [15, 30, 60]


def load_1m_bars(db_path: Path) -> pd.DataFrame:
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


def load_atr(db_path: Path) -> pd.DataFrame:
    """Load ATR_20 from daily_features."""
    print("Loading daily ATR...", end=" ", flush=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT trading_day, atr_20
            FROM daily_features
            WHERE symbol = 'MGC'
              AND orb_minutes = 5
              AND atr_20 IS NOT NULL
              AND trading_day >= ?
            ORDER BY trading_day
        """, [SCREEN_START]).fetchdf()
    finally:
        con.close()
    print(f"{len(df):,} days with ATR.")
    return df


def analyze_opening_drive(bars: pd.DataFrame, atr_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze opening drive for each session and window.

    Vectorized: pre-computes epoch seconds for fast slicing via searchsorted.
    """
    print("\nAnalyzing opening drives...")

    # Ensure UTC
    if bars["ts_utc"].dt.tz is not None:
        bars["ts_utc"] = bars["ts_utc"].dt.tz_convert("UTC")
    else:
        bars["ts_utc"] = bars["ts_utc"].dt.tz_localize("UTC")

    # Pre-compute epoch for fast range lookups
    # DuckDB returns datetime64[us] (microseconds), so divide by 10**6
    epoch = bars["ts_utc"].astype(np.int64) // 10**6
    epoch_vals = epoch.values
    open_vals = bars["open"].values
    close_vals = bars["close"].values
    high_vals = bars["high"].values
    low_vals = bars["low"].values

    # Build ATR lookup
    atr_df["td"] = pd.to_datetime(atr_df["trading_day"]).dt.date
    atr_lookup = dict(zip(atr_df["td"], atr_df["atr_20"]))

    # Get unique dates
    all_dates = sorted(bars["ts_utc"].dt.date.unique())
    print(f"  Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates):,} unique dates)")

    results = []

    for session_name, session_info in SESSIONS.items():
        start_h = session_info["start_utc"]
        dur_h = session_info["duration_h"]

        print(f"\n  Processing {session_name} ({session_info['description']})...", flush=True)

        session_count = 0

        for day_idx, cal_date in enumerate(all_dates):
            if day_idx % 1000 == 0 and day_idx > 0:
                print(f"    Day {day_idx}/{len(all_dates)}...", flush=True)

            # Session start/end as epoch seconds
            session_start_dt = datetime(cal_date.year, cal_date.month, cal_date.day,
                                        start_h, 0, 0, tzinfo=timezone.utc)
            session_start_ep = int(session_start_dt.timestamp())
            session_end_ep = session_start_ep + dur_h * 3600

            # Binary search for session bars
            i_start = np.searchsorted(epoch_vals, session_start_ep, side="left")
            i_end = np.searchsorted(epoch_vals, session_end_ep, side="left")

            if i_end - i_start < 30:
                continue

            session_open = open_vals[i_start]
            session_close = close_vals[i_end - 1]
            session_return = session_close - session_open

            if abs(session_return) < 0.01:
                continue

            session_direction = "up" if session_return > 0 else "down"

            # Determine trading day for ATR lookup
            if start_h == 23:
                trading_day = cal_date + timedelta(days=1)
            else:
                trading_day = cal_date

            atr = atr_lookup.get(trading_day)
            if atr is None or atr <= 0:
                continue

            session_count += 1

            for window_min in DRIVE_WINDOWS:
                window_end_ep = session_start_ep + window_min * 60
                i_win_end = np.searchsorted(epoch_vals, window_end_ep, side="left")

                n_drive = i_win_end - i_start
                if n_drive < max(5, window_min // 5):
                    continue

                drive_open = open_vals[i_start]
                drive_close = close_vals[i_win_end - 1]
                drive_return = drive_close - drive_open
                drive_range = high_vals[i_start:i_win_end].max() - low_vals[i_start:i_win_end].min()

                if abs(drive_return) < 0.01:
                    continue

                drive_direction = "up" if drive_return > 0 else "down"
                aligned = drive_direction == session_direction

                if drive_direction == "up":
                    trade_pnl_pts = session_close - drive_close
                else:
                    trade_pnl_pts = drive_close - session_close

                risk_pts = max(drive_range, SPEC.tick_size * SPEC.min_ticks_floor)
                pnl_dollars = trade_pnl_pts * SPEC.point_value - SPEC.total_friction
                risk_dollars = risk_pts * SPEC.point_value + SPEC.total_friction
                pnl_r = pnl_dollars / risk_dollars

                hold_pnl_pts = session_close - drive_close

                results.append({
                    "date": cal_date,
                    "trading_day": trading_day,
                    "session": session_name,
                    "window_min": window_min,
                    "drive_direction": drive_direction,
                    "session_direction": session_direction,
                    "aligned": aligned,
                    "drive_return_pts": drive_return,
                    "drive_range_pts": drive_range,
                    "drive_return_pct_atr": drive_return / atr * 100,
                    "session_return_pts": session_return,
                    "trade_pnl_pts": trade_pnl_pts,
                    "pnl_r": pnl_r,
                    "hold_pnl_pts": hold_pnl_pts,
                    "risk_pts": risk_pts,
                    "atr_20": atr,
                    "year": trading_day.year if hasattr(trading_day, "year") else pd.Timestamp(trading_day).year,
                    "remaining_bars": (i_end - i_start) - n_drive,
                })

        print(f"    {session_count:,} valid sessions found.")

    return pd.DataFrame(results)


def print_alignment_summary(data: pd.DataFrame) -> None:
    """Print alignment rates for each session/window combo."""
    print("\n" + "=" * 110)
    print("OPENING DRIVE ALIGNMENT: Does first X minutes predict session close?")
    print("=" * 110)

    header = f"{'Session':<15} {'Window':>7} {'N':>7} {'Align%':>8} {'T-stat':>8} {'Signif':>6}"
    print(header)
    print("-" * 110)

    for session in SESSIONS:
        for window in DRIVE_WINDOWS:
            subset = data[(data["session"] == session) & (data["window_min"] == window)]
            if len(subset) < 30:
                continue

            n = len(subset)
            align_rate = subset["aligned"].sum() / n
            # Test vs 50% null hypothesis
            se = np.sqrt(0.25 / n)  # std error of proportion under H0=0.5
            t_stat = (align_rate - 0.5) / se
            signif = "***" if abs(t_stat) > 2.576 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.645 else ""

            print(f"{session:<15} {window:>5}m {n:>7,} {align_rate*100:>7.1f}% {t_stat:>+8.2f} {signif:>6}")


def print_conditional_pnl(data: pd.DataFrame) -> None:
    """Print PnL if trading in drive direction."""
    print("\n" + "=" * 110)
    print("CONDITIONAL PnL: Enter at drive close in drive direction, hold to session close")
    print(f"  Cost: {SPEC.instrument}, friction={SPEC.total_friction:.2f} $/RT")
    print("=" * 110)

    header = f"{'Session':<15} {'Window':>7} {'N':>7} {'WR%':>6} {'ExpR':>8} {'Sharpe':>8} {'MaxDD_R':>8} {'Total_R':>9} {'ShANN':>7}"
    print(header)
    print("-" * 110)

    for session in SESSIONS:
        for window in DRIVE_WINDOWS:
            subset = data[(data["session"] == session) & (data["window_min"] == window)]
            if len(subset) < 30:
                continue

            pnl_r = subset["pnl_r"].values
            stats = compute_strategy_metrics(pnl_r)
            if stats is None:
                continue

            year_range = subset["year"].max() - subset["year"].min()
            n_years = max(year_range, 1)
            stats = annualize_sharpe(stats, n_years)

            sha = stats.get("sharpe_ann")
            sha_str = f"{sha:>+7.2f}" if sha is not None else "   N/A"

            print(f"{session:<15} {window:>5}m {stats['n']:>7,} {stats['wr']*100:>5.1f}% {stats['expr']:>+8.4f} {stats['sharpe']:>+8.4f} {stats['maxdd']:>+8.2f} {stats['total']:>+9.2f} {sha_str}")


def print_direction_breakdown(data: pd.DataFrame) -> None:
    """Break down by drive direction (up vs down)."""
    print("\n" + "=" * 110)
    print("DIRECTION BREAKDOWN: Up-drives vs Down-drives")
    print("=" * 110)

    for session in SESSIONS:
        print(f"\n--- {session} ---")
        header = f"  {'Window':>7} {'Dir':>5} {'N':>7} {'Align%':>8} {'WR%':>6} {'ExpR':>8} {'MeanMove':>10}"
        print(header)
        print("  " + "-" * 80)

        for window in DRIVE_WINDOWS:
            for direction in ["up", "down"]:
                subset = data[
                    (data["session"] == session)
                    & (data["window_min"] == window)
                    & (data["drive_direction"] == direction)
                ]
                if len(subset) < 30:
                    continue

                n = len(subset)
                align_pct = subset["aligned"].sum() / n * 100
                pnl_r = subset["pnl_r"].values
                stats = compute_strategy_metrics(pnl_r)
                if stats is None:
                    continue

                mean_move = subset["trade_pnl_pts"].mean()

                print(f"  {window:>5}m {direction:>5} {n:>7,} {align_pct:>7.1f}% {stats['wr']*100:>5.1f}% {stats['expr']:>+8.4f} {mean_move:>+10.3f}")


def print_drive_strength(data: pd.DataFrame) -> None:
    """Does a STRONGER opening drive predict better?"""
    print("\n" + "=" * 110)
    print("DRIVE STRENGTH: Does magnitude of opening drive predict outcome?")
    print("  (Comparing weak vs strong drives by ATR-normalized size)")
    print("=" * 110)

    for session in SESSIONS:
        for window in DRIVE_WINDOWS:
            subset = data[(data["session"] == session) & (data["window_min"] == window)]
            if len(subset) < 60:
                continue

            # Split into terciles by absolute drive return (as % of ATR)
            abs_drive = subset["drive_return_pct_atr"].abs()
            t1 = abs_drive.quantile(0.333)
            t2 = abs_drive.quantile(0.667)

            weak = subset[abs_drive <= t1]
            medium = subset[(abs_drive > t1) & (abs_drive <= t2)]
            strong = subset[abs_drive > t2]

            if len(weak) < 20 or len(strong) < 20:
                continue

            print(f"\n  {session} / {window}m drive:")
            print(f"    Weak   (|drive| <= {t1:.1f}% ATR): N={len(weak):>5}, Align={weak['aligned'].mean()*100:.1f}%, ExpR={weak['pnl_r'].mean():+.4f}")
            print(f"    Medium (|drive| <= {t2:.1f}% ATR): N={len(medium):>5}, Align={medium['aligned'].mean()*100:.1f}%, ExpR={medium['pnl_r'].mean():+.4f}")
            print(f"    Strong (|drive| >  {t2:.1f}% ATR): N={len(strong):>5}, Align={strong['aligned'].mean()*100:.1f}%, ExpR={strong['pnl_r'].mean():+.4f}")


def print_year_consistency(data: pd.DataFrame) -> None:
    """Year-by-year alignment rates."""
    print("\n" + "=" * 110)
    print("YEAR-BY-YEAR ALIGNMENT RATE")
    print("=" * 110)

    years = sorted(data["year"].unique())

    for session in SESSIONS:
        print(f"\n--- {session} ---")
        header = f"  {'Window':>7}" + "".join(f"{y:>10}" for y in years)
        print(header)
        print("  " + "-" * (7 + 10 * len(years)))

        for window in DRIVE_WINDOWS:
            row = f"  {window:>5}m"
            for y in years:
                subset = data[
                    (data["session"] == session)
                    & (data["window_min"] == window)
                    & (data["year"] == y)
                ]
                if len(subset) >= 10:
                    align = subset["aligned"].sum() / len(subset) * 100
                    row += f"  {align:5.1f}%({len(subset):>3})"
                else:
                    row += f"{'---':>10}"
            print(row)


def print_best_setups(data: pd.DataFrame) -> None:
    """Print the best session/window combinations."""
    print("\n" + "=" * 110)
    print("BEST OPENING DRIVE SETUPS (sorted by alignment rate)")
    print("=" * 110)

    results = []
    for session in SESSIONS:
        for window in DRIVE_WINDOWS:
            subset = data[(data["session"] == session) & (data["window_min"] == window)]
            if len(subset) < 50:
                continue

            n = len(subset)
            align = subset["aligned"].sum() / n
            se = np.sqrt(0.25 / n)
            t_stat = (align - 0.5) / se
            expr = subset["pnl_r"].mean()

            results.append({
                "session": session,
                "window": window,
                "n": n,
                "align_pct": align * 100,
                "t_stat": t_stat,
                "expr": expr,
            })

    results.sort(key=lambda x: x["align_pct"], reverse=True)

    for r in results:
        sig = "***" if abs(r["t_stat"]) > 2.576 else "**" if abs(r["t_stat"]) > 1.96 else "*" if abs(r["t_stat"]) > 1.645 else ""
        print(f"  {r['session']:<15} {r['window']:>3}m: Align={r['align_pct']:>5.1f}%, N={r['n']:>6,}, t={r['t_stat']:>+5.2f}{sig:>4}, ExpR={r['expr']:>+.4f}")

    print("\n  Interpretation:")
    print("    >55% alignment with t>1.96 = statistically significant predictive power")
    print("    >60% alignment = strong signal (rare in noisy markets)")
    print("    50% alignment = no predictive power (coin flip)")
    print("    <45% alignment = contrarian signal (fade the opening drive)")


def main():
    print("=" * 110)
    print(f"GOLD (MGC) OPENING DRIVE ANALYSIS  [screening: {SCREEN_START}+]")
    print("=" * 110)
    print()

    bars = load_1m_bars(DB_PATH)
    atr_df = load_atr(DB_PATH)
    data = analyze_opening_drive(bars, atr_df)

    if data.empty:
        print("No opening drive events found!")
        return

    print(f"\nTotal observations: {len(data):,}")

    print_alignment_summary(data)
    print_conditional_pnl(data)
    print_direction_breakdown(data)
    print_drive_strength(data)
    print_year_consistency(data)
    print_best_setups(data)

    print("\n" + "=" * 110)
    print("NOTES")
    print("=" * 110)
    print("  - This is NOT ORB analysis. ORB = breakout of a box. This = initial impulse direction.")
    print("  - Alignment rate = does the first X min direction predict end-of-session direction?")
    print("  - Conditional trade = enter in drive direction at window close, exit at session close.")
    print("  - Risk = drive range (high-low of the window), realistic stop placement.")
    print("  - Min N=30 for any conclusion. Significance tested vs 50% null hypothesis.")
    print(f"  - Cost model: {SPEC.instrument}, ${SPEC.total_friction:.2f}/RT ({SPEC.friction_in_points:.2f} pts)")
    print("  - Sessions: fixed Brisbane-time windows (not DST-aware). See pipeline/dst.py for actual market opens.")

    print("\n[Done]")


if __name__ == "__main__":
    main()

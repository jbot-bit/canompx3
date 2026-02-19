#!/usr/bin/env python3
"""
Multi-Day Trend Alignment: does prior-day trend direction predict ORB breakout quality?

Hypothesis: ORB breakouts aligned with a 2-3 day close trend have better follow-through.
Breakouts AGAINST a multi-day trend may be fakeouts.

Features engineered:
  - trend_2d: close[T-1] > close[T-2] -> "up", else "down"
  - trend_3d: close[T-1] > close[T-2] > close[T-3] -> "strong_up", etc.
  - alignment: does ORB break_dir match trend direction?

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_multiday_trend.py
    python scripts/analyze_multiday_trend.py --db-path C:/db/gold.db
    python scripts/analyze_multiday_trend.py --instrument MNQ
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(line_buffering=True)

from pipeline.asset_configs import get_enabled_sessions
from pipeline.cost_model import get_cost_spec
from research._alt_strategy_utils import compute_strategy_metrics, annualize_sharpe

def load_data(db_path: Path, instrument: str, orb_minutes: int = 5) -> pd.DataFrame:
    """Load daily_features with lagged daily closes for trend computation."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT *
            FROM daily_features
            WHERE symbol = ?
              AND orb_minutes = ?
            ORDER BY trading_day
        """, [instrument, orb_minutes]).fetchdf()
    finally:
        con.close()

    # Compute lagged closes
    df["close_lag1"] = df["daily_close"].shift(1)
    df["close_lag2"] = df["daily_close"].shift(2)
    df["close_lag3"] = df["daily_close"].shift(3)

    # 2-day trend: simple direction
    df["trend_2d"] = np.where(
        df["close_lag1"] > df["close_lag2"], "up", "down"
    )

    # 3-day trend: strong vs weak
    up2 = df["close_lag1"] > df["close_lag2"]
    up3 = df["close_lag2"] > df["close_lag3"]
    df["trend_3d"] = np.where(
        up2 & up3, "strong_up",
        np.where(~up2 & ~up3, "strong_down", "mixed")
    )

    # Gap direction
    df["gap_dir"] = np.where(df["gap_open_points"] > 0, "up",
                             np.where(df["gap_open_points"] < 0, "down", "flat"))

    return df

def load_outcomes(db_path: Path, instrument: str, session: str) -> pd.DataFrame:
    """Load orb_outcomes for a session, E1 CB2 only (standard baseline)."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT trading_day, entry_model, rr_target, confirm_bars,
                   pnl_r, outcome, entry_price, stop_price, target_price,
                   mfe_r, mae_r
            FROM orb_outcomes
            WHERE symbol = ? AND orb_label = ?
              AND orb_minutes = 5
              AND entry_ts IS NOT NULL
              AND pnl_r IS NOT NULL
            ORDER BY trading_day
        """, [instrument, session]).fetchall()
    finally:
        con.close()

    result = pd.DataFrame(df, columns=[
        "trading_day", "entry_model", "rr_target", "confirm_bars",
        "pnl_r", "outcome", "entry_price", "stop_price", "target_price",
        "mfe_r", "mae_r",
    ])
    result["trading_day"] = pd.to_datetime(result["trading_day"])
    return result

def fmt(m: dict, extra: str = "") -> str:
    if m is None or m["n"] == 0:
        return "N=0"
    s = (f"N={m['n']:<5d}  WR={m['wr']*100:5.1f}%  ExpR={m['expr']:+.3f}  "
         f"Sharpe={m['sharpe']:.3f}  MaxDD={m['maxdd']:.1f}R  Total={m['total']:+.1f}R")
    if extra:
        s += f"  {extra}"
    return s

def analyze_session(features: pd.DataFrame, outcomes: pd.DataFrame,
                    session: str, instrument: str):
    """Analyze alignment between multi-day trend and ORB breakout for one session."""
    break_dir_col = f"orb_{session}_break_dir"
    size_col = f"orb_{session}_size"

    if break_dir_col not in features.columns:
        print(f"  {session}: no break_dir column, skipping")
        return

    # Merge features with outcomes
    merged = outcomes.merge(
        features[["trading_day", "trend_2d", "trend_3d", "gap_dir",
                   break_dir_col, size_col, "atr_20"]],
        on="trading_day", how="inner"
    )

    if len(merged) < 30:
        print(f"  {session}: only {len(merged)} merged rows, skipping")
        return

    # Compute alignment: does break_dir match trend?
    merged["break_dir"] = merged[break_dir_col]
    merged["aligned_2d"] = (
        ((merged["break_dir"] == "long") & (merged["trend_2d"] == "up")) |
        ((merged["break_dir"] == "short") & (merged["trend_2d"] == "down"))
    )
    merged["aligned_3d"] = (
        ((merged["break_dir"] == "long") & (merged["trend_3d"] == "strong_up")) |
        ((merged["break_dir"] == "short") & (merged["trend_3d"] == "strong_down"))
    )
    merged["counter_3d"] = (
        ((merged["break_dir"] == "long") & (merged["trend_3d"] == "strong_down")) |
        ((merged["break_dir"] == "short") & (merged["trend_3d"] == "strong_up"))
    )

    # Filter to best entry: E1 CB2 (or E3 CB1 for retrace sessions)
    for em, cb_label in [("E1", 2), ("E3", 1)]:
        em_data = merged[(merged["entry_model"] == em) & (merged["confirm_bars"] == cb_label)]
        if len(em_data) < 30:
            continue

        print(f"\n  --- {session} {em} CB{cb_label} ---")

        for rr in sorted(em_data["rr_target"].unique()):
            rr_data = em_data[em_data["rr_target"] == rr]
            if len(rr_data) < 20:
                continue

            print(f"\n  RR {rr}:")

            # Baseline
            m = compute_strategy_metrics(rr_data["pnl_r"].values)
            print(f"    Baseline:              {fmt(m)}")

            # 2-day aligned vs counter
            aligned = rr_data[rr_data["aligned_2d"]]
            counter = rr_data[~rr_data["aligned_2d"]]
            if len(aligned) >= 10:
                m_a = compute_strategy_metrics(aligned["pnl_r"].values)
                print(f"    2d Aligned:            {fmt(m_a)}")
            if len(counter) >= 10:
                m_c = compute_strategy_metrics(counter["pnl_r"].values)
                print(f"    2d Counter:            {fmt(m_c)}")

            # 3-day strong aligned
            strong_aligned = rr_data[rr_data["aligned_3d"]]
            strong_counter = rr_data[rr_data["counter_3d"]]
            mixed = rr_data[rr_data["trend_3d"] == "mixed"]

            if len(strong_aligned) >= 10:
                m_sa = compute_strategy_metrics(strong_aligned["pnl_r"].values)
                print(f"    3d Strong Aligned:     {fmt(m_sa)}")
            if len(mixed) >= 10:
                m_mx = compute_strategy_metrics(mixed["pnl_r"].values)
                print(f"    3d Mixed:              {fmt(m_mx)}")
            if len(strong_counter) >= 10:
                m_sc = compute_strategy_metrics(strong_counter["pnl_r"].values)
                print(f"    3d Strong Counter:     {fmt(m_sc)}")

            # Interaction: alignment + ORB size filter (G4+)
            if size_col in rr_data.columns:
                g4 = rr_data[rr_data[size_col] >= 4.0]
                g4_aligned = g4[g4["aligned_2d"]]
                g4_counter = g4[~g4["aligned_2d"]]
                if len(g4) >= 10:
                    print(f"    G4+ Baseline:          {fmt(compute_strategy_metrics(g4['pnl_r'].values))}")
                if len(g4_aligned) >= 5:
                    print(f"    G4+ 2d Aligned:        {fmt(compute_strategy_metrics(g4_aligned['pnl_r'].values))}")
                if len(g4_counter) >= 5:
                    print(f"    G4+ 2d Counter:        {fmt(compute_strategy_metrics(g4_counter['pnl_r'].values))}")

            # Gap alignment: does gap direction match break direction?
            gap_aligned = rr_data[
                ((rr_data["break_dir"] == "long") & (rr_data["gap_dir"] == "up")) |
                ((rr_data["break_dir"] == "short") & (rr_data["gap_dir"] == "down"))
            ]
            gap_counter = rr_data[
                ((rr_data["break_dir"] == "long") & (rr_data["gap_dir"] == "down")) |
                ((rr_data["break_dir"] == "short") & (rr_data["gap_dir"] == "up"))
            ]
            if len(gap_aligned) >= 10:
                print(f"    Gap Aligned:           {fmt(compute_strategy_metrics(gap_aligned['pnl_r'].values))}")
            if len(gap_counter) >= 10:
                print(f"    Gap Counter:           {fmt(compute_strategy_metrics(gap_counter['pnl_r'].values))}")

def main():
    parser = argparse.ArgumentParser(description="Multi-day trend alignment analysis")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--instrument", default="MGC")
    args = parser.parse_args()

    instrument = args.instrument.upper()
    sessions = sorted(get_enabled_sessions(instrument))

    print("=" * 90)
    print(f"MULTI-DAY TREND ALIGNMENT ANALYSIS -- {instrument}")
    print(f"DB: {args.db_path}")
    print(f"Sessions: {', '.join(sessions)}")
    print("=" * 90)
    print()
    print("Hypothesis: ORB breakouts aligned with prior 2-3 day trend have better edge.")
    print("Counter: breakouts AGAINST multi-day trend may be fakeouts.")
    print()

    features = load_data(args.db_path, instrument)
    print(f"Loaded {len(features)} daily_features rows")
    print(f"Date range: {features['trading_day'].min()} to {features['trading_day'].max()}")

    # Show trend distribution
    print(f"\n2-day trend distribution:")
    print(features["trend_2d"].value_counts().to_string())
    print(f"\n3-day trend distribution:")
    print(features["trend_3d"].value_counts().to_string())
    print()

    for session in sessions:
        print(f"\n{'#' * 90}")
        print(f"  SESSION: {session}")
        print(f"{'#' * 90}")

        outcomes = load_outcomes(args.db_path, instrument, session)
        if outcomes.empty:
            print(f"  No outcomes for {session}")
            continue

        analyze_session(features, outcomes, session, instrument)

    print(f"\n{'=' * 90}")
    print("DONE")
    print("=" * 90)

if __name__ == "__main__":
    main()

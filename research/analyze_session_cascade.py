#!/usr/bin/env python3
"""
Session Cascade: does London/Asia range predict NY session ORB breakout quality?

Hypothesis: Wide London range -> wide 1800 ORB -> stronger breakout edge.
Narrow London range -> tight 1800 ORB -> chop/mean-reversion.

Also tests: London directional carry into 1800.
Also tests: Asia (0900/1000) range as predictor for 1800/2300.

Features:
  - london_range = session_london_high - session_london_low
  - london_dir = "up" if london_close > london_open
  - asia_range = MAX(orb_0900_high, orb_1000_high) - MIN(orb_0900_low, orb_1000_low)
  - range_percentile = london_range vs rolling 20-day median

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_session_cascade.py
    python scripts/analyze_session_cascade.py --db-path C:/db/gold.db
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(line_buffering=True)

from scripts._alt_strategy_utils import compute_strategy_metrics, annualize_sharpe


def load_features(db_path: Path, instrument: str = "MGC") -> pd.DataFrame:
    """Load daily_features with session stats."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT *
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5
            ORDER BY trading_day
        """, [instrument]).fetchdf()
    finally:
        con.close()

    # London range (session_london_high/low from daily_features)
    if "session_london_high" in df.columns and "session_london_low" in df.columns:
        df["london_range"] = df["session_london_high"] - df["session_london_low"]
        df["london_range_pct"] = df["london_range"].rolling(20).apply(
            lambda x: (x.iloc[-1] >= x.median()).astype(float) if len(x) == 20 else np.nan
        )
    else:
        df["london_range"] = np.nan

    # Asia range (composite of 0900 + 1000 ORBs)
    if "orb_0900_high" in df.columns and "orb_1000_high" in df.columns:
        asia_high = df[["orb_0900_high", "orb_1000_high"]].max(axis=1)
        asia_low = df[["orb_0900_low", "orb_1000_low"]].min(axis=1)
        df["asia_range"] = asia_high - asia_low
    else:
        df["asia_range"] = np.nan

    # NY range (0900+1000 composite in Brisbane time = actually Asia session)
    # session_ny_high/low is the actual NY session
    if "session_ny_high" in df.columns and "session_ny_low" in df.columns:
        df["ny_range"] = df["session_ny_high"] - df["session_ny_low"]
    else:
        df["ny_range"] = np.nan

    return df


def load_outcomes(db_path: Path, instrument: str, session: str) -> pd.DataFrame:
    """Load outcomes for a target session."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT trading_day, entry_model, rr_target, confirm_bars,
                   pnl_r, outcome, entry_price, stop_price
            FROM orb_outcomes
            WHERE symbol = ? AND orb_label = ?
              AND orb_minutes = 5
              AND entry_ts IS NOT NULL
              AND pnl_r IS NOT NULL
            ORDER BY trading_day
        """, [instrument, session]).fetchall()
    finally:
        con.close()

    result = pd.DataFrame(rows, columns=[
        "trading_day", "entry_model", "rr_target", "confirm_bars",
        "pnl_r", "outcome", "entry_price", "stop_price",
    ])
    result["trading_day"] = pd.to_datetime(result["trading_day"])
    return result


def fmt(m: dict) -> str:
    if m is None or m["n"] == 0:
        return "N=0"
    return (f"N={m['n']:<5d}  WR={m['wr']*100:5.1f}%  ExpR={m['expr']:+.3f}  "
            f"Sharpe={m['sharpe']:.3f}  MaxDD={m['maxdd']:.1f}R  Total={m['total']:+.1f}R")


def analyze_cascade(features: pd.DataFrame, outcomes: pd.DataFrame,
                    target_session: str, predictor_col: str, predictor_name: str):
    """Test if predictor_col range predicts target_session breakout quality."""
    merged = outcomes.merge(
        features[["trading_day", predictor_col, "atr_20"]].dropna(subset=[predictor_col]),
        on="trading_day", how="inner"
    )

    if len(merged) < 30:
        print(f"    Only {len(merged)} merged rows, skipping")
        return

    # Compute range terciles
    predictor_vals = merged[predictor_col]
    q33 = predictor_vals.quantile(0.33)
    q67 = predictor_vals.quantile(0.67)

    merged["range_bucket"] = np.where(
        predictor_vals <= q33, "narrow",
        np.where(predictor_vals <= q67, "medium", "wide")
    )

    # Also compute range relative to ATR
    if merged["atr_20"].notna().sum() > 0:
        merged["range_vs_atr"] = predictor_vals / merged["atr_20"]

    print(f"\n    {predictor_name} -> {target_session}")
    print(f"    Range terciles: narrow <= {q33:.1f}, medium <= {q67:.1f}, wide > {q67:.1f}")

    for em, cb in [("E1", 2), ("E3", 1)]:
        em_data = merged[(merged["entry_model"] == em) & (merged["confirm_bars"] == cb)]
        if len(em_data) < 20:
            continue

        for rr in sorted(em_data["rr_target"].unique()):
            rr_data = em_data[em_data["rr_target"] == rr]
            if len(rr_data) < 15:
                continue

            print(f"\n    {em} CB{cb} RR{rr}:")
            m_all = compute_strategy_metrics(rr_data["pnl_r"].values)
            print(f"      Baseline:    {fmt(m_all)}")

            for bucket in ["narrow", "medium", "wide"]:
                sub = rr_data[rr_data["range_bucket"] == bucket]
                if len(sub) >= 5:
                    m = compute_strategy_metrics(sub["pnl_r"].values)
                    print(f"      {bucket:10s}:  {fmt(m)}")

            # ATR-relative buckets
            if "range_vs_atr" in rr_data.columns and rr_data["range_vs_atr"].notna().sum() > 10:
                low_atr = rr_data[rr_data["range_vs_atr"] < 0.5]
                high_atr = rr_data[rr_data["range_vs_atr"] >= 0.5]
                if len(low_atr) >= 5:
                    print(f"      <0.5 ATR:    {fmt(compute_strategy_metrics(low_atr['pnl_r'].values))}")
                if len(high_atr) >= 5:
                    print(f"      >=0.5 ATR:   {fmt(compute_strategy_metrics(high_atr['pnl_r'].values))}")


def main():
    parser = argparse.ArgumentParser(description="Session cascade analysis")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--instrument", default="MGC")
    args = parser.parse_args()

    instrument = args.instrument.upper()

    print("=" * 90)
    print(f"SESSION CASCADE ANALYSIS -- {instrument}")
    print(f"DB: {args.db_path}")
    print("=" * 90)
    print()
    print("Does prior session range predict later session breakout quality?")
    print()

    features = load_features(args.db_path, instrument)
    print(f"Loaded {len(features)} daily_features rows")

    # Show range stats
    for col, name in [("london_range", "London"), ("asia_range", "Asia"), ("ny_range", "NY")]:
        if col in features.columns and features[col].notna().sum() > 0:
            vals = features[col].dropna()
            print(f"{name} range: mean={vals.mean():.2f}, median={vals.median():.2f}, "
                  f"std={vals.std():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")

    # Test cascades
    cascades = [
        # (target_session, predictor_column, label)
        ("1800", "london_range", "London Range"),
        ("1800", "asia_range", "Asia Range (0900+1000)"),
        ("2300", "london_range", "London Range"),
        ("2300", "asia_range", "Asia Range (0900+1000)"),
        ("0030", "ny_range", "NY Range"),
    ]

    # Also test dynamic sessions if available
    if "orb_LONDON_OPEN_size" in features.columns:
        cascades.append(("1800", "orb_LONDON_OPEN_size", "LONDON_OPEN ORB Size"))
    if "orb_CME_OPEN_size" in features.columns:
        cascades.append(("1000", "orb_CME_OPEN_size", "CME_OPEN ORB Size"))

    for target, predictor, label in cascades:
        if predictor not in features.columns or features[predictor].notna().sum() < 20:
            print(f"\n  Skipping {label} -> {target}: insufficient data")
            continue

        print(f"\n{'#' * 90}")
        print(f"  {label} -> {target} SESSION")
        print(f"{'#' * 90}")

        outcomes = load_outcomes(args.db_path, instrument, target)
        if outcomes.empty:
            print(f"  No outcomes for {target}")
            continue

        analyze_cascade(features, outcomes, target, predictor, label)

    print(f"\n{'=' * 90}")
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()

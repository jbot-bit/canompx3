#!/usr/bin/env python3
"""
Volume Confirmation: does volume at breakout bar predict follow-through?

Hypothesis: ORB break with SURGING volume = strong continuation.
ORB break with DECLINING volume = weak, likely reversion.

Method:
  1. Load 1m bars for each ORB break
  2. Compute median volume during ORB window (first N bars of session)
  3. Compute volume at the breakout bar (first bar closing outside ORB)
  4. Ratio = break_bar_volume / median_orb_volume
  5. Split into "surge" (>= 1.5x) vs "normal" (0.8-1.5x) vs "weak" (< 0.8x)
  6. Compare ExpR/WR/Sharpe across buckets

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_volume_confirmation.py
    python scripts/analyze_volume_confirmation.py --db-path C:/db/gold.db
    python scripts/analyze_volume_confirmation.py --instrument MNQ
"""

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(line_buffering=True)

from pipeline.asset_configs import get_enabled_sessions
from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.dst import (
    cme_open_brisbane, us_equity_open_brisbane,
    us_data_open_brisbane, london_open_brisbane,
)
from research._alt_strategy_utils import compute_strategy_metrics

def load_outcomes_with_entry_ts(db_path: Path, instrument: str, session: str) -> pd.DataFrame:
    """Load outcomes that have entry timestamps."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT trading_day, entry_model, rr_target, confirm_bars,
                   pnl_r, outcome, entry_ts, entry_price, stop_price,
                   mfe_r
            FROM orb_outcomes
            WHERE symbol = ? AND orb_label = ?
              AND orb_minutes = 5
              AND entry_ts IS NOT NULL
              AND pnl_r IS NOT NULL
            ORDER BY trading_day
        """, [instrument, session]).fetchall()
    finally:
        con.close()

    return pd.DataFrame(rows, columns=[
        "trading_day", "entry_model", "rr_target", "confirm_bars",
        "pnl_r", "outcome", "entry_ts", "entry_price", "stop_price",
        "mfe_r",
    ])

_DYNAMIC_RESOLVERS = {
    "CME_OPEN": cme_open_brisbane,
    "US_EQUITY_OPEN": us_equity_open_brisbane,
    "US_DATA_OPEN": us_data_open_brisbane,
    "LONDON_OPEN": london_open_brisbane,
}

_BRISBANE = ZoneInfo("Australia/Brisbane")

def _session_start_utc(session: str, trading_day) -> datetime | None:
    """Resolve session start to UTC datetime for a given trading day."""
    td = trading_day if isinstance(trading_day, date) else trading_day.date()

    if session in _DYNAMIC_RESOLVERS:
        hour, minute = _DYNAMIC_RESOLVERS[session](td)
    elif session.isdigit() and len(session) == 4:
        hour, minute = int(session[:2]), int(session[2:])
    else:
        return None

    local_dt = datetime.combine(td, datetime.min.time().replace(
        hour=hour, minute=minute), tzinfo=_BRISBANE)
    return local_dt.astimezone(timezone.utc)

def compute_volume_ratio_for_day(con, instrument: str, trading_day, session: str,
                                  entry_ts, orb_minutes: int = 5) -> float | None:
    """Compute volume ratio = break_bar_volume / median_orb_volume.

    Returns None if insufficient data.
    """
    session_start_utc = _session_start_utc(session, trading_day)
    if session_start_utc is None:
        return None

    orb_end_utc = session_start_utc + timedelta(minutes=orb_minutes)

    # Load ORB window bars
    orb_bars = con.execute("""
        SELECT volume FROM bars_1m
        WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?
        ORDER BY ts_utc
    """, [instrument, session_start_utc, orb_end_utc]).fetchall()

    if len(orb_bars) < 2:
        return None

    orb_volumes = [r[0] for r in orb_bars if r[0] is not None and r[0] > 0]
    if not orb_volumes:
        return None

    median_orb_vol = np.median(orb_volumes)
    if median_orb_vol == 0:
        return None

    # Get break bar volume (bar at entry_ts)
    # entry_ts might be TIMESTAMPTZ, ensure UTC
    break_bar = con.execute("""
        SELECT volume FROM bars_1m
        WHERE symbol = ? AND ts_utc = ?
    """, [instrument, entry_ts]).fetchone()

    if break_bar is None or break_bar[0] is None or break_bar[0] == 0:
        # Try nearest bar within 1 minute
        break_bar = con.execute("""
            SELECT volume FROM bars_1m
            WHERE symbol = ?
              AND ts_utc >= ? - INTERVAL '1 minute'
              AND ts_utc <= ? + INTERVAL '1 minute'
            ORDER BY ABS(EPOCH(ts_utc) - EPOCH(?::TIMESTAMPTZ))
            LIMIT 1
        """, [instrument, entry_ts, entry_ts, entry_ts]).fetchone()

        if break_bar is None or break_bar[0] is None:
            return None

    break_vol = break_bar[0]
    return break_vol / median_orb_vol

def fmt(m: dict) -> str:
    if m is None or m["n"] == 0:
        return "N=0"
    return (f"N={m['n']:<5d}  WR={m['wr']*100:5.1f}%  ExpR={m['expr']:+.3f}  "
            f"Sharpe={m['sharpe']:.3f}  MaxDD={m['maxdd']:.1f}R  Total={m['total']:+.1f}R")

def main():
    parser = argparse.ArgumentParser(description="Volume confirmation analysis")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--instrument", default="MGC")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Specific sessions to analyze (default: all enabled)")
    parser.add_argument("--sample-days", type=int, default=0,
                        help="Sample N random days per session (0 = all)")
    args = parser.parse_args()

    instrument = args.instrument.upper()
    sessions = args.sessions or sorted(get_enabled_sessions(instrument))

    # Focus on sessions with strongest known edge
    focus_sessions = [s for s in ["0900", "1000", "1800"] if s in sessions]
    if not focus_sessions:
        focus_sessions = sessions[:3]

    print("=" * 90)
    print(f"VOLUME CONFIRMATION ANALYSIS -- {instrument}")
    print(f"DB: {args.db_path}")
    print(f"Focus sessions: {', '.join(focus_sessions)}")
    print("=" * 90)
    print()
    print("Hypothesis: High volume at breakout bar = stronger continuation.")
    print("Volume ratio = break_bar_volume / median_ORB_window_volume")
    print()

    con = duckdb.connect(str(args.db_path), read_only=True)

    for session in focus_sessions:
        print(f"\n{'#' * 90}")
        print(f"  SESSION: {session}")
        print(f"{'#' * 90}")

        outcomes = load_outcomes_with_entry_ts(args.db_path, instrument, session)
        if outcomes.empty:
            print(f"  No outcomes for {session}")
            continue

        # Focus on E1 CB2 RR2.0 as reference (most representative)
        for em, cb in [("E1", 2), ("E3", 1)]:
            em_data = outcomes[(outcomes["entry_model"] == em) &
                               (outcomes["confirm_bars"] == cb)]
            if len(em_data) < 20:
                continue

            print(f"\n  --- {em} CB{cb} ---")

            for rr in [1.5, 2.0, 2.5, 3.0]:
                rr_data = em_data[em_data["rr_target"] == rr].copy()
                if len(rr_data) < 20:
                    continue

                # Sample if requested
                if args.sample_days > 0 and len(rr_data) > args.sample_days:
                    rr_data = rr_data.sample(args.sample_days, random_state=42)

                # Compute volume ratios
                ratios = []
                print(f"\n    RR{rr}: computing volume ratios for {len(rr_data)} trades...",
                      end="", flush=True)

                for _, row in rr_data.iterrows():
                    ratio = compute_volume_ratio_for_day(
                        con, instrument, row["trading_day"],
                        session, row["entry_ts"]
                    )
                    ratios.append(ratio)

                rr_data["vol_ratio"] = ratios
                has_ratio = rr_data[rr_data["vol_ratio"].notna()].copy()
                print(f" {len(has_ratio)} with volume data")

                if len(has_ratio) < 15:
                    print(f"    Insufficient volume data ({len(has_ratio)}), skipping")
                    continue

                # Stats on volume ratios
                vr = has_ratio["vol_ratio"]
                print(f"    Volume ratio stats: mean={vr.mean():.2f}, "
                      f"median={vr.median():.2f}, p25={vr.quantile(0.25):.2f}, "
                      f"p75={vr.quantile(0.75):.2f}")

                # Baseline
                m_all = compute_strategy_metrics(has_ratio["pnl_r"].values)
                print(f"    Baseline:      {fmt(m_all)}")

                # Buckets
                surge = has_ratio[has_ratio["vol_ratio"] >= 1.5]
                normal = has_ratio[(has_ratio["vol_ratio"] >= 0.8) &
                                   (has_ratio["vol_ratio"] < 1.5)]
                weak = has_ratio[has_ratio["vol_ratio"] < 0.8]

                if len(surge) >= 5:
                    print(f"    Surge (>=1.5x): {fmt(compute_strategy_metrics(surge['pnl_r'].values))}")
                if len(normal) >= 5:
                    print(f"    Normal (.8-1.5):{fmt(compute_strategy_metrics(normal['pnl_r'].values))}")
                if len(weak) >= 5:
                    print(f"    Weak (<0.8x):   {fmt(compute_strategy_metrics(weak['pnl_r'].values))}")

                # Also try 2x threshold
                strong_surge = has_ratio[has_ratio["vol_ratio"] >= 2.0]
                if len(strong_surge) >= 5:
                    print(f"    Strong (>=2.0x):{fmt(compute_strategy_metrics(strong_surge['pnl_r'].values))}")

                # MFE comparison: do surge trades run further?
                if "mfe_r" in has_ratio.columns and has_ratio["mfe_r"].notna().sum() > 10:
                    for label, sub in [("Surge", surge), ("Normal", normal), ("Weak", weak)]:
                        mfe_vals = sub["mfe_r"].dropna()
                        if len(mfe_vals) >= 5:
                            print(f"    {label} MFE: mean={mfe_vals.mean():.2f}R, "
                                  f"median={mfe_vals.median():.2f}R")

    con.close()

    print(f"\n{'=' * 90}")
    print("DONE")
    print("=" * 90)

if __name__ == "__main__":
    main()

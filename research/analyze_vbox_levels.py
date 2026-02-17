#!/usr/bin/env python3
"""
V-Box Volume S/R: classify trades by whether entry clears the highest-volume
bar's range during the ORB formation period.

Hypothesis: The highest-volume 1m bar during the ORB period acts as
intraday support/resistance. Trades that CLEAR this "V-Box" should
outperform trades that enter INSIDE it (CHOP).

For each (trading_day, orb_label):
  1. Load bars_1m for the ORB formation period (orb_minutes from session start)
  2. Find the bar with max volume -> vbox_high, vbox_low
  3. Classify each outcome:
     - LONG CLEAR:  entry_price > vbox_high
     - SHORT CLEAR: entry_price < vbox_low
     - CHOP:        entry inside [vbox_low, vbox_high]

Report: metrics split by CLEAR vs CHOP, by ORB size tier.

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_vbox_levels.py --db-path C:/db/gold.db
    python scripts/analyze_vbox_levels.py --sessions 0900,1000,1800
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.init_db import ORB_LABELS
from research._alt_strategy_utils import compute_strategy_metrics

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SESSIONS = ["0900", "1000", "1800", "2300"]
DEFAULT_MIN_ORB_SIZE = 4.0
ORB_MINUTES = 5  # standard 5-minute ORB formation

# Session start times in UTC hours (from Brisbane local)
# Brisbane 09:00 = UTC 23:00 (prev day)
# Brisbane 10:00 = UTC 00:00
# Brisbane 11:00 = UTC 01:00
# Brisbane 18:00 = UTC 08:00
# Brisbane 23:00 = UTC 13:00
# Brisbane 00:30 = UTC 14:30
SESSION_START_UTC = {
    "0900": (23, 0),   # prev day 23:00 UTC
    "1000": (0, 0),
    "1100": (1, 0),
    "1800": (8, 0),
    "2300": (13, 0),
    "0030": (14, 30),
}

SIZE_TIERS = {"G2": 2.0, "G4": 4.0, "G6": 6.0, "G8": 8.0}

# ---------------------------------------------------------------------------
# V-Box computation
# ---------------------------------------------------------------------------

def compute_vbox_for_day(
    bars_df: pd.DataFrame,
    session: str,
    trading_day: date,
) -> dict | None:
    """Find the highest-volume bar during ORB formation period.

    Returns {"vbox_high": float, "vbox_low": float, "vbox_volume": int}
    or None if no bars found.
    """
    hour, minute = SESSION_START_UTC[session]

    # Determine ORB start timestamp
    # 0900 session starts at 23:00 UTC on previous calendar day
    if session == "0900":
        orb_start = datetime(
            trading_day.year, trading_day.month, trading_day.day,
            hour, minute, tzinfo=timezone.utc
        ) - timedelta(days=1)
    else:
        orb_start = datetime(
            trading_day.year, trading_day.month, trading_day.day,
            hour, minute, tzinfo=timezone.utc
        )

    orb_end = orb_start + timedelta(minutes=ORB_MINUTES)

    # Filter bars to ORB formation window
    mask = (bars_df["ts_utc"] >= orb_start) & (bars_df["ts_utc"] < orb_end)
    orb_bars = bars_df[mask]

    if orb_bars.empty:
        return None

    # Find max volume bar
    max_vol_idx = orb_bars["volume"].idxmax()
    vbox_bar = orb_bars.loc[max_vol_idx]

    return {
        "vbox_high": float(vbox_bar["high"]),
        "vbox_low": float(vbox_bar["low"]),
        "vbox_volume": int(vbox_bar["volume"]),
    }

def classify_trade(
    entry_price: float,
    break_dir: str,
    vbox_high: float,
    vbox_low: float,
) -> str:
    """Classify trade as CLEAR or CHOP relative to V-Box.

    LONG CLEAR:  entry > vbox_high
    SHORT CLEAR: entry < vbox_low
    CHOP:        entry inside [vbox_low, vbox_high]
    """
    if break_dir == "long" and entry_price > vbox_high:
        return "CLEAR"
    if break_dir == "short" and entry_price < vbox_low:
        return "CLEAR"
    return "CHOP"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_outcomes(db_path: Path, sessions: list[str],
                  start: date, end: date) -> pd.DataFrame:
    """Load orb_outcomes with entry details."""
    session_ph = ", ".join(["?"] * len(sessions))

    dir_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_break_dir" for s in sessions
    )
    size_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_size" for s in sessions
    )

    query = f"""
        SELECT
            o.trading_day, o.orb_label, o.rr_target, o.confirm_bars,
            o.entry_model, o.entry_ts, o.entry_price, o.stop_price,
            o.target_price, o.outcome, o.pnl_r,
            CASE {size_cases} ELSE NULL END AS orb_size,
            CASE {dir_cases} ELSE NULL END AS break_dir
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.symbol = d.symbol
            AND o.trading_day = d.trading_day
            AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC'
            AND o.orb_minutes = 5
            AND o.orb_label IN ({session_ph})
            AND o.entry_ts IS NOT NULL
            AND o.outcome IS NOT NULL
            AND o.pnl_r IS NOT NULL
            AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day, o.orb_label
    """
    params = sessions + [start, end]

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(query, params).fetchdf()
    finally:
        con.close()

    # E3 always CB1
    df = df[~((df["entry_model"] == "E3") & (df["confirm_bars"] > 1))].copy()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    print(f"Loaded {len(df)} trades ({df['trading_day'].nunique()} days)")
    return df

def load_bars_for_day(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 1-minute bars for one trading day."""
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
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(db_path: Path, sessions: list[str],
                 min_orb_size: float, start: date, end: date) -> dict:
    """Run the V-Box classification analysis."""
    outcomes_df = load_outcomes(db_path, sessions, start, end)

    # Pre-compute V-Box levels per (trading_day, session)
    unique_days = sorted(outcomes_df["trading_day"].unique())
    print(f"Computing V-Box levels for {len(unique_days)} trading days...")
    t0 = time.time()

    vbox_cache = {}  # (trading_day, session) -> vbox dict
    bars_cache = {}

    for td in unique_days:
        if td not in bars_cache:
            bars_cache[td] = load_bars_for_day(db_path, td)
        bars = bars_cache[td]
        if bars.empty:
            continue
        for session in sessions:
            vbox = compute_vbox_for_day(bars, session, td)
            if vbox is not None:
                vbox_cache[(td, session)] = vbox

    print(f"  V-Box computed in {time.time() - t0:.1f}s "
          f"({len(vbox_cache)} day-sessions)")

    # Classify each trade
    # Results: {(session, em, size_tier, classification): [pnl_r, ...]}
    results = defaultdict(list)
    skipped = 0

    for _, row in outcomes_df.iterrows():
        td = row["trading_day"]
        session = row["orb_label"]
        orb_size = row["orb_size"]

        if orb_size is None or orb_size < min_orb_size:
            continue

        vbox = vbox_cache.get((td, session))
        if vbox is None:
            skipped += 1
            continue

        classification = classify_trade(
            row["entry_price"], row["break_dir"],
            vbox["vbox_high"], vbox["vbox_low"],
        )

        # Determine size tier
        size_tier = "ALL"
        for tier_name, threshold in sorted(SIZE_TIERS.items(), key=lambda x: x[1], reverse=True):
            if orb_size >= threshold:
                size_tier = tier_name
                break

        key = (session, row["entry_model"], size_tier, classification)
        results[key].append(row["pnl_r"])

        # Also aggregate across tiers
        key_all = (session, row["entry_model"], "ALL", classification)
        results[key_all].append(row["pnl_r"])

    if skipped:
        print(f"  Skipped {skipped} trades (no V-Box data)")

    return dict(results)

def print_results(results: dict) -> None:
    """Print formatted comparison table: CLEAR vs CHOP."""
    print("\n" + "=" * 80)
    print("V-BOX VOLUME S/R ANALYSIS: CLEAR vs CHOP")
    print("=" * 80)

    # Group by (session, em, tier)
    grouped = defaultdict(dict)
    for (session, em, tier, classification), pnls in results.items():
        grouped[(session, em, tier)][classification] = np.array(pnls)

    for (session, em, tier), class_data in sorted(grouped.items()):
        print(f"\n--- {session} / {em} / {tier} ---")
        print(f"  {'Class':<8} {'N':>6} {'WR':>7} {'ExpR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Total':>8}")
        print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

        for cls in ["CLEAR", "CHOP"]:
            arr = class_data.get(cls, np.array([]))
            if len(arr) == 0:
                continue
            m = compute_strategy_metrics(arr)
            if m is None:
                continue
            print(f"  {cls:<8} {m['n']:>6} {m['wr']:>7.3f} {m['expr']:>7.3f} "
                  f"{m['sharpe']:>7.3f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")

        # Delta
        clear_arr = class_data.get("CLEAR", np.array([]))
        chop_arr = class_data.get("CHOP", np.array([]))
        if len(clear_arr) > 0 and len(chop_arr) > 0:
            cm = compute_strategy_metrics(clear_arr)
            chm = compute_strategy_metrics(chop_arr)
            if cm and chm:
                delta_expr = cm["expr"] - chm["expr"]
                delta_sharpe = cm["sharpe"] - chm["sharpe"]
                print(f"  {'DELTA':<8} {'':>6} {'':>7} {delta_expr:>+7.3f} "
                      f"{delta_sharpe:>+7.3f}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V-Box Volume S/R Analysis")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--sessions", type=str, default=",".join(DEFAULT_SESSIONS))
    parser.add_argument("--min-orb-size", type=float, default=DEFAULT_MIN_ORB_SIZE)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 2, 5))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()

    sessions = [s.strip() for s in args.sessions.split(",")]

    print(f"Sessions: {sessions}")
    print(f"Min ORB size: {args.min_orb_size}")
    print(f"Date range: {args.start} to {args.end}")
    print()

    results = run_analysis(args.db_path, sessions, args.min_orb_size, args.start, args.end)
    print_results(results)

if __name__ == "__main__":
    main()

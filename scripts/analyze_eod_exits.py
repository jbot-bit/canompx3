#!/usr/bin/env python3
"""
EOD Exit Tournament: compare 3 exit strategies head-to-head per trade.

For each trade in orb_outcomes (G4+ filter, E1/E3, all sessions):
  1. Fixed RR  -- original stop/target from orb_outcomes (control)
  2. Session EOD -- mark-to-market at last bar before next 09:00 Brisbane
  3. 4h Time   -- mark-to-market at entry_ts + 4 hours

Different from early exit research (which tested killing losers early).
This tests REPLACING THE TARGET with time-based exits.

Report: metrics per (session, entry_model, exit_type).
Walk-forward: 12m train windows, select best exit per window, test OOS.

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_eod_exits.py --db-path C:/db/gold.db
    python scripts/analyze_eod_exits.py --sessions 0900,1800 --min-orb-size 4
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
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, to_r_multiple
from scripts._alt_strategy_utils import (
    compute_strategy_metrics,
    compute_walk_forward_windows,
)

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SESSIONS = ["0900", "1000", "1800", "2300"]
DEFAULT_ENTRY_MODELS = ["E1", "E3"]
DEFAULT_RR_TARGETS = [1.5, 2.0, 2.5]
DEFAULT_MIN_ORB_SIZE = 4.0
DEFAULT_CB = 2
FOUR_HOURS = timedelta(hours=4)

EXIT_TYPES = ["fixed_rr", "session_eod", "time_4h"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_outcomes(db_path: Path, sessions: list[str], entry_models: list[str],
                  rr_targets: list[float], min_orb_size: float,
                  start: date, end: date) -> pd.DataFrame:
    """Load orb_outcomes with entry_ts NOT NULL, G4+ filter."""
    session_ph = ", ".join(["?"] * len(sessions))
    em_ph = ", ".join(["?"] * len(entry_models))
    rr_ph = ", ".join(["?"] * len(rr_targets))

    size_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_size" for s in sessions
    )
    dir_cases = " ".join(
        f"WHEN o.orb_label = '{s}' THEN d.orb_{s}_break_dir" for s in sessions
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
            AND o.entry_model IN ({em_ph})
            AND o.rr_target IN ({rr_ph})
            AND o.entry_ts IS NOT NULL
            AND o.outcome IS NOT NULL
            AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day, o.orb_label
    """
    params = sessions + entry_models + rr_targets + [start, end]

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(query, params).fetchdf()
    finally:
        con.close()

    df = df[df["orb_size"] >= min_orb_size].copy()
    df = df[df["pnl_r"].notna()].copy()
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
# Exit computation
# ---------------------------------------------------------------------------

def compute_exits(
    bars_df: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    target_price: float,
    break_dir: str,
    original_outcome: str,
    original_pnl_r: float,
    trading_day: date,
) -> dict:
    """Compute all 3 exit types for one trade.

    Returns dict: {exit_type: pnl_r}.
    """
    is_long = break_dir == "long"
    risk_points = abs(entry_price - stop_price)
    if risk_points <= 0:
        return {et: 0.0 for et in EXIT_TYPES}

    results = {"fixed_rr": original_pnl_r}

    # Find entry bar index
    entry_mask = bars_df["ts_utc"] >= entry_ts
    if not entry_mask.any():
        results["session_eod"] = original_pnl_r
        results["time_4h"] = original_pnl_r
        return results

    entry_idx = entry_mask.idxmax()

    # --- Session EOD exit ---
    # Trading day ends at 23:00 UTC on trading_day (09:00 Brisbane next day)
    _, day_end_utc = compute_trading_day_utc_range(trading_day)
    # Scan bars from entry forward; check stop first, then mark-to-market at EOD
    eod_pnl = _scan_with_stop_and_time_exit(
        bars_df, entry_idx, entry_price, stop_price, is_long,
        cutoff_ts=day_end_utc,
    )
    results["session_eod"] = eod_pnl

    # --- 4h Time exit ---
    cutoff_4h = entry_ts + FOUR_HOURS
    time4h_pnl = _scan_with_stop_and_time_exit(
        bars_df, entry_idx, entry_price, stop_price, is_long,
        cutoff_ts=cutoff_4h,
    )
    results["time_4h"] = time4h_pnl

    return results


def _scan_with_stop_and_time_exit(
    bars_df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    is_long: bool,
    cutoff_ts: datetime,
) -> float:
    """Scan from entry_idx; stop takes priority, else mark-to-market at cutoff.

    Returns pnl_r via canonical to_r_multiple(spec, entry, stop, pnl_points).
    """
    last_close = entry_price  # fallback
    cost_spec = get_cost_spec("MGC")

    for i in range(entry_idx, len(bars_df)):
        bar = bars_df.iloc[i]
        bar_ts = bar["ts_utc"]

        # Stop check (before time cutoff on same bar)
        if is_long and bar["low"] <= stop_price:
            pnl_pts = stop_price - entry_price
            return to_r_multiple(cost_spec, entry_price, stop_price, pnl_pts)
        if not is_long and bar["high"] >= stop_price:
            pnl_pts = entry_price - stop_price
            return to_r_multiple(cost_spec, entry_price, stop_price, pnl_pts)

        last_close = bar["close"]

        # Time cutoff
        if bar_ts >= cutoff_ts:
            pnl_pts = (last_close - entry_price) if is_long else (entry_price - last_close)
            return to_r_multiple(cost_spec, entry_price, stop_price, pnl_pts)

    # Ran out of bars: mark-to-market at last available close
    pnl_pts = (last_close - entry_price) if is_long else (entry_price - last_close)
    return to_r_multiple(cost_spec, entry_price, stop_price, pnl_pts)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(db_path: Path, sessions: list[str], entry_models: list[str],
                 rr_targets: list[float], min_orb_size: float,
                 start: date, end: date) -> dict:
    """Run the full EOD exit tournament."""
    outcomes_df = load_outcomes(
        db_path, sessions, entry_models, rr_targets, min_orb_size, start, end,
    )

    # Group results by (session, entry_model, exit_type)
    results = defaultdict(lambda: defaultdict(list))

    # Cache bars by trading_day
    bars_cache = {}
    unique_days = sorted(outcomes_df["trading_day"].unique())
    print(f"Loading bars for {len(unique_days)} trading days...")
    t0 = time.time()

    for td in unique_days:
        bars_cache[td] = load_bars_for_day(db_path, td)

    print(f"  Bars loaded in {time.time() - t0:.1f}s")

    # Process each trade
    print(f"Processing {len(outcomes_df)} trades...")
    t0 = time.time()

    for _, row in outcomes_df.iterrows():
        td = row["trading_day"]
        bars = bars_cache.get(td)
        if bars is None or bars.empty:
            continue

        exits = compute_exits(
            bars, row["entry_ts"], row["entry_price"], row["stop_price"],
            row["target_price"], row["break_dir"],
            row["outcome"], row["pnl_r"], td,
        )

        key = (row["orb_label"], row["entry_model"])
        for exit_type, pnl_r in exits.items():
            results[key][exit_type].append(pnl_r)

    elapsed = time.time() - t0
    print(f"  Processed in {elapsed:.1f}s ({len(outcomes_df)/max(elapsed,0.1):.0f} trades/s)")

    return dict(results)


def print_results(results: dict) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("EOD EXIT TOURNAMENT RESULTS")
    print("=" * 80)

    for (session, em), exit_data in sorted(results.items()):
        print(f"\n--- {session} / {em} ---")
        print(f"  {'Exit Type':<15} {'N':>6} {'WR':>7} {'ExpR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Total':>8}")
        print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

        for exit_type in EXIT_TYPES:
            pnls = exit_data.get(exit_type, [])
            if not pnls:
                continue
            arr = np.array(pnls)
            m = compute_strategy_metrics(arr)
            if m is None:
                continue
            print(f"  {exit_type:<15} {m['n']:>6} {m['wr']:>7.3f} {m['expr']:>7.3f} "
                  f"{m['sharpe']:>7.3f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")


def run_walk_forward(results: dict) -> None:
    """Walk-forward: for each (session, em), select best exit in train window."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS (placeholder -- needs per-trade dates)")
    print("=" * 80)
    print("Note: Full walk-forward requires per-trade date tagging.")
    print("For now, the in-sample comparison above shows relative performance.")
    print("Run with --walk-forward after adding trade-level date support.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EOD Exit Tournament")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--sessions", type=str, default=",".join(DEFAULT_SESSIONS))
    parser.add_argument("--entry-models", type=str, default=",".join(DEFAULT_ENTRY_MODELS))
    parser.add_argument("--rr-targets", type=str, default=",".join(str(r) for r in DEFAULT_RR_TARGETS))
    parser.add_argument("--min-orb-size", type=float, default=DEFAULT_MIN_ORB_SIZE)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 2, 5))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()

    sessions = [s.strip() for s in args.sessions.split(",")]
    entry_models = [e.strip() for e in args.entry_models.split(",")]
    rr_targets = [float(r.strip()) for r in args.rr_targets.split(",")]

    print(f"Sessions: {sessions}")
    print(f"Entry models: {entry_models}")
    print(f"RR targets: {rr_targets}")
    print(f"Min ORB size: {args.min_orb_size}")
    print(f"Date range: {args.start} to {args.end}")
    print()

    results = run_analysis(
        args.db_path, sessions, entry_models, rr_targets,
        args.min_orb_size, args.start, args.end,
    )
    print_results(results)
    run_walk_forward(results)


if __name__ == "__main__":
    main()

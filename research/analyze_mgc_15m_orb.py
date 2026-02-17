#!/usr/bin/env python3
"""
Analyze whether 15m or 30m ORB windows improve MGC breakout performance.

Nested ORB research showed 1000 session benefits from 15m ORB (+0.208R premium).
This script does a proper full scan across all 6 sessions, comparing 5m baseline
(from orb_outcomes) against 15m and 30m ORB windows computed from bars_1m.

For each session, computes ORBs at 15m and 30m durations, runs bar-by-bar
breakout backtests with E1 (CB1, CB2) and E3 (CB1), at RR 1.5/2.0/2.5/3.0,
filtered by G4+ and G6+ ORB size.

Usage:
    python scripts/analyze_mgc_15m_orb.py
    python scripts/analyze_mgc_15m_orb.py --db-path C:/db/gold.db
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(line_buffering=True)

from pipeline.cost_model import get_cost_spec, to_r_multiple
from pipeline.build_daily_features import (
    compute_trading_day_utc_range,
    _orb_utc_window,
    _break_detection_window,
)
from pipeline.init_db import ORB_LABELS
from research._alt_strategy_utils import compute_strategy_metrics, annualize_sharpe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEC = get_cost_spec("MGC")

# Session start times in UTC (hour, minute)
SESSION_START_UTC = {
    "0900": (23, 0),   # 09:00 Brisbane = 23:00 UTC prev day
    "1000": (0, 0),    # 10:00 Brisbane = 00:00 UTC
    "1100": (1, 0),    # 11:00 Brisbane = 01:00 UTC
    "1800": (8, 0),    # 18:00 Brisbane = 08:00 UTC
    "2300": (13, 0),   # 23:00 Brisbane = 13:00 UTC
    "0030": (14, 30),  # 00:30 Brisbane = 14:30 UTC
}

RR_TARGETS = [1.5, 2.0, 2.5, 3.0]
ENTRY_CB_COMBOS = [
    ("E1", 1),
    ("E1", 2),
    ("E3", 1),
]
SIZE_FILTERS = {
    "G4+": 4.0,
    "G6+": 6.0,
}
ORB_DURATIONS = [15, 30]  # minutes (5m loaded from baseline)

START_DATE = date(2016, 2, 1)
END_DATE = date(2026, 2, 4)

# ---------------------------------------------------------------------------
# ORB computation from bars_1m
# ---------------------------------------------------------------------------

def compute_orb_from_bars(bars_df, trading_day, orb_label, orb_minutes):
    """Compute ORB high/low/size from bars_1m for a given duration.

    Uses the canonical _orb_utc_window from build_daily_features to get
    the correct UTC window, then filters bars.

    Returns (orb_high, orb_low, orb_size) or (None, None, None).
    """
    utc_start, utc_end = _orb_utc_window(trading_day, orb_label, orb_minutes)

    mask = (bars_df["ts_utc"] >= utc_start) & (bars_df["ts_utc"] < utc_end)
    orb_bars = bars_df.loc[mask]

    if orb_bars.empty:
        return None, None, None

    high = float(orb_bars["high"].max())
    low = float(orb_bars["low"].min())
    size = high - low
    return high, low, size

# ---------------------------------------------------------------------------
# Break detection
# ---------------------------------------------------------------------------

def detect_first_break(bars_df, trading_day, orb_label, orb_minutes,
                       orb_high, orb_low):
    """Find first 1m bar whose close breaks outside the ORB range.

    Returns (break_dir, break_ts) or (None, None).
    """
    if orb_high is None or orb_low is None:
        return None, None

    # Use canonical window calculation
    window_start, window_end = _break_detection_window(
        trading_day, orb_label, orb_minutes
    )

    mask = (bars_df["ts_utc"] >= window_start) & (bars_df["ts_utc"] < window_end)
    window_bars = bars_df.loc[mask]

    if window_bars.empty:
        return None, None

    closes = window_bars["close"].values
    ts_vals = window_bars["ts_utc"].values

    for i in range(len(closes)):
        c = float(closes[i])
        if c > orb_high:
            return "long", pd.Timestamp(ts_vals[i], tz="UTC")
        elif c < orb_low:
            return "short", pd.Timestamp(ts_vals[i], tz="UTC")

    return None, None

# ---------------------------------------------------------------------------
# Confirm bar detection (vectorized)
# ---------------------------------------------------------------------------

def find_confirm_bar(bars_df, break_ts, orb_high, orb_low,
                     break_dir, confirm_bars, window_end):
    """Find the Nth consecutive close outside ORB after break_ts.

    Returns (confirm_ts, confirm_close) or (None, None).
    """
    mask = (bars_df["ts_utc"] >= break_ts) & (bars_df["ts_utc"] < window_end)
    cand = bars_df.loc[mask]
    if cand.empty:
        return None, None

    closes = cand["close"].values
    ts_vals = cand["ts_utc"].values

    if break_dir == "long":
        outside = closes > orb_high
    else:
        outside = closes < orb_low

    run = 0
    for i in range(len(outside)):
        if outside[i]:
            run += 1
            if run >= confirm_bars:
                return pd.Timestamp(ts_vals[i], tz="UTC"), float(closes[i])
        else:
            run = 0

    return None, None

# ---------------------------------------------------------------------------
# E3 retrace detection
# ---------------------------------------------------------------------------

def find_e3_fill(bars_df, confirm_ts, orb_high, orb_low,
                 break_dir, stop_price, window_end):
    """For E3: find bar where price retraces to ORB level after confirm.

    Checks stop not breached before retrace.
    Returns (fill_ts, fill_price) or (None, None).
    """
    mask = (bars_df["ts_utc"] > confirm_ts) & (bars_df["ts_utc"] < window_end)
    post = bars_df.loc[mask]
    if post.empty:
        return None, None

    highs = post["high"].values
    lows = post["low"].values
    ts_vals = post["ts_utc"].values

    if break_dir == "long":
        # Limit buy at orb_high, stop at orb_low
        for i in range(len(lows)):
            if lows[i] <= stop_price:
                return None, None  # stop hit before fill
            if lows[i] <= orb_high:
                return pd.Timestamp(ts_vals[i], tz="UTC"), orb_high
    else:
        # Limit sell at orb_low, stop at orb_high
        for i in range(len(highs)):
            if highs[i] >= stop_price:
                return None, None
            if highs[i] >= orb_low:
                return pd.Timestamp(ts_vals[i], tz="UTC"), orb_low

    return None, None

# ---------------------------------------------------------------------------
# Bar-by-bar outcome resolution
# ---------------------------------------------------------------------------

def resolve_outcome(bars_df, entry_ts, entry_price, stop_price,
                    target_price, break_dir, trading_day_end):
    """Scan bars forward from entry to resolve win/loss/scratch.

    Returns pnl_r (float) using the canonical cost model.
    Stop priority on ambiguous bars.
    EOD exit if neither stop nor target hit.
    """
    risk_points = abs(entry_price - stop_price)
    if risk_points <= 0:
        return None

    rr_target_raw = abs(target_price - entry_price) / risk_points

    # Check fill bar first
    fill_mask = bars_df["ts_utc"] == entry_ts
    if fill_mask.any():
        bar = bars_df.loc[fill_mask].iloc[0]
        bh, bl = float(bar["high"]), float(bar["low"])
        if break_dir == "long":
            ht = bh >= target_price
            hs = bl <= stop_price
        else:
            ht = bl <= target_price
            hs = bh >= stop_price
        if ht and hs:
            return -1.0  # ambiguous -> loss
        if hs:
            return -1.0
        if ht:
            return round(to_r_multiple(SPEC, entry_price, stop_price,
                                       risk_points * rr_target_raw), 4)

    # Scan post-entry bars
    mask = (bars_df["ts_utc"] > entry_ts) & (bars_df["ts_utc"] < trading_day_end)
    post = bars_df.loc[mask]

    if post.empty:
        # Scratch: exit at entry (0 pnl points, minus friction)
        return round(to_r_multiple(SPEC, entry_price, stop_price, 0.0), 4)

    highs = post["high"].values
    lows = post["low"].values
    closes = post["close"].values

    if break_dir == "long":
        hit_target = highs >= target_price
        hit_stop = lows <= stop_price
    else:
        hit_target = lows <= target_price
        hit_stop = highs >= stop_price

    any_hit = hit_target | hit_stop
    if not any_hit.any():
        # Scratch/EOD exit at last close
        if break_dir == "long":
            eod_pnl = float(closes[-1]) - entry_price
        else:
            eod_pnl = entry_price - float(closes[-1])
        return round(to_r_multiple(SPEC, entry_price, stop_price, eod_pnl), 4)

    idx = int(np.argmax(any_hit))
    if hit_target[idx] and hit_stop[idx]:
        return -1.0  # ambiguous -> loss
    if hit_stop[idx]:
        return -1.0
    # Win
    return round(to_r_multiple(SPEC, entry_price, stop_price,
                               risk_points * rr_target_raw), 4)

# ---------------------------------------------------------------------------
# Single trade simulation
# ---------------------------------------------------------------------------

def simulate_trade(bars_df, trading_day, orb_label, orb_minutes,
                   orb_high, orb_low, break_dir, break_ts,
                   entry_model, confirm_bars, rr_target, td_end):
    """Run a single trade simulation. Returns pnl_r or None if no trade."""
    orb_size = orb_high - orb_low
    if orb_size <= 0:
        return None

    # Determine detection window end
    window_end = td_end

    # Find confirm bar
    confirm_ts, confirm_close = find_confirm_bar(
        bars_df, break_ts, orb_high, orb_low,
        break_dir, confirm_bars, window_end
    )
    if confirm_ts is None:
        return None

    # Determine entry
    if entry_model == "E1":
        # Next bar open after confirm
        next_mask = (bars_df["ts_utc"] > confirm_ts) & (bars_df["ts_utc"] < td_end)
        next_bars = bars_df.loc[next_mask]
        if next_bars.empty:
            return None
        entry_bar = next_bars.iloc[0]
        entry_ts = entry_bar["ts_utc"]
        entry_price = float(entry_bar["open"])
        stop_price = orb_low if break_dir == "long" else orb_high

    elif entry_model == "E3":
        stop_price = orb_low if break_dir == "long" else orb_high
        fill_ts, fill_price = find_e3_fill(
            bars_df, confirm_ts, orb_high, orb_low,
            break_dir, stop_price, td_end
        )
        if fill_ts is None:
            return None
        entry_ts = fill_ts
        entry_price = fill_price
    else:
        return None

    risk_points = abs(entry_price - stop_price)
    if risk_points <= 0:
        return None

    # Compute target
    if break_dir == "long":
        target_price = entry_price + risk_points * rr_target
    else:
        target_price = entry_price - risk_points * rr_target

    return resolve_outcome(
        bars_df, entry_ts, entry_price, stop_price,
        target_price, break_dir, td_end
    )

# ---------------------------------------------------------------------------
# Load 5m baseline from orb_outcomes
# ---------------------------------------------------------------------------

def load_5m_baseline(con, start_date, end_date):
    """Load existing 5m orb_outcomes as baseline for comparison.

    Returns dict keyed by (orb_label, entry_model, confirm_bars, rr_target, filter_name)
    with list of pnl_r values.
    """
    # Load orb_outcomes + daily_features for ORB size filtering
    query = """
        SELECT o.orb_label, o.entry_model, o.confirm_bars, o.rr_target,
               o.pnl_r, o.outcome,
               df.orb_0900_size, df.orb_1000_size, df.orb_1100_size,
               df.orb_1800_size, df.orb_2300_size, df.orb_0030_size
        FROM orb_outcomes o
        JOIN daily_features df
            ON o.trading_day = df.trading_day
            AND o.symbol = df.symbol
            AND df.orb_minutes = 5
        WHERE o.symbol = 'MGC'
          AND o.orb_minutes = 5
          AND o.trading_day >= ?
          AND o.trading_day <= ?
          AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL
    """
    rows = con.execute(query, [start_date, end_date]).fetchall()

    results = defaultdict(list)
    for row in rows:
        orb_label = row[0]
        entry_model = row[1]
        cb = row[2]
        rr = row[3]
        pnl_r = row[4]
        # outcome = row[5]

        # Get the ORB size for this session
        label_idx = ORB_LABELS.index(orb_label) if orb_label in ORB_LABELS else -1
        if label_idx < 0:
            continue
        orb_size = row[6 + label_idx]
        if orb_size is None:
            continue

        # Check (em, cb) combo is one we test
        if (entry_model, cb) not in ENTRY_CB_COMBOS:
            continue
        if rr not in RR_TARGETS:
            continue

        for filt_name, filt_min in SIZE_FILTERS.items():
            if orb_size >= filt_min:
                key = (orb_label, entry_model, cb, rr, filt_name)
                results[key].append(pnl_r)

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare 5m vs 15m vs 30m ORB windows for MGC breakouts"
    )
    parser.add_argument(
        "--db-path", type=str, default="C:/db/gold.db",
        help="Path to DuckDB database (default: C:/db/gold.db)"
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"FATAL: Database not found: {db_path}")
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)

    print("=" * 80)
    print("MGC ORB WINDOW ANALYSIS: 5m vs 15m vs 30m")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Sessions: {', '.join(ORB_LABELS)}")
    print(f"RR targets: {RR_TARGETS}")
    print(f"Entry combos: {ENTRY_CB_COMBOS}")
    print(f"Size filters: {list(SIZE_FILTERS.keys())}")
    print(f"Cost model: {SPEC.instrument} (friction={SPEC.total_friction:.2f})")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load 5m baseline from orb_outcomes
    # ------------------------------------------------------------------
    print("Loading 5m baseline from orb_outcomes...")
    t0 = time.monotonic()
    baseline = load_5m_baseline(con, START_DATE, END_DATE)
    print(f"  Loaded {len(baseline)} strategy keys, {sum(len(v) for v in baseline.values())} total trades")
    print(f"  Elapsed: {time.monotonic() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 2: Get trading days
    # ------------------------------------------------------------------
    print("Fetching trading days...")
    days_query = """
        SELECT DISTINCT
            CAST((ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours') AS DATE) AS trading_day
        FROM bars_1m
        WHERE symbol = 'MGC'
        AND CAST((ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours') AS DATE) >= ?
        AND CAST((ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours') AS DATE) <= ?
        ORDER BY trading_day
    """
    trading_days = [r[0] for r in con.execute(days_query, [START_DATE, END_DATE]).fetchall()]
    print(f"  {len(trading_days)} trading days")
    print()

    # ------------------------------------------------------------------
    # Step 3: Simulate 15m and 30m ORB trades
    # ------------------------------------------------------------------
    # Results keyed by (orb_minutes, orb_label, entry_model, cb, rr, filter_name)
    sim_results = defaultdict(list)

    total_days = len(trading_days)
    t0 = time.monotonic()

    for day_idx, td in enumerate(trading_days):
        # Load bars once per day
        td_start, td_end = compute_trading_day_utc_range(td)
        bars_df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = 'MGC'
              AND ts_utc >= ?::TIMESTAMPTZ
              AND ts_utc < ?::TIMESTAMPTZ
            ORDER BY ts_utc ASC
        """, [td_start.isoformat(), td_end.isoformat()]).fetchdf()

        if bars_df.empty:
            continue
        bars_df["ts_utc"] = pd.to_datetime(bars_df["ts_utc"], utc=True)

        for orb_minutes in ORB_DURATIONS:
            for orb_label in ORB_LABELS:
                # Compute ORB
                orb_high, orb_low, orb_size = compute_orb_from_bars(
                    bars_df, td, orb_label, orb_minutes
                )
                if orb_high is None or orb_size is None or orb_size <= 0:
                    continue

                # Detect break
                break_dir, break_ts = detect_first_break(
                    bars_df, td, orb_label, orb_minutes, orb_high, orb_low
                )
                if break_dir is None:
                    continue

                # For each filter / entry / rr combo
                for filt_name, filt_min in SIZE_FILTERS.items():
                    if orb_size < filt_min:
                        continue

                    for entry_model, cb in ENTRY_CB_COMBOS:
                        for rr in RR_TARGETS:
                            pnl_r = simulate_trade(
                                bars_df, td, orb_label, orb_minutes,
                                orb_high, orb_low, break_dir, break_ts,
                                entry_model, cb, rr, td_end
                            )
                            if pnl_r is not None:
                                key = (orb_minutes, orb_label, entry_model,
                                       cb, rr, filt_name)
                                sim_results[key].append(pnl_r)

        if (day_idx + 1) % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = (day_idx + 1) / elapsed
            remaining = (total_days - day_idx - 1) / rate if rate > 0 else 0
            print(f"  {day_idx + 1}/{total_days} days "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.monotonic() - t0
    print(f"  Simulation complete: {total_days} days in {elapsed:.1f}s")
    total_sim_trades = sum(len(v) for v in sim_results.values())
    print(f"  Total simulated trades: {total_sim_trades}")
    print()

    con.close()

    # ------------------------------------------------------------------
    # Step 4: Compute metrics and print results
    # ------------------------------------------------------------------
    years = (END_DATE - START_DATE).days / 365.25

    def fmt_metrics(pnls_list):
        """Return formatted metrics dict or None."""
        arr = np.array(pnls_list, dtype=float)
        stats = compute_strategy_metrics(arr)
        if stats is None:
            return None
        return annualize_sharpe(stats, years)

    def fmt_row(orb_tag, em, cb, rr, filt, stats, suffix=""):
        """Format a single output row."""
        if stats is None:
            return None
        return (
            f"{orb_tag:<7s} {em:<5s} {cb:<3d} {rr:<5.1f} {filt:<9s} "
            f"{stats['n']:>5d}  {stats['wr']*100:>5.1f}%  "
            f"{stats['expr']:>+6.3f}  {stats['sharpe']:>6.3f}  "
            f"{stats['maxdd']:>6.1f}  {stats['total']:>+7.1f}  {suffix}"
        )

    header = (
        f"{'ORB':<7s} {'Entry':<5s} {'CB':<3s} {'RR':<5s} {'Filter':<9s} "
        f"{'N':>5s}  {'WR':>6s}  {'ExpR':>6s}  {'Sharpe':>6s}  "
        f"{'MaxDD':>6s}  {'TotalR':>7s}"
    )

    # Collect best-per-session for summary
    session_bests = {}

    for orb_label in ORB_LABELS:
        print(f"=== SESSION {orb_label} ===")
        print(header)
        print("-" * len(header))

        for filt_name in SIZE_FILTERS:
            for em, cb in ENTRY_CB_COMBOS:
                for rr in RR_TARGETS:
                    # 5m baseline
                    bkey = (orb_label, em, cb, rr, filt_name)
                    b_pnls = baseline.get(bkey, [])
                    b_stats = fmt_metrics(b_pnls) if b_pnls else None
                    line = fmt_row("5m", em, cb, rr, filt_name, b_stats, "(baseline)")
                    if line:
                        print(line)

                    # 15m and 30m
                    for orb_min in ORB_DURATIONS:
                        tag = f"{orb_min}m"
                        skey = (orb_min, orb_label, em, cb, rr, filt_name)
                        s_pnls = sim_results.get(skey, [])
                        s_stats = fmt_metrics(s_pnls) if s_pnls else None
                        line = fmt_row(tag, em, cb, rr, filt_name, s_stats)
                        if line:
                            print(line)

                    # Track best for summary (use ExpR as criterion)
                    for tag, pnls in [
                        ("5m", b_pnls),
                        ("15m", sim_results.get((15, orb_label, em, cb, rr, filt_name), [])),
                        ("30m", sim_results.get((30, orb_label, em, cb, rr, filt_name), [])),
                    ]:
                        if not pnls:
                            continue
                        stats = fmt_metrics(pnls)
                        if stats is None:
                            continue
                        sk = orb_label
                        prev = session_bests.get(sk)
                        if prev is None or stats["expr"] > prev["expr"]:
                            session_bests[sk] = {
                                "orb_tag": tag, "em": em, "cb": cb,
                                "rr": rr, "filt": filt_name, **stats,
                            }

        print()

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    print("=" * 80)
    print("SUMMARY: Best ORB window per session (by ExpR)")
    print("=" * 80)
    print(f"{'Session':<9s} {'Best_ORB':<9s} {'EM':<5s} {'CB':<3s} {'RR':<5s} "
          f"{'Filter':<9s} {'N':>5s}  {'ExpR':>6s}  {'Sharpe':>6s}  {'Delta_vs_5m':>11s}")
    print("-" * 80)

    for orb_label in ORB_LABELS:
        best = session_bests.get(orb_label)
        if best is None:
            print(f"{orb_label:<9s} --  (no trades)")
            continue

        # Get 5m baseline ExpR for same combo
        bkey = (orb_label, best["em"], best["cb"], best["rr"], best["filt"])
        b_pnls = baseline.get(bkey, [])
        b_expr = 0.0
        if b_pnls:
            b_stats = fmt_metrics(b_pnls)
            if b_stats:
                b_expr = b_stats["expr"]

        delta = best["expr"] - b_expr
        delta_str = f"{delta:>+.3f}" if b_pnls else "N/A"

        print(
            f"{orb_label:<9s} {best['orb_tag']:<9s} {best['em']:<5s} "
            f"{best['cb']:<3d} {best['rr']:<5.1f} {best['filt']:<9s} "
            f"{best['n']:>5d}  {best['expr']:>+6.3f}  {best['sharpe']:>6.3f}  "
            f"{delta_str:>11s}"
        )

    print()
    print("Done.")

if __name__ == "__main__":
    main()

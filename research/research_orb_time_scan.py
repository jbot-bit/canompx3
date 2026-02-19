#!/usr/bin/env python3
"""
Full 24-Hour ORB Time Scan — find the best breakout times across the CME trading day.

Scans every 15-minute increment (96 candidate start times) for each instrument,
computes 5-minute ORBs, detects breaks, resolves outcomes at RR2.0, and ranks
times by total R earned. No friction applied — raw edge signal only.

Method:
  1. Load all 1m bars into numpy arrays indexed by (trading_day, minute_offset)
  2. For each candidate Brisbane time, vectorize ORB computation, then scan
     bar-by-bar for breaks and outcomes on G4+ days only
  3. Aggregate per (instrument, time), rank, and highlight current sessions

Usage:
  python research/research_orb_time_scan.py --db-path gold.db
  python research/research_orb_time_scan.py --db-path C:/db/gold.db
"""

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np

# Suppress expected NaN warnings from nanmax/nanmin on all-NaN days (weekends/holidays)
warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
import pandas as pd

_US_EASTERN = ZoneInfo("America/New_York")

# =========================================================================
# Constants
# =========================================================================

INSTRUMENTS = ["MGC", "MNQ", "MES", "MCL"]

# 96 candidate times: every 15 minutes from 00:00 to 23:45 Brisbane
CANDIDATE_TIMES = [(h, m) for h in range(24) for m in (0, 15, 30, 45)]

# Current sessions (Brisbane times) to highlight with *
CURRENT_SESSIONS = {
    (8, 0): "CME_OPEN(CDT)",
    (9, 0): "0900/CME_OPEN(CST)",
    (10, 0): "1000",
    (11, 0): "1100",
    (11, 30): "1130",
    (17, 0): "LONDON(BST)",
    (18, 0): "1800/LONDON(GMT)",
    (22, 30): "US_DATA(EDT)",
    (23, 0): "2300",
    (23, 30): "US_DATA(EST)/US_EQ(EDT)",
    (0, 30): "0030/US_EQ(EST)",
}

# ORB parameters
ORB_BARS = 5          # 5-minute ORB
G4_MIN = 4.0          # Minimum ORB size (points)
BREAK_WINDOW = 240    # 4 hours in minutes
OUTCOME_WINDOW = 480  # 8 hours in minutes
RR_TARGET = 2.0
MIN_TRADES = 30       # Minimum for ranking


# =========================================================================
# Data Loading
# =========================================================================

def load_bars(con, instrument):
    """Load all 1m bars for an instrument into a DataFrame."""
    return con.execute("""
        SELECT ts_utc, open, high, low, close
        FROM bars_1m
        WHERE symbol = ?
        ORDER BY ts_utc
    """, [instrument]).fetchdf()


def build_day_arrays(bars_df):
    """Convert bars DataFrame into 2D numpy arrays indexed by (day, minute_offset).

    Returns (trading_days_list, opens, highs, lows, closes) where each array
    has shape (n_days, 1440) with NaN for missing bars.
    """
    df = bars_df.copy()

    # Brisbane = UTC + 10 (constant, no DST)
    df["bris_dt"] = df["ts_utc"] + pd.Timedelta(hours=10)
    df["bris_hour"] = df["bris_dt"].dt.hour
    df["bris_minute"] = df["bris_dt"].dt.minute

    # Trading day: if Brisbane hour >= 9, same date; else previous date
    df["trading_day"] = df["bris_dt"].dt.normalize()
    mask = df["bris_hour"] < 9
    df.loc[mask, "trading_day"] -= pd.Timedelta(days=1)
    df["trading_day"] = df["trading_day"].dt.date

    # Minute offset from 09:00 Brisbane (0 = 09:00, 1439 = 08:59 next day)
    df["min_offset"] = ((df["bris_hour"] - 9) % 24) * 60 + df["bris_minute"]

    # Build day index
    all_days = sorted(df["trading_day"].unique())
    day_to_idx = {d: i for i, d in enumerate(all_days)}
    n_days = len(all_days)

    # Allocate arrays
    opens = np.full((n_days, 1440), np.nan)
    highs = np.full((n_days, 1440), np.nan)
    lows = np.full((n_days, 1440), np.nan)
    closes = np.full((n_days, 1440), np.nan)

    # Fill from DataFrame (vectorized)
    day_idx = df["trading_day"].map(day_to_idx).values
    min_idx = df["min_offset"].values

    opens[day_idx, min_idx] = df["open"].values
    highs[day_idx, min_idx] = df["high"].values
    lows[day_idx, min_idx] = df["low"].values
    closes[day_idx, min_idx] = df["close"].values

    return all_days, opens, highs, lows, closes


def build_dst_mask(trading_days):
    """Build boolean array: True if US is in DST (EDT) on that trading day."""
    mask = np.zeros(len(trading_days), dtype=bool)
    for i, td in enumerate(trading_days):
        dt = datetime(td.year, td.month, td.day, 12, 0, 0, tzinfo=_US_EASTERN)
        mask[i] = dt.utcoffset().total_seconds() == -4 * 3600
    return mask


def _aggregate_trades(trades):
    """Compute stats from a list of R-multiples."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "avg_r": np.nan, "total_r": 0.0, "win_rate": np.nan}
    arr = np.array(trades)
    return {
        "n": n,
        "avg_r": float(arr.mean()),
        "total_r": float(arr.sum()),
        "win_rate": float((arr >= RR_TARGET - 0.01).sum() / n),
    }


# =========================================================================
# Core Scan
# =========================================================================

def scan_candidate_time(bris_h, bris_m, highs, lows, closes, dst_mask):
    """Scan one candidate ORB start time across all trading days.

    Returns dict of aggregate metrics (with winter/summer split) or None if
    time wraps past trading day.
    """
    start_min = ((bris_h - 9) % 24) * 60 + bris_m

    # ORB minute columns
    orb_mins = [start_min + i for i in range(ORB_BARS)]
    if orb_mins[-1] >= 1440:
        return None  # ORB would wrap past end of trading day

    # Vectorized ORB computation across all days
    orb_h = np.column_stack([highs[:, m] for m in orb_mins])
    orb_l = np.column_stack([lows[:, m] for m in orb_mins])
    orb_c = np.column_stack([closes[:, m] for m in orb_mins])

    valid_orb = np.all(~np.isnan(orb_c), axis=1)
    orb_high = np.nanmax(orb_h, axis=1)
    orb_low = np.nanmin(orb_l, axis=1)
    orb_size = orb_high - orb_low

    g4_mask = valid_orb & (orb_size >= G4_MIN)
    g4_indices = np.where(g4_mask)[0]

    n_days_valid = int(valid_orb.sum())
    n_g4 = len(g4_indices)

    if n_g4 == 0:
        return _empty_result(n_days_valid)

    avg_orb_size = float(orb_size[g4_mask].mean())

    # Break detection and outcome resolution (per-day loop on G4+ days only)
    trades_winter = []
    trades_summer = []
    n_breaks = 0
    n_long = 0
    n_short = 0

    break_start = start_min + ORB_BARS
    max_break_min = min(break_start + BREAK_WINDOW, 1440)

    for day_idx in g4_indices:
        oh = orb_high[day_idx]
        ol = orb_low[day_idx]
        os_val = orb_size[day_idx]

        # --- Break detection ---
        break_dir = None
        entry = np.nan
        break_at = -1

        for m in range(break_start, max_break_min):
            c = closes[day_idx, m]
            if np.isnan(c):
                continue
            if c > oh:
                break_dir = "long"
                entry = c
                break_at = m
                break
            elif c < ol:
                break_dir = "short"
                entry = c
                break_at = m
                break

        if break_dir is None:
            continue

        n_breaks += 1
        if break_dir == "long":
            n_long += 1
        else:
            n_short += 1

        # --- Outcome resolution ---
        if break_dir == "long":
            target = entry + RR_TARGET * os_val
            stop = ol
        else:
            target = entry - RR_TARGET * os_val
            stop = oh

        max_outcome_min = min(break_at + 1 + OUTCOME_WINDOW, 1440)
        outcome_r = None
        last_close = entry

        for m in range(break_at + 1, max_outcome_min):
            h_val = highs[day_idx, m]
            l_val = lows[day_idx, m]
            c_val = closes[day_idx, m]

            if np.isnan(c_val):
                continue

            last_close = c_val

            if break_dir == "long":
                # Gate C: both sides hit on same bar → loss
                if l_val <= stop and h_val >= target:
                    outcome_r = -1.0
                    break
                if l_val <= stop:
                    outcome_r = -1.0
                    break
                if h_val >= target:
                    outcome_r = RR_TARGET
                    break
            else:
                if h_val >= stop and l_val <= target:
                    outcome_r = -1.0
                    break
                if h_val >= stop:
                    outcome_r = -1.0
                    break
                if l_val <= target:
                    outcome_r = RR_TARGET
                    break

        if outcome_r is None:
            # Timeout: mark-to-market
            if break_dir == "long":
                outcome_r = (last_close - entry) / os_val
            else:
                outcome_r = (entry - last_close) / os_val

        if dst_mask[day_idx]:
            trades_summer.append(outcome_r)
        else:
            trades_winter.append(outcome_r)

    # Aggregate
    trades = trades_winter + trades_summer
    all_stats = _aggregate_trades(trades)
    w_stats = _aggregate_trades(trades_winter)
    s_stats = _aggregate_trades(trades_summer)

    return {
        "n_days": n_days_valid,
        "n_g4": n_g4,
        "n_breaks": n_breaks,
        "break_rate": n_breaks / n_g4 if n_g4 > 0 else np.nan,
        "n_trades": all_stats["n"],
        "avg_r": all_stats["avg_r"],
        "total_r": all_stats["total_r"],
        "win_rate": all_stats["win_rate"],
        "avg_orb_size": avg_orb_size,
        "direction_bias": (n_long - n_short) / n_breaks if n_breaks > 0 else np.nan,
        "n_long": n_long,
        "n_short": n_short,
        # Winter/summer split (US DST)
        "n_winter": w_stats["n"],
        "avg_r_winter": w_stats["avg_r"],
        "total_r_winter": w_stats["total_r"],
        "wr_winter": w_stats["win_rate"],
        "n_summer": s_stats["n"],
        "avg_r_summer": s_stats["avg_r"],
        "total_r_summer": s_stats["total_r"],
        "wr_summer": s_stats["win_rate"],
    }


def _empty_result(n_days_valid):
    return {
        "n_days": n_days_valid,
        "n_g4": 0,
        "n_breaks": 0,
        "break_rate": np.nan,
        "n_trades": 0,
        "avg_r": np.nan,
        "total_r": 0.0,
        "win_rate": np.nan,
        "avg_orb_size": np.nan,
        "direction_bias": np.nan,
        "n_long": 0,
        "n_short": 0,
        "n_winter": 0,
        "avg_r_winter": np.nan,
        "total_r_winter": 0.0,
        "wr_winter": np.nan,
        "n_summer": 0,
        "avg_r_summer": np.nan,
        "total_r_summer": 0.0,
        "wr_summer": np.nan,
    }


# =========================================================================
# Output Formatting
# =========================================================================

def print_ranked_table(inst_df, instrument, top_n=20, bottom_n=10):
    """Print top and bottom times for one instrument."""
    qualified = inst_df[inst_df["n_trades"] >= MIN_TRADES].copy()

    # --- Top N ---
    print(f"\n{'=' * 90}")
    print(f"  {instrument} — TOP {top_n} TIMES by total_r "
          f"(min {MIN_TRADES} trades)")
    print(f"{'=' * 90}")

    if len(qualified) == 0:
        print(f"  No candidate times with >= {MIN_TRADES} trades")
        return

    top = qualified.nlargest(top_n, "total_r")
    _print_table(top)

    # --- Bottom N ---
    print(f"\n  {instrument} — BOTTOM {bottom_n} TIMES (worst avgR):")
    bottom = qualified.nsmallest(bottom_n, "avg_r")
    _print_table(bottom)

    # --- DST stability of top times ---
    print(f"\n  {instrument} — DST STABILITY (top 20 times):")
    print(f"    STBL = |winter-summer| <= 0.10R  W>> = winter much better  S>> = summer much better")
    top_for_dst = qualified.nlargest(min(top_n, len(qualified)), "total_r")
    n_stable = 0
    n_winter_hot = 0
    n_summer_hot = 0
    for _, row in top_for_dst.iterrows():
        v = _dst_verdict(row["avg_r_winter"], row["avg_r_summer"],
                         int(row["n_winter"]), int(row["n_summer"]))
        if v == "STBL":
            n_stable += 1
        elif v == "W>>":
            n_winter_hot += 1
        elif v == "S>>":
            n_summer_hot += 1
    print(f"    {n_stable} STABLE | {n_winter_hot} winter-dominant | "
          f"{n_summer_hot} summer-dominant")

    # Flag any top-20 time where the edge vanishes in one regime
    vanished = []
    for _, row in top_for_dst.iterrows():
        nw, ns = int(row["n_winter"]), int(row["n_summer"])
        aw, asr = row["avg_r_winter"], row["avg_r_summer"]
        if nw >= 10 and ns >= 10:
            if aw > 0 and asr <= 0:
                vanished.append(f"    {row['brisbane_time']}  W:+{aw:.3f}({nw}t) S:{asr:+.3f}({ns}t) — edge DIES in summer")
            elif asr > 0 and aw <= 0:
                vanished.append(f"    {row['brisbane_time']}  W:{aw:+.3f}({nw}t) S:+{asr:.3f}({ns}t) — edge DIES in winter")
    if vanished:
        print(f"    WARNING — edge vanishes in one DST regime:")
        for v in vanished:
            print(v)

    # --- Summary ---
    pos = len(qualified[qualified["avg_r"] > 0])
    neg = len(qualified[qualified["avg_r"] < 0])
    zero = len(qualified) - pos - neg
    print(f"\n  {instrument} summary: {len(qualified)} times with >= {MIN_TRADES} trades "
          f"| {pos} positive avgR | {neg} negative | {zero} zero")


def _dst_verdict(avg_w, avg_s, n_w, n_s):
    """Classify DST stability from winter/summer avgR."""
    if n_w < 10 or n_s < 10:
        return "LOW-N"
    if np.isnan(avg_w) or np.isnan(avg_s):
        return "  -- "
    delta = avg_w - avg_s
    if abs(delta) <= 0.10:
        return "STBL"
    if delta > 0.10:
        return "W>>"
    return "S>>"


def _print_table(df):
    header = (f"  {'Time':>6s} {'':1s} {'N':>5s} {'avgR':>7s} {'totR':>8s} "
              f"{'WR%':>6s} {'ORBsz':>6s}"
              f" | {'W_N':>4s} {'W_avgR':>7s} {'S_N':>4s} {'S_avgR':>7s} {'DST':>4s}"
              f"  Session")
    print(header)
    print(f"  {'-' * 6} {'-'} {'-' * 5} {'-' * 7} {'-' * 8} "
          f"{'-' * 6} {'-' * 6}"
          f" + {'-' * 4} {'-' * 7} {'-' * 4} {'-' * 7} {'-' * 4}"
          f"  {'-' * 22}")
    for _, row in df.iterrows():
        marker = "*" if row["current_session"] else " "
        sess = row["current_session"] if row["current_session"] else ""
        orb_str = f"{row['avg_orb_size']:5.1f}" if not np.isnan(row['avg_orb_size']) else "   --"
        n_w = int(row["n_winter"])
        n_s = int(row["n_summer"])
        avg_w = row["avg_r_winter"]
        avg_s = row["avg_r_summer"]
        w_str = f"{avg_w:+7.3f}" if not np.isnan(avg_w) else "     --"
        s_str = f"{avg_s:+7.3f}" if not np.isnan(avg_s) else "     --"
        verdict = _dst_verdict(avg_w, avg_s, n_w, n_s)
        print(
            f"  {row['brisbane_time']:>6s} {marker} "
            f"{row['n_trades']:5d} {row['avg_r']:+7.3f} {row['total_r']:+8.1f} "
            f"{row['win_rate'] * 100:5.1f}% "
            f"{orb_str}"
            f" | {n_w:4d} {w_str} {n_s:4d} {s_str} {verdict:>4s}"
            f"  {sess}"
        )


def print_analysis_summary(results_df):
    """Print cross-instrument analysis answering the prompt's key questions."""
    print(f"\n{'=' * 90}")
    print(f"  CROSS-INSTRUMENT ANALYSIS")
    print(f"{'=' * 90}")

    # Q1: Are our current 11 sessions the best times?
    print(f"\n  Q1: Are current sessions the best times?")
    for instrument in INSTRUMENTS:
        idf = results_df[
            (results_df["instrument"] == instrument)
            & (results_df["n_trades"] >= MIN_TRADES)
        ]
        if len(idf) == 0:
            continue

        top5 = idf.nlargest(5, "total_r")
        top_times = ", ".join(top5["brisbane_time"].values)
        current_in_top5 = sum(1 for s in top5["current_session"].values if s)
        print(f"    {instrument} best 5: {top_times}  "
              f"({current_in_top5}/5 are current sessions)")

    # Q2: Clusters of good times?
    print(f"\n  Q2: Time clusters with edge (top quartile by avgR, >= {MIN_TRADES} trades):")
    for instrument in INSTRUMENTS:
        idf = results_df[
            (results_df["instrument"] == instrument)
            & (results_df["n_trades"] >= MIN_TRADES)
        ].copy()
        if len(idf) == 0:
            continue

        q75 = idf["avg_r"].quantile(0.75)
        hot = idf[idf["avg_r"] >= q75].sort_values("bris_h")
        hot_times = [f"{r['brisbane_time']}" for _, r in hot.iterrows()]
        print(f"    {instrument} (avgR >= {q75:+.3f}): {', '.join(hot_times)}")

    # Q3: DST-shifted event times as separate opportunities?
    print(f"\n  Q3: DST-shifted times (summer vs winter CME/London):")
    dst_pairs = [
        ("08:00", "09:00", "CME_OPEN"),
        ("17:00", "18:00", "LONDON_OPEN"),
        ("22:30", "23:30", "US_DATA_OPEN"),
        ("23:30", "00:30", "US_EQUITY_OPEN"),
    ]
    for summer, winter, label in dst_pairs:
        for instrument in INSTRUMENTS:
            idf = results_df[results_df["instrument"] == instrument]
            s_row = idf[idf["brisbane_time"] == summer]
            w_row = idf[idf["brisbane_time"] == winter]
            if len(s_row) == 0 or len(w_row) == 0:
                continue
            s_r = s_row.iloc[0]
            w_r = w_row.iloc[0]
            if s_r["n_trades"] < 10 and w_r["n_trades"] < 10:
                continue
            print(f"    {instrument} {label}: summer {summer} avgR={s_r['avg_r']:+.3f} N={s_r['n_trades']} | "
                  f"winter {winter} avgR={w_r['avg_r']:+.3f} N={w_r['n_trades']}")

    # Q4: Dead zones
    print(f"\n  Q4: Dead zones (no instrument has positive avgR, >= {MIN_TRADES} trades):")
    dead_times = []
    for bris_h, bris_m in CANDIDATE_TIMES:
        time_str = f"{bris_h:02d}:{bris_m:02d}"
        rows = results_df[
            (results_df["brisbane_time"] == time_str)
            & (results_df["n_trades"] >= MIN_TRADES)
        ]
        if len(rows) == 0 or (rows["avg_r"] <= 0).all():
            dead_times.append(time_str)

    if dead_times:
        print(f"    {len(dead_times)} times: {', '.join(dead_times)}")
    else:
        print("    None — every time has at least one instrument with positive avgR")

    # Q5: Shared time structure across instruments?
    print(f"\n  Q5: Do instruments share the same peak times?")
    for instrument in INSTRUMENTS:
        idf = results_df[
            (results_df["instrument"] == instrument)
            & (results_df["n_trades"] >= MIN_TRADES)
        ]
        if len(idf) == 0:
            continue
        best = idf.nlargest(1, "total_r").iloc[0]
        print(f"    {instrument} #1 time: {best['brisbane_time']} "
              f"(avgR={best['avg_r']:+.3f}, N={best['n_trades']})")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full 24-hour ORB time scan across CME micro futures"
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to gold.db (default: auto-resolve via pipeline.paths)"
    )
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path)
    else:
        from pipeline.paths import GOLD_DB_PATH
        db_path = GOLD_DB_PATH

    print(f"\n{'=' * 90}")
    print(f"  FULL 24-HOUR ORB TIME SCAN")
    print(f"  Database: {db_path}")
    print(f"  Parameters: {ORB_BARS}m ORB | G{G4_MIN:.0f}+ filter | "
          f"RR{RR_TARGET:.1f} target | {BREAK_WINDOW // 60}h break window | "
          f"{OUTCOME_WINDOW // 60}h outcome window")
    print(f"  Scanning {len(CANDIDATE_TIMES)} candidate times × "
          f"{len(INSTRUMENTS)} instruments")
    print(f"{'=' * 90}")

    con = duckdb.connect(str(db_path), read_only=True)
    all_results = []

    try:
        for instrument in INSTRUMENTS:
            print(f"\n  Loading {instrument} bars...")
            t_load = time.time()
            bars_df = load_bars(con, instrument)

            if len(bars_df) == 0:
                print(f"    No data for {instrument}, skipping.")
                continue

            print(f"    {len(bars_df):,} bars loaded in {time.time() - t_load:.1f}s. "
                  f"Building day arrays...")
            t_build = time.time()
            all_days, opens, highs, lows, closes = build_day_arrays(bars_df)
            n_days = len(all_days)
            print(f"    {n_days} trading days ({all_days[0]} to {all_days[-1]}) "
                  f"built in {time.time() - t_build:.1f}s")

            # Free the DataFrame to save memory
            del bars_df

            # Build US DST mask for winter/summer split
            dst_mask = build_dst_mask(all_days)
            n_summer = int(dst_mask.sum())
            n_winter = n_days - n_summer
            print(f"    DST split: {n_winter} winter (EST) / {n_summer} summer (EDT)")

            print(f"    Scanning 96 candidate times...")
            t_scan = time.time()

            for i, (bris_h, bris_m) in enumerate(CANDIDATE_TIMES):
                result = scan_candidate_time(
                    bris_h, bris_m, highs, lows, closes, dst_mask
                )

                if result is not None:
                    result["instrument"] = instrument
                    result["brisbane_time"] = f"{bris_h:02d}:{bris_m:02d}"
                    result["bris_h"] = bris_h
                    result["bris_m"] = bris_m
                    result["current_session"] = CURRENT_SESSIONS.get(
                        (bris_h, bris_m), ""
                    )
                    all_results.append(result)

                if (i + 1) % 24 == 0:
                    elapsed = time.time() - t_scan
                    print(f"      {instrument}: {i + 1}/96 times "
                          f"({elapsed:.0f}s elapsed)")

            print(f"    {instrument} scan complete in "
                  f"{time.time() - t_scan:.0f}s")

            # Free arrays before next instrument
            del opens, highs, lows, closes
    finally:
        con.close()

    if not all_results:
        print("\nNo results to report.")
        return

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save CSV
    output_dir = Path("research/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "orb_time_scan_full.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  CSV saved: {csv_path} ({len(results_df)} rows)")

    # Per-instrument ranked tables
    for instrument in INSTRUMENTS:
        inst_df = results_df[results_df["instrument"] == instrument].copy()
        if len(inst_df) > 0:
            print_ranked_table(inst_df, instrument)

    # Cross-instrument analysis
    print_analysis_summary(results_df)

    print()


if __name__ == "__main__":
    main()

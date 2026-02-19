#!/usr/bin/env python3
"""
Month-of-year seasonality analysis for ORB breakout strategies.

Answers: does breakout edge vary by calendar month? Plausible mechanisms include
summer liquidity thinning (Jun-Aug), year-end rebalancing (Dec), and January
effect. Follows the same methodology as research_day_of_week.py.

  Q1: Month-by-month breakdown for every session x instrument x filter
  Q2: Skip-month filter simulation (what if we skip the worst months?)
  Q3: Yearly stability — does the month pattern hold across years?

Usage:
  python research/research_month_seasonality.py
  python research/research_month_seasonality.py --db-path C:/db/gold.db
"""

import argparse
import time
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

# =========================================================================
# Constants
# =========================================================================

OUTCOME_WINDOW = 480
RR_TARGET = 2.0
BREAK_WINDOW = 240
APERTURE_MIN = 5

INSTRUMENTS = ["MGC", "MNQ", "MES"]
SESSIONS = ["0900", "1000", "1100", "1800", "2300"]
FILTERS = [
    ("G4+", 4.0, None),
    ("G5+", 5.0, None),
    ("G6+", 6.0, None),
]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# =========================================================================
# Data loading (standalone engine — same as research_day_of_week.py)
# =========================================================================

def load_bars(con, instrument):
    return con.execute("""
        SELECT ts_utc, open, high, low, close
        FROM bars_1m WHERE symbol = ? ORDER BY ts_utc
    """, [instrument]).fetchdf()


def build_day_arrays(bars_df):
    df = bars_df.copy()
    df["bris_dt"] = df["ts_utc"] + pd.Timedelta(hours=10)
    df["bris_hour"] = df["bris_dt"].dt.hour
    df["bris_minute"] = df["bris_dt"].dt.minute
    df["trading_day"] = df["bris_dt"].dt.normalize()
    mask = df["bris_hour"] < 9
    df.loc[mask, "trading_day"] -= pd.Timedelta(days=1)
    df["trading_day"] = df["trading_day"].dt.date
    df["min_offset"] = ((df["bris_hour"] - 9) % 24) * 60 + df["bris_minute"]

    all_days = sorted(df["trading_day"].unique())
    day_to_idx = {d: i for i, d in enumerate(all_days)}
    n_days = len(all_days)

    highs = np.full((n_days, 1440), np.nan)
    lows = np.full((n_days, 1440), np.nan)
    closes = np.full((n_days, 1440), np.nan)

    day_idx = df["trading_day"].map(day_to_idx).values
    min_idx = df["min_offset"].values
    highs[day_idx, min_idx] = df["high"].values
    lows[day_idx, min_idx] = df["low"].values
    closes[day_idx, min_idx] = df["close"].values

    return all_days, highs, lows, closes


def scan_session(highs, lows, closes, bris_h, bris_m):
    """Return per-day results for all break-days (no size filter)."""
    n_days = highs.shape[0]
    start_min = ((bris_h - 9) % 24) * 60 + bris_m
    orb_mins = [start_min + i for i in range(APERTURE_MIN)]
    if orb_mins[-1] >= 1440:
        return {}

    orb_h = np.column_stack([highs[:, m] for m in orb_mins])
    orb_l = np.column_stack([lows[:, m] for m in orb_mins])
    orb_c = np.column_stack([closes[:, m] for m in orb_mins])

    valid_orb = np.all(~np.isnan(orb_c), axis=1)
    orb_high = np.nanmax(orb_h, axis=1)
    orb_low = np.nanmin(orb_l, axis=1)
    orb_size = orb_high - orb_low

    results = {}
    for di in range(n_days):
        if not valid_orb[di] or orb_size[di] < 0.01:
            continue
        oh, ol, os_val = orb_high[di], orb_low[di], orb_size[di]

        break_start = start_min + APERTURE_MIN
        max_bm = min(break_start + BREAK_WINDOW, 1440)
        break_dir, entry, break_at = None, np.nan, -1
        for m in range(break_start, max_bm):
            c = closes[di, m]
            if np.isnan(c):
                continue
            if c > oh:
                break_dir, entry, break_at = "long", c, m; break
            elif c < ol:
                break_dir, entry, break_at = "short", c, m; break
        if break_dir is None:
            continue

        target = entry + (RR_TARGET if break_dir == "long" else -RR_TARGET) * os_val
        stop = ol if break_dir == "long" else oh
        max_om = min(break_at + 1 + OUTCOME_WINDOW, 1440)
        outcome_r, last_close = None, entry
        for m in range(break_at + 1, max_om):
            h, l, c = highs[di, m], lows[di, m], closes[di, m]
            if np.isnan(c):
                continue
            last_close = c
            if break_dir == "long":
                if l <= stop and h >= target: outcome_r = -1.0; break
                if l <= stop: outcome_r = -1.0; break
                if h >= target: outcome_r = RR_TARGET; break
            else:
                if h >= stop and l <= target: outcome_r = -1.0; break
                if h >= stop: outcome_r = -1.0; break
                if l <= target: outcome_r = RR_TARGET; break
        if outcome_r is None:
            outcome_r = ((last_close - entry) / os_val if break_dir == "long"
                         else (entry - last_close) / os_val)

        results[di] = {
            "orb_size": float(os_val), "direction": break_dir,
            "outcome_r": float(outcome_r),
        }
    return results


# =========================================================================
# Q1: Month-by-month breakdown
# =========================================================================

def q1_month_breakdown(data_cache):
    print(f"\n{'=' * 100}")
    print(f"  Q1: MONTH-BY-MONTH BREAKDOWN")
    print(f"{'=' * 100}")

    all_rows = []

    for instrument in INSTRUMENTS:
        if instrument not in data_cache:
            continue
        all_days, highs, lows, closes = data_cache[instrument]

        for session in SESSIONS:
            bh, bm = int(session[:2]), int(session[2:])
            per_day = scan_session(highs, lows, closes, bh, bm)

            for fname, flo, fhi in FILTERS:
                filtered = {}
                for di, r in per_day.items():
                    sz = r["orb_size"]
                    if sz < flo:
                        continue
                    if fhi is not None and sz >= fhi:
                        continue
                    filtered[di] = r

                if len(filtered) < 30:
                    continue

                # Group by month (1-12)
                month_groups = defaultdict(list)
                for di, r in filtered.items():
                    td = all_days[di]
                    month_groups[td.month].append(r["outcome_r"])

                header_printed = False
                for month in range(1, 13):
                    rs = month_groups.get(month, [])
                    if not rs:
                        continue
                    n = len(rs)
                    avg = float(np.mean(rs))
                    wr = float(np.mean([1 if r > 0 else 0 for r in rs]))
                    tot = float(np.sum(rs))

                    if not header_printed:
                        print(f"\n  {instrument} {session} {fname} (N={len(filtered)}):")
                        print(f"    {'Month':>5s} {'N':>5s} {'avgR':>7s} {'WR':>6s} {'totR':>8s}")
                        print(f"    {'-' * 35}")
                        header_printed = True

                    marker = ""
                    if avg > 0.15:
                        marker = "  ++"
                    elif avg < -0.10:
                        marker = "  --"

                    print(f"    {MONTH_NAMES[month - 1]:>5s} {n:5d} {avg:+7.3f} {wr:5.1%} {tot:+8.1f}{marker}")

                    all_rows.append({
                        "instrument": instrument, "session": session, "filter": fname,
                        "month": month, "month_name": MONTH_NAMES[month - 1],
                        "n": n, "avg_r": avg, "wr": wr, "total_r": tot,
                    })

    return all_rows


# =========================================================================
# Q2: Skip-month filter simulation
# =========================================================================

def q2_skip_month(data_cache, q1_rows):
    print(f"\n{'=' * 100}")
    print(f"  Q2: SKIP-MONTH FILTER SIMULATION")
    print(f"  Testing: skip worst 1 month, skip worst 2 months, best-quarter only")
    print(f"{'=' * 100}")

    all_rows = []

    combos = set()
    for r in q1_rows:
        combos.add((r["instrument"], r["session"], r["filter"]))

    for instrument, session, fname in sorted(combos):
        if instrument not in data_cache:
            continue
        combo_rows = [r for r in q1_rows
                      if r["instrument"] == instrument
                      and r["session"] == session
                      and r["filter"] == fname]

        if not combo_rows:
            continue

        combo_rows.sort(key=lambda x: x["avg_r"])
        worst_1 = combo_rows[0]
        worst_2 = combo_rows[:2]
        best_quarter = sorted(combo_rows, key=lambda x: x["avg_r"], reverse=True)[:3]

        # Baseline
        total_n = sum(r["n"] for r in combo_rows)
        total_r = sum(r["total_r"] for r in combo_rows)
        baseline_avg = total_r / total_n if total_n > 0 else 0

        # Skip worst 1
        skip1_rows = [r for r in combo_rows if r["month"] != worst_1["month"]]
        s1_n = sum(r["n"] for r in skip1_rows)
        s1_tot = sum(r["total_r"] for r in skip1_rows)
        s1_avg = s1_tot / s1_n if s1_n > 0 else 0

        # Skip worst 2
        skip2_months = {r["month"] for r in worst_2}
        skip2_rows = [r for r in combo_rows if r["month"] not in skip2_months]
        s2_n = sum(r["n"] for r in skip2_rows)
        s2_tot = sum(r["total_r"] for r in skip2_rows)
        s2_avg = s2_tot / s2_n if s2_n > 0 else 0

        # Best quarter (top 3 months)
        bq_n = sum(r["n"] for r in best_quarter)
        bq_tot = sum(r["total_r"] for r in best_quarter)
        bq_avg = bq_tot / bq_n if bq_n > 0 else 0

        print(f"\n  {instrument} {session} {fname}:")
        print(f"    Worst month: {MONTH_NAMES[worst_1['month'] - 1]} (avgR={worst_1['avg_r']:+.3f}, N={worst_1['n']})")
        best_3_names = ", ".join(MONTH_NAMES[r["month"] - 1] for r in best_quarter)
        print(f"    Best quarter: {best_3_names}")
        print(f"    {'':>20s} {'N':>5s} {'avgR':>7s} {'totR':>8s} {'vs base':>8s}")
        print(f"    {'Baseline (all months)':>20s} {total_n:5d} {baseline_avg:+7.3f} {total_r:+8.1f} {'':>8s}")
        print(f"    {'Skip worst 1':>20s} {s1_n:5d} {s1_avg:+7.3f} {s1_tot:+8.1f} {s1_tot - total_r:+8.1f}")
        print(f"    {'Skip worst 2':>20s} {s2_n:5d} {s2_avg:+7.3f} {s2_tot:+8.1f} {s2_tot - total_r:+8.1f}")
        print(f"    {'Best quarter only':>20s} {bq_n:5d} {bq_avg:+7.3f} {bq_tot:+8.1f} {bq_tot - total_r:+8.1f}")

        all_rows.append({
            "instrument": instrument, "session": session, "filter": fname,
            "worst_month": MONTH_NAMES[worst_1["month"] - 1],
            "best_quarter": best_3_names,
            "baseline_n": total_n, "baseline_avgR": baseline_avg, "baseline_totR": total_r,
            "skip1_n": s1_n, "skip1_avgR": s1_avg, "skip1_totR": s1_tot,
            "skip2_n": s2_n, "skip2_avgR": s2_avg, "skip2_totR": s2_tot,
            "bestQ_n": bq_n, "bestQ_avgR": bq_avg, "bestQ_totR": bq_tot,
        })

    return all_rows


# =========================================================================
# Q3: Yearly stability — does the month pattern hold across years?
# =========================================================================

def q3_yearly_stability(data_cache):
    print(f"\n{'=' * 100}")
    print(f"  Q3: YEARLY STABILITY — Does the month pattern hold per-year?")
    print(f"  (MGC 0900 G4+ and MGC 1000 G4+ only)")
    print(f"{'=' * 100}")

    for instrument in ["MGC"]:
        if instrument not in data_cache:
            continue
        all_days, highs, lows, closes = data_cache[instrument]

        for session in ["0900", "1000"]:
            bh, bm = int(session[:2]), int(session[2:])
            per_day = scan_session(highs, lows, closes, bh, bm)
            filtered = {di: r for di, r in per_day.items() if r["orb_size"] >= 4.0}

            if len(filtered) < 30:
                continue

            # Group by (year, month)
            year_month = defaultdict(lambda: defaultdict(list))
            for di, r in filtered.items():
                td = all_days[di]
                year_month[td.year][td.month].append(r["outcome_r"])

            years = sorted(year_month.keys())
            if len(years) < 2:
                continue

            # Show quarterly aggregates per year for readability
            print(f"\n  {instrument} {session} G4+ — avgR by year x quarter:")
            q_names = ["Q1 (JFM)", "Q2 (AMJ)", "Q3 (JAS)", "Q4 (OND)"]
            header = f"    {'Year':>6s}"
            for qn in q_names:
                header += f"  {qn:>12s}"
            header += f"  {'Spread':>7s}"
            print(header)
            print(f"    {'-' * (8 + 14 * 4 + 9)}")

            for year in years:
                row = f"    {year:>6d}"
                q_avgs = []
                for qi, months in enumerate([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]):
                    rs = []
                    for m in months:
                        rs.extend(year_month[year].get(m, []))
                    if rs:
                        avg = float(np.mean(rs))
                        n = len(rs)
                        q_avgs.append(avg)
                        row += f"  {avg:+6.3f}({n:3d})"
                    else:
                        row += f"  {'--':>12s}"
                spread = max(q_avgs) - min(q_avgs) if len(q_avgs) >= 2 else 0
                row += f"  {spread:7.3f}"
                print(row)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Month-of-year seasonality analysis")
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path)
    else:
        try:
            from pipeline.paths import GOLD_DB_PATH
            db_path = GOLD_DB_PATH
        except ImportError:
            db_path = Path("gold.db")

    print(f"\n{'=' * 100}")
    print(f"  MONTH-OF-YEAR SEASONALITY — ORB Breakout Edge by Calendar Month")
    print(f"  Database: {db_path}")
    print(f"  RR{RR_TARGET} | {BREAK_WINDOW}min break | {APERTURE_MIN}min aperture")
    print(f"{'=' * 100}")

    con = duckdb.connect(str(db_path), read_only=True)
    t_total = time.time()

    try:
        data_cache = {}
        for instrument in INSTRUMENTS:
            print(f"\n  Loading {instrument}...")
            t = time.time()
            bars_df = load_bars(con, instrument)
            if len(bars_df) == 0:
                continue
            all_days, highs, lows, closes = build_day_arrays(bars_df)
            del bars_df
            print(f"    {len(all_days)} days in {time.time() - t:.1f}s")
            data_cache[instrument] = (all_days, highs, lows, closes)

        if not data_cache:
            print("  No data.")
            return

        q1_rows = q1_month_breakdown(data_cache)
        q2_rows = q2_skip_month(data_cache, q1_rows)
        q3_yearly_stability(data_cache)

        # Save CSVs
        out = Path("research/output")
        out.mkdir(parents=True, exist_ok=True)

        if q1_rows:
            pd.DataFrame(q1_rows).to_csv(out / "month_seasonality_breakdown.csv",
                                          index=False, float_format="%.4f")
        if q2_rows:
            pd.DataFrame(q2_rows).to_csv(out / "month_seasonality_skip_filter.csv",
                                          index=False, float_format="%.4f")

        print(f"\n  CSVs saved to research/output/month_seasonality_*.csv")
        print(f"  Total: {time.time() - t_total:.1f}s")

    finally:
        con.close()


if __name__ == "__main__":
    main()

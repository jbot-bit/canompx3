#!/usr/bin/env python3
"""
P2 Calendar Effects — Day-of-week filter + FOMC/NFP/opex overlay.

Rolling eval already showed the signal:
  1000: Wed best (+0.133), Thu/Fri poison (-0.22 to -0.24)
  0900: Fri best (+0.39 to +0.64)

This script formalizes it:
  Q1: Day-of-week breakdown for every session x instrument x filter
  Q2: Skip-day filter simulation (what if we skip the worst days?)
  Q3: FOMC/NFP/opex day overlay (do macro events explain the pattern?)

Usage:
  python research/research_day_of_week.py
  python research/research_day_of_week.py --db-path C:/db/gold.db
"""

import argparse
import time
import warnings
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

# =========================================================================
# Timezone helpers
# =========================================================================

_US_EASTERN = ZoneInfo("America/New_York")
_UK_LONDON = ZoneInfo("Europe/London")

def is_us_dst(trading_day: date) -> bool:
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_US_EASTERN)
    return dt.utcoffset().total_seconds() == -4 * 3600

def is_uk_dst(trading_day: date) -> bool:
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_UK_LONDON)
    return dt.utcoffset().total_seconds() == 1 * 3600

# =========================================================================
# Constants
# =========================================================================

OUTCOME_WINDOW = 480
RR_TARGET = 2.0
BREAK_WINDOW = 240
APERTURE_MIN = 5

SESSION_DST_TYPE = {
    "0900": "US", "1000": "CLEAN", "1100": "CLEAN",
    "1130": "CLEAN", "1800": "UK", "2300": "US", "0030": "US",
}

INSTRUMENTS = ["MGC", "MNQ", "MES"]
SESSIONS = ["0900", "1000", "1100", "1800", "2300"]
FILTERS = [
    ("G4+", 4.0, None),
    ("G5+", 5.0, None),
    ("G6+", 6.0, None),
]

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# =========================================================================
# FOMC / NFP / Opex calendar
# =========================================================================

def build_fomc_dates(start_year=2020, end_year=2026):
    """Known FOMC announcement dates (Wed at 2pm ET → Thu Brisbane).
    Trading day in Brisbane = the day the announcement impacts overnight session.
    FOMC Wed 2pm ET = Thu ~5am Brisbane. Affects Thu trading day."""
    # Major FOMC dates (announcement days, US calendar)
    fomc_us = [
        # 2020
        "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
        "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
        # 2021
        "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
        "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        # 2022
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
        "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
        # 2026
        "2026-01-28", "2026-03-18",
    ]
    dates = set()
    for d_str in fomc_us:
        d = date.fromisoformat(d_str)
        # FOMC day itself + next day (reaction continues)
        dates.add(d)
        dates.add(d + timedelta(days=1))
    return dates


def build_nfp_dates(start_year=2020, end_year=2026):
    """NFP = first Friday of each month. Impact on Brisbane trading day = that Friday."""
    dates = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = date(year, month, 1)
            # Find first Friday
            while d.weekday() != 4:  # Friday = 4
                d += timedelta(days=1)
            dates.add(d)
    return dates


def build_opex_dates(start_year=2020, end_year=2026):
    """Monthly options expiry = third Friday of each month."""
    dates = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = date(year, month, 1)
            friday_count = 0
            while friday_count < 3:
                if d.weekday() == 4:
                    friday_count += 1
                    if friday_count == 3:
                        break
                d += timedelta(days=1)
            dates.add(d)
    return dates


# =========================================================================
# Data loading (standalone engine)
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
# Q1: Day-of-week breakdown
# =========================================================================

def q1_day_of_week(data_cache):
    print(f"\n{'=' * 100}")
    print(f"  Q1: DAY-OF-WEEK BREAKDOWN")
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
                # Filter to size band
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

                # Group by day of week
                dow_groups = defaultdict(list)
                for di, r in filtered.items():
                    td = all_days[di]
                    dow = td.weekday()  # 0=Mon .. 4=Fri
                    dow_groups[dow].append(r["outcome_r"])

                # Print
                header_printed = False
                for dow in range(5):  # Mon-Fri only
                    rs = dow_groups.get(dow, [])
                    if not rs:
                        continue
                    n = len(rs)
                    avg = float(np.mean(rs))
                    wr = float(np.mean([1 if r > 0 else 0 for r in rs]))
                    tot = float(np.sum(rs))

                    if not header_printed:
                        print(f"\n  {instrument} {session} {fname} (N={len(filtered)}):")
                        print(f"    {'Day':>5s} {'N':>5s} {'avgR':>7s} {'WR':>6s} {'totR':>8s}")
                        print(f"    {'-' * 35}")
                        header_printed = True

                    marker = ""
                    if avg > 0.15:
                        marker = "  ++"
                    elif avg < -0.10:
                        marker = "  --"

                    print(f"    {DOW_NAMES[dow]:>5s} {n:5d} {avg:+7.3f} {wr:5.1%} {tot:+8.1f}{marker}")

                    all_rows.append({
                        "instrument": instrument, "session": session, "filter": fname,
                        "dow": dow, "dow_name": DOW_NAMES[dow],
                        "n": n, "avg_r": avg, "wr": wr, "total_r": tot,
                    })

    return all_rows


# =========================================================================
# Q2: Skip-day filter simulation
# =========================================================================

def q2_skip_filter(data_cache, q1_rows):
    print(f"\n{'=' * 100}")
    print(f"  Q2: SKIP-DAY FILTER SIMULATION")
    print(f"  Testing: skip worst 1 day, skip worst 2 days, keep best 1 day only")
    print(f"{'=' * 100}")

    all_rows = []

    # Group q1 data by (instrument, session, filter)
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

        # Sort by avgR to find best/worst days
        combo_rows.sort(key=lambda x: x["avg_r"])
        worst_1 = combo_rows[0]
        worst_2 = combo_rows[:2]
        best_1 = combo_rows[-1]

        # Baseline
        total_n = sum(r["n"] for r in combo_rows)
        total_r = sum(r["total_r"] for r in combo_rows)
        baseline_avg = total_r / total_n if total_n > 0 else 0

        # Skip worst 1
        skip1_rows = [r for r in combo_rows if r["dow"] != worst_1["dow"]]
        s1_n = sum(r["n"] for r in skip1_rows)
        s1_tot = sum(r["total_r"] for r in skip1_rows)
        s1_avg = s1_tot / s1_n if s1_n > 0 else 0

        # Skip worst 2
        skip2_dows = {r["dow"] for r in worst_2}
        skip2_rows = [r for r in combo_rows if r["dow"] not in skip2_dows]
        s2_n = sum(r["n"] for r in skip2_rows)
        s2_tot = sum(r["total_r"] for r in skip2_rows)
        s2_avg = s2_tot / s2_n if s2_n > 0 else 0

        # Best day only
        b1_n = best_1["n"]
        b1_tot = best_1["total_r"]
        b1_avg = best_1["avg_r"]

        # Avoided losses
        avoided_1 = total_r - s1_tot  # negative = losses avoided
        saved_1 = s1_tot - total_r  # positive = improvement

        print(f"\n  {instrument} {session} {fname}:")
        print(f"    Worst day: {DOW_NAMES[worst_1['dow']]} (avgR={worst_1['avg_r']:+.3f}, N={worst_1['n']})")
        print(f"    Best day:  {DOW_NAMES[best_1['dow']]} (avgR={best_1['avg_r']:+.3f}, N={best_1['n']})")
        print(f"    {'':>20s} {'N':>5s} {'avgR':>7s} {'totR':>8s} {'vs base':>8s}")
        print(f"    {'Baseline (all days)':>20s} {total_n:5d} {baseline_avg:+7.3f} {total_r:+8.1f} {'':>8s}")
        print(f"    {'Skip worst 1':>20s} {s1_n:5d} {s1_avg:+7.3f} {s1_tot:+8.1f} {s1_tot - total_r:+8.1f}")
        print(f"    {'Skip worst 2':>20s} {s2_n:5d} {s2_avg:+7.3f} {s2_tot:+8.1f} {s2_tot - total_r:+8.1f}")
        print(f"    {'Best day only':>20s} {b1_n:5d} {b1_avg:+7.3f} {b1_tot:+8.1f} {b1_tot - total_r:+8.1f}")

        all_rows.append({
            "instrument": instrument, "session": session, "filter": fname,
            "worst_dow": DOW_NAMES[worst_1["dow"]],
            "best_dow": DOW_NAMES[best_1["dow"]],
            "baseline_n": total_n, "baseline_avgR": baseline_avg, "baseline_totR": total_r,
            "skip1_n": s1_n, "skip1_avgR": s1_avg, "skip1_totR": s1_tot,
            "skip2_n": s2_n, "skip2_avgR": s2_avg, "skip2_totR": s2_tot,
            "best_only_n": b1_n, "best_only_avgR": b1_avg, "best_only_totR": b1_tot,
        })

    return all_rows


# =========================================================================
# Q3: FOMC / NFP / Opex overlay
# =========================================================================

def q3_macro_overlay(data_cache):
    print(f"\n{'=' * 100}")
    print(f"  Q3: FOMC / NFP / OPEX OVERLAY")
    print(f"{'=' * 100}")

    fomc_dates = build_fomc_dates()
    nfp_dates = build_nfp_dates()
    opex_dates = build_opex_dates()

    all_rows = []

    for instrument in INSTRUMENTS:
        if instrument not in data_cache:
            continue
        all_days, highs, lows, closes = data_cache[instrument]

        for session in SESSIONS:
            bh, bm = int(session[:2]), int(session[2:])
            per_day = scan_session(highs, lows, closes, bh, bm)

            for fname, flo, fhi in [("G4+", 4.0, None), ("G6+", 6.0, None)]:
                filtered = {}
                for di, r in per_day.items():
                    if r["orb_size"] < flo:
                        continue
                    if fhi is not None and r["orb_size"] >= fhi:
                        continue
                    filtered[di] = r

                if len(filtered) < 30:
                    continue

                # Split by event type
                for event_name, event_dates in [("FOMC", fomc_dates),
                                                 ("NFP", nfp_dates),
                                                 ("OPEX", opex_dates)]:
                    on_event = []
                    off_event = []
                    for di, r in filtered.items():
                        td = all_days[di]
                        if td in event_dates:
                            on_event.append(r["outcome_r"])
                        else:
                            off_event.append(r["outcome_r"])

                    if len(on_event) < 5:
                        continue

                    on_n = len(on_event)
                    on_avg = float(np.mean(on_event))
                    on_wr = float(np.mean([1 if r > 0 else 0 for r in on_event]))
                    off_n = len(off_event)
                    off_avg = float(np.mean(off_event))
                    off_wr = float(np.mean([1 if r > 0 else 0 for r in off_event]))
                    delta = on_avg - off_avg

                    marker = ""
                    if abs(delta) > 0.15:
                        marker = " <<" if delta < -0.15 else " >>"

                    print(f"  {instrument} {session} {fname} {event_name}: "
                          f"ON={on_n:3d} avgR={on_avg:+.3f} WR={on_wr:.0%} | "
                          f"OFF={off_n:3d} avgR={off_avg:+.3f} WR={off_wr:.0%} | "
                          f"delta={delta:+.3f}{marker}")

                    all_rows.append({
                        "instrument": instrument, "session": session, "filter": fname,
                        "event": event_name,
                        "on_n": on_n, "on_avgR": on_avg, "on_wr": on_wr,
                        "off_n": off_n, "off_avgR": off_avg, "off_wr": off_wr,
                        "delta": delta,
                    })

    return all_rows


# =========================================================================
# Consistency check: is the pattern stable across years?
# =========================================================================

def q4_yearly_stability(data_cache):
    print(f"\n{'=' * 100}")
    print(f"  Q4: YEARLY STABILITY — Does the day-of-week pattern hold per-year?")
    print(f"  (1000 G4+ and 0900 G4+ only)")
    print(f"{'=' * 100}")

    for instrument in INSTRUMENTS:
        if instrument not in data_cache:
            continue
        all_days, highs, lows, closes = data_cache[instrument]

        for session in ["0900", "1000"]:
            bh, bm = int(session[:2]), int(session[2:])
            per_day = scan_session(highs, lows, closes, bh, bm)
            filtered = {di: r for di, r in per_day.items() if r["orb_size"] >= 4.0}

            if len(filtered) < 30:
                continue

            # Group by (year, dow)
            year_dow = defaultdict(lambda: defaultdict(list))
            for di, r in filtered.items():
                td = all_days[di]
                year_dow[td.year][td.weekday()].append(r["outcome_r"])

            years = sorted(year_dow.keys())
            if len(years) < 2:
                continue

            print(f"\n  {instrument} {session} G4+ — avgR by year x day:")
            header = f"    {'Year':>6s}"
            for dow in range(5):
                header += f"  {DOW_NAMES[dow]:>7s}"
            header += f"  {'Spread':>7s}"
            print(header)
            print(f"    {'-' * (8 + 10 * 5 + 9)}")

            for year in years:
                row = f"    {year:>6d}"
                avgs = []
                for dow in range(5):
                    rs = year_dow[year].get(dow, [])
                    if rs:
                        avg = float(np.mean(rs))
                        n = len(rs)
                        avgs.append(avg)
                        row += f"  {avg:+6.3f}({n:2d})" if n < 100 else f"  {avg:+7.3f}"
                    else:
                        row += f"  {'--':>7s}"
                spread = max(avgs) - min(avgs) if len(avgs) >= 2 else 0
                row += f"  {spread:7.3f}"
                print(row)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="P2 Calendar Effects — day-of-week + macro overlay")
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
    print(f"  P2 CALENDAR EFFECTS — Day-of-Week + FOMC/NFP/Opex")
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

        q1_rows = q1_day_of_week(data_cache)
        q2_rows = q2_skip_filter(data_cache, q1_rows)
        q3_rows = q3_macro_overlay(data_cache)
        q4_yearly_stability(data_cache)

        # Save CSVs
        out = Path("research/output")
        out.mkdir(parents=True, exist_ok=True)

        if q1_rows:
            pd.DataFrame(q1_rows).to_csv(out / "day_of_week_breakdown.csv",
                                          index=False, float_format="%.4f")
        if q2_rows:
            pd.DataFrame(q2_rows).to_csv(out / "day_of_week_skip_filter.csv",
                                          index=False, float_format="%.4f")
        if q3_rows:
            pd.DataFrame(q3_rows).to_csv(out / "day_of_week_macro_overlay.csv",
                                          index=False, float_format="%.4f")

        print(f"\n  CSVs saved to research/output/day_of_week_*.csv")
        print(f"  Total: {time.time() - t_total:.1f}s")

    finally:
        con.close()


if __name__ == "__main__":
    main()

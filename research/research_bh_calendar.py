#!/usr/bin/env python3
"""
BH FDR correction on FOMC/NFP/OPEX calendar overlay.

Re-runs the Q3 macro overlay from research_day_of_week.py with proper
Welch two-sample t-tests, then applies Benjamini-Hochberg FDR correction
across all 80 tests to find which calendar signals genuinely survive.

Usage:
  python research/research_bh_calendar.py
  python research/research_bh_calendar.py --db-path C:/db/gold.db
"""

import argparse
import time
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_ind
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

# =========================================================================
# Constants (mirror research_day_of_week.py)
# =========================================================================

RR_TARGET    = 2.0
BREAK_WINDOW = 240
APERTURE_MIN = 5
OUTCOME_WINDOW = 480

INSTRUMENTS = ["MGC", "MNQ", "MES"]
SESSIONS    = ["0900", "1000", "1100", "1800", "2300"]
FILTERS     = [("G4+", 4.0, None), ("G6+", 6.0, None)]

_US_EASTERN = ZoneInfo("America/New_York")

# =========================================================================
# Calendar builders (copied verbatim from research_day_of_week.py)
# =========================================================================

def build_fomc_dates():
    fomc_us = [
        "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
        "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
        "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
        "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
        "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
        "2026-01-28", "2026-03-18",
    ]
    dates = set()
    for d_str in fomc_us:
        d = date.fromisoformat(d_str)
        dates.add(d)
        dates.add(d + timedelta(days=1))
    return dates


def build_nfp_dates():
    dates = set()
    for year in range(2020, 2027):
        for month in range(1, 13):
            d = date(year, month, 1)
            while d.weekday() != 4:
                d += timedelta(days=1)
            dates.add(d)
    return dates


def build_opex_dates():
    dates = set()
    for year in range(2020, 2027):
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
# Data loading (mirror research_day_of_week.py)
# =========================================================================

def load_bars(con, instrument):
    return con.execute(
        "SELECT ts_utc, open, high, low, close FROM bars_1m WHERE symbol = ? ORDER BY ts_utc",
        [instrument],
    ).fetchdf()


def build_day_arrays(bars_df):
    df = bars_df.copy()
    df["bris_dt"]    = df["ts_utc"] + pd.Timedelta(hours=10)
    df["bris_hour"]  = df["bris_dt"].dt.hour
    df["bris_minute"] = df["bris_dt"].dt.minute
    df["trading_day"] = df["bris_dt"].dt.normalize()
    mask = df["bris_hour"] < 9
    df.loc[mask, "trading_day"] -= pd.Timedelta(days=1)
    df["trading_day"] = df["trading_day"].dt.date
    df["min_offset"]  = ((df["bris_hour"] - 9) % 24) * 60 + df["bris_minute"]

    all_days   = sorted(df["trading_day"].unique())
    day_to_idx = {d: i for i, d in enumerate(all_days)}
    n_days     = len(all_days)

    highs  = np.full((n_days, 1440), np.nan)
    lows   = np.full((n_days, 1440), np.nan)
    closes = np.full((n_days, 1440), np.nan)

    day_idx = df["trading_day"].map(day_to_idx).values
    min_idx = df["min_offset"].values
    highs [day_idx, min_idx] = df["high"].values
    lows  [day_idx, min_idx] = df["low"].values
    closes[day_idx, min_idx] = df["close"].values

    return all_days, highs, lows, closes


def scan_session(highs, lows, closes, bris_h, bris_m):
    """Return per-day {di: {orb_size, outcome_r}} for all break-days."""
    n_days    = highs.shape[0]
    start_min = ((bris_h - 9) % 24) * 60 + bris_m
    orb_mins  = [start_min + i for i in range(APERTURE_MIN)]
    if orb_mins[-1] >= 1440:
        return {}

    orb_h = np.column_stack([highs[:, m] for m in orb_mins])
    orb_l = np.column_stack([lows[:, m]  for m in orb_mins])
    orb_c = np.column_stack([closes[:, m] for m in orb_mins])

    valid_orb = np.all(~np.isnan(orb_c), axis=1)
    orb_high  = np.nanmax(orb_h, axis=1)
    orb_low   = np.nanmin(orb_l, axis=1)
    orb_size  = orb_high - orb_low

    results = {}
    for di in range(n_days):
        if not valid_orb[di] or orb_size[di] < 0.01:
            continue
        oh, ol, os_val = orb_high[di], orb_low[di], orb_size[di]

        break_start = start_min + APERTURE_MIN
        max_bm      = min(break_start + BREAK_WINDOW, 1440)
        break_dir, entry, break_at = None, np.nan, -1
        for m in range(break_start, max_bm):
            c = closes[di, m]
            if np.isnan(c):
                continue
            if c > oh:
                break_dir, entry, break_at = "long",  c, m; break
            elif c < ol:
                break_dir, entry, break_at = "short", c, m; break
        if break_dir is None:
            continue

        target  = entry + (RR_TARGET if break_dir == "long" else -RR_TARGET) * os_val
        stop    = ol if break_dir == "long" else oh
        max_om  = min(break_at + 1 + OUTCOME_WINDOW, 1440)
        outcome_r, last_close = None, entry
        for m in range(break_at + 1, max_om):
            h, l, c = highs[di, m], lows[di, m], closes[di, m]
            if np.isnan(c):
                continue
            last_close = c
            if break_dir == "long":
                if l <= stop:  outcome_r = -1.0;       break
                if h >= target: outcome_r = RR_TARGET;  break
            else:
                if h >= stop:  outcome_r = -1.0;       break
                if l <= target: outcome_r = RR_TARGET;  break
        if outcome_r is None:
            outcome_r = ((last_close - entry) / os_val if break_dir == "long"
                         else (entry - last_close) / os_val)

        results[di] = {"orb_size": float(os_val), "outcome_r": float(outcome_r)}
    return results


# =========================================================================
# BH FDR correction
# =========================================================================

def bh_fdr_correction(p_values: list) -> list:
    """Benjamini-Hochberg FDR correction. NaN-safe."""
    p_arr    = np.array(p_values, dtype=float)
    n        = len(p_arr)
    adjusted = np.full(n, np.nan)

    valid_mask = ~np.isnan(p_arr)
    valid_idx  = np.where(valid_mask)[0]
    valid_p    = p_arr[valid_idx]
    if len(valid_p) == 0:
        return adjusted.tolist()

    m            = len(valid_p)
    sorted_order = np.argsort(valid_p)
    sorted_p     = valid_p[sorted_order]

    bh    = np.zeros(m)
    bh[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        bh[i] = min(bh[i + 1], sorted_p[i] * m / (i + 1))

    bh = np.clip(bh, 0.0, 1.0)
    unsorted = np.zeros(m)
    unsorted[sorted_order] = bh
    adjusted[valid_idx] = unsorted
    return adjusted.tolist()


# =========================================================================
# Main analysis
# =========================================================================

def run(db_path: str):
    t0 = time.time()

    if not HAS_SCIPY:
        print("ERROR: scipy not installed. Run: pip install scipy")
        return

    calendars = {
        "FOMC": build_fomc_dates(),
        "NFP":  build_nfp_dates(),
        "OPEX": build_opex_dates(),
    }

    print("=" * 90)
    print("  BH FDR CORRECTION — FOMC / NFP / OPEX CALENDAR OVERLAY")
    print(f"  DB: {db_path}  |  RR{RR_TARGET} | {BREAK_WINDOW}min break | {APERTURE_MIN}min aperture")
    print("=" * 90)

    con = duckdb.connect(db_path, read_only=True)

    # Load data and build arrays for each instrument
    data_cache = {}
    for inst in INSTRUMENTS:
        t1 = time.time()
        bars = load_bars(con, inst)
        if bars.empty:
            print(f"  {inst}: no bar data — skipping")
            continue
        all_days, highs, lows, closes = build_day_arrays(bars)
        data_cache[inst] = (all_days, highs, lows, closes)
        print(f"  {inst}: {len(all_days)} days loaded in {time.time()-t1:.1f}s")
    con.close()

    # Collect all tests
    rows = []
    for inst in INSTRUMENTS:
        if inst not in data_cache:
            continue
        all_days, highs, lows, closes = data_cache[inst]

        for session in SESSIONS:
            bh_hr, bh_mn = int(session[:2]), int(session[2:])
            per_day = scan_session(highs, lows, closes, bh_hr, bh_mn)

            for fname, flo, fhi in FILTERS:
                filtered = {
                    di: r for di, r in per_day.items()
                    if r["orb_size"] >= flo and (fhi is None or r["orb_size"] < fhi)
                }
                if len(filtered) < 30:
                    continue

                for event_name, event_dates in calendars.items():
                    on_vals  = [r["outcome_r"] for di, r in filtered.items()
                                if all_days[di] in event_dates]
                    off_vals = [r["outcome_r"] for di, r in filtered.items()
                                if all_days[di] not in event_dates]

                    if len(on_vals) < 5:
                        continue

                    on_arr  = np.array(on_vals)
                    off_arr = np.array(off_vals)

                    on_avg  = float(on_arr.mean())
                    off_avg = float(off_arr.mean())
                    delta   = on_avg - off_avg

                    # Welch two-sample t-test (unequal variance, unequal N)
                    _, p_val = ttest_ind(on_arr, off_arr, equal_var=False)

                    rows.append({
                        "instrument": inst,
                        "session":    session,
                        "filter":     fname,
                        "event":      event_name,
                        "on_n":       len(on_vals),
                        "off_n":      len(off_vals),
                        "on_avgR":    round(on_avg, 4),
                        "off_avgR":   round(off_avg, 4),
                        "delta":      round(delta, 4),
                        "p_raw":      round(float(p_val), 4),
                        "p_bh":       np.nan,   # filled below
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No tests generated — check data.")
        return

    # Apply BH correction across ALL tests together
    df["p_bh"] = bh_fdr_correction(df["p_raw"].tolist())
    df["p_bh"] = df["p_bh"].round(4)

    n_total   = len(df)
    n_raw05   = int((df["p_raw"] < 0.05).sum())
    n_bh05    = int((df["p_bh"] < 0.05).sum())
    n_bh10    = int((df["p_bh"] < 0.10).sum())

    print(f"\n  Tests run: {n_total}")
    print(f"  Raw p<0.05: {n_raw05}   BH-adjusted p<0.05: {n_bh05}   BH-adjusted p<0.10: {n_bh10}")

    # -----------------------------------------------------------------------
    # Report: ALL tests sorted by p_bh
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print(f"  ALL TESTS RANKED BY BH-ADJUSTED p-VALUE")
    print(f"{'=' * 90}")
    print(f"  {'Instrument':<5} {'Sess':<5} {'Flt':<4} {'Event':<5}  "
          f"{'N_on':>5} {'on_avgR':>8} {'N_off':>6} {'off_avgR':>9} "
          f"{'delta':>7} {'p_raw':>7} {'p_bh':>7}  sig")
    print(f"  {'-' * 84}")

    for _, r in df.sort_values("p_bh").iterrows():
        sig = ""
        if r["p_bh"] < 0.05:
            sig = "*** BH-SIG"
        elif r["p_bh"] < 0.10:
            sig = "  * BH-10%"
        elif r["p_raw"] < 0.05:
            sig = "    raw<.05"
        print(f"  {r['instrument']:<5} {r['session']:<5} {r['filter']:<4} {r['event']:<5}  "
              f"{r['on_n']:>5d} {r['on_avgR']:>+8.3f} {r['off_n']:>6d} {r['off_avgR']:>+9.3f} "
              f"{r['delta']:>+7.3f} {r['p_raw']:>7.4f} {r['p_bh']:>7.4f}  {sig}")

    # -----------------------------------------------------------------------
    # Summary: survivors only
    # -----------------------------------------------------------------------
    survivors = df[df["p_bh"] < 0.10].sort_values("p_bh")
    print(f"\n{'=' * 90}")
    print(f"  SURVIVORS (BH-adjusted p < 0.10)")
    print(f"{'=' * 90}")
    if survivors.empty:
        print("  None — no calendar signal survives BH correction at q=0.10.")
    else:
        for _, r in survivors.iterrows():
            direction = "BOOST" if r["delta"] > 0 else "DRAG"
            sample_grade = ("REGIME" if r["on_n"] < 30 else
                            "PRELIMINARY" if r["on_n"] < 100 else "CORE")
            print(f"\n  {r['instrument']} {r['session']} {r['filter']} — {r['event']} is a {direction}")
            print(f"    ON: avgR={r['on_avgR']:+.3f}  N={r['on_n']} | "
                  f"OFF: avgR={r['off_avgR']:+.3f}  N={r['off_n']}")
            print(f"    delta={r['delta']:+.3f}  p_raw={r['p_raw']:.4f}  p_bh={r['p_bh']:.4f}")
            print(f"    Grade: {sample_grade}")

    # Save CSV
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    csv_path = out / "bh_calendar_results.csv"
    df.sort_values("p_bh").to_csv(csv_path, index=False)
    print(f"\n  CSV: {csv_path}")
    print(f"  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BH FDR correction on calendar overlay")
    parser.add_argument("--db-path", default="gold.db")
    args = parser.parse_args()
    run(args.db_path)

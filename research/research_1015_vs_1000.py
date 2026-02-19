#!/usr/bin/env python3
"""
1015 vs 1000 — is the 15-minute delay real edge or noise?

Institutional lens: Tokyo opens at 9AM JST (= 1000 Brisbane). First 15 min
is opening auction chaos — orders filling, spreads wide, locals front-running.
By 1015, the real flow is visible. If 1015 ORB captures post-auction intent
while 1000 captures noise, the outperformance is structural.

Questions:
  Q1: Head-to-head comparison — 1015 vs 1000, all filters, DST split
  Q2: What happens in the 1000-1015 window on win vs loss days?
      (Does the first 15 min predict the trade outcome?)
  Q3: Overlap — are these the same trades or different?
  Q4: Does the pattern hold across ORB sizes? (band analysis)

Usage:
  python research/research_1015_vs_1000.py
  python research/research_1015_vs_1000.py --db-path C:/db/gold.db
"""

import argparse
import time
import warnings
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

# =========================================================================
# Timezone / Data helpers (standalone)
# =========================================================================

_US_EASTERN = ZoneInfo("America/New_York")

def is_us_dst(trading_day: date) -> bool:
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_US_EASTERN)
    return dt.utcoffset().total_seconds() == -4 * 3600

OUTCOME_WINDOW = 480
RR_TARGET = 2.0
BREAK_WINDOW = 240
APERTURE_MIN = 5

INSTRUMENTS = ["MNQ", "MES", "MGC"]
SESSIONS = ["1000", "1015"]
FILTERS = [
    ("G3+", 3.0, None),
    ("G4+", 4.0, None),
    ("G5+", 5.0, None),
    ("G6+", 6.0, None),
    ("G8+", 8.0, None),
    ("G4-G6", 4.0, 6.0),
    ("G6-G8", 6.0, 8.0),
    ("G8-G12", 8.0, 12.0),
    ("G12+", 12.0, None),
]


def load_bars(con, instrument):
    return con.execute("""
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = ?
        ORDER BY ts_utc
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

    opens = np.full((n_days, 1440), np.nan)
    highs = np.full((n_days, 1440), np.nan)
    lows = np.full((n_days, 1440), np.nan)
    closes = np.full((n_days, 1440), np.nan)
    volumes = np.full((n_days, 1440), np.nan)

    day_idx = df["trading_day"].map(day_to_idx).values
    min_idx = df["min_offset"].values

    opens[day_idx, min_idx] = df["open"].values
    highs[day_idx, min_idx] = df["high"].values
    lows[day_idx, min_idx] = df["low"].values
    closes[day_idx, min_idx] = df["close"].values
    volumes[day_idx, min_idx] = df["volume"].values

    return all_days, opens, highs, lows, closes, volumes


def scan_session(highs, lows, closes, bris_h, bris_m):
    """Return all break-day results (no size filter)."""
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
    for day_idx in range(n_days):
        if not valid_orb[day_idx]:
            continue
        oh = orb_high[day_idx]
        ol = orb_low[day_idx]
        os_val = orb_size[day_idx]
        if os_val < 0.01:
            continue

        break_start = start_min + APERTURE_MIN
        max_break_min = min(break_start + BREAK_WINDOW, 1440)

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
            results[day_idx] = {
                "orb_size": float(os_val), "broke": False,
                "direction": None, "outcome_r": np.nan,
                "entry": np.nan, "orb_high": float(oh), "orb_low": float(ol),
                "break_min": -1,
            }
            continue

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
                if l_val <= stop and h_val >= target:
                    outcome_r = -1.0; break
                if l_val <= stop:
                    outcome_r = -1.0; break
                if h_val >= target:
                    outcome_r = RR_TARGET; break
            else:
                if h_val >= stop and l_val <= target:
                    outcome_r = -1.0; break
                if h_val >= stop:
                    outcome_r = -1.0; break
                if l_val <= target:
                    outcome_r = RR_TARGET; break

        if outcome_r is None:
            if break_dir == "long":
                outcome_r = (last_close - entry) / os_val
            else:
                outcome_r = (entry - last_close) / os_val

        results[day_idx] = {
            "orb_size": float(os_val), "broke": True,
            "direction": break_dir, "outcome_r": float(outcome_r),
            "entry": float(entry), "orb_high": float(oh), "orb_low": float(ol),
            "break_min": break_at - start_min,
        }
    return results


def filter_breaks(results, lo, hi):
    """Filter break-days to a size band."""
    out = {}
    for d, r in results.items():
        if not r["broke"]:
            continue
        sz = r["orb_size"]
        if sz < lo:
            continue
        if hi is not None and sz >= hi:
            continue
        out[d] = r
    return out


def stats(filtered):
    if not filtered:
        return 0, np.nan, np.nan, np.nan
    rs = [r["outcome_r"] for r in filtered.values()]
    n = len(rs)
    avg = float(np.mean(rs))
    wr = float(np.mean([1 if r > 0 else 0 for r in rs]))
    tot = float(np.sum(rs))
    return n, avg, wr, tot


# =========================================================================
# Q1: Head-to-head comparison
# =========================================================================

def q1_head_to_head(data_cache):
    print(f"\n{'=' * 90}")
    print(f"  Q1: HEAD-TO-HEAD — 1000 vs 1015")
    print(f"{'=' * 90}")

    rows = []
    for instrument in INSTRUMENTS:
        if instrument not in data_cache:
            continue
        all_days, _o, highs, lows, closes, _v = data_cache[instrument]

        # Build DST mask (1000/1015 are CLEAN — Japan no DST — but scan anyway)
        n = len(all_days)

        s1000 = scan_session(highs, lows, closes, 10, 0)
        s1015 = scan_session(highs, lows, closes, 10, 15)

        print(f"\n  {instrument}  ({len(all_days)} days)")
        print(f"  {'Filter':>8s} | {'--- 1000 ---':>32s} | {'--- 1015 ---':>32s} | {'Delta':>7s}")
        print(f"  {'':>8s} | {'N':>5s} {'avgR':>7s} {'WR':>6s} {'totR':>7s} | "
              f"{'N':>5s} {'avgR':>7s} {'WR':>6s} {'totR':>7s} | {'avgR':>7s}")
        print(f"  {'-' * 85}")

        for fname, flo, fhi in FILTERS:
            f1000 = filter_breaks(s1000, flo, fhi)
            f1015 = filter_breaks(s1015, flo, fhi)

            n0, a0, w0, t0 = stats(f1000)
            n1, a1, w1, t1 = stats(f1015)

            delta = a1 - a0 if not (np.isnan(a0) or np.isnan(a1)) else np.nan

            def _f(v): return f"{v:+7.3f}" if not np.isnan(v) else "     --"
            def _w(v): return f"{v:5.1%}" if not np.isnan(v) else "   --"

            marker = " <<" if (not np.isnan(delta) and delta > 0.05) else ""
            print(f"  {fname:>8s} | {n0:5d} {_f(a0)} {_w(w0)} {_f(t0)} | "
                  f"{n1:5d} {_f(a1)} {_w(w1)} {_f(t1)} | {_f(delta)}{marker}")

            rows.append({
                "instrument": instrument, "filter": fname,
                "n_1000": n0, "avgR_1000": a0, "wr_1000": w0, "totR_1000": t0,
                "n_1015": n1, "avgR_1015": a1, "wr_1015": w1, "totR_1015": t1,
                "delta_avgR": delta,
            })

    return rows


# =========================================================================
# Q2: What happens in the 1000-1015 window?
# =========================================================================

def q2_opening_noise(data_cache):
    print(f"\n{'=' * 90}")
    print(f"  Q2: WHAT HAPPENS 1000-1015 ON WIN vs LOSS DAYS?")
    print(f"  (Using 1000 G4+ break-days)")
    print(f"{'=' * 90}")

    rows = []
    for instrument in INSTRUMENTS:
        if instrument not in data_cache:
            continue
        all_days, opens, highs, lows, closes, volumes = data_cache[instrument]

        s1000 = scan_session(highs, lows, closes, 10, 0)
        breaks_g4 = filter_breaks(s1000, 4.0, None)

        if not breaks_g4:
            print(f"\n  {instrument}: no G4+ breaks at 1000")
            continue

        # Minute offset for 1000 = (10-9)*60 = 60
        start_1000 = 60
        # The 15 bars from 1000 to 1014
        window_mins = list(range(start_1000, start_1000 + 15))

        wins = {d: r for d, r in breaks_g4.items() if r["outcome_r"] > 0}
        losses = {d: r for d, r in breaks_g4.items() if r["outcome_r"] <= 0}

        print(f"\n  {instrument}: {len(wins)} wins, {len(losses)} losses at 1000 G4+")

        # For each minute in the window, compute:
        # - avg volume
        # - avg |price move from 1000 open|
        # - avg range (high-low)
        # - fraction where price has already crossed ORB
        print(f"\n  {'Min':>5s} | {'--- WINS ---':>36s} | {'--- LOSSES ---':>36s}")
        print(f"  {'':>5s} | {'Vol':>6s} {'|Move|':>7s} {'Range':>6s} {'%Cross':>7s} | "
              f"{'Vol':>6s} {'|Move|':>7s} {'Range':>6s} {'%Cross':>7s}")
        print(f"  {'-' * 80}")

        for offset_from_1000 in range(15):
            m = start_1000 + offset_from_1000

            def bar_stats(day_set):
                vols, moves, ranges, crosses = [], [], [], []
                for d, r in day_set.items():
                    v = volumes[d, m]
                    h = highs[d, m]
                    l = lows[d, m]
                    c = closes[d, m]
                    ref_open = opens[d, start_1000]  # 1000 bar open

                    if np.isnan(c) or np.isnan(ref_open):
                        continue
                    vols.append(v if not np.isnan(v) else 0)
                    moves.append(abs(c - ref_open))
                    ranges.append(h - l if not np.isnan(h) else 0)
                    # Has price crossed the ORB boundary?
                    oh = r["orb_high"]
                    ol = r["orb_low"]
                    crossed = (h > oh or l < ol) if not np.isnan(h) else False
                    crosses.append(1 if crossed else 0)

                if not vols:
                    return np.nan, np.nan, np.nan, np.nan
                return (float(np.mean(vols)), float(np.mean(moves)),
                        float(np.mean(ranges)), float(np.mean(crosses)))

            wv, wm, wr_val, wc = bar_stats(wins)
            lv, lm, lr, lc = bar_stats(losses)

            time_label = f"10:{offset_from_1000:02d}"
            def _f2(v): return f"{v:7.2f}" if not np.isnan(v) else "     --"
            def _p(v): return f"{v:6.1%}" if not np.isnan(v) else "    --"

            print(f"  {time_label:>5s} | {_f2(wv)} {_f2(wm)} {_f2(wr_val)} {_p(wc)} | "
                  f"{_f2(lv)} {_f2(lm)} {_f2(lr)} {_p(lc)}")

            rows.append({
                "instrument": instrument, "minute": time_label,
                "win_vol": wv, "win_move": wm, "win_range": wr_val, "win_cross_pct": wc,
                "loss_vol": lv, "loss_move": lm, "loss_range": lr, "loss_cross_pct": lc,
            })

    return rows


# =========================================================================
# Q3: Overlap — same trades or different?
# =========================================================================

def q3_overlap(data_cache):
    print(f"\n{'=' * 90}")
    print(f"  Q3: OVERLAP — ARE 1000 AND 1015 THE SAME TRADES?")
    print(f"{'=' * 90}")

    for instrument in INSTRUMENTS:
        if instrument not in data_cache:
            continue
        all_days, _o, highs, lows, closes, _v = data_cache[instrument]

        s1000 = scan_session(highs, lows, closes, 10, 0)
        s1015 = scan_session(highs, lows, closes, 10, 15)

        for fname, flo, fhi in [("G4+", 4.0, None), ("G6+", 6.0, None), ("G8+", 8.0, None)]:
            f1000 = filter_breaks(s1000, flo, fhi)
            f1015 = filter_breaks(s1015, flo, fhi)

            d1000 = set(f1000.keys())
            d1015 = set(f1015.keys())

            both = d1000 & d1015
            only_1000 = d1000 - d1015
            only_1015 = d1015 - d1000

            n_both = len(both)
            n_1000 = len(d1000)
            n_1015 = len(d1015)

            shared_pct_of_1000 = n_both / n_1000 if n_1000 > 0 else 0
            shared_pct_of_1015 = n_both / n_1015 if n_1015 > 0 else 0

            # Direction agreement on shared days
            same_dir = 0
            diff_dir = 0
            r_vals_1000 = []
            r_vals_1015 = []
            for d in both:
                r0 = f1000[d]
                r1 = f1015[d]
                if r0["direction"] == r1["direction"]:
                    same_dir += 1
                else:
                    diff_dir += 1
                r_vals_1000.append(r0["outcome_r"])
                r_vals_1015.append(r1["outcome_r"])

            dir_agree = same_dir / n_both if n_both > 0 else 0

            # R-correlation on shared days
            r_corr = np.nan
            if n_both >= 10:
                a0 = np.array(r_vals_1000)
                a1 = np.array(r_vals_1015)
                if np.std(a0) > 0 and np.std(a1) > 0:
                    r_corr = float(np.corrcoef(a0, a1)[0, 1])

            # 1015-only days: what's their avgR?
            only_1015_rs = [f1015[d]["outcome_r"] for d in only_1015]
            only_1015_avgR = float(np.mean(only_1015_rs)) if only_1015_rs else np.nan

            rc_s = f"{r_corr:+.3f}" if not np.isnan(r_corr) else "  --"
            o15_s = f"{only_1015_avgR:+.3f}" if not np.isnan(only_1015_avgR) else "  --"

            print(f"\n  {instrument} {fname}:")
            print(f"    1000 breaks: {n_1000:4d}  |  1015 breaks: {n_1015:4d}")
            print(f"    Shared days: {n_both:4d} ({shared_pct_of_1000:.0%} of 1000, "
                  f"{shared_pct_of_1015:.0%} of 1015)")
            print(f"    Only-1000: {len(only_1000):4d}  |  Only-1015: {len(only_1015):4d}")
            print(f"    Direction agreement: {dir_agree:.0%}  |  R-correlation: {rc_s}")
            print(f"    1015-only avgR: {o15_s} (N={len(only_1015_rs)})")

            # On shared days: which session has better outcomes?
            if n_both >= 20:
                avg_1000_shared = float(np.mean(r_vals_1000))
                avg_1015_shared = float(np.mean(r_vals_1015))
                delta = avg_1015_shared - avg_1000_shared
                print(f"    Shared-day head-to-head: 1000={avg_1000_shared:+.3f} "
                      f"vs 1015={avg_1015_shared:+.3f} (delta={delta:+.3f})")


# =========================================================================
# Q4: LONG vs SHORT at 1015 (1000 is confirmed LONG-ONLY — is 1015 same?)
# =========================================================================

def q4_direction(data_cache):
    print(f"\n{'=' * 90}")
    print(f"  Q4: DIRECTION — LONG vs SHORT at 1015 (1000 = LONG-ONLY confirmed)")
    print(f"{'=' * 90}")

    for instrument in INSTRUMENTS:
        if instrument not in data_cache:
            continue
        all_days, _o, highs, lows, closes, _v = data_cache[instrument]

        s1015 = scan_session(highs, lows, closes, 10, 15)

        print(f"\n  {instrument}:")
        print(f"  {'Filter':>8s} | {'--- LONG ---':>22s} | {'--- SHORT ---':>22s} | {'BOTH':>7s}")
        print(f"  {'':>8s} | {'N':>5s} {'avgR':>7s} {'WR':>6s} | "
              f"{'N':>5s} {'avgR':>7s} {'WR':>6s} | {'avgR':>7s}")
        print(f"  {'-' * 70}")

        for fname, flo, fhi in [("G4+", 4.0, None), ("G5+", 5.0, None),
                                  ("G6+", 6.0, None), ("G8+", 8.0, None)]:
            fb = filter_breaks(s1015, flo, fhi)
            longs = {d: r for d, r in fb.items() if r["direction"] == "long"}
            shorts = {d: r for d, r in fb.items() if r["direction"] == "short"}

            nl, al, wl, _ = stats(longs)
            ns, a_s, ws, _ = stats(shorts)
            nb, ab, wb, _ = stats(fb)

            def _f(v): return f"{v:+7.3f}" if not np.isnan(v) else "     --"
            def _w(v): return f"{v:5.1%}" if not np.isnan(v) else "   --"

            print(f"  {fname:>8s} | {nl:5d} {_f(al)} {_w(wl)} | "
                  f"{ns:5d} {_f(a_s)} {_w(ws)} | {_f(ab)}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="1015 vs 1000 head-to-head analysis")
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

    print(f"\n{'=' * 90}")
    print(f"  1015 vs 1000 — IS THE 15-MINUTE DELAY REAL EDGE?")
    print(f"  Database: {db_path}")
    print(f"  RR{RR_TARGET} | {BREAK_WINDOW}min break window | {APERTURE_MIN}min aperture")
    print(f"{'=' * 90}")

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
            print(f"    {len(bars_df):,} bars in {time.time() - t:.1f}s")
            all_days, opens, highs, lows, closes, volumes = build_day_arrays(bars_df)
            del bars_df
            print(f"    {len(all_days)} trading days")
            data_cache[instrument] = (all_days, opens, highs, lows, closes, volumes)

        if not data_cache:
            print("  No data. Exiting.")
            return

        q1_rows = q1_head_to_head(data_cache)
        q2_rows = q2_opening_noise(data_cache)
        q3_overlap(data_cache)
        q4_direction(data_cache)

        # Save CSVs
        output_dir = Path("research/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        if q1_rows:
            pd.DataFrame(q1_rows).to_csv(
                output_dir / "1015_vs_1000_head_to_head.csv",
                index=False, float_format="%.4f")
            print(f"\n  Q1 CSV: research/output/1015_vs_1000_head_to_head.csv")

        if q2_rows:
            pd.DataFrame(q2_rows).to_csv(
                output_dir / "1015_vs_1000_opening_noise.csv",
                index=False, float_format="%.4f")
            print(f"  Q2 CSV: research/output/1015_vs_1000_opening_noise.csv")

        print(f"\n  Total: {time.time() - t_total:.1f}s")

    finally:
        con.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Transition ORB scan — test whether dead-zone / one-sided periods produce
better ORB follow-through than established session opens.

Hypothesis: During transition windows (one major market closing, another not
yet open), order flow is one-sided. An ORB formed during that period captures
directional conviction with no counter-pressure. The break should have
stronger follow-through than ORBs at contested session opens.

Method:
  1. Scan ORBs every 15 min across the full trading day (0900-0830 Bris)
  2. Compute avgR, WR, N, median MFE proxy at each time slot
  3. Overlay average volume at each slot
  4. Classify each slot by market dominance regime
  5. Identify dead-zone ORBs that outperform established sessions
  6. Test whether dead-zone ORBs are independent (shared-break analysis)

Engine: Copied from research_edge_structure.py (zero pipeline dependencies).
"""

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

# ── Timezone helpers ──────────────────────────────────────────────────────

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

# ── Constants ─────────────────────────────────────────────────────────────

G4_MIN = 4.0
OUTCOME_WINDOW = 480  # 8h in minutes
RR_TARGET = 2.0

# Market dominance by Brisbane hour
# Based on volume analysis: MNQ vol by Bris hour
DOMINANCE_BY_HOUR = {
    # Bris hour: (label, description)
    8:  ("DEAD-ZONE",    "US closed, Asia not open"),
    9:  ("HANDOFF",      "CME open, US→Asia transition"),
    10: ("ASIA-ONLY",    "Tokyo open, US/EU asleep"),
    11: ("ASIA-ONLY",    "Singapore/HK, US/EU asleep"),
    12: ("ASIA-ONLY",    "Asia mid-session"),
    13: ("ASIA-FADE",    "Asia winding down"),
    14: ("DEAD-ZONE",    "Asia closed, EU not open"),
    15: ("EU-EARLY",     "Europe pre-market"),
    16: ("EU-EARLY",     "Europe opening"),
    17: ("EU-EARLY",     "Europe morning"),
    18: ("EU-MAIN",      "London open"),
    19: ("EU-MAIN",      "Europe mid-session"),
    20: ("EU-MAIN",      "Europe afternoon"),
    21: ("EU-US-TRANS",  "Europe PM, US pre-market"),
    22: ("CONTESTED",    "US+Europe overlap"),
    23: ("CONTESTED",    "US+Europe overlap (peak)"),
    0:  ("CONTESTED",    "US equity hours (peak vol)"),
    1:  ("CONTESTED",    "US mid-session"),
    2:  ("US-FADE",      "US afternoon, EU closed"),
    3:  ("US-FADE",      "US late session"),
    4:  ("US-FADE",      "US closing"),
    5:  ("US-CLOSE",     "US post-close"),
    6:  ("DEAD-ZONE",    "US closed, quiet"),
    7:  ("DEAD-ZONE",    "US closed, Asia not open"),
}

# DST type by Brisbane hour (simplified: sessions before 15:00 = CLEAN or US, after = UK possible)
# For this scan we classify all as CLEAN unless they fall in known DST-affected windows
def get_dst_type(bris_h, bris_m):
    """Approximate DST classification for arbitrary Brisbane times."""
    t = bris_h * 100 + bris_m
    if t == 900:
        return "US"
    elif t == 1800:
        return "UK"
    elif t == 30:  # 0030
        return "US"
    elif t == 2300:
        return "US"
    # All Asia times are CLEAN (no DST anywhere)
    elif 1000 <= t <= 1330:
        return "CLEAN"
    # Pre-Europe could be UK DST affected
    elif 1500 <= t <= 1700:
        return "UK"
    # Most other times: report ALL (not splitting)
    return "ALL"

# ── Data loading ──────────────────────────────────────────────────────────

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

def build_dst_masks(trading_days):
    n = len(trading_days)
    us_mask = np.zeros(n, dtype=bool)
    uk_mask = np.zeros(n, dtype=bool)
    for i, td in enumerate(trading_days):
        us_mask[i] = is_us_dst(td)
        uk_mask[i] = is_uk_dst(td)
    return us_mask, uk_mask

# ── ORB scan engine ──────────────────────────────────────────────────────

def scan_session_per_day(highs, lows, closes, bris_h, bris_m,
                         aperture_min=5, break_window=240):
    """Scan one session across all days. Returns dict[day_idx -> result]."""
    n_days = highs.shape[0]
    start_min = ((bris_h - 9) % 24) * 60 + bris_m
    orb_mins = [start_min + i for i in range(aperture_min)]
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

        if os_val < 0.001:  # Zero-width ORB
            results[day_idx] = {
                "valid_orb": True, "g4_pass": False, "orb_size": float(os_val),
                "broke": False, "direction": None, "outcome_r": np.nan, "entry": np.nan,
            }
            continue

        break_start = start_min + aperture_min
        max_break_min = min(break_start + break_window, 1440)

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
                "valid_orb": True, "g4_pass": os_val >= G4_MIN,
                "orb_size": float(os_val),
                "broke": False, "direction": None, "outcome_r": np.nan, "entry": np.nan,
            }
            continue

        # Outcome resolution
        if break_dir == "long":
            target = entry + RR_TARGET * os_val
            stop = ol
        else:
            target = entry - RR_TARGET * os_val
            stop = oh

        max_outcome_min = min(break_at + 1 + OUTCOME_WINDOW, 1440)
        outcome_r = None
        last_close = entry
        max_fav = 0.0

        for m in range(break_at + 1, max_outcome_min):
            h_val = highs[day_idx, m]
            l_val = lows[day_idx, m]
            c_val = closes[day_idx, m]
            if np.isnan(c_val):
                continue
            last_close = c_val

            # Track MFE
            if break_dir == "long":
                fav = (h_val - entry) / os_val
            else:
                fav = (entry - l_val) / os_val
            if fav > max_fav:
                max_fav = fav

            if break_dir == "long":
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
            if break_dir == "long":
                outcome_r = (last_close - entry) / os_val
            else:
                outcome_r = (entry - last_close) / os_val

        results[day_idx] = {
            "valid_orb": True, "g4_pass": os_val >= G4_MIN,
            "orb_size": float(os_val),
            "broke": True, "direction": break_dir,
            "outcome_r": float(outcome_r), "entry": float(entry),
            "mfe_r": float(max_fav),
        }

    return results

# ── Volume profiling ─────────────────────────────────────────────────────

def compute_volume_profile(volumes):
    """Average volume per minute-offset across all days."""
    with np.errstate(all='ignore'):
        avg_vol = np.nanmean(volumes, axis=0)
    return avg_vol

# ── Main scan ────────────────────────────────────────────────────────────

def run_time_scan(instrument, all_days, highs, lows, closes, volumes,
                  us_mask, uk_mask):
    """Scan every 15 minutes and compute stats."""
    vol_profile = compute_volume_profile(volumes)
    n_days = len(all_days)

    rows = []

    # Scan every 15 minutes from 09:00 to 08:45 next day
    for slot in range(96):  # 96 x 15min = 24h
        bris_h = (9 + (slot * 15) // 60) % 24
        bris_m = (slot * 15) % 60
        label = f"{bris_h:02d}{bris_m:02d}"

        results = scan_session_per_day(highs, lows, closes, bris_h, bris_m)
        if not results:
            continue

        # Volume at this time slot (average over the 5 ORB minutes)
        start_min = ((bris_h - 9) % 24) * 60 + bris_m
        vol_slice = vol_profile[start_min:start_min + 5]
        avg_vol = float(np.nanmean(vol_slice)) if len(vol_slice) > 0 else 0.0

        # Dominance classification
        dom_label, dom_desc = DOMINANCE_BY_HOUR.get(bris_h, ("??", "??"))
        dst_type = get_dst_type(bris_h, bris_m)

        # Split by DST if needed, otherwise report ALL
        if dst_type == "CLEAN" or dst_type == "ALL":
            regimes = [("ALL", np.ones(n_days, dtype=bool))]
        elif dst_type == "US":
            regimes = [("WINTER", ~us_mask), ("SUMMER", us_mask)]
        elif dst_type == "UK":
            regimes = [("WINTER", ~uk_mask), ("SUMMER", uk_mask)]
        else:
            regimes = [("ALL", np.ones(n_days, dtype=bool))]

        for regime_name, regime_mask in regimes:
            # Filter to regime days that have results
            regime_days = {
                di: r for di, r in results.items()
                if regime_mask[di]
            }

            # All breaks (regardless of size filter)
            break_days = [r for r in regime_days.values()
                          if r.get("broke", False)]

            # G4+ breaks
            g4_breaks = [r for r in break_days if r.get("g4_pass", False)]

            # G8+ breaks
            g8_breaks = [r for r in break_days
                         if r.get("g4_pass", False) and r["orb_size"] >= 8.0]

            for filter_name, filtered in [("ALL_BREAKS", break_days),
                                          ("G4+", g4_breaks),
                                          ("G8+", g8_breaks)]:
                if len(filtered) < 5:
                    continue

                outcomes = [r["outcome_r"] for r in filtered
                            if not np.isnan(r["outcome_r"])]
                mfes = [r.get("mfe_r", 0) for r in filtered
                        if r.get("mfe_r") is not None]

                if len(outcomes) < 5:
                    continue

                avg_r = float(np.mean(outcomes))
                wr = float(np.mean([1 if o > 0 else 0 for o in outcomes]))
                tot_r = float(np.sum(outcomes))
                med_orb = float(np.median([r["orb_size"] for r in filtered]))
                med_mfe = float(np.median(mfes)) if mfes else 0.0

                rows.append({
                    "instrument": instrument,
                    "bris_time": label,
                    "bris_h": bris_h,
                    "bris_m": bris_m,
                    "dst_regime": regime_name,
                    "size_filter": filter_name,
                    "dominance": dom_label,
                    "dom_desc": dom_desc,
                    "avg_vol": round(avg_vol, 1),
                    "n_breaks": len(filtered),
                    "avg_r": round(avg_r, 4),
                    "wr": round(wr, 4),
                    "total_r": round(tot_r, 2),
                    "median_orb_size": round(med_orb, 2),
                    "median_mfe_r": round(med_mfe, 3),
                })

    return pd.DataFrame(rows)

# ── Shared-break analysis ────────────────────────────────────────────────

def shared_break_analysis(results_a, results_b, n_days, mask=None):
    """Compute % of shared break-days between two session scans."""
    if mask is None:
        mask = np.ones(n_days, dtype=bool)

    breaks_a = {di for di, r in results_a.items()
                if mask[di] and r.get("broke", False) and r.get("g4_pass", False)}
    breaks_b = {di for di, r in results_b.items()
                if mask[di] and r.get("broke", False) and r.get("g4_pass", False)}

    if not breaks_a or not breaks_b:
        return 0, 0, 0, 0.0

    shared = breaks_a & breaks_b
    pct = len(shared) / min(len(breaks_a), len(breaks_b)) if min(len(breaks_a), len(breaks_b)) > 0 else 0

    # R correlation on shared days
    if len(shared) > 5:
        r_a = [results_a[d]["outcome_r"] for d in shared
               if not np.isnan(results_a[d]["outcome_r"])]
        r_b = [results_b[d]["outcome_r"] for d in shared
               if not np.isnan(results_b[d]["outcome_r"])]
        if len(r_a) > 5 and len(r_a) == len(r_b):
            corr = float(np.corrcoef(r_a, r_b)[0, 1])
        else:
            corr = np.nan
    else:
        corr = np.nan

    return len(breaks_a), len(breaks_b), len(shared), pct, corr if not np.isnan(corr) else 0.0

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--instruments", default="MNQ,MES",
                        help="Comma-separated instruments")
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path)
    else:
        db_path = Path(__file__).resolve().parent.parent / "local_db" / "gold.db"

    con = duckdb.connect(str(db_path), read_only=True)
    instruments = [s.strip() for s in args.instruments.split(",")]

    t0 = time.time()
    print("=" * 80)
    print("  TRANSITION ORB SCAN")
    print(f"  Database: {db_path}")
    print(f"  Instruments: {', '.join(instruments)}")
    print(f"  Parameters: RR{RR_TARGET} | {240}min break window | 5min aperture")
    print(f"  Time slots: every 15min across 24h trading day")
    print("=" * 80)

    all_results = []
    data_cache = {}

    for instrument in instruments:
        print(f"\n  Loading {instrument}...")
        bars_df = load_bars(con, instrument)
        print(f"    {len(bars_df):,} bars loaded")

        all_days, opens, highs, lows, closes, volumes = build_day_arrays(bars_df)
        us_mask, uk_mask = build_dst_masks(all_days)
        print(f"    {len(all_days)} trading days")

        data_cache[instrument] = {
            "all_days": all_days, "highs": highs, "lows": lows,
            "closes": closes, "volumes": volumes,
            "us_mask": us_mask, "uk_mask": uk_mask,
        }

        df = run_time_scan(instrument, all_days, highs, lows, closes, volumes,
                           us_mask, uk_mask)
        all_results.append(df)

        # Print summary: best time slots by dominance category
        print(f"\n  {instrument} — Top ORB times by dominance (G4+ filter, avgR):")
        g4 = df[df["size_filter"] == "G4+"].copy()
        if len(g4) == 0:
            g4 = df[df["size_filter"] == "G8+"].copy()
        if len(g4) == 0:
            g4 = df[df["size_filter"] == "ALL_BREAKS"].copy()

        for dom in ["DEAD-ZONE", "ASIA-FADE", "ASIA-ONLY", "EU-EARLY",
                     "EU-MAIN", "EU-US-TRANS", "CONTESTED", "HANDOFF",
                     "US-FADE", "US-CLOSE"]:
            sub = g4[(g4["dominance"] == dom) & (g4["n_breaks"] >= 20)]
            if len(sub) == 0:
                continue
            best = sub.sort_values("avg_r", ascending=False).head(3)
            print(f"\n    [{dom}]")
            for _, r in best.iterrows():
                print(f"      {r['bris_time']} {r['dst_regime']:8s}  "
                      f"avgR={r['avg_r']:+.3f}  N={r['n_breaks']:4d}  "
                      f"vol={r['avg_vol']:7.0f}  "
                      f"medMFE={r['median_mfe_r']:.2f}R  "
                      f"medORB={r['median_orb_size']:.1f}")

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path(__file__).resolve().parent / "output" / "transition_orb_scan.csv"
    out_path.parent.mkdir(exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\n  CSV saved: {out_path} ({len(combined)} rows)")

    # ── Shared-break analysis for top dead-zone candidates ──
    print("\n" + "=" * 80)
    print("  SHARED-BREAK ANALYSIS: Dead-zone ORBs vs Established Sessions")
    print("=" * 80)

    # Established sessions to compare against
    established = {
        "MNQ": ["1000", "1100", "1245"],
        "MES": ["1000", "1100", "1245"],
    }

    for instrument in instruments:
        cache = data_cache[instrument]
        h, l, c = cache["highs"], cache["lows"], cache["closes"]
        n_days = len(cache["all_days"])

        # Find top dead-zone candidates from the scan
        g4 = combined[(combined["instrument"] == instrument) &
                       (combined["size_filter"].isin(["G4+", "G8+", "ALL_BREAKS"])) &
                       (combined["dominance"].isin(["DEAD-ZONE", "ASIA-FADE"])) &
                       (combined["n_breaks"] >= 20) &
                       (combined["avg_r"] > 0)]
        if len(g4) == 0:
            continue

        # Deduplicate by time (take best filter per time)
        g4_best = g4.sort_values("avg_r", ascending=False).drop_duplicates(
            subset=["bris_time", "dst_regime"], keep="first").head(5)

        print(f"\n  {instrument} — Top dead-zone candidates:")
        for _, cand in g4_best.iterrows():
            bh, bm = int(cand["bris_time"][:2]), int(cand["bris_time"][2:])
            cand_results = scan_session_per_day(h, l, c, bh, bm)

            print(f"\n    {cand['bris_time']} ({cand['dominance']}) "
                  f"avgR={cand['avg_r']:+.3f} N={cand['n_breaks']}")

            for est_label in established.get(instrument, []):
                eh, em = int(est_label[:2]), int(est_label[2:])
                est_results = scan_session_per_day(h, l, c, eh, em)

                result = shared_break_analysis(
                    cand_results, est_results, n_days)
                if len(result) == 5:
                    na, nb, shared, pct, corr = result
                    print(f"      vs {est_label}: shared={pct:.0%} "
                          f"({shared}/{min(na,nb)})  R-corr={corr:+.3f}")

    # ── Honest Summary ──
    print("\n" + "=" * 80)
    print("  HONEST SUMMARY")
    print("=" * 80)

    # For each instrument, compare dead-zone avgR vs established
    for instrument in instruments:
        print(f"\n  {instrument}:")
        inst_df = combined[combined["instrument"] == instrument]

        # Best established session (1000)
        est_1000 = inst_df[(inst_df["bris_time"] == "1000") &
                            (inst_df["size_filter"].isin(["G4+", "G8+"]))]
        if len(est_1000) > 0:
            best_est = est_1000.sort_values("avg_r", ascending=False).iloc[0]
            print(f"    Established benchmark: 1000 {best_est['size_filter']} "
                  f"avgR={best_est['avg_r']:+.3f} N={best_est['n_breaks']}")

        # Dead zone candidates that beat 1000
        benchmark = best_est["avg_r"] if len(est_1000) > 0 else 0
        dead = inst_df[(inst_df["dominance"].isin(["DEAD-ZONE", "ASIA-FADE"])) &
                        (inst_df["n_breaks"] >= 20) &
                        (inst_df["avg_r"] > benchmark)]
        if len(dead) > 0:
            print(f"    Dead-zone slots BEATING benchmark:")
            for _, r in dead.sort_values("avg_r", ascending=False).head(5).iterrows():
                print(f"      {r['bris_time']} {r['dst_regime']:8s} {r['size_filter']:12s} "
                      f"avgR={r['avg_r']:+.3f} N={r['n_breaks']:4d} vol={r['avg_vol']:.0f}")
        else:
            print(f"    No dead-zone slots beat the 1000 benchmark")

        # All time slots sorted by avgR (top 10)
        best_overall = inst_df[inst_df["n_breaks"] >= 30].sort_values(
            "avg_r", ascending=False).head(10)
        print(f"\n    Top 10 time slots overall (N>=30):")
        for _, r in best_overall.iterrows():
            marker = " ◄ DEAD-ZONE" if r["dominance"] in ["DEAD-ZONE", "ASIA-FADE"] else ""
            marker2 = " ◄ ESTABLISHED" if r["bris_time"] in ["0900","1000","1100","1130","1800","0030","2300"] else ""
            print(f"      {r['bris_time']} {r['dst_regime']:8s} {r['dominance']:14s} "
                  f"{r['size_filter']:12s} avgR={r['avg_r']:+.3f} N={r['n_breaks']:4d} "
                  f"vol={r['avg_vol']:7.0f}{marker}{marker2}")

    print(f"\n  CAVEATS:")
    print(f"    - IN-SAMPLE only (~500 MNQ/MES days, ~2900 MGC days)")
    print(f"    - 96 time slots x 2 instruments x filters = many tests (multiple comparison risk)")
    print(f"    - RR2.0, E1, CB1, 5min aperture only")
    print(f"    - DST-affected times split by regime; CLEAN times report ALL")
    print(f"    - Dead-zone ORBs may have wider spreads / worse fills in practice")
    print(f"    - Volume at dead-zone times is genuinely thin — slippage risk")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")

    con.close()

if __name__ == "__main__":
    main()

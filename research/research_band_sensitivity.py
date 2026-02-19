#!/usr/bin/env python3
"""
Band-filter sensitivity test — do Q2 band boundaries survive ±20% shifts?

The edge structure analysis (Q2) found that at the SAME G6-G8 size band,
different sessions produce different avgR (spread = 0.884). This suggests
band filters (G4-G6, G6-G8) may capture session-specific sweet spots that
floor-only filters (G4+, G6+) miss.

This script tests whether those band boundaries are real or curve-fitted
by shifting boundaries ±20% and checking survival.

Survival criteria (from RESEARCH_RULES.md):
  1. avgR stays positive across ALL 25 boundary shifts
  2. N stays above 20 in all shifts (REGIME minimum)
  3. avgR doesn't drop more than 50% from baseline at ±20% shift
  4. Floor-only comparison: band must beat uncapped floor by > 0.05 avgR

Usage:
  python research/research_band_sensitivity.py
  python research/research_band_sensitivity.py --db-path C:/db/gold.db
"""

import argparse
import csv
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
# Timezone helpers (standalone — no imports from pipeline)
# =========================================================================

_US_EASTERN = ZoneInfo("America/New_York")
_UK_LONDON = ZoneInfo("Europe/London")


def is_us_dst(trading_day: date) -> bool:
    """True if US Eastern is in DST (EDT, UTC-4) on this date."""
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_US_EASTERN)
    return dt.utcoffset().total_seconds() == -4 * 3600


def is_uk_dst(trading_day: date) -> bool:
    """True if UK is in BST (UTC+1) on this date."""
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=_UK_LONDON)
    return dt.utcoffset().total_seconds() == 1 * 3600


# =========================================================================
# Constants
# =========================================================================

OUTCOME_WINDOW = 480  # 8 hours in minutes
RR_TARGET = 2.0
BREAK_WINDOW = 240
APERTURE_MIN = 5

# DST type for each session label
SESSION_DST_TYPE = {
    "0900": "US", "0930": "US", "1000": "CLEAN", "1015": "CLEAN",
    "1100": "CLEAN", "1130": "CLEAN", "1245": "CLEAN",
    "1545": "UK", "1615": "UK", "1645": "UK",
    "1800": "UK", "1815": "UK", "1900": "CLEAN", "1915": "CLEAN",
    "2300": "US", "0030": "US",
}

# Band candidates from Q2 data
BAND_CANDIDATES = [
    # (instrument, session, baseline_lo, baseline_hi, description)
    ("MES", "1000", 4.0, 6.0, "MES 1000 G4-G6"),
    ("MES", "1000", 6.0, 8.0, "MES 1000 G6-G8"),
    ("MES", "1245", 6.0, 8.0, "MES 1245 G6-G8"),
    ("MGC", "1900", 6.0, 8.0, "MGC 1900 G6-G8"),
    ("MGC", "1100", 6.0, 8.0, "MGC 1100 G6-G8"),
    ("MNQ", "2300", 6.0, 8.0, "MNQ 2300 G6-G8"),
]

SHIFT_PCTS = [-0.20, -0.10, 0.0, 0.10, 0.20]


# =========================================================================
# Data Loading (copied from research_edge_structure.py — standalone)
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
    """Convert bars DataFrame into 2D numpy arrays indexed by (day, minute_offset)."""
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

    day_idx = df["trading_day"].map(day_to_idx).values
    min_idx = df["min_offset"].values

    opens[day_idx, min_idx] = df["open"].values
    highs[day_idx, min_idx] = df["high"].values
    lows[day_idx, min_idx] = df["low"].values
    closes[day_idx, min_idx] = df["close"].values

    return all_days, opens, highs, lows, closes


def build_dst_masks(trading_days):
    """Build US and UK DST boolean masks. True = summer/DST."""
    n = len(trading_days)
    us_mask = np.zeros(n, dtype=bool)
    uk_mask = np.zeros(n, dtype=bool)
    for i, td in enumerate(trading_days):
        us_mask[i] = is_us_dst(td)
        uk_mask[i] = is_uk_dst(td)
    return us_mask, uk_mask


def parse_session_label(label):
    """Parse session label like '0930' into (bris_h, bris_m)."""
    h = int(label[:2])
    m = int(label[2:])
    return h, m


# =========================================================================
# Per-Day Scan Engine (no G4 floor — returns ALL valid ORBs)
# =========================================================================

def scan_session_all_orbs(highs, lows, closes, bris_h, bris_m):
    """Scan one session across all trading days, returning ALL valid ORBs.

    Unlike research_edge_structure.py which filters at G4, this returns
    every day with a valid ORB so the caller can apply any band filter.

    Returns list of dicts with keys: day_idx, orb_size, broke, direction, outcome_r.
    """
    n_days = highs.shape[0]
    start_min = ((bris_h - 9) % 24) * 60 + bris_m

    orb_mins = [start_min + i for i in range(APERTURE_MIN)]
    if orb_mins[-1] >= 1440:
        return []

    orb_h = np.column_stack([highs[:, m] for m in orb_mins])
    orb_l = np.column_stack([lows[:, m] for m in orb_mins])
    orb_c = np.column_stack([closes[:, m] for m in orb_mins])

    valid_orb = np.all(~np.isnan(orb_c), axis=1)
    orb_high = np.nanmax(orb_h, axis=1)
    orb_low = np.nanmin(orb_l, axis=1)
    orb_size = orb_high - orb_low

    results = []

    for day_idx in range(n_days):
        if not valid_orb[day_idx]:
            continue

        oh = orb_high[day_idx]
        ol = orb_low[day_idx]
        os_val = orb_size[day_idx]

        if os_val < 0.01:  # degenerate ORB
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
            continue  # no break — skip (not a trade day)

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

        for m in range(break_at + 1, max_outcome_min):
            h_val = highs[day_idx, m]
            l_val = lows[day_idx, m]
            c_val = closes[day_idx, m]

            if np.isnan(c_val):
                continue
            last_close = c_val

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

        results.append({
            "day_idx": day_idx,
            "orb_size": float(os_val),
            "broke": True,
            "direction": break_dir,
            "outcome_r": float(outcome_r),
        })

    return results


# =========================================================================
# Band & Floor Statistics
# =========================================================================

def compute_band_stats(break_days, lo, hi):
    """Compute stats for break-days within a size band [lo, hi)."""
    filtered = [d for d in break_days
                if d["orb_size"] >= lo and (d["orb_size"] < hi if hi is not None else True)]
    n = len(filtered)
    if n == 0:
        return {"n": 0, "avg_r": np.nan, "wr": np.nan, "total_r": np.nan}
    rs = [d["outcome_r"] for d in filtered]
    avg_r = float(np.mean(rs))
    wr = float(np.mean([1 if r > 0 else 0 for r in rs]))
    total_r = float(np.sum(rs))
    return {"n": n, "avg_r": avg_r, "wr": wr, "total_r": total_r}


# =========================================================================
# Sensitivity Test
# =========================================================================

def run_sensitivity(data_cache):
    """Run the band sensitivity test for all candidates."""
    all_rows = []

    for instrument, session, base_lo, base_hi, desc in BAND_CANDIDATES:
        if instrument not in data_cache:
            print(f"\n  SKIP {desc} — no data for {instrument}")
            continue

        all_days, _opens, highs, lows, closes, us_mask, uk_mask = data_cache[instrument]
        bh, bm = parse_session_label(session)

        # Scan session — get ALL break-days (no size filter)
        break_days = scan_session_all_orbs(highs, lows, closes, bh, bm)

        dst_type = SESSION_DST_TYPE.get(session, "CLEAN")

        if dst_type == "CLEAN":
            regimes = [("ALL", None)]
        elif dst_type == "US":
            regimes = [("WINTER", False), ("SUMMER", True)]
        else:
            regimes = [("WINTER", False), ("SUMMER", True)]

        print(f"\n{'=' * 80}")
        print(f"  {desc}")
        print(f"  Baseline: [{base_lo:.1f}, {base_hi:.1f})  |  Total break-days: {len(break_days)}")
        print(f"{'=' * 80}")

        for regime_name, regime_test in regimes:
            # Filter to regime
            if regime_test is None:
                regime_days = break_days
            else:
                mask = us_mask if dst_type == "US" else uk_mask
                regime_days = [d for d in break_days if mask[d["day_idx"]] == regime_test]

            if not regime_days:
                print(f"\n  [{regime_name}] No break-days, skipping.")
                continue

            print(f"\n  [{regime_name}] Break-days: {len(regime_days)}")

            # ---- Band tests: 5x5 grid ----
            candidate_rows = []

            print(f"\n  Band sensitivity grid (avgR):")
            lo_hi_label = "lo\\hi"
            header = f"  {lo_hi_label:>10s}"
            for hi_s in SHIFT_PCTS:
                adj_hi = base_hi * (1 + hi_s)
                header += f"  hi={adj_hi:5.2f}"
            print(header)
            print(f"  {'-' * (12 + 10 * len(SHIFT_PCTS))}")

            for lo_s in SHIFT_PCTS:
                adj_lo = base_lo * (1 + lo_s)
                row_str = f"  lo={adj_lo:5.2f}  "

                for hi_s in SHIFT_PCTS:
                    adj_hi = base_hi * (1 + hi_s)

                    if adj_lo >= adj_hi:
                        row_str += f"  {'N/A':>7s}"
                        continue

                    stats = compute_band_stats(regime_days, adj_lo, adj_hi)

                    marker = ""
                    if lo_s == 0.0 and hi_s == 0.0:
                        marker = "*"  # baseline

                    if stats["n"] == 0:
                        row_str += f"  {'--':>7s}"
                    else:
                        row_str += f"  {stats['avg_r']:+6.3f}{marker}"

                    candidate_rows.append({
                        "instrument": instrument,
                        "session": session,
                        "dst_regime": regime_name,
                        "baseline_lo": base_lo,
                        "baseline_hi": base_hi,
                        "lo_shift_pct": lo_s,
                        "hi_shift_pct": hi_s,
                        "actual_lo": adj_lo,
                        "actual_hi": adj_hi,
                        "n_breaks": stats["n"],
                        "avg_r": stats["avg_r"],
                        "wr": stats["wr"],
                        "total_r": stats["total_r"],
                        "test_type": "band",
                    })

                print(row_str)

            # ---- Floor tests: 3 levels ----
            print(f"\n  Floor comparison (no ceiling):")
            for floor_mult in [0.80, 1.0, 1.20]:
                floor_val = base_lo * floor_mult
                stats = compute_band_stats(regime_days, floor_val, None)

                marker = " (baseline floor)" if floor_mult == 1.0 else ""
                if stats["n"] == 0:
                    print(f"    floor>={floor_val:5.2f}: N=0{marker}")
                else:
                    print(f"    floor>={floor_val:5.2f}: N={stats['n']:4d}, "
                          f"avgR={stats['avg_r']:+.3f}, WR={stats['wr']:.1%}, "
                          f"totR={stats['total_r']:+.1f}{marker}")

                candidate_rows.append({
                    "instrument": instrument,
                    "session": session,
                    "dst_regime": regime_name,
                    "baseline_lo": base_lo,
                    "baseline_hi": base_hi,
                    "lo_shift_pct": floor_mult - 1.0,
                    "hi_shift_pct": np.nan,
                    "actual_lo": floor_val,
                    "actual_hi": np.nan,
                    "n_breaks": stats["n"],
                    "avg_r": stats["avg_r"],
                    "wr": stats["wr"],
                    "total_r": stats["total_r"],
                    "test_type": "floor",
                })

            all_rows.extend(candidate_rows)

    return all_rows


# =========================================================================
# Honest Summary
# =========================================================================

def print_honest_summary(all_rows):
    """Print survival verdicts per candidate."""
    print(f"\n{'=' * 80}")
    print(f"  HONEST SUMMARY")
    print(f"{'=' * 80}")

    survived = []
    did_not_survive = []

    # Group by (instrument, session, dst_regime, baseline_lo, baseline_hi)
    candidates = set()
    for r in all_rows:
        candidates.add((r["instrument"], r["session"], r["dst_regime"],
                         r["baseline_lo"], r["baseline_hi"]))

    for inst, sess, regime, blo, bhi in sorted(candidates):
        cand_rows = [r for r in all_rows
                     if r["instrument"] == inst and r["session"] == sess
                     and r["dst_regime"] == regime
                     and r["baseline_lo"] == blo and r["baseline_hi"] == bhi]

        band_rows = [r for r in cand_rows if r["test_type"] == "band"]
        floor_rows = [r for r in cand_rows if r["test_type"] == "floor"]

        label = f"{inst} {sess} [{regime}] G{blo:.0f}-G{bhi:.0f}"

        # Get baseline band stats
        baseline_band = [r for r in band_rows
                         if r["lo_shift_pct"] == 0.0 and r["hi_shift_pct"] == 0.0]
        baseline_floor = [r for r in floor_rows
                          if abs(r["lo_shift_pct"]) < 0.01]

        if not baseline_band:
            did_not_survive.append((label, "No baseline data"))
            continue

        bl = baseline_band[0]
        bl_avgr = bl["avg_r"]
        bl_n = bl["n_breaks"]

        # Criterion 1: avgR positive at ALL band shifts
        valid_band = [r for r in band_rows
                      if r["actual_lo"] < r["actual_hi"]
                      and r["n_breaks"] > 0]
        any_negative = any(r["avg_r"] < 0 for r in valid_band if not np.isnan(r["avg_r"]))

        # Criterion 2: N >= 20 at all shifts
        any_low_n = any(r["n_breaks"] < 20 for r in valid_band)

        # Criterion 3: avgR doesn't drop >50% at ±20% shifts
        extreme_shifts = [r for r in valid_band
                          if (abs(r["lo_shift_pct"]) >= 0.19
                              or abs(r["hi_shift_pct"]) >= 0.19)
                          and r["n_breaks"] > 0
                          and not np.isnan(r["avg_r"])]
        big_drop = False
        if bl_avgr > 0:
            big_drop = any(r["avg_r"] < bl_avgr * 0.5 for r in extreme_shifts)

        # Criterion 4: band beats floor by > 0.05 avgR
        floor_beat = False
        if baseline_floor and not np.isnan(bl_avgr):
            fl = baseline_floor[0]
            if not np.isnan(fl["avg_r"]):
                floor_beat = bl_avgr > fl["avg_r"] + 0.05

        # Verdict
        reasons = []
        if any_negative:
            reasons.append("avgR negative at some shift")
        if any_low_n:
            reasons.append("N<20 at some shift")
        if big_drop:
            reasons.append("avgR drops >50% at ±20%")
        if not floor_beat:
            reasons.append("does not beat floor by >0.05")

        if not reasons:
            survived.append((label, bl_avgr, bl_n))
        else:
            did_not_survive.append((label, "; ".join(reasons)))

    print(f"\n  SURVIVED (avgR positive at all ±20% shifts, N>20, beats floor):")
    if survived:
        for label, avgr, n in survived:
            print(f"    {label}: baseline avgR={avgr:+.3f}, N={n}")
    else:
        print(f"    (none)")

    print(f"\n  DID NOT SURVIVE:")
    if did_not_survive:
        for label, reason in did_not_survive:
            print(f"    {label}: {reason}")
    else:
        print(f"    (none — all survived)")

    print(f"\n  CAVEATS:")
    print(f"    - IN-SAMPLE only (no walk-forward)")
    print(f"    - {len(BAND_CANDIDATES)} candidates x 28 tests = {len(BAND_CANDIDATES) * 28} tests"
          f" -> Benjamini-Hochberg FDR required for discovery claims")
    print(f"    - Band boundaries are fitted to Q2 data from the same sample")
    print(f"    - Selection bias: these 6 candidates were chosen because they looked good in Q2")
    print(f"    - ~500 MNQ/MES days — PRELIMINARY sample size for per-band tests")
    print(f"    - RR2.0, E1, CB1, 5min aperture only")

    print(f"\n  NEXT STEPS:")
    print(f"    - Survivors: add as candidate filters in strategy_discovery grid")
    print(f"    - Priority: 10-year rebuild to test bands with proper IS/OOS split")
    print()


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Band-filter sensitivity test — do Q2 band boundaries survive ±20% shifts?"
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to gold.db (default: auto-resolve via pipeline.paths)"
    )
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path)
    else:
        try:
            from pipeline.paths import GOLD_DB_PATH
            db_path = GOLD_DB_PATH
        except ImportError:
            db_path = Path("gold.db")

    print(f"\n{'=' * 80}")
    print(f"  BAND-FILTER SENSITIVITY TEST")
    print(f"  Database: {db_path}")
    print(f"  Parameters: RR{RR_TARGET:.1f} | {BREAK_WINDOW}min break window | "
          f"{APERTURE_MIN}min aperture | {OUTCOME_WINDOW // 60}h outcome")
    print(f"  Candidates: {len(BAND_CANDIDATES)}")
    print(f"  Tests per candidate: 25 band + 3 floor = 28")
    print(f"{'=' * 80}")

    con = duckdb.connect(str(db_path), read_only=True)
    t_total = time.time()

    try:
        # Load instruments needed
        needed = sorted(set(c[0] for c in BAND_CANDIDATES))
        data_cache = {}

        for instrument in needed:
            print(f"\n  Loading {instrument}...")
            t_load = time.time()
            bars_df = load_bars(con, instrument)
            if len(bars_df) == 0:
                print(f"    No data for {instrument}, skipping.")
                continue
            print(f"    {len(bars_df):,} bars loaded in {time.time() - t_load:.1f}s")

            t_build = time.time()
            all_days, opens, highs, lows, closes = build_day_arrays(bars_df)
            del bars_df
            us_mask, uk_mask = build_dst_masks(all_days)
            print(f"    {len(all_days)} trading days built in {time.time() - t_build:.1f}s")

            data_cache[instrument] = (all_days, opens, highs, lows, closes,
                                      us_mask, uk_mask)

        if not data_cache:
            print("\n  No instrument data loaded. Exiting.")
            return

        # Run sensitivity
        all_rows = run_sensitivity(data_cache)

        # Save CSV
        output_dir = Path("research/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        if all_rows:
            df = pd.DataFrame(all_rows)
            csv_path = output_dir / "band_sensitivity_results.csv"
            df.to_csv(csv_path, index=False, float_format="%.4f")
            print(f"\n  CSV saved: {csv_path} ({len(df)} rows)")

            # Count by type
            n_band = len([r for r in all_rows if r["test_type"] == "band"])
            n_floor = len([r for r in all_rows if r["test_type"] == "floor"])
            print(f"  Band tests: {n_band}, Floor tests: {n_floor}")

        # Honest summary
        print_honest_summary(all_rows)

        print(f"  Total runtime: {time.time() - t_total:.1f}s")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()

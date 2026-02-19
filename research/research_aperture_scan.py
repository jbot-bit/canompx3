#!/usr/bin/env python3
"""
Multi-Aperture ORB Scan — does a wider ORB window improve edge quality?

Tests 7 aperture widths (5/10/15/20/30/45/60 min) across 4 instruments and
4 sessions. Reuses the vectorized numpy engine from research_orb_time_scan.py.

Key question: The 4 fixed sessions (0900/1800/2300/0030) drift ±1hr vs their
market events during DST. A wider aperture might capture the actual event even
when the fixed clock time drifts. This is a screening pass — if something
survives, THEN rebuild the pipeline.

Anti-bias measures:
  1. Report ALL 112 combos (no cherry-picking)
  2. Benjamini-Hochberg FDR correction across all p-values
  3. Sensitivity: ±2min perturbation on positive findings → ROBUST/FRAGILE
  4. DST splits: US DST for 0900/0030/2300, UK DST for 1800
  5. Every aperture vs 5m baseline (delta_avgR)
  6. Sample classification per RESEARCH_RULES.md

Method:
  1. Load all 1m bars into numpy (day × 1440min) arrays (once per instrument)
  2. For each (session, aperture): compute ORB, detect break, resolve outcome
  3. Aggregate with DST split, sensitivity check, FDR correction
  4. Output CSV + honest summary

Usage:
  python research/research_aperture_scan.py
  python research/research_aperture_scan.py --db-path C:/db/gold.db
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

try:
    from scipy.stats import ttest_1samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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

INSTRUMENTS = ["MGC", "MNQ", "MES", "MCL"]

# Sessions to test (Brisbane clock times)
# 0900/0030/2300 use US DST; 1800 uses UK DST
SESSIONS = {
    "0900": {"bris_h": 9, "bris_m": 0, "dst_type": "US"},
    "1000": {"bris_h": 10, "bris_m": 0, "dst_type": "CLEAN"},
    "1100": {"bris_h": 11, "bris_m": 0, "dst_type": "CLEAN"},
    "1800": {"bris_h": 18, "bris_m": 0, "dst_type": "UK"},
    "2300": {"bris_h": 23, "bris_m": 0, "dst_type": "US"},
    "0030": {"bris_h": 0, "bris_m": 30, "dst_type": "US"},
}

APERTURES = [5, 10, 15, 20, 30, 45, 60]

# Fixed parameters (match production: E1 entry, CB1, RR2.0, G4+)
G4_MIN = 4.0
BREAK_WINDOW = 240    # 4 hours in minutes
OUTCOME_WINDOW = 480  # 8 hours in minutes
RR_TARGET = 2.0


# =========================================================================
# Data Loading (copied from research_orb_time_scan.py — standalone)
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


def build_dst_masks(trading_days):
    """Build US and UK DST boolean masks. True = summer/DST."""
    n = len(trading_days)
    us_mask = np.zeros(n, dtype=bool)
    uk_mask = np.zeros(n, dtype=bool)
    for i, td in enumerate(trading_days):
        us_mask[i] = is_us_dst(td)
        uk_mask[i] = is_uk_dst(td)
    return us_mask, uk_mask


# =========================================================================
# Core Scan Engine
# =========================================================================

def scan_session_aperture(highs, lows, closes, bris_h, bris_m,
                          aperture_min, dst_mask):
    """Scan one session at one aperture across all trading days.

    Returns dict with aggregated metrics + raw trade lists for p-value.
    """
    start_min = ((bris_h - 9) % 24) * 60 + bris_m

    # ORB minute columns — first `aperture_min` bars from session start
    orb_mins = [start_min + i for i in range(aperture_min)]
    if orb_mins[-1] >= 1440:
        return None  # Would wrap past trading day boundary

    # Vectorized ORB computation
    orb_h = np.column_stack([highs[:, m] for m in orb_mins])
    orb_l = np.column_stack([lows[:, m] for m in orb_mins])
    orb_c = np.column_stack([closes[:, m] for m in orb_mins])

    # Valid ORB = ALL bars present (stricter for wider apertures — correct)
    valid_orb = np.all(~np.isnan(orb_c), axis=1)
    orb_high = np.nanmax(orb_h, axis=1)
    orb_low = np.nanmin(orb_l, axis=1)
    orb_size = orb_high - orb_low

    # G4+ filter
    g4_mask = valid_orb & (orb_size >= G4_MIN)
    g4_indices = np.where(g4_mask)[0]

    n_days_valid = int(valid_orb.sum())
    n_g4 = len(g4_indices)

    if n_g4 == 0:
        return _empty_result(n_days_valid)

    avg_orb_size = float(orb_size[g4_mask].mean())

    # Break detection starts AFTER ORB ends
    break_start = start_min + aperture_min
    max_break_min = min(break_start + BREAK_WINDOW, 1440)

    trades_winter = []
    trades_summer = []
    n_breaks = 0

    for day_idx in g4_indices:
        oh = orb_high[day_idx]
        ol = orb_low[day_idx]
        os_val = orb_size[day_idx]

        # --- Break detection (first 1m close outside ORB) ---
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

    trades_all = trades_winter + trades_summer
    return {
        "n_days": n_days_valid,
        "n_g4": n_g4,
        "n_breaks": n_breaks,
        "n_trades": len(trades_all),
        "avg_r": _safe_mean(trades_all),
        "total_r": sum(trades_all) if trades_all else 0.0,
        "win_rate": _safe_winrate(trades_all),
        "avg_orb_size": avg_orb_size,
        "n_winter": len(trades_winter),
        "avg_r_winter": _safe_mean(trades_winter),
        "wr_winter": _safe_winrate(trades_winter),
        "n_summer": len(trades_summer),
        "avg_r_summer": _safe_mean(trades_summer),
        "wr_summer": _safe_winrate(trades_summer),
        "trades_raw": trades_all,  # For p-value computation
    }


def _safe_mean(lst):
    return float(np.mean(lst)) if lst else np.nan


def _safe_winrate(lst):
    if not lst:
        return np.nan
    arr = np.array(lst)
    return float((arr >= RR_TARGET - 0.01).sum() / len(arr))


def _empty_result(n_days_valid):
    return {
        "n_days": n_days_valid,
        "n_g4": 0,
        "n_breaks": 0,
        "n_trades": 0,
        "avg_r": np.nan,
        "total_r": 0.0,
        "win_rate": np.nan,
        "avg_orb_size": np.nan,
        "n_winter": 0,
        "avg_r_winter": np.nan,
        "wr_winter": np.nan,
        "n_summer": 0,
        "avg_r_summer": np.nan,
        "wr_summer": np.nan,
        "trades_raw": [],
    }


# =========================================================================
# Statistical Tools
# =========================================================================

def compute_pvalue(trades_raw):
    """One-sample t-test: is mean significantly different from 0?"""
    if not HAS_SCIPY:
        return np.nan
    if len(trades_raw) < 2:
        return np.nan
    arr = np.array(trades_raw)
    _, p = ttest_1samp(arr, 0.0)
    return float(p)


def bh_fdr_correction(p_values):
    """Benjamini-Hochberg FDR correction. NaN-safe."""
    p_arr = np.array(p_values, dtype=float)
    n = len(p_arr)
    adjusted = np.full(n, np.nan)

    # Get indices of non-NaN p-values
    valid_mask = ~np.isnan(p_arr)
    valid_idx = np.where(valid_mask)[0]
    valid_p = p_arr[valid_idx]

    if len(valid_p) == 0:
        return adjusted.tolist()

    m = len(valid_p)
    sorted_order = np.argsort(valid_p)
    sorted_p = valid_p[sorted_order]

    # BH adjustment
    bh_adjusted = np.zeros(m)
    bh_adjusted[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        rank = i + 1
        bh_adjusted[i] = min(bh_adjusted[i + 1], sorted_p[i] * m / rank)

    # Clip to [0, 1]
    bh_adjusted = np.clip(bh_adjusted, 0.0, 1.0)

    # Unsort back to original order
    unsorted = np.zeros(m)
    unsorted[sorted_order] = bh_adjusted

    # Place back into full array
    adjusted[valid_idx] = unsorted
    return adjusted.tolist()


def classify_sample(n):
    """Per RESEARCH_RULES.md thresholds."""
    if n < 30:
        return "INVALID"
    elif n < 100:
        return "REGIME"
    elif n < 200:
        return "PRELIMINARY"
    else:
        return "CORE"


def sensitivity_check(highs, lows, closes, bris_h, bris_m,
                      aperture_min, dst_mask):
    """Run scan at aperture±2min. Returns ROBUST, FRAGILE, or SKIP."""
    if aperture_min <= 6:
        return "SKIP"  # Too small to perturb

    results = []
    for delta in [-2, +2]:
        test_aperture = aperture_min + delta
        if test_aperture < 3:
            continue
        r = scan_session_aperture(highs, lows, closes, bris_h, bris_m,
                                  test_aperture, dst_mask)
        if r is None or r["n_trades"] < 5:
            results.append(np.nan)
        else:
            results.append(r["avg_r"])

    if not results or all(np.isnan(r) for r in results):
        return "SKIP"

    # ROBUST = both neighbors positive; FRAGILE = either neighbor <= 0
    valid = [r for r in results if not np.isnan(r)]
    if all(r > 0 for r in valid):
        return "ROBUST"
    return "FRAGILE"


# =========================================================================
# Output Formatting
# =========================================================================

def print_instrument_table(inst_rows, instrument):
    """Print aperture comparison table for one instrument."""
    print(f"\n{'=' * 110}")
    print(f"  {instrument} — APERTURE SCAN")
    print(f"{'=' * 110}")

    sessions_present = sorted(set(r["session"] for r in inst_rows))

    for session in sessions_present:
        sess_rows = [r for r in inst_rows if r["session"] == session]
        sess_rows.sort(key=lambda r: r["aperture_min"])

        # Find 5m baseline
        baseline_avgr = np.nan
        for r in sess_rows:
            if r["aperture_min"] == 5:
                baseline_avgr = r["avg_r"]
                break

        print(f"\n  {session} (DST: {SESSIONS[session]['dst_type']})")
        header = (f"  {'Aper':>4s} {'N':>5s} {'avgR':>8s} {'delta':>7s} "
                  f"{'totR':>8s} {'WR%':>6s} {'ORBsz':>6s} "
                  f"{'p_raw':>8s} {'p_adj':>8s} "
                  f"{'W_N':>4s} {'W_avgR':>7s} {'S_N':>4s} {'S_avgR':>7s} "
                  f"{'Sens':>7s} {'Class':>11s}")
        print(header)
        print(f"  {'-' * 106}")

        for r in sess_rows:
            n = r["n_trades"]
            avg_r = r["avg_r"]
            delta = avg_r - baseline_avgr if not np.isnan(avg_r) and not np.isnan(baseline_avgr) else np.nan

            # Star if beats 5m baseline
            marker = " *" if not np.isnan(delta) and delta > 0 and r["aperture_min"] != 5 else "  "

            avg_str = f"{avg_r:+8.4f}" if not np.isnan(avg_r) else "      --"
            delta_str = f"{delta:+7.4f}" if not np.isnan(delta) else "     --"
            tot_str = f"{r['total_r']:+8.1f}" if r['total_r'] != 0 else "     0.0"
            wr_str = f"{r['win_rate'] * 100:5.1f}%" if not np.isnan(r['win_rate']) else "    --%"
            orb_str = f"{r['avg_orb_size']:6.1f}" if not np.isnan(r['avg_orb_size']) else "    --"
            p_raw = f"{r['p_value']:8.4f}" if not np.isnan(r['p_value']) else "      --"
            p_adj = f"{r['p_adj_bh']:8.4f}" if not np.isnan(r['p_adj_bh']) else "      --"
            nw = r["n_winter"]
            ns = r["n_summer"]
            aw = r["avg_r_winter"]
            asr = r["avg_r_summer"]
            w_str = f"{aw:+7.3f}" if not np.isnan(aw) else "     --"
            s_str = f"{asr:+7.3f}" if not np.isnan(asr) else "     --"
            sens = r["sensitivity_verdict"]
            cls = r["sample_class"]

            print(f"  {r['aperture_min']:4d}{marker} {n:5d} {avg_str} {delta_str} "
                  f"{tot_str} {wr_str} {orb_str} "
                  f"{p_raw} {p_adj} "
                  f"{nw:4d} {w_str} {ns:4d} {s_str} "
                  f"{sens:>7s} {cls:>11s}")


def print_honest_summary(all_rows):
    """RESEARCH_RULES.md mandated honest summary."""
    print(f"\n{'=' * 110}")
    print(f"  HONEST SUMMARY")
    print(f"{'=' * 110}")

    # Count what we tested
    n_combos = len(all_rows)
    n_positive = sum(1 for r in all_rows if not np.isnan(r["avg_r"]) and r["avg_r"] > 0)
    n_sig_raw = sum(1 for r in all_rows if not np.isnan(r["p_value"]) and r["p_value"] < 0.05)
    n_sig_bh = sum(1 for r in all_rows if not np.isnan(r["p_adj_bh"]) and r["p_adj_bh"] < 0.05)

    print(f"\n  SCOPE: {n_combos} combinations tested "
          f"({len(INSTRUMENTS)} instruments x {len(SESSIONS)} sessions x {len(APERTURES)} apertures)")
    print(f"  POSITIVE avg_r: {n_positive}/{n_combos}")
    print(f"  RAW p < 0.05: {n_sig_raw} (before correction)")
    print(f"  BH-ADJUSTED p < 0.05: {n_sig_bh} (after FDR correction)")

    # Find combos that beat 5m baseline
    baselines = {}  # (instrument, session) -> 5m avg_r
    for r in all_rows:
        if r["aperture_min"] == 5:
            baselines[(r["instrument"], r["session"])] = r["avg_r"]

    beats_baseline = []
    for r in all_rows:
        if r["aperture_min"] == 5:
            continue
        key = (r["instrument"], r["session"])
        base = baselines.get(key, np.nan)
        if (not np.isnan(r["avg_r"]) and not np.isnan(base)
                and r["avg_r"] > base and r["n_trades"] >= 30):
            delta = r["avg_r"] - base
            beats_baseline.append(r | {"delta_vs_5m": delta})

    beats_baseline.sort(key=lambda r: r["delta_vs_5m"], reverse=True)

    print(f"\n  SURVIVED SCRUTINY (beats 5m baseline, N>=30):")
    if not beats_baseline:
        print("    None — no aperture consistently outperformed 5m.")
    else:
        for r in beats_baseline[:15]:
            p_str = f"p_adj={r['p_adj_bh']:.4f}" if not np.isnan(r["p_adj_bh"]) else "p_adj=NaN"
            print(f"    {r['instrument']} {r['session']} {r['aperture_min']}m: "
                  f"avgR={r['avg_r']:+.4f} (delta={r['delta_vs_5m']:+.4f} vs 5m) "
                  f"N={r['n_trades']} {p_str} sens={r['sensitivity_verdict']} "
                  f"[{r['sample_class']}]")

    # Specifically flag robust + significant + core/preliminary
    print(f"\n  STRONGEST CANDIDATES (beats 5m, BH p<0.05, ROBUST, N>=30):")
    strong = [r for r in beats_baseline
              if not np.isnan(r["p_adj_bh"]) and r["p_adj_bh"] < 0.05
              and r["sensitivity_verdict"] == "ROBUST"]
    if not strong:
        print("    None.")
    else:
        for r in strong:
            print(f"    {r['instrument']} {r['session']} {r['aperture_min']}m: "
                  f"avgR={r['avg_r']:+.4f} (delta={r['delta_vs_5m']:+.4f}) "
                  f"N={r['n_trades']} p_adj={r['p_adj_bh']:.4f} [{r['sample_class']}]")

    # DID NOT SURVIVE
    print(f"\n  DID NOT SURVIVE:")
    worse = []
    for r in all_rows:
        if r["aperture_min"] == 5:
            continue
        key = (r["instrument"], r["session"])
        base = baselines.get(key, np.nan)
        if not np.isnan(r["avg_r"]) and not np.isnan(base) and r["avg_r"] < base:
            worse.append(r)
    n_worse = len(worse)
    n_fragile = sum(1 for r in beats_baseline if r["sensitivity_verdict"] == "FRAGILE")
    print(f"    {n_worse} combos performed WORSE than 5m baseline")
    print(f"    {n_fragile} combos beat 5m but were FRAGILE (±2min kills the edge)")

    # CAVEATS
    print(f"\n  CAVEATS:")
    print(f"    - This is IN-SAMPLE research (not walk-forward validated)")
    print(f"    - E1 entry + CB1 only (production uses CB1-5)")
    print(f"    - RR2.0 only (production tests 1.0-4.0)")
    print(f"    - G4+ only (production tests G2-G8)")
    print(f"    - Any finding needs full pipeline rebuild to confirm")

    # Instrument-specific data caveats
    for inst in INSTRUMENTS:
        inst_rows_local = [r for r in all_rows if r["instrument"] == inst and r["aperture_min"] == 5]
        if inst_rows_local:
            n_days = inst_rows_local[0]["n_days"]
            if n_days < 1000:
                print(f"    - {inst}: only {n_days} trading days (short history)")

    # NEXT STEPS
    print(f"\n  NEXT STEPS:")
    print(f"    - If any combo is ROBUST + BH p<0.05 + CORE class:")
    print(f"      → Run outcome_builder at that aperture (~60-90s)")
    print(f"      → Run strategy_discovery + validation")
    print(f"      → Compare family-level Sharpe to 5m families")
    print(f"    - If nothing survives: 5m is optimal, stop here")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-aperture ORB scan across CME micro futures"
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

    n_combos = len(INSTRUMENTS) * len(SESSIONS) * len(APERTURES)
    print(f"\n{'=' * 110}")
    print(f"  MULTI-APERTURE ORB SCAN")
    print(f"  Database: {db_path}")
    print(f"  Parameters: G{G4_MIN:.0f}+ filter | RR{RR_TARGET:.1f} target | "
          f"{BREAK_WINDOW // 60}h break window | {OUTCOME_WINDOW // 60}h outcome window")
    print(f"  Grid: {len(INSTRUMENTS)} instruments x {len(SESSIONS)} sessions x "
          f"{len(APERTURES)} apertures = {n_combos} combinations")
    print(f"  scipy available: {HAS_SCIPY}")
    print(f"{'=' * 110}")

    con = duckdb.connect(str(db_path), read_only=True)
    all_rows = []
    t_total = time.time()

    try:
        for instrument in INSTRUMENTS:
            print(f"\n  Loading {instrument}...")
            t_load = time.time()
            bars_df = load_bars(con, instrument)

            if len(bars_df) == 0:
                print(f"    No data for {instrument}, skipping.")
                continue

            print(f"    {len(bars_df):,} bars loaded in {time.time() - t_load:.1f}s")
            t_build = time.time()
            all_days, opens, highs, lows, closes = build_day_arrays(bars_df)
            n_days = len(all_days)
            print(f"    {n_days} trading days ({all_days[0]} to {all_days[-1]}) "
                  f"built in {time.time() - t_build:.1f}s")
            del bars_df

            # Build DST masks once
            us_mask, uk_mask = build_dst_masks(all_days)
            print(f"    US DST: {int(us_mask.sum())} summer / "
                  f"{n_days - int(us_mask.sum())} winter")
            print(f"    UK DST: {int(uk_mask.sum())} summer / "
                  f"{n_days - int(uk_mask.sum())} winter")

            t_scan = time.time()
            combo_count = 0

            for sess_name, sess_cfg in SESSIONS.items():
                bris_h = sess_cfg["bris_h"]
                bris_m = sess_cfg["bris_m"]
                dst_type = sess_cfg["dst_type"]

                # Pick correct DST mask
                if dst_type == "US":
                    dst_mask = us_mask
                elif dst_type == "UK":
                    dst_mask = uk_mask
                else:
                    # CLEAN sessions: no meaningful DST split, use zeros
                    dst_mask = np.zeros(n_days, dtype=bool)

                for aperture in APERTURES:
                    result = scan_session_aperture(
                        highs, lows, closes, bris_h, bris_m,
                        aperture, dst_mask
                    )

                    if result is None:
                        result = _empty_result(n_days)

                    # Compute p-value from raw trades
                    p_val = compute_pvalue(result.get("trades_raw", []))

                    # Sensitivity check (only if positive avgR)
                    sens = "--"
                    if not np.isnan(result["avg_r"]) and result["avg_r"] > 0:
                        sens = sensitivity_check(
                            highs, lows, closes, bris_h, bris_m,
                            aperture, dst_mask
                        )

                    row = {
                        "instrument": instrument,
                        "session": sess_name,
                        "aperture_min": aperture,
                        "n_days": result["n_days"],
                        "n_g4": result["n_g4"],
                        "n_breaks": result["n_breaks"],
                        "n_trades": result["n_trades"],
                        "avg_r": result["avg_r"],
                        "total_r": result["total_r"],
                        "win_rate": result["win_rate"],
                        "avg_orb_size": result["avg_orb_size"],
                        "p_value": p_val,
                        "p_adj_bh": np.nan,  # Filled after all combos
                        "n_winter": result["n_winter"],
                        "avg_r_winter": result["avg_r_winter"],
                        "wr_winter": result["wr_winter"],
                        "n_summer": result["n_summer"],
                        "avg_r_summer": result["avg_r_summer"],
                        "wr_summer": result["wr_summer"],
                        "sensitivity_verdict": sens,
                        "sample_class": classify_sample(result["n_trades"]),
                    }
                    all_rows.append(row)
                    combo_count += 1

            print(f"    {instrument}: {combo_count} combos scanned in "
                  f"{time.time() - t_scan:.1f}s")
            del opens, highs, lows, closes

    finally:
        con.close()

    if not all_rows:
        print("\nNo results to report.")
        return

    # BH FDR correction across ALL p-values
    p_values = [r["p_value"] for r in all_rows]
    adjusted = bh_fdr_correction(p_values)
    for i, r in enumerate(all_rows):
        r["p_adj_bh"] = adjusted[i]

    # Compute delta_vs_5m for each row
    baselines = {}
    for r in all_rows:
        if r["aperture_min"] == 5:
            baselines[(r["instrument"], r["session"])] = r["avg_r"]
    for r in all_rows:
        key = (r["instrument"], r["session"])
        base = baselines.get(key, np.nan)
        if not np.isnan(r["avg_r"]) and not np.isnan(base):
            r["delta_vs_5m"] = r["avg_r"] - base
        else:
            r["delta_vs_5m"] = np.nan

    # Save CSV
    output_dir = Path("research/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "aperture_scan_results.csv"

    # Remove trades_raw from CSV output (it's a list)
    csv_rows = [{k: v for k, v in r.items() if k != "trades_raw"} for r in all_rows]
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  CSV saved: {csv_path} ({len(df)} rows)")

    # Per-instrument tables
    for instrument in INSTRUMENTS:
        inst_rows = [r for r in all_rows if r["instrument"] == instrument]
        if inst_rows:
            print_instrument_table(inst_rows, instrument)

    # Honest summary
    print_honest_summary(all_rows)

    print(f"\n  Total runtime: {time.time() - t_total:.1f}s")
    print()


if __name__ == "__main__":
    main()

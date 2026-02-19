#!/usr/bin/env python3
"""
Overlap Analysis — are candidate sessions independent edges or redundant?

The 24h edge scan found candidate times (10:15, 12:45, 15:45, etc.) with strong
metrics. Before adding any as pipeline sessions, this script answers: are they
genuinely independent edges, or the same edge as existing sessions measured at a
different clock time?

Method:
  1. Reuse vectorized numpy engine from research_aperture_scan.py (standalone copy)
  2. scan_session_per_day() returns per-day results (not aggregated)
  3. compute_overlap_metrics() computes paired-day overlap between two sessions
  4. classify_overlap() labels INDEPENDENT / REDUNDANT / GREY-ZONE
  5. DST splits: affected pairs get WINTER + SUMMER only; clean pairs get ALL only

Anti-bias measures:
  1. Report ALL pairs (no cherry-picking)
  2. Pearson r with p-value on shared break-days
  3. Exclusive-day conditional avgR with N count
  4. DST regime split where applicable
  5. Honest summary per RESEARCH_RULES.md

Usage:
  python research/research_overlap_analysis.py
  python research/research_overlap_analysis.py --db-path C:/db/gold.db
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
    from scipy.stats import pearsonr
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

# Fixed parameters (match production: E1 entry, CB1, RR2.0, G4+)
G4_MIN = 4.0
BREAK_WINDOW = 240    # 4 hours in minutes
OUTCOME_WINDOW = 480  # 8 hours in minutes
RR_TARGET = 2.0

# DST type for each session label
# US DST: sessions near CME open (5PM ET) — shifts ±1hr with US DST
# UK DST: sessions near London open (8AM London) — shifts ±1hr with UK DST
# CLEAN: sessions near Asia opens (no DST) or mid-European-morning (no shifting event)
SESSION_DST_TYPE = {
    "0900": "US",     # CME open in winter, +1hr late in summer
    "0930": "US",     # CME +30min winter, +1hr30 summer
    "1000": "CLEAN",  # Tokyo open, no DST
    "1015": "CLEAN",  # Tokyo +15min, no DST
    "1100": "CLEAN",  # Singapore open, no DST
    "1130": "CLEAN",  # HK/Shanghai open, no DST
    "1245": "CLEAN",  # HK/SG afternoon, no DST
    "1545": "UK",     # 05:45 UTC — relationship to London open shifts ±1hr
    "1615": "UK",     # 06:15 UTC — near London open, shifts ±1hr
    "1645": "UK",     # 06:45 UTC — very close to London open, shifts ±1hr
    "1800": "UK",     # London open in winter, +1hr late in summer
    "1815": "UK",     # London +15min winter, +1hr15 summer
    "1900": "CLEAN",  # 09:00 UTC — mid-European-morning, not near any DST-shifting open
    "1915": "CLEAN",  # 09:15 UTC — same
    "2300": "US",     # Pre/post US data release depending on DST
    "0030": "US",     # US equity open in winter, +1hr late in summer
}

# Candidate pairs: (existing_session, candidate_session) per instrument
CANDIDATE_PAIRS = {
    "MNQ": [("1000", "1015"), ("1130", "1245"), ("1800", "1545"), ("1800", "1815")],
    "MES": [("1000", "1015"), ("0030", "1900"), ("0030", "1915")],
    "MGC": [("0900", "0930"), ("1800", "1815"), ("1800", "1615"), ("1800", "1645")],
}


# =========================================================================
# Data Loading (copied from research_aperture_scan.py — standalone)
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
# Per-Day Scan Engine
# =========================================================================

def parse_session_label(label):
    """Parse session label like '0930' or '1815' into (bris_h, bris_m)."""
    h = int(label[:2])
    m = int(label[2:])
    return h, m


def scan_session_per_day(highs, lows, closes, bris_h, bris_m, aperture_min=5):
    """Scan one session across all trading days, returning per-day results.

    Adapted from scan_session_aperture (aperture_scan.py). Same ORB computation,
    break detection, and outcome resolution — but returns per-day detail instead
    of aggregated metrics.

    Returns dict[day_index -> per-day result dict].
    """
    n_days = highs.shape[0]
    start_min = ((bris_h - 9) % 24) * 60 + bris_m

    # ORB minute columns
    orb_mins = [start_min + i for i in range(aperture_min)]
    if orb_mins[-1] >= 1440:
        return {}  # Would wrap past trading day boundary

    # Vectorized ORB computation
    orb_h = np.column_stack([highs[:, m] for m in orb_mins])
    orb_l = np.column_stack([lows[:, m] for m in orb_mins])
    orb_c = np.column_stack([closes[:, m] for m in orb_mins])

    # Valid ORB = ALL bars present
    valid_orb = np.all(~np.isnan(orb_c), axis=1)
    orb_high = np.nanmax(orb_h, axis=1)
    orb_low = np.nanmin(orb_l, axis=1)
    orb_size = orb_high - orb_low

    # G4+ filter
    g4_mask = valid_orb & (orb_size >= G4_MIN)

    results = {}

    for day_idx in range(n_days):
        if not valid_orb[day_idx]:
            continue  # No valid ORB — skip entirely

        if not g4_mask[day_idx]:
            # Valid ORB but sub-G4
            results[day_idx] = {
                "valid_orb": True,
                "g4_pass": False,
                "orb_size": float(orb_size[day_idx]),
                "broke": False,
                "direction": None,
                "outcome_r": np.nan,
                "entry": np.nan,
            }
            continue

        oh = orb_high[day_idx]
        ol = orb_low[day_idx]
        os_val = orb_size[day_idx]

        # --- Break detection (first 1m close outside ORB) ---
        break_start = start_min + aperture_min
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
            # G4+ but no break within window
            results[day_idx] = {
                "valid_orb": True,
                "g4_pass": True,
                "orb_size": float(os_val),
                "broke": False,
                "direction": None,
                "outcome_r": np.nan,
                "entry": np.nan,
            }
            continue

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

        results[day_idx] = {
            "valid_orb": True,
            "g4_pass": True,
            "orb_size": float(os_val),
            "broke": True,
            "direction": break_dir,
            "outcome_r": float(outcome_r),
            "entry": float(entry),
        }

    return results


# =========================================================================
# Overlap Metrics
# =========================================================================

def compute_overlap_metrics(existing_days, candidate_days):
    """Compute overlap metrics between two sessions' per-day results.

    Operates on G4+ eligible days common to both sessions.

    Returns dict with overlap metrics, or None if insufficient data.
    """
    # Find common G4+ days
    existing_g4 = {d for d, r in existing_days.items() if r["g4_pass"]}
    candidate_g4 = {d for d, r in candidate_days.items() if r["g4_pass"]}
    common_g4 = existing_g4 & candidate_g4

    if len(common_g4) == 0:
        return None

    n_both_broke = 0
    n_existing_only = 0
    n_candidate_only = 0
    n_neither = 0

    existing_r_shared = []
    candidate_r_shared = []
    n_same_direction = 0

    candidate_r_exclusive = []

    existing_all_r = []
    candidate_all_r = []

    for d in common_g4:
        e = existing_days[d]
        c = candidate_days[d]
        e_broke = e["broke"]
        c_broke = c["broke"]

        if e_broke:
            existing_all_r.append(e["outcome_r"])
        if c_broke:
            candidate_all_r.append(c["outcome_r"])

        if e_broke and c_broke:
            n_both_broke += 1
            existing_r_shared.append(e["outcome_r"])
            candidate_r_shared.append(c["outcome_r"])
            if e["direction"] == c["direction"]:
                n_same_direction += 1
        elif e_broke and not c_broke:
            n_existing_only += 1
        elif not e_broke and c_broke:
            n_candidate_only += 1
            candidate_r_exclusive.append(c["outcome_r"])
        else:
            n_neither += 1

    n_candidate_broke = n_both_broke + n_candidate_only

    # Shared break percentage (candidate denominator)
    shared_break_pct = (n_both_broke / n_candidate_broke
                        if n_candidate_broke > 0 else np.nan)

    # Direction concordance
    direction_concordance = (n_same_direction / n_both_broke
                             if n_both_broke > 0 else np.nan)

    # Pearson r on shared break-days
    r_correlation = np.nan
    r_pvalue = np.nan
    if HAS_SCIPY and n_both_broke >= 3:
        arr_e = np.array(existing_r_shared)
        arr_c = np.array(candidate_r_shared)
        # Check for zero variance
        if np.std(arr_e) > 0 and np.std(arr_c) > 0:
            r_correlation, r_pvalue = pearsonr(arr_e, arr_c)
    elif not HAS_SCIPY and n_both_broke >= 3:
        print("  WARNING: scipy not available, Pearson r will be NaN")

    # Conditional avgR on exclusive days
    cond_avg_r_exclusive = (float(np.mean(candidate_r_exclusive))
                            if candidate_r_exclusive else np.nan)

    # Overall avgR for each session (on common G4+ days only)
    existing_avg_r = (float(np.mean(existing_all_r))
                      if existing_all_r else np.nan)
    candidate_avg_r = (float(np.mean(candidate_all_r))
                       if candidate_all_r else np.nan)

    return {
        "n_common_g4": len(common_g4),
        "n_both_broke": n_both_broke,
        "n_existing_only": n_existing_only,
        "n_candidate_only": n_candidate_only,
        "n_neither": n_neither,
        "shared_break_pct": shared_break_pct,
        "direction_concordance": direction_concordance,
        "r_correlation": r_correlation,
        "r_pvalue": r_pvalue,
        "existing_avg_r": existing_avg_r,
        "existing_n": len(existing_all_r),
        "candidate_avg_r": candidate_avg_r,
        "candidate_n": len(candidate_all_r),
        "cond_avg_r_exclusive": cond_avg_r_exclusive,
        "n_exclusive": len(candidate_r_exclusive),
    }


def classify_overlap(shared_break_pct, r_correlation):
    """Classify the overlap between two sessions.

    | Verdict     | Criteria                           |
    |-------------|-----------------------------------|
    | INDEPENDENT | shared < 50% AND |r| < 0.3       |
    | REDUNDANT   | shared > 80% AND r > 0.6          |
    | GREY-ZONE   | everything else                   |
    """
    if np.isnan(shared_break_pct):
        return "GREY-ZONE"

    r_abs = abs(r_correlation) if not np.isnan(r_correlation) else 0.0

    if shared_break_pct < 0.50 and r_abs < 0.3:
        return "INDEPENDENT"
    elif shared_break_pct > 0.80 and (not np.isnan(r_correlation)
                                      and r_correlation > 0.6):
        return "REDUNDANT"
    else:
        return "GREY-ZONE"


# =========================================================================
# Output Formatting
# =========================================================================

def print_instrument_table(inst_rows, instrument):
    """Print overlap analysis table for one instrument."""
    print(f"\n{'=' * 120}")
    print(f"  {instrument} — OVERLAP ANALYSIS")
    print(f"{'=' * 120}")

    header = (f"  {'Existing':>8s} {'Candidate':>9s} {'DST':>6s} "
              f"{'G4':>5s} {'Both':>5s} {'E_only':>6s} {'C_only':>6s} "
              f"{'Neith':>5s} {'Shrd%':>6s} {'DirCon':>6s} "
              f"{'r':>6s} {'r_p':>7s} "
              f"{'E_avgR':>7s} {'E_N':>4s} {'C_avgR':>7s} {'C_N':>4s} "
              f"{'Excl_R':>7s} {'X_N':>4s} {'Verdict':>12s}")
    print(header)
    print(f"  {'-' * 116}")

    for r in inst_rows:
        sbp = f"{r['shared_break_pct'] * 100:5.1f}%" if not np.isnan(r['shared_break_pct']) else "    --%"
        dc = f"{r['direction_concordance'] * 100:5.1f}%" if not np.isnan(r['direction_concordance']) else "    --%"
        rc = f"{r['r_correlation']:+6.3f}" if not np.isnan(r['r_correlation']) else "    --"
        rp = f"{r['r_pvalue']:7.4f}" if not np.isnan(r['r_pvalue']) else "     --"
        ea = f"{r['existing_avg_r']:+7.3f}" if not np.isnan(r['existing_avg_r']) else "     --"
        ca = f"{r['candidate_avg_r']:+7.3f}" if not np.isnan(r['candidate_avg_r']) else "     --"
        xa = f"{r['cond_avg_r_exclusive']:+7.3f}" if not np.isnan(r['cond_avg_r_exclusive']) else "     --"

        excl_flag = ""
        if r['n_exclusive'] > 0 and r['n_exclusive'] < 15:
            excl_flag = " *"  # Flag small exclusive N

        print(f"  {r['existing_session']:>8s} {r['candidate_session']:>9s} "
              f"{r['dst_regime']:>6s} "
              f"{r['n_common_g4']:5d} {r['n_both_broke']:5d} "
              f"{r['n_existing_only']:6d} {r['n_candidate_only']:6d} "
              f"{r['n_neither']:5d} {sbp} {dc} "
              f"{rc} {rp} "
              f"{ea} {r['existing_n']:4d} {ca} {r['candidate_n']:4d} "
              f"{xa} {r['n_exclusive']:4d}{excl_flag} {r['classification']:>12s}")


def print_honest_summary(all_rows):
    """RESEARCH_RULES.md mandated honest summary."""
    print(f"\n{'=' * 120}")
    print(f"  HONEST SUMMARY")
    print(f"{'=' * 120}")

    # Group rows by unique pair
    pairs_by_key = {}
    for r in all_rows:
        key = (r["instrument"], r["existing_session"], r["candidate_session"])
        pairs_by_key.setdefault(key, []).append(r)

    n_pairs = len(pairs_by_key)

    # Count operative verdicts (ALL for clean, WINTER+SUMMER for DST)
    n_independent = sum(1 for r in all_rows if r["classification"] == "INDEPENDENT")
    n_redundant = sum(1 for r in all_rows if r["classification"] == "REDUNDANT")
    n_grey = sum(1 for r in all_rows if r["classification"] == "GREY-ZONE")
    n_verdicts = n_independent + n_redundant + n_grey

    print(f"\n  SCOPE: {n_pairs} candidate pairs tested across "
          f"{len(CANDIDATE_PAIRS)} instruments")
    print(f"  scipy available: {HAS_SCIPY}")

    print(f"\n  CLASSIFICATION ({n_verdicts} operative verdicts across {n_pairs} pairs):")
    print(f"    INDEPENDENT: {n_independent}")
    print(f"    REDUNDANT:   {n_redundant}")
    print(f"    GREY-ZONE:   {n_grey}")
    print(f"    (Clean pairs = 1 verdict each; DST pairs = 2 verdicts each: WINTER + SUMMER)")

    # --- Per-pair results ---
    def _fmt_row(r):
        rc = f"r={r['r_correlation']:+.3f}" if not np.isnan(r['r_correlation']) else "r=NaN"
        sbp = f"shared={r['shared_break_pct'] * 100:.1f}%" if not np.isnan(r['shared_break_pct']) else "shared=NaN"
        xa = f"excl_avgR={r['cond_avg_r_exclusive']:.3f}" if not np.isnan(r['cond_avg_r_exclusive']) else "excl_avgR=NaN"
        return sbp, rc, xa

    # SURVIVED
    print(f"\n  SURVIVED (INDEPENDENT — genuinely new edges):")
    found_independent = False
    for key, rows in pairs_by_key.items():
        inst, ex, cand = key
        indep_rows = [r for r in rows if r["classification"] == "INDEPENDENT"]
        if not indep_rows:
            continue
        found_independent = True
        if len(rows) == 1 and rows[0]["dst_regime"] == "ALL":
            # Clean pair
            r = rows[0]
            sbp, rc, xa = _fmt_row(r)
            print(f"    {inst} {ex}->{cand}: {sbp}, {rc}, "
                  f"exclusive_days={r['n_exclusive']}, {xa}")
        else:
            # DST pair — report both regimes
            for r in rows:
                if r["classification"] == "INDEPENDENT":
                    sbp, rc, xa = _fmt_row(r)
                    print(f"    {inst} {ex}->{cand} [{r['dst_regime']}]: {sbp}, {rc}, "
                          f"exclusive_days={r['n_exclusive']}, {xa}")
    if not found_independent:
        print("    None — all candidates overlap with existing sessions.")

    # DID NOT SURVIVE
    print(f"\n  DID NOT SURVIVE (REDUNDANT — same edge, different clock):")
    found_redundant = False
    for key, rows in pairs_by_key.items():
        inst, ex, cand = key
        redun_rows = [r for r in rows if r["classification"] == "REDUNDANT"]
        if not redun_rows:
            continue
        found_redundant = True
        if len(rows) == 1 and rows[0]["dst_regime"] == "ALL":
            r = rows[0]
            sbp, rc, _ = _fmt_row(r)
            print(f"    {inst} {ex}->{cand}: {sbp}, {rc}")
        else:
            for r in rows:
                if r["classification"] == "REDUNDANT":
                    sbp, rc, _ = _fmt_row(r)
                    print(f"    {inst} {ex}->{cand} [{r['dst_regime']}]: {sbp}, {rc}")
    if not found_redundant:
        print("    None — no pairs clearly redundant.")

    # GREY-ZONE
    print(f"\n  GREY-ZONE (ambiguous — needs deeper investigation):")
    found_grey = False
    for key, rows in pairs_by_key.items():
        inst, ex, cand = key
        grey_rows = [r for r in rows if r["classification"] == "GREY-ZONE"]
        if not grey_rows:
            continue
        found_grey = True
        if len(rows) == 1 and rows[0]["dst_regime"] == "ALL":
            r = rows[0]
            sbp, rc, _ = _fmt_row(r)
            print(f"    {inst} {ex}->{cand}: {sbp}, {rc}")
        else:
            for r in rows:
                if r["classification"] == "GREY-ZONE":
                    sbp, rc, _ = _fmt_row(r)
                    print(f"    {inst} {ex}->{cand} [{r['dst_regime']}]: {sbp}, {rc}")
    if not found_grey:
        print("    None.")

    # DST DIVERGENCE — flag pairs where winter/summer classifications differ
    print(f"\n  DST REGIME DIVERGENCE (winter vs summer classification differs):")
    divergence_found = False
    for key, rows in pairs_by_key.items():
        if len(rows) < 2:
            continue
        regimes = {r["dst_regime"]: r for r in rows}
        if "WINTER" in regimes and "SUMMER" in regimes:
            w = regimes["WINTER"]
            s = regimes["SUMMER"]
            if w["classification"] != s["classification"]:
                divergence_found = True
                inst, ex, cand = key
                w_sbp = f"{w['shared_break_pct'] * 100:.1f}%" if not np.isnan(w['shared_break_pct']) else "NaN"
                s_sbp = f"{s['shared_break_pct'] * 100:.1f}%" if not np.isnan(s['shared_break_pct']) else "NaN"
                print(f"    ** {inst} {ex}->{cand}: "
                      f"WINTER={w['classification']} (shared={w_sbp}), "
                      f"SUMMER={s['classification']} (shared={s_sbp})")
                print(f"       DIFFERENT TRADE in each regime.")

    if not divergence_found:
        print("    None — all DST-affected pairs have consistent winter/summer classification.")

    # MECHANISM NOTES for UK-DST pre-London candidates
    # London opens at 08:00 UTC in winter (GMT), 07:00 UTC in summer (BST).
    # Pre-London candidates are CLOSER in summer, FARTHER in winter.
    _PRE_LONDON_NOTES = {
        "1545": ("In winter, 15:45 Brisbane (05:45 UTC) = 2hr15 pre-London-open (08:00 UTC). "
                 "In summer, 15:45 Brisbane (05:45 UTC) = 1hr15 pre-London-open (07:00 UTC)."),
        "1615": ("In winter, 16:15 Brisbane (06:15 UTC) = 1hr45 pre-London-open (08:00 UTC). "
                 "In summer, 16:15 Brisbane (06:15 UTC) = 45min pre-London-open (07:00 UTC)."),
        "1645": ("In winter, 16:45 Brisbane (06:45 UTC) = 1hr15 pre-London-open (08:00 UTC). "
                 "In summer, 16:45 Brisbane (06:45 UTC) = 15min pre-London-open (07:00 UTC)."),
        "1815": ("In summer, 18:15 Brisbane (08:15 UTC) = 1hr15 post-London-open (07:00 UTC). "
                 "In winter, 18:15 Brisbane (08:15 UTC) = 15min post-London-open (08:00 UTC)."),
    }
    pre_london_candidates = set()
    for r in all_rows:
        cand = r["candidate_session"]
        if cand in _PRE_LONDON_NOTES:
            pre_london_candidates.add((r["instrument"], cand))
    if pre_london_candidates:
        print(f"\n  MECHANISM NOTES (UK-DST pre/post-London candidates):")
        for inst, cand in sorted(pre_london_candidates):
            print(f"    {inst} {cand}: {_PRE_LONDON_NOTES[cand]}")
            print(f"       The edge may be a different trade in each regime.")

    # CAVEATS
    print(f"\n  CAVEATS:")
    print(f"    - This is IN-SAMPLE analysis (no walk-forward)")
    print(f"    - E1 entry + CB1 only (5min aperture, G4+ filter)")
    print(f"    - RR2.0 only (production tests 1.0-4.0)")
    print(f"    - Overlap != causation. Two sessions can share break-days")
    print(f"      by coincidence if the underlying move is large enough")
    print(f"    - Small exclusive-day N (< 15) flagged with * in tables")
    print(f"    - DST-affected pairs report WINTER and SUMMER separately (no blended number)")

    # NEXT STEPS
    print(f"\n  NEXT STEPS:")
    print(f"    - INDEPENDENT candidates: consider adding as pipeline sessions")
    print(f"    - REDUNDANT candidates: do NOT add (would double-count same edge)")
    print(f"    - GREY-ZONE candidates: investigate further before deciding")
    print(f"      (e.g., conditional analysis, volume profiles, entry timing)")
    print(f"    - DST-DIVERGENT pairs: treat winter and summer as separate")
    print(f"      research questions — one regime may be independent while")
    print(f"      the other is redundant")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overlap analysis: are candidate sessions independent edges?"
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

    n_total_pairs = sum(len(pairs) for pairs in CANDIDATE_PAIRS.values())
    print(f"\n{'=' * 120}")
    print(f"  OVERLAP ANALYSIS — Session Independence Test")
    print(f"  Database: {db_path}")
    print(f"  Parameters: G{G4_MIN:.0f}+ filter | RR{RR_TARGET:.1f} target | "
          f"{BREAK_WINDOW // 60}h break window | {OUTCOME_WINDOW // 60}h outcome window")
    print(f"  Pairs: {n_total_pairs} across {len(CANDIDATE_PAIRS)} instruments")
    print(f"  scipy available: {HAS_SCIPY}")
    print(f"{'=' * 120}")

    con = duckdb.connect(str(db_path), read_only=True)
    all_rows = []
    t_total = time.time()

    try:
        for instrument, pairs in CANDIDATE_PAIRS.items():
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

            # Cache per-day results for each unique session label
            session_cache = {}
            unique_sessions = set()
            for existing, candidate in pairs:
                unique_sessions.add(existing)
                unique_sessions.add(candidate)

            t_scan = time.time()
            for sess_label in unique_sessions:
                bris_h, bris_m = parse_session_label(sess_label)
                session_cache[sess_label] = scan_session_per_day(
                    highs, lows, closes, bris_h, bris_m
                )

            print(f"    {len(unique_sessions)} sessions scanned in "
                  f"{time.time() - t_scan:.1f}s")

            # Process each pair
            inst_rows = []
            for existing, candidate in pairs:
                existing_days = session_cache[existing]
                candidate_days_data = session_cache[candidate]

                # Determine DST type for this pair
                e_dst = SESSION_DST_TYPE.get(existing, "CLEAN")
                c_dst = SESSION_DST_TYPE.get(candidate, "CLEAN")

                # Fail-closed: conflicting DST types (US vs UK) would make
                # the split meaningless — abort rather than produce bad data
                if (e_dst != "CLEAN" and c_dst != "CLEAN"
                        and e_dst != c_dst):
                    raise ValueError(
                        f"Conflicting DST types in pair {existing}({e_dst}) "
                        f"vs {candidate}({c_dst}). Cannot determine split mask."
                    )

                # Use the affected session's DST type
                if e_dst != "CLEAN":
                    pair_dst_type = e_dst
                elif c_dst != "CLEAN":
                    pair_dst_type = c_dst
                else:
                    pair_dst_type = "CLEAN"

                # Pick DST mask
                if pair_dst_type == "US":
                    dst_mask = us_mask
                elif pair_dst_type == "UK":
                    dst_mask = uk_mask
                else:
                    dst_mask = None

                if pair_dst_type == "CLEAN":
                    # --- CLEAN pair: single ALL row ---
                    metrics = compute_overlap_metrics(existing_days, candidate_days_data)
                    if metrics is None:
                        print(f"    {existing}->{candidate}: no common G4+ days, skipping")
                        continue

                    classification = classify_overlap(
                        metrics["shared_break_pct"], metrics["r_correlation"]
                    )

                    row = {
                        "instrument": instrument,
                        "existing_session": existing,
                        "candidate_session": candidate,
                        "dst_regime": "ALL",
                        **metrics,
                        "classification": classification,
                    }
                    inst_rows.append(row)
                    all_rows.append(row)

                else:
                    # --- DST-affected pair: WINTER + SUMMER only (no blended ALL) ---
                    pair_has_data = False
                    for regime_name, regime_test in [("WINTER", False), ("SUMMER", True)]:
                        e_filtered = {d: r for d, r in existing_days.items()
                                      if dst_mask[d] == regime_test}
                        c_filtered = {d: r for d, r in candidate_days_data.items()
                                      if dst_mask[d] == regime_test}

                        regime_metrics = compute_overlap_metrics(e_filtered, c_filtered)
                        if regime_metrics is None:
                            continue

                        pair_has_data = True
                        regime_class = classify_overlap(
                            regime_metrics["shared_break_pct"],
                            regime_metrics["r_correlation"],
                        )

                        regime_row = {
                            "instrument": instrument,
                            "existing_session": existing,
                            "candidate_session": candidate,
                            "dst_regime": regime_name,
                            **regime_metrics,
                            "classification": regime_class,
                        }
                        inst_rows.append(regime_row)
                        all_rows.append(regime_row)

                    if not pair_has_data:
                        print(f"    {existing}->{candidate}: no common G4+ days in either regime, skipping")

            if inst_rows:
                print_instrument_table(inst_rows, instrument)

            del opens, highs, lows, closes

    finally:
        con.close()

    if not all_rows:
        print("\nNo results to report.")
        return

    # Save CSV
    output_dir = Path("research/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "overlap_analysis.csv"

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  CSV saved: {csv_path} ({len(df)} rows)")

    # Honest summary
    print_honest_summary(all_rows)

    print(f"\n  Total runtime: {time.time() - t_total:.1f}s")
    print()


if __name__ == "__main__":
    main()

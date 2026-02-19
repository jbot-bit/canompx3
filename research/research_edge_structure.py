#!/usr/bin/env python3
"""
Edge structure analysis — three structural questions about overlap results.

Phase 0A found ALL 11 candidate pairs in GREY-ZONE: 95-100% shared break-days
but R-correlation only 0.03-0.24. Same days breaking, different outcomes.

This script answers three structural questions before any pipeline changes:

Q1: Break Window Sensitivity
    Is the 100% shared-break overlap an artifact of the 240-min window?
    Test 3 CLEAN pairs at [30, 60, 120, 180, 240, 360] min windows.

Q2: ORB Size Distribution by Time
    Does the edge come from WHEN you trade or HOW BIG the ORB is?
    All 16 sessions × 3 instruments, with size-band avgR.

Q3: ORB Size Correlation Between Low-R-Corr Pairs
    Are different outcomes because of different risk structures (ORB sizes)?
    Pairs from overlap_analysis.csv where r<0.3 AND shared>90%.

Usage:
  python research/research_edge_structure.py
  python research/research_edge_structure.py --db-path C:/db/gold.db
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

G4_MIN = 4.0
OUTCOME_WINDOW = 480  # 8 hours in minutes
RR_TARGET = 2.0

# DST type for each session label
SESSION_DST_TYPE = {
    "0900": "US",
    "0930": "US",
    "1000": "CLEAN",
    "1015": "CLEAN",
    "1100": "CLEAN",
    "1130": "CLEAN",
    "1245": "CLEAN",
    "1545": "UK",
    "1615": "UK",
    "1645": "UK",
    "1800": "UK",
    "1815": "UK",
    "1900": "CLEAN",
    "1915": "CLEAN",
    "2300": "US",
    "0030": "US",
}

ALL_SESSIONS = list(SESSION_DST_TYPE.keys())
INSTRUMENTS = ["MGC", "MNQ", "MES"]


# =========================================================================
# Data Loading (copied from research_overlap_analysis.py — standalone)
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


# =========================================================================
# Per-Day Scan Engine (break_window parameterized)
# =========================================================================

def parse_session_label(label):
    """Parse session label like '0930' or '1815' into (bris_h, bris_m)."""
    h = int(label[:2])
    m = int(label[2:])
    return h, m


def scan_session_per_day(highs, lows, closes, bris_h, bris_m,
                         aperture_min=5, break_window=240):
    """Scan one session across all trading days, returning per-day results.

    Returns dict[day_index -> per-day result dict].
    """
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

    g4_mask = valid_orb & (orb_size >= G4_MIN)

    results = {}

    for day_idx in range(n_days):
        if not valid_orb[day_idx]:
            continue

        if not g4_mask[day_idx]:
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
                "valid_orb": True,
                "g4_pass": True,
                "orb_size": float(os_val),
                "broke": False,
                "direction": None,
                "outcome_r": np.nan,
                "entry": np.nan,
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
# Overlap Metrics (copied from research_overlap_analysis.py)
# =========================================================================

def compute_overlap_metrics(existing_days, candidate_days):
    """Compute overlap metrics between two sessions' per-day results."""
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

    for d in common_g4:
        e = existing_days[d]
        c = candidate_days[d]
        e_broke = e["broke"]
        c_broke = c["broke"]

        if e_broke and c_broke:
            n_both_broke += 1
            existing_r_shared.append(e["outcome_r"])
            candidate_r_shared.append(c["outcome_r"])
        elif e_broke and not c_broke:
            n_existing_only += 1
        elif not e_broke and c_broke:
            n_candidate_only += 1
        else:
            n_neither += 1

    n_candidate_broke = n_both_broke + n_candidate_only

    shared_break_pct = (n_both_broke / n_candidate_broke
                        if n_candidate_broke > 0 else np.nan)

    r_correlation = np.nan
    r_pvalue = np.nan
    if HAS_SCIPY and n_both_broke >= 3:
        arr_e = np.array(existing_r_shared)
        arr_c = np.array(candidate_r_shared)
        if np.std(arr_e) > 0 and np.std(arr_c) > 0:
            r_correlation, r_pvalue = pearsonr(arr_e, arr_c)

    return {
        "n_common_g4": len(common_g4),
        "n_both_broke": n_both_broke,
        "n_candidate_only": n_candidate_only,
        "shared_break_pct": shared_break_pct,
        "r_correlation": r_correlation,
        "r_pvalue": r_pvalue,
    }


# =========================================================================
# Q1: Break Window Sensitivity
# =========================================================================

Q1_PAIRS = [
    ("MNQ", "1000", "1015"),
    ("MNQ", "1130", "1245"),
    ("MES", "1000", "1015"),
]

Q1_WINDOWS = [30, 60, 120, 180, 240, 360]


def window_sensitivity(data_cache):
    """Q1: Does overlap change with break window size?"""
    print(f"\n{'=' * 100}")
    print(f"  Q1: BREAK WINDOW SENSITIVITY")
    print(f"  Testing 3 CLEAN pairs at windows: {Q1_WINDOWS}")
    print(f"{'=' * 100}")

    rows = []

    for instrument, existing, candidate in Q1_PAIRS:
        all_days, _opens, highs, lows, closes, _us, _uk = data_cache[instrument]
        bh_e, bm_e = parse_session_label(existing)
        bh_c, bm_c = parse_session_label(candidate)

        print(f"\n  {instrument} {existing} vs {candidate}:")
        print(f"  {'Window':>8s} {'G4':>6s} {'Both':>6s} {'C_only':>7s} "
              f"{'Shrd%':>8s} {'r':>8s} {'r_p':>8s}")
        print(f"  {'-' * 55}")

        for bw in Q1_WINDOWS:
            e_results = scan_session_per_day(highs, lows, closes, bh_e, bm_e,
                                             break_window=bw)
            c_results = scan_session_per_day(highs, lows, closes, bh_c, bm_c,
                                             break_window=bw)
            metrics = compute_overlap_metrics(e_results, c_results)

            if metrics is None:
                print(f"  {bw:>6d}m   -- no common G4+ days --")
                continue

            sbp = metrics["shared_break_pct"]
            rc = metrics["r_correlation"]
            rp = metrics["r_pvalue"]

            sbp_s = f"{sbp * 100:6.1f}%" if not np.isnan(sbp) else "     --"
            rc_s = f"{rc:+7.3f}" if not np.isnan(rc) else "     --"
            rp_s = f"{rp:7.4f}" if not np.isnan(rp) else "     --"

            print(f"  {bw:>6d}m {metrics['n_common_g4']:6d} "
                  f"{metrics['n_both_broke']:6d} {metrics['n_candidate_only']:7d} "
                  f"{sbp_s} {rc_s} {rp_s}")

            rows.append({
                "instrument": instrument,
                "existing_session": existing,
                "candidate_session": candidate,
                "break_window": bw,
                "n_common_g4": metrics["n_common_g4"],
                "n_both_broke": metrics["n_both_broke"],
                "n_candidate_only": metrics["n_candidate_only"],
                "shared_break_pct": sbp,
                "r_correlation": rc,
                "r_pvalue": rp,
            })

    # Highlight thresholds
    print(f"\n  THRESHOLD ANALYSIS:")
    for instrument, existing, candidate in Q1_PAIRS:
        pair_rows = [r for r in rows
                     if r["instrument"] == instrument
                     and r["existing_session"] == existing
                     and r["candidate_session"] == candidate]
        below_80 = [r for r in pair_rows
                    if not np.isnan(r["shared_break_pct"])
                    and r["shared_break_pct"] < 0.80]
        below_50 = [r for r in pair_rows
                    if not np.isnan(r["shared_break_pct"])
                    and r["shared_break_pct"] < 0.50]

        t80 = f"{below_80[0]['break_window']}m" if below_80 else "never"
        t50 = f"{below_50[0]['break_window']}m" if below_50 else "never"
        print(f"    {instrument} {existing}v{candidate}: "
              f"drops below 80% at {t80}, below 50% at {t50}")

    return rows


# =========================================================================
# Q2: ORB Size Distribution by Time
# =========================================================================

def size_distribution(data_cache):
    """Q2: Does the edge come from WHEN or HOW BIG?"""
    print(f"\n{'=' * 100}")
    print(f"  Q2: ORB SIZE DISTRIBUTION BY TIME")
    print(f"  16 sessions × 3 instruments, size-band avgR comparison")
    print(f"{'=' * 100}")

    rows = []

    for instrument in INSTRUMENTS:
        all_days, _opens, highs, lows, closes, us_mask, uk_mask = data_cache[instrument]

        print(f"\n  {instrument}")
        print(f"  {'Sess':>6s} {'DST':>7s} {'N_orb':>6s} {'MedSz':>7s} {'MnSz':>7s} "
              f"{'%G4':>5s} {'%G5':>5s} {'%G6':>5s} {'%G8':>5s} "
              f"{'G4-6R':>7s} {'N':>4s} {'G6-8R':>7s} {'N':>4s} "
              f"{'G8+R':>7s} {'N':>4s}")
        print(f"  {'-' * 98}")

        for session in ALL_SESSIONS:
            bh, bm = parse_session_label(session)
            per_day = scan_session_per_day(highs, lows, closes, bh, bm,
                                           break_window=240)

            dst_type = SESSION_DST_TYPE[session]

            if dst_type == "CLEAN":
                regimes = [("ALL", None)]
            elif dst_type == "US":
                regimes = [("WINTER", False), ("SUMMER", True)]
            else:  # UK
                regimes = [("WINTER", False), ("SUMMER", True)]

            for regime_name, regime_test in regimes:
                # Filter to regime
                if regime_test is None:
                    regime_days = per_day
                else:
                    mask = us_mask if dst_type == "US" else uk_mask
                    regime_days = {d: r for d, r in per_day.items()
                                   if mask[d] == regime_test}

                # Valid ORB days (any valid_orb, regardless of G4)
                valid_orb_days = {d: r for d, r in regime_days.items()
                                  if r["valid_orb"]}
                n_valid_orb = len(valid_orb_days)

                if n_valid_orb == 0:
                    continue

                # ORB sizes for all valid days
                sizes = np.array([r["orb_size"] for r in valid_orb_days.values()])
                median_orb = float(np.median(sizes))
                mean_orb = float(np.mean(sizes))

                # Grade percentages (fraction of valid-ORB days at each threshold)
                pct_g4 = float(np.mean(sizes >= 4.0))
                pct_g5 = float(np.mean(sizes >= 5.0))
                pct_g6 = float(np.mean(sizes >= 6.0))
                pct_g8 = float(np.mean(sizes >= 8.0))

                # Size-band avgR (only on break-days within band)
                def band_stats(lo, hi):
                    band_days = [r for r in regime_days.values()
                                 if r["g4_pass"] and r["broke"]
                                 and r["orb_size"] >= lo
                                 and (r["orb_size"] < hi if hi is not None
                                      else True)]
                    if not band_days:
                        return np.nan, 0
                    avg_r = float(np.mean([r["outcome_r"] for r in band_days]))
                    return avg_r, len(band_days)

                g4g6_r, g4g6_n = band_stats(4.0, 6.0)
                g6g8_r, g6g8_n = band_stats(6.0, 8.0)
                g8p_r, g8p_n = band_stats(8.0, None)

                # Console output
                def _fr(v):
                    return f"{v:+7.3f}" if not np.isnan(v) else "     --"

                print(f"  {session:>6s} {regime_name:>7s} {n_valid_orb:6d} "
                      f"{median_orb:7.2f} {mean_orb:7.2f} "
                      f"{pct_g4 * 100:4.0f}% {pct_g5 * 100:4.0f}% "
                      f"{pct_g6 * 100:4.0f}% {pct_g8 * 100:4.0f}% "
                      f"{_fr(g4g6_r)} {g4g6_n:4d} "
                      f"{_fr(g6g8_r)} {g6g8_n:4d} "
                      f"{_fr(g8p_r)} {g8p_n:4d}")

                rows.append({
                    "instrument": instrument,
                    "session": session,
                    "dst_regime": regime_name,
                    "n_valid_orb": n_valid_orb,
                    "median_orb_size": median_orb,
                    "mean_orb_size": mean_orb,
                    "pct_g4": pct_g4,
                    "pct_g5": pct_g5,
                    "pct_g6": pct_g6,
                    "pct_g8": pct_g8,
                    "band_g4g6_avgR": g4g6_r,
                    "band_g4g6_n": g4g6_n,
                    "band_g6g8_avgR": g6g8_r,
                    "band_g6g8_n": g6g8_n,
                    "band_g8p_avgR": g8p_r,
                    "band_g8p_n": g8p_n,
                })

    # Key comparison
    print(f"\n  KEY COMPARISON: At same size band, do different sessions produce different avgR?")
    for instrument in INSTRUMENTS:
        inst_rows = [r for r in rows if r["instrument"] == instrument]
        # G6-G8 band comparison (most common actionable band)
        g6g8_rows = [(r["session"], r["dst_regime"], r["band_g6g8_avgR"], r["band_g6g8_n"])
                     for r in inst_rows if r["band_g6g8_n"] >= 20]
        if g6g8_rows:
            g6g8_rows.sort(key=lambda x: x[2] if not np.isnan(x[2]) else -999,
                           reverse=True)
            print(f"\n    {instrument} G6-G8 band (N>=20):")
            for sess, regime, avgr, n in g6g8_rows:
                print(f"      {sess:>6s} [{regime:>7s}]: avgR={avgr:+.3f} (N={n})")

    return rows


# =========================================================================
# Q3: ORB Size Correlation Between Low-R-Corr Pairs
# =========================================================================

def size_correlation(data_cache):
    """Q3: Are different outcomes because of different ORB sizes or noise?"""
    print(f"\n{'=' * 100}")
    print(f"  Q3: ORB SIZE CORRELATION BETWEEN LOW-R-CORR PAIRS")
    print(f"{'=' * 100}")

    # Load overlap_analysis.csv
    csv_path = Path("research/output/overlap_analysis.csv")
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found. Run research_overlap_analysis.py first.")
        return []

    oa_df = pd.read_csv(csv_path,
                        dtype={"existing_session": str, "candidate_session": str})
    # Pad session labels back to 4 chars (CSV may drop leading zeros)
    for col in ("existing_session", "candidate_session"):
        oa_df[col] = oa_df[col].str.zfill(4)
    # Filter: r_correlation < 0.3 (or NaN) AND shared_break_pct > 0.90
    qualifying = oa_df[
        ((oa_df["r_correlation"].isna()) | (oa_df["r_correlation"].abs() < 0.3))
        & (oa_df["shared_break_pct"] > 0.90)
    ]

    # Get unique pairs (instrument, existing, candidate)
    pair_keys = qualifying[["instrument", "existing_session", "candidate_session"]].drop_duplicates()
    n_pairs = len(pair_keys)
    print(f"  Qualifying pairs (|r|<0.3 AND shared>90%): {n_pairs}")

    if n_pairs == 0:
        print("  No qualifying pairs found.")
        return []

    rows = []

    print(f"\n  {'Inst':>4s} {'Exist':>6s} {'Cand':>6s} {'DST':>7s} "
          f"{'N_shr':>6s} {'OrbSzR':>7s} {'OrbSzP':>8s} "
          f"{'MnAbsDf':>8s} {'SzConc':>7s}")
    print(f"  {'-' * 72}")

    for _, pair in pair_keys.iterrows():
        instrument = pair["instrument"]
        existing = pair["existing_session"]
        candidate = pair["candidate_session"]

        if instrument not in data_cache:
            continue

        all_days, _opens, highs, lows, closes, us_mask, uk_mask = data_cache[instrument]
        bh_e, bm_e = parse_session_label(existing)
        bh_c, bm_c = parse_session_label(candidate)

        e_results = scan_session_per_day(highs, lows, closes, bh_e, bm_e,
                                         break_window=240)
        c_results = scan_session_per_day(highs, lows, closes, bh_c, bm_c,
                                         break_window=240)

        # Determine DST type
        e_dst = SESSION_DST_TYPE.get(existing, "CLEAN")
        c_dst = SESSION_DST_TYPE.get(candidate, "CLEAN")

        if e_dst != "CLEAN":
            pair_dst_type = e_dst
        elif c_dst != "CLEAN":
            pair_dst_type = c_dst
        else:
            pair_dst_type = "CLEAN"

        if pair_dst_type == "CLEAN":
            regimes = [("ALL", None)]
        elif pair_dst_type == "US":
            regimes = [("WINTER", False), ("SUMMER", True)]
        else:
            regimes = [("WINTER", False), ("SUMMER", True)]

        for regime_name, regime_test in regimes:
            if regime_test is None:
                e_filt = e_results
                c_filt = c_results
            else:
                mask = us_mask if pair_dst_type == "US" else uk_mask
                e_filt = {d: r for d, r in e_results.items()
                          if mask[d] == regime_test}
                c_filt = {d: r for d, r in c_results.items()
                          if mask[d] == regime_test}

            # Shared break-days (both G4+ AND broke)
            shared_days = []
            for d in set(e_filt.keys()) & set(c_filt.keys()):
                e = e_filt[d]
                c = c_filt[d]
                if (e["g4_pass"] and e["broke"]
                        and c["g4_pass"] and c["broke"]):
                    shared_days.append((e["orb_size"], c["orb_size"]))

            n_shared = len(shared_days)
            if n_shared < 3:
                continue

            e_sizes = np.array([s[0] for s in shared_days])
            c_sizes = np.array([s[1] for s in shared_days])

            # Pearson r on ORB sizes
            orb_r = np.nan
            orb_p = np.nan
            if HAS_SCIPY and np.std(e_sizes) > 0 and np.std(c_sizes) > 0:
                orb_r, orb_p = pearsonr(e_sizes, c_sizes)

            # Mean absolute size difference
            mean_abs_diff = float(np.mean(np.abs(e_sizes - c_sizes)))

            # Size direction concordance: both above or both below respective medians
            e_med = np.median(e_sizes)
            c_med = np.median(c_sizes)
            concordant = np.sum(
                ((e_sizes >= e_med) & (c_sizes >= c_med))
                | ((e_sizes < e_med) & (c_sizes < c_med))
            )
            concordance = float(concordant / n_shared)

            # Console
            orb_r_s = f"{orb_r:+7.3f}" if not np.isnan(orb_r) else "     --"
            orb_p_s = f"{orb_p:8.4f}" if not np.isnan(orb_p) else "      --"

            print(f"  {instrument:>4s} {existing:>6s} {candidate:>6s} "
                  f"{regime_name:>7s} {n_shared:6d} "
                  f"{orb_r_s} {orb_p_s} {mean_abs_diff:8.2f} "
                  f"{concordance:6.1%}")

            rows.append({
                "instrument": instrument,
                "existing_session": existing,
                "candidate_session": candidate,
                "dst_regime": regime_name,
                "n_shared_breaks": n_shared,
                "orb_size_r": orb_r,
                "orb_size_p": orb_p,
                "mean_abs_size_diff": mean_abs_diff,
                "size_direction_concordance": concordance,
            })

    return rows


# =========================================================================
# Honest Summary
# =========================================================================

def print_honest_summary(q1_rows, q2_rows, q3_rows):
    """Print honest summary of all three questions."""
    print(f"\n{'=' * 100}")
    print(f"  HONEST SUMMARY")
    print(f"{'=' * 100}")

    # --- Q1 verdict ---
    print(f"\n  Q1 VERDICT — BREAK WINDOW SENSITIVITY:")
    if q1_rows:
        # Check overlap at smallest window
        min_window = min(Q1_WINDOWS)
        at_min = [r for r in q1_rows if r["break_window"] == min_window]
        at_240 = [r for r in q1_rows if r["break_window"] == 240]

        if at_min and at_240:
            avg_min_sbp = np.nanmean([r["shared_break_pct"] for r in at_min])
            avg_240_sbp = np.nanmean([r["shared_break_pct"] for r in at_240])
            drop = avg_240_sbp - avg_min_sbp

            if avg_min_sbp > 0.80:
                print(f"    REAL OVERLAP — even at {min_window}min window, "
                      f"shared break = {avg_min_sbp * 100:.1f}% "
                      f"(vs {avg_240_sbp * 100:.1f}% at 240min)")
            elif avg_min_sbp < 0.50:
                print(f"    ARTIFACT — at {min_window}min window, "
                      f"shared break drops to {avg_min_sbp * 100:.1f}% "
                      f"(from {avg_240_sbp * 100:.1f}% at 240min)")
            else:
                print(f"    PARTIAL ARTIFACT — at {min_window}min window, "
                      f"shared break = {avg_min_sbp * 100:.1f}% "
                      f"(from {avg_240_sbp * 100:.1f}% at 240min)")

            # Per-pair details
            for instrument, existing, candidate in Q1_PAIRS:
                pair_min = [r for r in at_min
                            if r["instrument"] == instrument
                            and r["existing_session"] == existing]
                pair_240 = [r for r in at_240
                            if r["instrument"] == instrument
                            and r["existing_session"] == existing]
                if pair_min and pair_240:
                    m_sbp = pair_min[0]["shared_break_pct"]
                    f_sbp = pair_240[0]["shared_break_pct"]
                    m_s = f"{m_sbp * 100:.1f}%" if not np.isnan(m_sbp) else "N/A"
                    f_s = f"{f_sbp * 100:.1f}%" if not np.isnan(f_sbp) else "N/A"
                    print(f"      {instrument} {existing}v{candidate}: "
                          f"{min_window}min={m_s}, 240min={f_s}")
        else:
            print(f"    INSUFFICIENT DATA for verdict")
    else:
        print(f"    NO DATA")

    # --- Q2 verdict ---
    print(f"\n  Q2 VERDICT — SIZE vs TIME:")
    if q2_rows:
        # Compare avgR at G6-G8 band across sessions
        g6g8 = [(r["instrument"], r["session"], r["dst_regime"],
                 r["band_g6g8_avgR"], r["band_g6g8_n"])
                for r in q2_rows if r["band_g6g8_n"] >= 20]

        if g6g8:
            avgrs = [r[3] for r in g6g8 if not np.isnan(r[3])]
            if avgrs:
                spread = max(avgrs) - min(avgrs)
                best = max(g6g8, key=lambda x: x[3] if not np.isnan(x[3]) else -999)
                worst = min(g6g8, key=lambda x: x[3] if not np.isnan(x[3]) else 999)

                if spread > 0.3:
                    print(f"    TIME MATTERS TOO — at G6-G8, avgR spread = {spread:.3f}")
                elif spread < 0.1:
                    print(f"    SIZE IS EVERYTHING — at G6-G8, avgR spread = {spread:.3f} "
                          f"(sessions converge)")
                else:
                    print(f"    MIXED — at G6-G8, avgR spread = {spread:.3f}")

                print(f"      Best:  {best[0]} {best[1]} [{best[2]}] "
                      f"avgR={best[3]:+.3f} (N={best[4]})")
                print(f"      Worst: {worst[0]} {worst[1]} [{worst[2]}] "
                      f"avgR={worst[3]:+.3f} (N={worst[4]})")
            else:
                print(f"    INSUFFICIENT DATA (no G6-G8 rows with N>=20)")
        else:
            print(f"    INSUFFICIENT DATA (no G6-G8 rows with N>=20)")
    else:
        print(f"    NO DATA")

    # --- Q3 verdict ---
    print(f"\n  Q3 VERDICT — ORB SIZE CORRELATION:")
    if q3_rows:
        valid = [r for r in q3_rows if not np.isnan(r["orb_size_r"])]
        if valid:
            avg_orb_r = np.mean([r["orb_size_r"] for r in valid])
            avg_abs_diff = np.mean([r["mean_abs_size_diff"] for r in valid])
            avg_conc = np.mean([r["size_direction_concordance"] for r in valid])

            if avg_orb_r > 0.7:
                print(f"    SAME RISK STRUCTURE — orb_size_r = {avg_orb_r:+.3f} "
                      f"(sizes move together)")
            elif avg_orb_r < 0.3:
                print(f"    GENUINE DIVERSIFIERS — orb_size_r = {avg_orb_r:+.3f} "
                      f"(different ORB sizes on same days)")
            else:
                print(f"    MODERATE CORRELATION — orb_size_r = {avg_orb_r:+.3f}")

            print(f"      Mean |size_diff| = {avg_abs_diff:.2f}, "
                  f"size concordance = {avg_conc:.1%}")
        else:
            print(f"    INSUFFICIENT DATA (no valid r values)")
    else:
        print(f"    NO DATA")

    # --- Combined implications ---
    print(f"\n  COMBINED IMPLICATIONS:")
    print(f"    [See Q1/Q2/Q3 verdicts above for data-driven conclusions]")
    print(f"    If Q1=REAL, Q2=SIZE, Q3=SAME: sessions are truly redundant")
    print(f"    If Q1=ARTIFACT, Q2=TIME, Q3=DIVERSIFIERS: new sessions add value")
    print(f"    Mixed results -> session-specific decisions needed")

    # --- Caveats ---
    print(f"\n  CAVEATS:")
    print(f"    - IN-SAMPLE analysis (no walk-forward)")
    print(f"    - E1 entry + CB1 only (5min aperture)")
    print(f"    - RR2.0 only")
    print(f"    - DST-affected sessions report WINTER and SUMMER separately (no blended number)")
    print(f"    - Q3 pair selection based on overlap_analysis.csv thresholds (|r|<0.3, shared>90%)")

    # --- Next steps ---
    print(f"\n  NEXT STEPS:")
    print(f"    - If overlap is window-artifact: tighter break windows may reveal independence")
    print(f"    - If size explains everything: single best-time per size band suffices")
    print(f"    - If time matters at same size: adding sessions has structural justification")
    print(f"    - Review DST-split results for regime-specific divergence")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Edge structure analysis — three structural questions about overlap results."
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

    print(f"\n{'=' * 100}")
    print(f"  EDGE STRUCTURE ANALYSIS — Three Structural Questions")
    print(f"  Database: {db_path}")
    print(f"  Parameters: G{G4_MIN:.0f}+ filter | RR{RR_TARGET:.1f} target | "
          f"{OUTCOME_WINDOW // 60}h outcome window")
    print(f"  scipy available: {HAS_SCIPY}")
    print(f"{'=' * 100}")

    con = duckdb.connect(str(db_path), read_only=True)
    t_total = time.time()

    try:
        # Load all instruments upfront, cache data
        data_cache = {}
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
            del bars_df
            us_mask, uk_mask = build_dst_masks(all_days)
            print(f"    {len(all_days)} trading days built in {time.time() - t_build:.1f}s")

            data_cache[instrument] = (all_days, opens, highs, lows, closes,
                                      us_mask, uk_mask)

        if not data_cache:
            print("\n  No instrument data loaded. Exiting.")
            return

        # Run Q1, Q2, Q3
        q1_rows = window_sensitivity(data_cache)
        q2_rows = size_distribution(data_cache)
        q3_rows = size_correlation(data_cache)

        # Save CSVs
        output_dir = Path("research/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        if q1_rows:
            df1 = pd.DataFrame(q1_rows)
            p1 = output_dir / "edge_structure_window_sensitivity.csv"
            df1.to_csv(p1, index=False, float_format="%.4f")
            print(f"\n  Q1 CSV saved: {p1} ({len(df1)} rows)")

        if q2_rows:
            df2 = pd.DataFrame(q2_rows)
            p2 = output_dir / "edge_structure_size_distribution.csv"
            df2.to_csv(p2, index=False, float_format="%.4f")
            print(f"  Q2 CSV saved: {p2} ({len(df2)} rows)")

        if q3_rows:
            df3 = pd.DataFrame(q3_rows)
            p3 = output_dir / "edge_structure_size_correlation.csv"
            df3.to_csv(p3, index=False, float_format="%.4f")
            print(f"  Q3 CSV saved: {p3} ({len(df3)} rows)")

        # Honest summary
        print_honest_summary(q1_rows, q2_rows, q3_rows)

        print(f"\n  Total runtime: {time.time() - t_total:.1f}s")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()

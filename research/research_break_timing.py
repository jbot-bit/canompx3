#!/usr/bin/env python3
"""
Break Timing Research: Does WHEN the ORB breaks predict outcome quality?

Investigates whether early breaks (first 5-15 min) outperform late breaks
(60+ min) within a session window. Uses the pre-computed
orb_{label}_break_delay_min column from daily_features.

No look-ahead concern: break timing is known at entry for all entry models
(E1, E2, E3).

Read-only: no writes to gold.db.

Output:
  research/output/break_timing_detail.csv   -- per-test detail
  research/output/break_timing_summary.md   -- findings summary
  Console output

Usage:
    python research/research_break_timing.py
    python research/research_break_timing.py --db-path C:/db/gold.db
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
from scipy import stats

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, get_enabled_sessions
from pipeline.paths import GOLD_DB_PATH

# Force unbuffered stdout (Windows)
sys.stdout.reconfigure(line_buffering=True)


# -- Configuration --------------------------------------------------------

# Timing buckets: (label, lo_min_inclusive, hi_min_exclusive)
TIMING_BUCKETS = [
    ("IMM",   0,    5),
    ("EARLY", 5,   15),
    ("MID_E", 15,  30),
    ("MID_L", 30,  60),
    ("LATE",  60, 99999),
]

# Representative combos: (entry_model, rr_target, confirm_bars)
# One outcome per day per combo -> valid t-test (no N-inflation)
REPRESENTATIVE_COMBOS = [
    ("E2", 2.0, 1),
    ("E2", 2.5, 1),
    ("E1", 2.0, 2),
    ("E1", 2.5, 2),
]

# Cutoff thresholds for fast vs slow split (minutes)
CUTOFF_THRESHOLDS = [5, 10, 15, 20, 30, 45, 60]

# Minimum group size for t-tests
MIN_GROUP_SIZE = 30

# G4 filter minimum ORB size (match validated strategy universe)
G4_MIN_SIZE = 4.0

# BH FDR q-value
FDR_Q = 0.05


# -- Helpers --------------------------------------------------------------

def classify_bucket(delay_min: float) -> str:
    """Classify break delay into a timing bucket label."""
    for label, lo, hi in TIMING_BUCKETS:
        if lo <= delay_min < hi:
            return label
    return "LATE"


def compute_metrics(pnl_rs: list[float]) -> dict:
    """Compute summary metrics from a list of pnl_r values."""
    n = len(pnl_rs)
    if n == 0:
        return {"n": 0, "avg_r": None, "total_r": 0.0, "wr": None, "sharpe": None}

    avg_r = float(np.mean(pnl_rs))
    total_r = float(np.sum(pnl_rs))
    wins = sum(1 for r in pnl_rs if r > 0)
    wr = wins / n

    if n >= 2:
        std = float(np.std(pnl_rs, ddof=1))
        sharpe = avg_r / std if std > 0 else None
    else:
        sharpe = None

    return {
        "n": n,
        "avg_r": round(avg_r, 4),
        "total_r": round(total_r, 2),
        "wr": round(wr, 4),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
    }


def apply_bh_fdr(results: list[dict], q: float = FDR_Q) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction. Returns survivors."""
    if not results:
        return []
    results_sorted = sorted(results, key=lambda r: r["p_val"])
    m = len(results_sorted)
    survivors = []

    print(f"\n{'=' * 80}")
    print(f"  BH FDR Correction ({m} tests, q={q})")
    print(f"{'=' * 80}")

    for i, r in enumerate(results_sorted):
        rank = i + 1
        bh_threshold = q * rank / m
        survives = r["p_val"] <= bh_threshold
        r["bh_rank"] = rank
        r["bh_threshold"] = bh_threshold
        r["bh_survives"] = survives
        tag = "SURVIVES" if survives else "rejected"
        print(f"  [{rank}/{m}] p={r['p_val']:.6f} vs BH={bh_threshold:.6f} "
              f"-> {tag}  ({r['label']})")
        if survives:
            survivors.append(r)

    return survivors


def year_by_year_check(delays: np.ndarray, pnl_rs: np.ndarray,
                       trading_days, cutoff: float, overall_direction: float) -> tuple[int, int]:
    """Check year-by-year consistency for a cutoff split.

    Returns (years_consistent, years_total).
    """
    # Extract years
    years_arr = np.array([
        d.year if hasattr(d, 'year') else int(str(d)[:4])
        for d in trading_days
    ])
    unique_years = sorted(set(years_arr))

    consistent = 0
    total = 0

    for year in unique_years:
        mask_yr = years_arr == year
        yr_delays = delays[mask_yr]
        yr_pnls = pnl_rs[mask_yr]

        fast_mask = yr_delays <= cutoff
        slow_mask = yr_delays > cutoff

        n_fast = fast_mask.sum()
        n_slow = slow_mask.sum()

        if n_fast >= 5 and n_slow >= 5:
            yr_delta = float(np.mean(yr_pnls[fast_mask]) - np.mean(yr_pnls[slow_mask]))
            total += 1
            if (yr_delta > 0) == (overall_direction > 0):
                consistent += 1

    return consistent, total


def year_by_year_detail(delays: np.ndarray, pnl_rs: np.ndarray,
                        trading_days, cutoff: float) -> list[dict]:
    """Detailed year-by-year breakdown for a cutoff split."""
    years_arr = np.array([
        d.year if hasattr(d, 'year') else int(str(d)[:4])
        for d in trading_days
    ])
    unique_years = sorted(set(years_arr))
    detail = []

    for year in unique_years:
        mask_yr = years_arr == year
        yr_delays = delays[mask_yr]
        yr_pnls = pnl_rs[mask_yr]

        fast_mask = yr_delays <= cutoff
        slow_mask = yr_delays > cutoff

        n_fast = int(fast_mask.sum())
        n_slow = int(slow_mask.sum())

        entry = {"year": year, "n_fast": n_fast, "n_slow": n_slow}

        if n_fast >= 5 and n_slow >= 5:
            avg_fast = float(np.mean(yr_pnls[fast_mask]))
            avg_slow = float(np.mean(yr_pnls[slow_mask]))
            entry["avg_r_fast"] = round(avg_fast, 4)
            entry["avg_r_slow"] = round(avg_slow, 4)
            entry["delta"] = round(avg_fast - avg_slow, 4)
        else:
            entry["avg_r_fast"] = None
            entry["avg_r_slow"] = None
            entry["delta"] = None

        detail.append(entry)

    return detail


# -- Data Loading ---------------------------------------------------------

def load_data(con, instrument: str, session: str) -> list[dict]:
    """Load joined daily_features + orb_outcomes for one (instrument, session).

    Applies:
      - orb_minutes = 5 (standard ORB aperture for research)
      - G4 filter (orb_size >= 4.0)
      - outcome in ('win', 'loss')
      - break_delay_min IS NOT NULL

    Returns list of dicts, one per (trading_day, entry_model, rr_target, confirm_bars).
    """
    delay_col = f"orb_{session}_break_delay_min"
    size_col = f"orb_{session}_size"

    # Check if column exists first
    try:
        col_check = con.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'daily_features'
            AND column_name = '{delay_col}'
        """).fetchall()
        if not col_check:
            return []
    except Exception:
        return []

    rows = []
    for entry_model, rr_target, confirm_bars in REPRESENTATIVE_COMBOS:
        result = con.execute(f"""
            SELECT
                o.trading_day,
                o.symbol,
                o.orb_label,
                o.entry_model,
                o.rr_target,
                o.confirm_bars,
                o.outcome,
                o.pnl_r,
                df.{delay_col} AS break_delay_min,
                df.{size_col} AS orb_size
            FROM orb_outcomes o
            JOIN daily_features df
              ON o.trading_day = df.trading_day
              AND o.symbol = df.symbol
              AND o.orb_minutes = df.orb_minutes
            WHERE o.symbol = ?
              AND o.orb_label = ?
              AND o.orb_minutes = 5
              AND o.entry_model = ?
              AND o.rr_target = ?
              AND o.confirm_bars = ?
              AND o.outcome IN ('win', 'loss')
              AND o.pnl_r IS NOT NULL
              AND df.{delay_col} IS NOT NULL
              AND df.{size_col} >= {G4_MIN_SIZE}
            ORDER BY o.trading_day
        """, [instrument, session, entry_model, rr_target, confirm_bars]).fetchall()

        col_names = [
            "trading_day", "symbol", "orb_label", "entry_model",
            "rr_target", "confirm_bars", "outcome", "pnl_r",
            "break_delay_min", "orb_size",
        ]

        for row in result:
            d = dict(zip(col_names, row))
            d["combo_tag"] = f"{entry_model}_RR{rr_target}_CB{confirm_bars}"
            d["bucket"] = classify_bucket(d["break_delay_min"])
            rows.append(d)

    return rows


# -- Analysis Functions ---------------------------------------------------

def analyze_distribution(rows: list[dict], instrument: str, session: str) -> dict:
    """Compute empirical distribution of break_delay_min for a session."""
    delays = [r["break_delay_min"] for r in rows]
    if not delays:
        return {}

    # Deduplicate to unique trading days (avoid counting same day 4x for 4 combos)
    seen_days = set()
    unique_delays = []
    for r in rows:
        if r["trading_day"] not in seen_days:
            seen_days.add(r["trading_day"])
            unique_delays.append(r["break_delay_min"])

    arr = np.array(unique_delays)

    dist = {
        "instrument": instrument,
        "session": session,
        "n_unique_days": len(unique_delays),
        "mean": round(float(np.mean(arr)), 1),
        "median": round(float(np.median(arr)), 1),
        "std": round(float(np.std(arr, ddof=1)), 1) if len(arr) > 1 else 0,
        "p5": round(float(np.percentile(arr, 5)), 1),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "max": round(float(np.max(arr)), 1),
    }

    # Bucket counts
    bucket_counts = defaultdict(int)
    for d in unique_delays:
        bucket_counts[classify_bucket(d)] += 1

    dist["bucket_counts"] = dict(bucket_counts)
    return dist


def run_bucket_test(rows: list[dict], bucket_label: str,
                    combo_tag: str, instrument: str, session: str) -> dict | None:
    """Welch's t-test: bucket vs rest for one combo.

    Returns result dict or None if insufficient data.
    """
    combo_rows = [r for r in rows if r["combo_tag"] == combo_tag]
    if not combo_rows:
        return None

    in_bucket = [r["pnl_r"] for r in combo_rows if r["bucket"] == bucket_label]
    rest = [r["pnl_r"] for r in combo_rows if r["bucket"] != bucket_label]

    if len(in_bucket) < MIN_GROUP_SIZE or len(rest) < MIN_GROUP_SIZE:
        return None

    arr_in = np.array(in_bucket)
    arr_rest = np.array(rest)

    t_stat, p_val = stats.ttest_ind(arr_in, arr_rest, equal_var=False)
    mean_in = float(np.mean(arr_in))
    mean_rest = float(np.mean(arr_rest))
    delta = mean_in - mean_rest

    label = f"BUCKET {bucket_label} vs REST | {combo_tag} | {instrument} {session}"

    return {
        "test_type": "bucket",
        "label": label,
        "instrument": instrument,
        "session": session,
        "combo_tag": combo_tag,
        "bucket": bucket_label,
        "n_in": len(in_bucket),
        "n_rest": len(rest),
        "mean_in": round(mean_in, 4),
        "mean_rest": round(mean_rest, 4),
        "delta": round(delta, 4),
        "t_stat": round(t_stat, 4),
        "p_val": p_val,
    }


def run_spearman_test(rows: list[dict], combo_tag: str,
                      instrument: str, session: str) -> dict | None:
    """Spearman rank correlation: break_delay vs pnl_r for monotonic trend."""
    combo_rows = [r for r in rows if r["combo_tag"] == combo_tag]
    if len(combo_rows) < MIN_GROUP_SIZE:
        return None

    delays = np.array([r["break_delay_min"] for r in combo_rows])
    pnl_rs = np.array([r["pnl_r"] for r in combo_rows])

    rho, p_val = stats.spearmanr(delays, pnl_rs)

    label = f"SPEARMAN delay~pnl_r | {combo_tag} | {instrument} {session}"

    return {
        "test_type": "spearman",
        "label": label,
        "instrument": instrument,
        "session": session,
        "combo_tag": combo_tag,
        "n": len(combo_rows),
        "rho": round(float(rho), 4),
        "p_val": p_val,
    }


def run_cutoff_test(rows: list[dict], cutoff_min: float,
                    combo_tag: str, instrument: str, session: str) -> dict | None:
    """Welch's t-test: fast (<= cutoff) vs slow (> cutoff) split."""
    combo_rows = [r for r in rows if r["combo_tag"] == combo_tag]
    if not combo_rows:
        return None

    fast = [r["pnl_r"] for r in combo_rows if r["break_delay_min"] <= cutoff_min]
    slow = [r["pnl_r"] for r in combo_rows if r["break_delay_min"] > cutoff_min]

    if len(fast) < MIN_GROUP_SIZE or len(slow) < MIN_GROUP_SIZE:
        return None

    arr_fast = np.array(fast)
    arr_slow = np.array(slow)

    t_stat, p_val = stats.ttest_ind(arr_fast, arr_slow, equal_var=False)
    mean_fast = float(np.mean(arr_fast))
    mean_slow = float(np.mean(arr_slow))
    delta = mean_fast - mean_slow

    label = f"CUTOFF {cutoff_min}min | {combo_tag} | {instrument} {session}"

    # Year-by-year for context
    delays_arr = np.array([r["break_delay_min"] for r in combo_rows])
    pnl_arr = np.array([r["pnl_r"] for r in combo_rows])
    tdays = [r["trading_day"] for r in combo_rows]
    yr_consistent, yr_total = year_by_year_check(
        delays_arr, pnl_arr, tdays, cutoff_min, delta
    )

    return {
        "test_type": "cutoff",
        "label": label,
        "instrument": instrument,
        "session": session,
        "combo_tag": combo_tag,
        "cutoff_min": cutoff_min,
        "n_fast": len(fast),
        "n_slow": len(slow),
        "mean_fast": round(mean_fast, 4),
        "mean_slow": round(mean_slow, 4),
        "delta": round(delta, 4),
        "t_stat": round(t_stat, 4),
        "p_val": p_val,
        "yr_consistent": yr_consistent,
        "yr_total": yr_total,
    }


# -- Main -----------------------------------------------------------------

def run_research(db_path: Path) -> None:
    output_dir = PROJECT_ROOT / "research" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []      # All t-test / cutoff results for BH FDR
    spearman_results = []  # Spearman tests (separate pool)
    distributions = []     # Per-session delay distributions
    detail_rows = []       # CSV detail rows

    with duckdb.connect(str(db_path), read_only=True) as con:
        for instrument in ACTIVE_ORB_INSTRUMENTS:
            sessions = get_enabled_sessions(instrument)
            if not sessions:
                print(f"\n  {instrument}: no enabled sessions, skipping.")
                continue

            for session in sessions:
                print(f"\n{'#' * 70}")
                print(f"#  {instrument} {session}")
                print(f"{'#' * 70}")

                # Load data
                rows = load_data(con, instrument, session)
                if not rows:
                    print(f"  No data (G4+ breaks with resolved outcomes). Skipping.")
                    continue

                # Unique day count
                unique_days = len(set(r["trading_day"] for r in rows))
                combo_tags = sorted(set(r["combo_tag"] for r in rows))
                print(f"  Loaded {len(rows)} outcomes across {unique_days} unique break days")
                print(f"  Combos: {combo_tags}")

                # -- Distribution ------------------------------------------
                dist = analyze_distribution(rows, instrument, session)
                if dist:
                    distributions.append(dist)
                    print(f"\n  Break Delay Distribution ({dist['n_unique_days']} days):")
                    print(f"    Mean={dist['mean']}m, Median={dist['median']}m, "
                          f"Std={dist['std']}m")
                    print(f"    P5={dist['p5']}m, P25={dist['p25']}m, "
                          f"P75={dist['p75']}m, P95={dist['p95']}m, Max={dist['max']}m")
                    print(f"    Buckets: ", end="")
                    for bl, _, _ in TIMING_BUCKETS:
                        cnt = dist["bucket_counts"].get(bl, 0)
                        pct = cnt / dist["n_unique_days"] * 100 if dist["n_unique_days"] > 0 else 0
                        print(f"{bl}={cnt}({pct:.0f}%) ", end="")
                    print()

                # -- Per-combo analysis ------------------------------------
                for combo_tag in combo_tags:
                    combo_rows = [r for r in rows if r["combo_tag"] == combo_tag]
                    n_combo = len(combo_rows)
                    if n_combo < MIN_GROUP_SIZE:
                        print(f"\n  {combo_tag}: N={n_combo} < {MIN_GROUP_SIZE}, skipping.")
                        continue

                    combo_m = compute_metrics([r["pnl_r"] for r in combo_rows])
                    print(f"\n  {combo_tag}: N={n_combo}, avgR={combo_m['avg_r']}, "
                          f"WR={combo_m['wr']}, Sharpe={combo_m['sharpe']}")

                    # -- Bucket breakdown ----------------------------------
                    print(f"    {'Bucket':<8} {'N':>5} {'avgR':>8} {'WR':>7} {'Sharpe':>8}")
                    print(f"    {'-' * 40}")
                    for bl, _, _ in TIMING_BUCKETS:
                        bucket_rows = [r for r in combo_rows if r["bucket"] == bl]
                        if not bucket_rows:
                            continue
                        bm = compute_metrics([r["pnl_r"] for r in bucket_rows])
                        wr_str = f"{bm['wr']:.1%}" if bm['wr'] is not None else "N/A"
                        avg_str = f"{bm['avg_r']}" if bm['avg_r'] is not None else "N/A"
                        sh_str = f"{bm['sharpe']}" if bm['sharpe'] is not None else "N/A"
                        print(f"    {bl:<8} {bm['n']:>5} {avg_str:>8} {wr_str:>7} {sh_str:>8}")

                    # -- Bucket t-tests ------------------------------------
                    for bl, _, _ in TIMING_BUCKETS:
                        result = run_bucket_test(rows, bl, combo_tag, instrument, session)
                        if result:
                            all_results.append(result)
                            sig = " *" if result["p_val"] < 0.05 else ""
                            print(f"    {bl} vs rest: delta={result['delta']:+.4f}, "
                                  f"p={result['p_val']:.4f}{sig}")
                            detail_rows.append(result)

                    # -- Spearman correlation ------------------------------
                    sp = run_spearman_test(rows, combo_tag, instrument, session)
                    if sp:
                        spearman_results.append(sp)
                        sig = " *" if sp["p_val"] < 0.05 else ""
                        print(f"    Spearman: rho={sp['rho']:+.4f}, p={sp['p_val']:.4f}{sig}")

                    # -- Cutoff scans --------------------------------------
                    print(f"    Cutoff scan:")
                    print(f"      {'Cut':>5} {'N_fast':>7} {'N_slow':>7} {'avg_fast':>9} "
                          f"{'avg_slow':>9} {'delta':>8} {'p':>8} {'yr':>6}")
                    for cutoff in CUTOFF_THRESHOLDS:
                        result = run_cutoff_test(
                            rows, cutoff, combo_tag, instrument, session
                        )
                        if result:
                            all_results.append(result)
                            sig = " *" if result["p_val"] < 0.05 else ""
                            print(f"      {cutoff:>5} {result['n_fast']:>7} {result['n_slow']:>7} "
                                  f"{result['mean_fast']:>+9.4f} {result['mean_slow']:>+9.4f} "
                                  f"{result['delta']:>+8.4f} {result['p_val']:>8.4f}{sig}"
                                  f"  {result['yr_consistent']}/{result['yr_total']}")
                            detail_rows.append(result)
                        else:
                            # Not enough data for this cutoff
                            pass

    # -- BH FDR Correction -------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print(f"STATISTICAL CORRECTION")
    print(f"{'=' * 80}")

    # Separate bucket and cutoff tests
    bucket_tests = [r for r in all_results if r.get("test_type") == "bucket"]
    cutoff_tests = [r for r in all_results if r.get("test_type") == "cutoff"]

    print(f"\nTotal tests: {len(all_results)} ({len(bucket_tests)} bucket, "
          f"{len(cutoff_tests)} cutoff)")
    print(f"Spearman tests (separate, not in FDR pool): {len(spearman_results)}")

    # Apply BH FDR across ALL bucket + cutoff tests together
    survivors = apply_bh_fdr(all_results, q=FDR_Q)

    # -- Spearman summary --------------------------------------------------
    if spearman_results:
        print(f"\n{'=' * 80}")
        print(f"  SPEARMAN RANK CORRELATIONS (monotonic trend test)")
        print(f"{'=' * 80}")
        for sp in sorted(spearman_results, key=lambda x: x["p_val"]):
            sig = " *" if sp["p_val"] < 0.05 else ""
            print(f"  rho={sp['rho']:+.4f}, p={sp['p_val']:.4f}{sig}, "
                  f"N={sp['n']}  {sp['label']}")

    # -- Year-by-year for BH survivors ------------------------------------
    if survivors:
        print(f"\n{'=' * 80}")
        print(f"  YEAR-BY-YEAR FOR BH SURVIVORS ({len(survivors)} tests)")
        print(f"{'=' * 80}")

        # Re-load data for year-by-year deep dive
        with duckdb.connect(str(db_path), read_only=True) as con:
            for surv in survivors:
                inst = surv["instrument"]
                sess = surv["session"]
                combo = surv["combo_tag"]
                rows = load_data(con, inst, sess)
                combo_rows = [r for r in rows if r["combo_tag"] == combo]

                if surv["test_type"] == "cutoff":
                    cutoff = surv["cutoff_min"]
                    print(f"\n  --- {surv['label']} ---")
                    delays_arr = np.array([r["break_delay_min"] for r in combo_rows])
                    pnl_arr = np.array([r["pnl_r"] for r in combo_rows])
                    tdays = [r["trading_day"] for r in combo_rows]

                    detail = year_by_year_detail(delays_arr, pnl_arr, tdays, cutoff)
                    for yd in detail:
                        if yd["delta"] is not None:
                            print(f"    {yd['year']}: fast N={yd['n_fast']}, "
                                  f"slow N={yd['n_slow']}, "
                                  f"avgR_fast={yd['avg_r_fast']:+.4f}, "
                                  f"avgR_slow={yd['avg_r_slow']:+.4f}, "
                                  f"delta={yd['delta']:+.4f}")
                        else:
                            print(f"    {yd['year']}: fast N={yd['n_fast']}, "
                                  f"slow N={yd['n_slow']} (too few for split)")

                elif surv["test_type"] == "bucket":
                    bucket = surv["bucket"]
                    print(f"\n  --- {surv['label']} ---")
                    # Year-by-year bucket vs rest
                    years_arr = np.array([
                        r["trading_day"].year if hasattr(r["trading_day"], 'year')
                        else int(str(r["trading_day"])[:4])
                        for r in combo_rows
                    ])
                    for year in sorted(set(years_arr)):
                        yr_rows = [r for r, y in zip(combo_rows, years_arr) if y == year]
                        in_b = [r["pnl_r"] for r in yr_rows if r["bucket"] == bucket]
                        out_b = [r["pnl_r"] for r in yr_rows if r["bucket"] != bucket]
                        if len(in_b) >= 5 and len(out_b) >= 5:
                            delta = float(np.mean(in_b) - np.mean(out_b))
                            print(f"    {year}: in_bucket N={len(in_b)}, "
                                  f"rest N={len(out_b)}, delta={delta:+.4f}")
                        else:
                            print(f"    {year}: in_bucket N={len(in_b)}, "
                                  f"rest N={len(out_b)} (too few)")

    # -- Connection to existing BRK_FAST5/FAST10 ---------------------------
    print(f"\n{'=' * 80}")
    print(f"  CONNECTION TO EXISTING FILTERS")
    print(f"{'=' * 80}")

    relevant_cutoffs_5 = [r for r in all_results
                          if r.get("test_type") == "cutoff" and r.get("cutoff_min") == 5]
    relevant_cutoffs_10 = [r for r in all_results
                           if r.get("test_type") == "cutoff" and r.get("cutoff_min") == 10]

    if relevant_cutoffs_5:
        print(f"\n  BRK_FAST5 (5 min cutoff) — existing filter in discovery grid:")
        for rc in sorted(relevant_cutoffs_5, key=lambda x: x["p_val"]):
            sig = " *" if rc.get("bh_survives") else ""
            print(f"    p={rc['p_val']:.4f}, delta={rc['delta']:+.4f}, "
                  f"N_fast={rc['n_fast']}, N_slow={rc['n_slow']}  "
                  f"{rc['instrument']} {rc['session']} {rc['combo_tag']}{sig}")

    if relevant_cutoffs_10:
        print(f"\n  BRK_FAST10 (10 min cutoff) — existing filter in discovery grid:")
        for rc in sorted(relevant_cutoffs_10, key=lambda x: x["p_val"]):
            sig = " *" if rc.get("bh_survives") else ""
            print(f"    p={rc['p_val']:.4f}, delta={rc['delta']:+.4f}, "
                  f"N_fast={rc['n_fast']}, N_slow={rc['n_slow']}  "
                  f"{rc['instrument']} {rc['session']} {rc['combo_tag']}{sig}")

    # Find the best cutoff across all tests
    cutoff_tests_with_p = [r for r in cutoff_tests if r["p_val"] < 0.05]
    if cutoff_tests_with_p:
        print(f"\n  Best raw cutoffs (p < 0.05, before FDR):")
        for ct in sorted(cutoff_tests_with_p, key=lambda x: x["p_val"])[:10]:
            surv = " [BH SURVIVOR]" if ct.get("bh_survives") else ""
            print(f"    {ct['cutoff_min']}min: p={ct['p_val']:.4f}, "
                  f"delta={ct['delta']:+.4f}, yr={ct['yr_consistent']}/{ct['yr_total']} "
                  f"  {ct['instrument']} {ct['session']} {ct['combo_tag']}{surv}")

    # -- Summary -----------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n  Tests conducted: {len(all_results)} (bucket + cutoff)")
    print(f"  Spearman tests: {len(spearman_results)}")
    print(f"  Raw p < 0.05: {sum(1 for r in all_results if r['p_val'] < 0.05)}")
    print(f"  BH FDR survivors (q={FDR_Q}): {len(survivors)}")

    if not survivors:
        print(f"\n  VERDICT: NO BH FDR SURVIVORS.")
        print(f"  Break timing does NOT predict outcome quality after multiple-testing correction.")
        print(f"  The existing BRK_FAST5/BRK_FAST10 filters may be capturing noise, not signal.")
        print(f"  Recommendation: WATCH ONLY. Do not expand break speed filters.")
    else:
        print(f"\n  VERDICT: {len(survivors)} BH FDR SURVIVOR(S).")
        for s in survivors:
            yr_str = ""
            if "yr_consistent" in s and "yr_total" in s:
                yr_str = f", yr={s['yr_consistent']}/{s['yr_total']}"
            print(f"    {s['label']}: p={s['p_val']:.6f}, delta={s['delta']:+.4f}{yr_str}")

    # -- Distribution summary table ----------------------------------------
    if distributions:
        print(f"\n  Session Window Break Delay Distributions:")
        print(f"  {'Instrument':<6} {'Session':<16} {'N':>5} {'Mean':>6} {'Med':>5} "
              f"{'P25':>5} {'P75':>5} {'P95':>6} {'Max':>6}")
        for d in distributions:
            print(f"  {d['instrument']:<6} {d['session']:<16} {d['n_unique_days']:>5} "
                  f"{d['mean']:>6.1f} {d['median']:>5.1f} {d['p25']:>5.1f} "
                  f"{d['p75']:>5.1f} {d['p95']:>6.1f} {d['max']:>6.1f}")

    # -- Write detail CSV --------------------------------------------------
    detail_path = output_dir / "break_timing_detail.csv"
    if detail_rows:
        fieldnames = sorted(set().union(*(r.keys() for r in detail_rows)))
        with open(detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in sorted(detail_rows, key=lambda x: x.get("p_val", 1)):
                writer.writerow(row)
        print(f"\n  Detail CSV: {detail_path}")

    # -- Write summary markdown --------------------------------------------
    summary_path = output_dir / "break_timing_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Break Timing Research Results\n\n")
        f.write(f"**Date:** 2026-03-01\n")
        f.write(f"**Script:** `research/research_break_timing.py`\n\n")

        f.write("## Hypothesis\n\n")
        f.write("**H0:** ORB break outcomes are independent of when the break occurs.\n")
        f.write("**H1:** Early and late breaks have systematically different outcomes.\n\n")

        f.write("## Method\n\n")
        f.write(f"- Instruments: {', '.join(ACTIVE_ORB_INSTRUMENTS)}\n")
        f.write(f"- Filter: G4+ (orb_size >= {G4_MIN_SIZE})\n")
        f.write(f"- ORB aperture: 5 minutes\n")
        f.write(f"- Representative combos: {len(REPRESENTATIVE_COMBOS)}\n")
        f.write(f"- Timing buckets: {', '.join(bl for bl, _, _ in TIMING_BUCKETS)}\n")
        f.write(f"- Cutoff thresholds: {CUTOFF_THRESHOLDS} minutes\n")
        f.write(f"- Tests: {len(all_results)} (bucket + cutoff), "
                f"{len(spearman_results)} Spearman\n")
        f.write(f"- Correction: BH FDR at q={FDR_Q}\n\n")

        f.write("## Break Delay Distributions\n\n")
        if distributions:
            f.write("| Instrument | Session | N | Mean | Median | P25 | P75 | P95 | Max |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for d in distributions:
                f.write(f"| {d['instrument']} | {d['session']} | {d['n_unique_days']} | "
                        f"{d['mean']} | {d['median']} | {d['p25']} | {d['p75']} | "
                        f"{d['p95']} | {d['max']} |\n")
        f.write("\n")

        f.write("## Results\n\n")
        f.write(f"- Total tests: {len(all_results)}\n")
        f.write(f"- Raw p < 0.05: {sum(1 for r in all_results if r['p_val'] < 0.05)}\n")
        f.write(f"- **BH FDR survivors: {len(survivors)}**\n\n")

        if survivors:
            f.write("### BH FDR Survivors\n\n")
            f.write("| Test | p-value | BH threshold | Delta (R) | Year consistency |\n")
            f.write("|---|---|---|---|---|\n")
            for s in survivors:
                yr = f"{s.get('yr_consistent', '?')}/{s.get('yr_total', '?')}" if "yr_consistent" in s else "N/A"
                # Escape pipe characters in label to avoid breaking markdown table
                safe_label = s['label'].replace('|', '/')
                f.write(f"| {safe_label} | {s['p_val']:.6f} | {s.get('bh_threshold', 0):.6f} | "
                        f"{s['delta']:+.4f} | {yr} |\n")
            f.write("\n")
        else:
            f.write("**No BH FDR survivors.** Break timing does not predict outcome "
                    "quality after multiple-testing correction.\n\n")

        f.write("## Connection to Existing Filters\n\n")
        f.write("- `BRK_FAST5` (5 min) and `BRK_FAST10` (10 min) exist in `config.py`\n")
        f.write("- These are composite filters applied to CME_REOPEN, TOKYO_OPEN, LONDON_METALS\n")
        if not survivors:
            f.write("- **Finding: No BH FDR evidence supports these thresholds.** "
                    "Break speed filters may be capturing noise.\n")
            f.write("- Recommendation: WATCH ONLY. Do not expand break speed filters "
                    "without additional out-of-sample evidence.\n")
        else:
            surv_cutoffs = [s for s in survivors if s.get("test_type") == "cutoff"]
            if surv_cutoffs:
                best = min(surv_cutoffs, key=lambda x: x["p_val"])
                f.write(f"- **Best surviving cutoff: {best.get('cutoff_min', '?')} min** "
                        f"(p={best['p_val']:.6f}, delta={best['delta']:+.4f})\n")
        f.write("\n")

        f.write("## Spearman Correlations\n\n")
        if spearman_results:
            sig_sp = [s for s in spearman_results if s["p_val"] < 0.05]
            f.write(f"- {len(sig_sp)}/{len(spearman_results)} with p < 0.05 (raw)\n")
            if sig_sp:
                f.write("\n| Combo | Instrument | Session | rho | p-value | N |\n")
                f.write("|---|---|---|---|---|---|\n")
                for s in sorted(sig_sp, key=lambda x: x["p_val"]):
                    f.write(f"| {s['combo_tag']} | {s['instrument']} | {s['session']} | "
                            f"{s['rho']:+.4f} | {s['p_val']:.4f} | {s['n']} |\n")
        else:
            f.write("No Spearman tests passed minimum sample size.\n")
        f.write("\n")

        # Honest label per RESEARCH_RULES.md
        if not survivors:
            f.write("## Classification\n\n")
            f.write("**NO-GO** -- Break timing as a standalone predictor. "
                    "After BH FDR correction across all tests, no systematic relationship "
                    "between break delay and trade outcome quality survives.\n")
        else:
            f.write("## Classification\n\n")
            f.write("**Statistical observation** -- requires walk-forward validation "
                    "before deployment.\n")

    print(f"\n  Summary: {summary_path}")
    print(f"\n{'=' * 80}")
    print(f"  DONE")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Break timing research")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH,
                        help="Path to gold.db")
    args = parser.parse_args()

    run_research(args.db_path)


if __name__ == "__main__":
    main()

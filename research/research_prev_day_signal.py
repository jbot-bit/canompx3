#!/usr/bin/env python3
"""
Prior-day ORB outcome as a next-day entry signal.

Tests whether yesterday's session outcome (win/loss/scratch) predicts
today's pnl_r for the same session. This is a REAL-TIME observable —
yesterday's outcome is known before today's session opens.

Methodology:
  - LAG(orb_{session}_outcome) over daily_features (all calendar days)
    so prev_outcome = NULL means no break yesterday (true prior knowledge)
  - Current day must pass G4+ filter (orb_size / atr_20 >= 4.0)
  - Test each prev_outcome bucket with one-sample t-test (H0: avgR = 0)
  - Benjamini-Hochberg FDR correction across ALL tests (q = 0.10)
  - Year-by-year breakdown for any BH-surviving cell

Sessions tested: active validated sessions per instrument
  MGC:  0900, 1000, 1800
  MNQ:  0900, 1000, 1100, 1800
  MES:  0900, 1000

Usage:
  python research/research_prev_day_signal.py
  python research/research_prev_day_signal.py --db-path C:/db/gold.db
  python research/research_prev_day_signal.py --g6  # stricter G6+ filter
"""

import argparse
import math
import sys
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Representative production config per (instrument, session)
# (entry_model, rr_target, confirm_bars)
SESSION_CONFIGS = {
    ("MGC", "0900"): ("E1", 2.5, 2),
    ("MGC", "1000"): ("E1", 2.5, 2),
    ("MGC", "1800"): ("E1", 2.5, 2),
    ("MNQ", "0900"): ("E1", 2.5, 2),
    ("MNQ", "1000"): ("E1", 3.0, 2),
    ("MNQ", "1100"): ("E1", 2.0, 3),
    ("MNQ", "1800"): ("E1", 2.5, 3),
    ("MES", "0900"): ("E1", 2.5, 2),
    ("MES", "1000"): ("E1", 3.0, 2),
}

MIN_N = 30  # minimum per bucket for t-test
BH_Q  = 0.10

# G-filter thresholds in absolute ORB points (NOT ATR multiples)
# G4 = orb_size >= 4.0 pts,  G6 = orb_size >= 6.0 pts
G4_THRESHOLD = 4.0
G6_THRESHOLD = 6.0


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def se(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var / n)

def t_stat(vals):
    s = se(vals)
    if math.isnan(s) or s == 0:
        return float("nan")
    return mean(vals) / s

def p_from_t(t, df):
    """Two-tailed p-value from t-statistic (numerical approximation)."""
    if math.isnan(t) or df <= 0:
        return float("nan")
    # Use incomplete beta function approximation (Abramowitz & Stegun)
    x = df / (df + t * t)
    # Regularized incomplete beta I_x(a, b) with a=df/2, b=1/2
    # Use scipy if available, otherwise approximate
    try:
        from scipy.stats import t as t_dist
        return float(2 * t_dist.sf(abs(t), df))
    except ImportError:
        # Simple normal approximation for large df
        z = abs(t)
        p = 2 * (1 - _normal_cdf(z))
        return p

def _normal_cdf(z):
    """Approximate CDF of standard normal."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def bh_correct(tests, q=BH_Q):
    """
    Benjamini-Hochberg FDR correction.
    tests: list of (label, p_value) — skip NaN p-values
    Returns dict label -> p_bh (adjusted p), and set of surviving labels.
    """
    valid = [(lbl, p) for lbl, p in tests if not math.isnan(p)]
    m = len(valid)
    if m == 0:
        return {}, set()
    sorted_tests = sorted(valid, key=lambda x: x[1])
    adjusted = {}
    survivors = set()
    prev_p_bh = 1.0
    for k, (lbl, p) in enumerate(sorted_tests, start=1):
        threshold = k / m * q
        p_bh = min(prev_p_bh, p * m / k)
        adjusted[lbl] = p_bh
        if p <= threshold:
            survivors.add(lbl)
        prev_p_bh = p_bh
    return adjusted, survivors

def stars(p):
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(con, instrument, session, entry_model, rr_target, confirm_bars,
              min_orb_pts):
    """
    Returns rows: (trading_day, year, prev_outcome, pnl_r)
    prev_outcome: yesterday's outcome for this session (None/NULL = no break).
    Current day must have a break AND pass the G4+ filter.
    """
    orb_col = f"orb_{session}_outcome"
    size_col = f"orb_{session}_size"

    # Guard: check column exists
    cols = {r[0] for r in con.execute("DESCRIBE daily_features").fetchall()}
    if orb_col not in cols or size_col not in cols:
        return []

    sql = f"""
        WITH lag_feats AS (
            SELECT
                trading_day,
                symbol,
                {orb_col}                                      AS curr_outcome,
                LAG({orb_col}) OVER (
                    PARTITION BY symbol ORDER BY trading_day
                )                                              AS prev_outcome,
                {size_col}                                     AS orb_size_pts
            FROM daily_features
            WHERE orb_minutes = 5
              AND symbol = ?
        )
        SELECT
            l.trading_day,
            YEAR(l.trading_day)    AS yr,
            l.prev_outcome,
            o.pnl_r
        FROM lag_feats l
        JOIN orb_outcomes o
          ON  l.trading_day  = o.trading_day
          AND l.symbol        = o.symbol
          AND o.orb_label     = ?
          AND o.orb_minutes   = 5
          AND o.entry_model   = ?
          AND o.rr_target     = ?
          AND o.confirm_bars  = ?
          AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL
        WHERE l.curr_outcome IS NOT NULL
          AND l.orb_size_pts  IS NOT NULL
          AND l.orb_size_pts  >= ?
        ORDER BY l.trading_day
    """
    rows = con.execute(sql, [
        instrument, session, entry_model, rr_target, confirm_bars, min_orb_pts
    ]).fetchall()
    return rows  # (trading_day, yr, prev_outcome, pnl_r)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_cell(rows):
    """
    Group rows by prev_outcome bucket and compute per-bucket stats.
    Returns dict: bucket -> {n, avgR, se, t, p, pnl_list}
    Also returns baseline (all rows).
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    all_r = []
    for _day, _yr, prev, pnl in rows:
        bucket = prev if prev else "no_break"
        buckets[bucket].append(pnl)
        all_r.append(pnl)

    results = {}
    for bucket, vals in buckets.items():
        n = len(vals)
        avg = mean(vals)
        s = se(vals)
        t = t_stat(vals)
        p = p_from_t(t, n - 1) if n >= 2 else float("nan")
        results[bucket] = dict(n=n, avgR=avg, se=s, t=t, p=p, vals=vals)

    # Baseline
    n_all = len(all_r)
    results["_ALL"] = dict(
        n=n_all, avgR=mean(all_r), se=se(all_r),
        t=t_stat(all_r),
        p=p_from_t(t_stat(all_r), n_all - 1),
        vals=all_r,
    )
    return results


def year_by_year(rows, bucket):
    """Year-by-year stats for a specific prev_outcome bucket."""
    from collections import defaultdict
    by_year = defaultdict(list)
    for _day, yr, prev, pnl in rows:
        b = prev if prev else "no_break"
        if b == bucket:
            by_year[yr].append(pnl)

    out = []
    for yr in sorted(by_year):
        vals = by_year[yr]
        n = len(vals)
        avg = mean(vals)
        s = se(vals)
        t = t_stat(vals)
        p = p_from_t(t, n - 1) if n >= 2 else float("nan")
        out.append((yr, n, avg, s, t, p))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="C:/db/gold.db")
    parser.add_argument("--g6", action="store_true",
                        help="Use G6+ filter instead of G4+")
    args = parser.parse_args()

    min_orb_pts = G6_THRESHOLD if args.g6 else G4_THRESHOLD
    filter_name  = "G6+" if args.g6 else "G4+"

    con = duckdb.connect(args.db_path, read_only=True)

    print("=" * 72)
    print(f"PRIOR-DAY OUTCOME SIGNAL  |  filter={filter_name}  |  BH q={BH_Q}")
    print("=" * 72)
    print()
    print("Hypothesis: yesterday's same-session outcome (known before open)")
    print("predicts today's pnl_r. Real-time observable — no look-ahead.")
    print()

    all_tests = []      # (label, p_value) for BH
    cell_data  = {}     # label -> (rows, bucket, stats) for year-by-year

    # ------------------------------------------------------------------
    for (inst, sess), (em, rr, cb) in sorted(SESSION_CONFIGS.items()):
        rows = load_data(con, inst, sess, em, rr, cb, min_orb_pts)
        if not rows:
            continue

        cell_stats = analyse_cell(rows)
        baseline = cell_stats.get("_ALL", {})
        n_total = baseline.get("n", 0)

        print(f"  {inst} {sess}  E{em[1]} RR{rr} CB{cb} {filter_name}")
        print(f"    Baseline: N={n_total}  avgR={baseline.get('avgR',0):.3f}")

        for bucket in ["win", "loss", "scratch"]:
            bdata = cell_stats.get(bucket, {})
            n = bdata.get("n", 0)
            if n < MIN_N:
                print(f"    prev={bucket:<8}  N={n:4d}  (skip, N<{MIN_N})")
                continue
            avg = bdata["avgR"]
            t   = bdata["t"]
            p   = bdata["p"]
            lbl = f"{inst}_{sess}_prev={bucket}"
            all_tests.append((lbl, p))
            cell_data[lbl] = (rows, bucket, bdata)
            print(f"    prev={bucket:<8}  N={n:4d}  avgR={avg:+.3f}  "
                  f"SE={bdata['se']:.3f}  t={t:+.2f}  p={p:.3f}  {stars(p)}")
        print()

    # ------------------------------------------------------------------
    # BH correction
    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"BENJAMINI-HOCHBERG FDR  (q={BH_Q}, m={len(all_tests)} tests)")
    print("=" * 72)

    p_bh_map, survivors = bh_correct(all_tests, q=BH_Q)

    # Sort by raw p-value
    sorted_tests = sorted(all_tests, key=lambda x: x[1])
    print(f"\n{'Rank':>4}  {'Label':<40}  {'p_raw':>7}  {'p_bh':>7}  {'Status'}")
    print("-" * 72)
    for rank, (lbl, p) in enumerate(sorted_tests, start=1):
        p_bh = p_bh_map.get(lbl, float("nan"))
        status = "SURVIVE *" if lbl in survivors else ""
        print(f"{rank:4d}  {lbl:<40}  {p:7.4f}  {p_bh:7.4f}  {status}")

    if not survivors:
        print("\nNO tests survive BH FDR at q=0.10.")
        print("Conclusion: prior-day outcome has NO reliable edge after")
        print("multiple-comparison correction. Do NOT trade this signal.")
        con.close()
        return

    # ------------------------------------------------------------------
    # Year-by-year for survivors
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("YEAR-BY-YEAR STRESS TEST  (BH survivors only)")
    print("=" * 72)

    for lbl in sorted(survivors):
        if lbl not in cell_data:
            continue
        rows, bucket, bdata = cell_data[lbl]
        # Parse inst/sess from label: {inst}_{sess}_prev={bucket}
        parts = lbl.split("_")
        inst = parts[0]
        sess = parts[1]

        print(f"\n  {lbl}")
        print(f"  Aggregate: N={bdata['n']}  avgR={bdata['avgR']:+.3f}  "
              f"t={bdata['t']:+.2f}  p={bdata['p']:.4f}")
        print(f"  {'Year':>4}  {'N':>5}  {'avgR':>7}  {'SE':>6}  {'t':>6}  {'p':>6}")
        print(f"  {'-'*44}")

        yy = year_by_year(rows, bucket)
        pos_years = 0
        for yr, n, avg, s, t, p in yy:
            if n < 5:
                print(f"  {yr:4d}  {n:5d}  (skip, N<5)")
                continue
            sig = stars(p)
            direction = "+" if avg >= 0 else "-"
            if avg >= 0:
                pos_years += 1
            print(f"  {yr:4d}  {n:5d}  {avg:+7.3f}  {s:6.3f}  {t:+6.2f}  {p:6.3f}  {sig}")

        total_years = len([y for y in yy if y[1] >= 5])
        pct_pos = 100 * pos_years / total_years if total_years > 0 else 0
        print(f"\n  Positive years: {pos_years}/{total_years} ({pct_pos:.0f}%)")

        if pct_pos >= 75:
            print(f"  -> PROMISING: consistent across {pct_pos:.0f}% of years")
        elif pct_pos >= 50:
            print(f"  -> WATCH: majority-positive but not stable")
        else:
            print(f"  -> NO-GO: majority-negative years, aggregate is artifact")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print(f"Tests run: {len(all_tests)}")
    print(f"BH survivors at q={BH_Q}: {len(survivors)}")
    print()
    for lbl in sorted(survivors):
        bdata = cell_data.get(lbl, {})
        if len(bdata) >= 3:
            _, bucket, stats = cell_data[lbl]
            print(f"  [SURVIVED] {lbl}")
            print(f"             N={stats['n']}  avgR={stats['avgR']:+.3f}  "
                  f"p_bh={p_bh_map.get(lbl, float('nan')):.4f}")
    if not survivors:
        print("  NO survivors.")

    print()
    print("Interpretation guide:")
    print("  prev=win  avgR > baseline  => momentum: trade after wins")
    print("  prev=win  avgR < baseline  => reversion: skip after wins")
    print("  prev=loss avgR > baseline  => mean-rev: trade after losses")
    print("  Any surviving signal requires 75%+ positive years to be actionable.")

    con.close()


if __name__ == "__main__":
    main()

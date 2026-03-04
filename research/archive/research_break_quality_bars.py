"""
Break Quality Research: Pre-Break Compression & Explosion Ratio

Investigates whether bar-level compression/explosion patterns predict
trade outcomes for ORB breakout strategies.

Two metrics:
  Layer 1 - Pre-Break Compression = avg(pre_break_bar_ranges) / orb_range
    -> Universal (valid for all entry models including E0 CB1)
  Layer 2 - Explosion Ratio = break_bar_range / avg(pre_break_bar_ranges)
    -> Valid for E1, E3, E0 CB2+ (lookahead for E0 CB1)

Usage:
    python research/research_break_quality_bars.py
    python research/research_break_quality_bars.py --instrument MNQ --sessions 1100 LONDON_OPEN 0030
    python research/research_break_quality_bars.py --db-path C:/db/gold.db
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import duckdb
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_INSTRUMENT = "MGC"
DEFAULT_SESSIONS = ["0900", "1000"]

MIN_BARS_LAYER1 = 2   # >= 2 pre-break bars for compression ratio
MIN_BARS_LAYER2 = 1   # >= 1 pre-break bar for explosion ratio denominator

# Representative strategy combos: (entry_model, rr_target, confirm_bars)
# One outcome per day per combo → valid t-test (no N-inflation)
REPRESENTATIVE_COMBOS = [
    ("E1", 2.0, 2),    # E1 CB2 RR2.0 — most common baseline
    ("E0", 2.5, 1),    # E0 CB1 RR2.5 — top MGC 0900 strategy
    ("E0", 2.0, 1),    # E0 CB1 RR2.0
    ("E3", 2.0, 1),    # E3 CB1 RR2.0 — retrace entry
]

MIN_GROUP_SIZE = 30    # minimum per-group N for t-test


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_break_days(con, instrument, session):
    """Load all days with breaks for given session from daily_features."""
    return con.execute(f"""
        SELECT
            trading_day,
            orb_{session}_high   AS orb_high,
            orb_{session}_low    AS orb_low,
            orb_{session}_size   AS orb_size,
            orb_{session}_break_ts         AS break_ts,
            orb_{session}_break_dir        AS break_dir,
            orb_{session}_break_delay_min  AS break_delay_min,
            orb_{session}_break_bar_continues AS break_bar_continues,
            us_dst
        FROM daily_features
        WHERE symbol = ?
          AND orb_minutes = 5
          AND orb_{session}_break_dir IS NOT NULL
          AND orb_{session}_break_ts IS NOT NULL
        ORDER BY trading_day
    """, [instrument]).fetchdf()


def load_outcomes(con, instrument, session, entry_model, rr_target, confirm_bars):
    """Load outcomes for one (session, entry_model, rr, cb) combo.

    Returns at most one row per trading_day → no N-inflation.
    """
    return con.execute("""
        SELECT
            trading_day, outcome, pnl_r
        FROM orb_outcomes
        WHERE symbol = ?
          AND orb_label = ?
          AND orb_minutes = 5
          AND entry_model = ?
          AND rr_target = ?
          AND confirm_bars = ?
          AND outcome IN ('win', 'loss')
          AND pnl_r IS NOT NULL
        ORDER BY trading_day
    """, [instrument, session, entry_model, rr_target, confirm_bars]).fetchdf()


def load_all_bars_1m(con, instrument, start_date, end_date):
    """Bulk-load 1m bars for full date range (with 1-day buffer each side)."""
    utc_start = pd.Timestamp(start_date) - pd.Timedelta(days=2)
    utc_end = pd.Timestamp(end_date) + pd.Timedelta(days=2)

    bars = con.execute("""
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?::TIMESTAMPTZ
          AND ts_utc < ?::TIMESTAMPTZ
        ORDER BY ts_utc
    """, [instrument, utc_start.isoformat(), utc_end.isoformat()]).fetchdf()

    # Ensure timezone-aware
    if len(bars) > 0 and bars["ts_utc"].dt.tz is None:
        bars["ts_utc"] = bars["ts_utc"].dt.tz_localize("UTC")

    return bars


# ─── Metric Computation ─────────────────────────────────────────────────────

def compute_bar_metrics(bars_df, break_ts, orb_end_utc, orb_size):
    """Compute pre-break compression and explosion ratio for one day.

    Args:
        bars_df: Full bars_1m DataFrame (pre-filtered to relevant range)
        break_ts: Timestamp of the break bar (tz-aware UTC)
        orb_end_utc: Timestamp of ORB window end (derived from break_ts - delay)
        orb_size: ORB high - low (points)

    Returns dict with all computed metrics (None where insufficient data).
    """
    result = {
        "pre_break_bars": 0,
        "pre_break_compression": None,
        "explosion_ratio": None,
        "break_bar_range": None,
        "avg_pre_break_range": None,
        "immediate_break": False,
    }

    if orb_size is None or orb_size <= 0:
        return result

    # Pre-break bars: ORB end (inclusive) to break bar (exclusive)
    pre_mask = (bars_df["ts_utc"] >= orb_end_utc) & (bars_df["ts_utc"] < break_ts)
    pre_break = bars_df.loc[pre_mask]

    # Break bar itself
    break_mask = bars_df["ts_utc"] == break_ts
    break_bar = bars_df.loc[break_mask]

    n_pre = len(pre_break)
    result["pre_break_bars"] = n_pre
    result["immediate_break"] = (n_pre == 0)

    # Break bar range
    if len(break_bar) > 0:
        bb = break_bar.iloc[0]
        result["break_bar_range"] = float(bb["high"] - bb["low"])

    # Pre-break bar ranges
    if n_pre > 0:
        pre_ranges = (pre_break["high"] - pre_break["low"]).values.astype(float)
        avg_pre = float(np.mean(pre_ranges))
        result["avg_pre_break_range"] = avg_pre

        # Layer 1: compression ratio (>= 2 pre-break bars)
        if n_pre >= MIN_BARS_LAYER1 and orb_size > 0:
            result["pre_break_compression"] = avg_pre / orb_size

        # Layer 2: explosion ratio (>= 1 pre-break bar + break bar)
        if (n_pre >= MIN_BARS_LAYER2
                and avg_pre > 0
                and result["break_bar_range"] is not None):
            result["explosion_ratio"] = result["break_bar_range"] / avg_pre

    return result


def compute_all_metrics(break_days, bars_df):
    """Compute bar metrics for all break days.

    Returns DataFrame with one row per trading day + metric columns.
    """
    rows = []
    for _, day in break_days.iterrows():
        break_ts = day["break_ts"]
        delay_min = day["break_delay_min"]
        orb_size = day["orb_size"]

        if break_ts is None or delay_min is None or pd.isna(delay_min):
            continue

        # Ensure tz-aware
        if hasattr(break_ts, "tzinfo") and break_ts.tzinfo is None:
            break_ts = pd.Timestamp(break_ts, tz="UTC")

        # Derive ORB end: break_ts minus break delay
        orb_end_utc = break_ts - pd.Timedelta(minutes=float(delay_min))

        m = compute_bar_metrics(bars_df, break_ts, orb_end_utc, orb_size)
        m["trading_day"] = day["trading_day"]
        m["break_dir"] = day["break_dir"]
        m["break_delay_min"] = delay_min
        m["orb_size"] = orb_size
        m["us_dst"] = day.get("us_dst")
        rows.append(m)

    return pd.DataFrame(rows)


# ─── Statistical Analysis ───────────────────────────────────────────────────

def run_ttest(data, metric_col, label):
    """Run t-test and quartile analysis for one metric.

    Returns result dict or None if insufficient data.
    """
    valid = data[data[metric_col].notna()].copy()
    wins = valid[valid["outcome"] == "win"][metric_col].values
    losses = valid[valid["outcome"] == "loss"][metric_col].values

    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")
    print(f"  N = {len(valid)} ({len(wins)} W / {len(losses)} L)")

    if len(wins) < MIN_GROUP_SIZE or len(losses) < MIN_GROUP_SIZE:
        print(f"  SKIP: need >= {MIN_GROUP_SIZE} in each group")
        return None

    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(wins, losses, equal_var=False)
    mean_w = float(np.mean(wins))
    mean_l = float(np.mean(losses))
    delta = mean_w - mean_l

    print(f"  Mean (wins):   {mean_w:.4f}")
    print(f"  Mean (losses): {mean_l:.4f}")
    print(f"  Delta:         {delta:+.4f}")
    print(f"  t-stat:        {t_stat:.4f}")
    print(f"  p-value:       {p_val:.6f}")

    if p_val < 0.005:
        verdict = "DISCOVERY (p < 0.005)"
    elif p_val < 0.01:
        verdict = "ACTIONABLE (p < 0.01)"
    elif p_val < 0.05:
        verdict = "NOTABLE (p < 0.05)"
    else:
        verdict = "NOT SIGNIFICANT"
    print(f"  -> {verdict}")

    # Quartile breakdown
    try:
        quartiles = pd.qcut(valid[metric_col], 4, labels=["Q1", "Q2", "Q3", "Q4"],
                            duplicates="drop")
    except ValueError:
        quartiles = None

    if quartiles is not None:
        valid = valid.copy()
        valid["quartile"] = quartiles
        print(f"\n  Quartile Breakdown:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            qd = valid[valid["quartile"] == q]
            if len(qd) == 0:
                continue
            q_wr = (qd["outcome"] == "win").sum() / len(qd)
            q_pnl = qd["pnl_r"].mean()
            q_range = qd[metric_col]
            print(f"    {q}: N={len(qd):>4}, WR={q_wr:.1%}, avgR={q_pnl:+.3f}, "
                  f"range=[{q_range.min():.4f}, {q_range.max():.4f}]")

    # Year-by-year stability
    valid_yr = valid.copy()
    valid_yr["year"] = valid_yr["trading_day"].apply(
        lambda d: d.year if hasattr(d, "year") else int(str(d)[:4])
    )
    years_consistent = 0
    years_total = 0

    print(f"\n  Year-by-Year:")
    for year in sorted(valid_yr["year"].unique()):
        yr = valid_yr[valid_yr["year"] == year]
        yr_w = yr[yr["outcome"] == "win"][metric_col].values
        yr_l = yr[yr["outcome"] == "loss"][metric_col].values
        if len(yr_w) >= 5 and len(yr_l) >= 5:
            yr_delta = float(np.mean(yr_w) - np.mean(yr_l))
            _, yr_p = stats.ttest_ind(yr_w, yr_l, equal_var=False)
            years_total += 1
            # "Consistent" = same sign as overall delta
            if (yr_delta > 0) == (delta > 0):
                years_consistent += 1
            print(f"    {year}: N={len(yr):>4}, delta={yr_delta:+.4f}, p={yr_p:.4f}")
        else:
            print(f"    {year}: N={len(yr):>4} (too few for split)")

    if years_total > 0:
        pct = years_consistent / years_total
        print(f"  Years consistent: {years_consistent}/{years_total} ({pct:.0%})")

    return {
        "label": label,
        "n": len(valid),
        "t_stat": t_stat,
        "p_val": p_val,
        "mean_wins": mean_w,
        "mean_losses": mean_l,
        "delta": delta,
        "years_consistent": years_consistent,
        "years_total": years_total,
    }


def analyze_immediate_breaks(merged, metrics_df, session):
    """Analyze the immediate-break cohort (zero pre-break bars)."""
    imm_days = set(metrics_df[metrics_df["immediate_break"]]["trading_day"])
    imm = merged[merged["trading_day"].isin(imm_days)]
    non_imm = merged[~merged["trading_day"].isin(imm_days)]

    if len(imm) < 10:
        print(f"\n  Immediate-break cohort: only {len(imm)} trades, skipping.")
        return

    print(f"\n  --- IMMEDIATE BREAK COHORT ({session}) ---")
    print(f"  Days: {len(imm_days)}")

    imm_wr = (imm["outcome"] == "win").sum() / len(imm) if len(imm) > 0 else 0
    imm_pnl = imm["pnl_r"].mean()
    print(f"  Immediate:     N={len(imm):>4}, WR={imm_wr:.1%}, avgR={imm_pnl:+.3f}")

    if len(non_imm) >= 10:
        non_wr = (non_imm["outcome"] == "win").sum() / len(non_imm)
        non_pnl = non_imm["pnl_r"].mean()
        print(f"  Non-immediate: N={len(non_imm):>4}, WR={non_wr:.1%}, avgR={non_pnl:+.3f}")

        # Fisher exact test for WR difference
        a = int((imm["outcome"] == "win").sum())
        b = int((imm["outcome"] == "loss").sum())
        c = int((non_imm["outcome"] == "win").sum())
        d = int((non_imm["outcome"] == "loss").sum())
        _, fisher_p = stats.fisher_exact([[a, b], [c, d]])
        print(f"  Fisher exact p: {fisher_p:.6f}")


def print_distributions(metrics_df, session):
    """Print summary distributions for all metrics."""
    print(f"\n  --- Metric Distributions ({session}) ---")
    for col, label in [
        ("pre_break_compression", "Pre-Break Compression"),
        ("explosion_ratio", "Explosion Ratio"),
        ("pre_break_bars", "Pre-Break Bar Count"),
        ("break_bar_range", "Break Bar Range (pts)"),
    ]:
        vals = metrics_df[col].dropna()
        if len(vals) > 0:
            print(f"  {label}: N={len(vals)}, "
                  f"mean={vals.mean():.4f}, med={vals.median():.4f}, "
                  f"std={vals.std():.4f}, "
                  f"P5={vals.quantile(0.05):.4f}, P95={vals.quantile(0.95):.4f}")


def apply_bh_fdr(results):
    """Apply Benjamini-Hochberg FDR correction to a list of test results."""
    if not results:
        return
    # Sort by p-value
    results.sort(key=lambda r: r["p_val"])
    m = len(results)

    print(f"\n{'='*64}")
    print(f"  BH FDR Correction ({m} tests)")
    print(f"{'='*64}")

    for i, r in enumerate(results):
        rank = i + 1
        bh_threshold = 0.05 * rank / m
        survives = r["p_val"] <= bh_threshold
        tag = "SURVIVES" if survives else "rejected"
        print(f"  [{rank}/{m}] p={r['p_val']:.6f} vs BH={bh_threshold:.6f} "
              f"-> {tag}  ({r['label']})")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Break quality bar-level research"
    )
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT,
                        help="Instrument symbol (default: MGC)")
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="Session labels (default: 0900 1000)")
    args = parser.parse_args()

    instrument = args.instrument
    sessions = args.sessions or DEFAULT_SESSIONS

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)

    all_results = []

    try:
        for session in sessions:
            print(f"\n{'#'*70}")
            print(f"#  {instrument} {session} — Break Quality Bar-Level Analysis")
            print(f"{'#'*70}")

            # ── Load break days ──────────────────────────────────────────
            break_days = load_break_days(con, instrument, session)
            if len(break_days) == 0:
                print(f"  No break days found for {session}")
                continue

            start_date = break_days["trading_day"].min()
            end_date = break_days["trading_day"].max()
            print(f"  Date range: {start_date} to {end_date}")
            print(f"  Break days: {len(break_days)}")

            # ── Bulk load bars ───────────────────────────────────────────
            bars = load_all_bars_1m(con, instrument, start_date, end_date)
            if len(bars) == 0:
                print(f"  No 1m bars found!")
                continue
            print(f"  Loaded {len(bars):,} 1m bars")

            # ── Compute metrics ──────────────────────────────────────────
            metrics_df = compute_all_metrics(break_days, bars)
            print(f"  Computed metrics for {len(metrics_df)} days")

            n_immediate = int(metrics_df["immediate_break"].sum())
            n_l1 = int(metrics_df["pre_break_compression"].notna().sum())
            n_l2 = int(metrics_df["explosion_ratio"].notna().sum())
            print(f"  Immediate breaks (0 pre-break bars): "
                  f"{n_immediate} ({n_immediate/len(metrics_df):.1%})")
            print(f"  Layer 1 computable (>= {MIN_BARS_LAYER1} pre-break bars): {n_l1}")
            print(f"  Layer 2 computable (>= {MIN_BARS_LAYER2} pre-break bars + break bar): {n_l2}")

            print_distributions(metrics_df, session)

            # ── Per-combo analysis ───────────────────────────────────────
            for em, rr, cb in REPRESENTATIVE_COMBOS:
                outcomes = load_outcomes(con, instrument, session, em, rr, cb)
                if len(outcomes) == 0:
                    continue

                merged = outcomes.merge(metrics_df, on="trading_day", how="inner")
                if len(merged) < 20:
                    continue

                combo_tag = f"{em}_RR{rr}_CB{cb}"

                # Layer 1: pre-break compression (ALL entry types)
                r = run_ttest(
                    merged, "pre_break_compression",
                    f"L1 Compression | {combo_tag} | {session}"
                )
                if r:
                    all_results.append(r)

                # Layer 2: explosion ratio
                if em == "E0" and cb == 1:
                    # E0 CB1: descriptive only (lookahead)
                    print(f"\n  ** E0 CB1 explosion ratio is DESCRIPTIVE (lookahead) **")
                    r = run_ttest(
                        merged, "explosion_ratio",
                        f"L2 Explosion [DESCRIPTIVE] | {combo_tag} | {session}"
                    )
                    # Don't add to FDR pool — not a filter candidate
                else:
                    r = run_ttest(
                        merged, "explosion_ratio",
                        f"L2 Explosion | {combo_tag} | {session}"
                    )
                    if r:
                        all_results.append(r)

            # ── DST split (0900 only) ────────────────────────────────────
            if session == "0900" and "us_dst" in metrics_df.columns:
                for dst_val, dst_label in [(True, "DST-ON"), (False, "DST-OFF")]:
                    dst_days = set(
                        metrics_df[metrics_df["us_dst"] == dst_val]["trading_day"]
                    )
                    # Use E1 RR2.0 CB2 as baseline for DST analysis
                    outcomes = load_outcomes(con, instrument, session, "E1", 2.0, 2)
                    if len(outcomes) == 0:
                        continue
                    merged = outcomes.merge(metrics_df, on="trading_day", how="inner")
                    dst_merged = merged[merged["trading_day"].isin(dst_days)]
                    if len(dst_merged) >= 40:
                        r = run_ttest(
                            dst_merged, "pre_break_compression",
                            f"L1 Compression | E1_RR2.0_CB2 | {session} {dst_label}"
                        )
                        if r:
                            all_results.append(r)

            # ── Immediate-break cohort ───────────────────────────────────
            # Use E1 RR2.0 CB2 as baseline
            outcomes = load_outcomes(con, instrument, session, "E1", 2.0, 2)
            if len(outcomes) > 0:
                merged = outcomes.merge(metrics_df, on="trading_day", how="inner")
                analyze_immediate_breaks(merged, metrics_df, session)

        # ── BH FDR across all tests ──────────────────────────────────────
        actionable = [r for r in all_results if r is not None]
        if actionable:
            apply_bh_fdr(actionable)

        # ── Summary table ────────────────────────────────────────────────
        print(f"\n{'#'*70}")
        print(f"#  SUMMARY")
        print(f"{'#'*70}")
        for r in sorted(actionable, key=lambda x: x["p_val"]):
            sig = "*" if r["p_val"] < 0.01 else ""
            print(f"  p={r['p_val']:.6f}{sig:1s}  delta={r['delta']:+.4f}  "
                  f"N={r['n']:>4}  "
                  f"yr={r['years_consistent']}/{r['years_total']}  "
                  f"{r['label']}")

    finally:
        con.close()


if __name__ == "__main__":
    main()

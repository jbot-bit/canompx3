"""
Cross-Session Deep Dive: Concordance + Friday/Vol
==================================================
Follow-up to research_cross_session.py. Two BH survivors need stress-testing:

1. ORB Size Concordance (all_wide/all_narrow) — year-by-year consistency check.
2. Friday toxicity x vol regime — threshold sensitivity and stability.

ALSO: test concordance at multiple RR targets, with G4+ filter, and at other sessions.

LOOK-AHEAD: NONE. All medians/percentiles use expanding window with shift(1)
(each day only sees strictly prior data). First 20 days dropped for stability.

Output: research/output/cross_session_deep_findings.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
from scipy import stats
from pipeline.paths import GOLD_DB_PATH

DB_PATH = str(GOLD_DB_PATH)
OUTPUT = Path(__file__).resolve().parent / "output" / "cross_session_deep_findings.md"


def clean(arr):
    a = np.array(arr, dtype=float)
    return a[~np.isnan(a)]


def welch_t(group_a, group_b):
    a, b = clean(group_a), clean(group_b)
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan, np.nan)
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.nanmean(a) - np.nanmean(b)) / pooled_std if pooled_std > 0 else 0
    return (t_stat, p_val, d)


def one_sample_t(arr):
    """One-sample t-test vs 0."""
    a = clean(arr)
    if len(a) < 5:
        return (np.nan, np.nan)
    t_stat, p_val = stats.ttest_1samp(a, 0)
    return (t_stat, p_val)


def bh_fdr(p_values, q=0.10):
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result


def fmt(p):
    if np.isnan(p): return "N/A"
    if p < 0.001: return f"{p:.4f}"
    return f"{p:.3f}"


def sample_label(n):
    if n < 30: return "INVALID"
    if n < 100: return "REGIME"
    if n < 200: return "PRELIMINARY"
    if n < 500: return "CORE"
    return "HIGH-CONFIDENCE"


con = duckdb.connect(DB_PATH, read_only=True)
report_lines = []

def report(line=""):
    print(line)
    report_lines.append(line)


# ============================================================
# PART 1: CONCORDANCE YEAR-BY-YEAR DEEP DIVE
# ============================================================
report("=" * 70)
report("PART 1: ORB SIZE CONCORDANCE -- YEAR-BY-YEAR HONESTY CHECK")
report("=" * 70)

# Get pivoted ORB sizes (no median — computed via expanding window below)
concordance_sql = """
WITH sizes AS (
    SELECT trading_day, symbol, orb_1000_size
    FROM daily_features
    WHERE orb_minutes = 5
      AND symbol IN ('MGC', 'MES', 'MNQ')
      AND orb_1000_size IS NOT NULL
)
SELECT trading_day,
       MAX(CASE WHEN symbol = 'MGC' THEN orb_1000_size END) AS mgc_size,
       MAX(CASE WHEN symbol = 'MES' THEN orb_1000_size END) AS mes_size,
       MAX(CASE WHEN symbol = 'MNQ' THEN orb_1000_size END) AS mnq_size
FROM sizes
GROUP BY trading_day
HAVING mgc_size IS NOT NULL AND mes_size IS NOT NULL AND mnq_size IS NOT NULL
ORDER BY trading_day
"""

df_conc = con.execute(concordance_sql).fetchdf().sort_values("trading_day").reset_index(drop=True)

# Expanding-window median — shift(1) means each day only sees PRIOR days (no lookahead)
for col in ["mgc_size", "mes_size", "mnq_size"]:
    df_conc[f"{col}_med"] = df_conc[col].expanding(min_periods=20).median().shift(1)
df_conc = df_conc.dropna(subset=["mgc_size_med", "mes_size_med", "mnq_size_med"])

# Classify concordance (strictly backward-looking)
all_wide = ((df_conc["mgc_size"] > df_conc["mgc_size_med"]) &
            (df_conc["mes_size"] > df_conc["mes_size_med"]) &
            (df_conc["mnq_size"] > df_conc["mnq_size_med"]))
all_narrow = ((df_conc["mgc_size"] < df_conc["mgc_size_med"]) &
              (df_conc["mes_size"] < df_conc["mes_size_med"]) &
              (df_conc["mnq_size"] < df_conc["mnq_size_med"]))
df_conc["concordance"] = "mixed"
df_conc.loc[all_wide, "concordance"] = "all_wide"
df_conc.loc[all_narrow, "concordance"] = "all_narrow"

report(f"\n  Days with expanding concordance (20-day min lookback): {len(df_conc)}")
for c in ["all_wide", "all_narrow", "mixed"]:
    n = len(df_conc[df_conc["concordance"] == c])
    report(f"    {c}: {n} days ({100*n/len(df_conc):.1f}%)")

# Now test across multiple RR targets and entry models
report("\n--- Year-by-year for each instrument x concordance x RR ---")
all_tests = []

for sym in ["MGC", "MES", "MNQ"]:
    for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for em in ["E0", "E1"]:
            outcome_sql = f"""
            SELECT o.trading_day, o.pnl_r,
                   EXTRACT(YEAR FROM o.trading_day) AS year
            FROM orb_outcomes o
            WHERE o.orb_label = '1000' AND o.orb_minutes = 5
              AND o.symbol = '{sym}' AND o.entry_model = '{em}'
              AND o.confirm_bars = 1 AND o.rr_target = {rr}
            """
            df_out = con.execute(outcome_sql).fetchdf()
            df_merged = df_out.merge(df_conc[["trading_day", "concordance"]], on="trading_day")

            for conc_type in ["all_wide", "all_narrow"]:
                grp = df_merged[df_merged["concordance"] == conc_type]
                other = df_merged[df_merged["concordance"] != conc_type]
                vals = clean(grp["pnl_r"].values)
                other_vals = clean(other["pnl_r"].values)
                if len(vals) < 10:
                    continue

                avg_r = np.nanmean(vals)
                base_r = np.nanmean(other_vals)
                t, p, d = welch_t(vals, other_vals)

                # Year-by-year
                years = sorted(grp["year"].unique())
                yr_results = []
                for y in years:
                    yv = clean(grp[grp["year"] == y]["pnl_r"].values)
                    if len(yv) >= 5:
                        yr_results.append({"year": int(y), "N": len(yv),
                                           "avg_r": np.nanmean(yv),
                                           "positive": np.nanmean(yv) > 0})

                pos_years = sum(1 for yr in yr_results if yr["positive"])
                total_years = len(yr_results)

                all_tests.append({
                    "sym": sym, "rr": rr, "em": em, "conc": conc_type,
                    "N": len(vals), "avg_r": avg_r, "baseline": base_r,
                    "delta": avg_r - base_r, "p": p, "d": d,
                    "pos_years": pos_years, "total_years": total_years,
                    "yr_detail": yr_results,
                    "label": sample_label(len(vals))
                })

# Sort by p-value and show top results
all_tests.sort(key=lambda x: x["p"] if not np.isnan(x["p"]) else 999)

report("\n  TOP 20 CONCORDANCE RESULTS (by raw p-value):")
report(f"  {'Sym':<5} {'EM':<4} {'RR':<5} {'Conc':<12} {'N':<6} {'avgR':>7} {'base':>7} "
       f"{'delta':>7} {'p':>8} {'yrs+':>5} {'label':<15}")
report("  " + "-" * 90)

for t in all_tests[:20]:
    report(f"  {t['sym']:<5} {t['em']:<4} {t['rr']:<5.1f} {t['conc']:<12} {t['N']:<6} "
           f"{t['avg_r']:>+7.3f} {t['baseline']:>+7.3f} {t['delta']:>+7.3f} "
           f"{fmt(t['p']):>8} {t['pos_years']}/{t['total_years']:>2} {t['label']:<15}")

# Show year-by-year detail for top 5
report("\n  YEAR-BY-YEAR DETAIL (top 5 by p-value):")
for t in all_tests[:5]:
    report(f"\n  {t['sym']} {t['em']} RR{t['rr']} {t['conc']} (N={t['N']}, avgR={t['avg_r']:+.3f}, p={fmt(t['p'])}):")
    for yr in t["yr_detail"]:
        marker = "+" if yr["positive"] else "-"
        report(f"    {yr['year']}: N={yr['N']:>4}, avgR={yr['avg_r']:>+7.3f} [{marker}]")


# ============================================================
# PART 1b: CONCORDANCE + G4 FILTER STACKING
# ============================================================
report("\n" + "=" * 70)
report("PART 1b: CONCORDANCE + G4 FILTER -- DO THEY STACK?")
report("=" * 70)

part1b_tests = []  # Collect for BH pooling
for sym in ["MGC", "MES", "MNQ"]:
    g4_sql = f"""
    SELECT o.trading_day, o.pnl_r, d.orb_1000_size,
           EXTRACT(YEAR FROM o.trading_day) AS year
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '1000' AND o.orb_minutes = 5
      AND o.symbol = '{sym}' AND o.entry_model = 'E0'
      AND o.confirm_bars = 1 AND o.rr_target = 2.0
      AND d.orb_1000_size >= 4
    """
    df_g4 = con.execute(g4_sql).fetchdf()
    df_g4_conc = df_g4.merge(df_conc[["trading_day", "concordance"]], on="trading_day")

    for conc_type in ["all_wide", "all_narrow", "mixed"]:
        vals = clean(df_g4_conc[df_g4_conc["concordance"] == conc_type]["pnl_r"].values)
        if len(vals) >= 10:
            avg = np.nanmean(vals)
            t_stat, p_1s = one_sample_t(vals)
            report(f"  {sym} E0 RR2.0 G4+ {conc_type}: N={len(vals)}, avgR={avg:+.3f}, "
                   f"p_vs_0={fmt(p_1s)}, {sample_label(len(vals))}")
            if not np.isnan(p_1s):
                part1b_tests.append({"label": f"G4_{sym}_{conc_type}", "p": p_1s})


# ============================================================
# PART 2: FRIDAY x VOL REGIME DEEP DIVE
# ============================================================
report("\n" + "=" * 70)
report("PART 2: FRIDAY TOXICITY x VOL REGIME -- THRESHOLD SENSITIVITY")
report("=" * 70)

# Test multiple ATR thresholds to find the right cutoff
fri_sql = """
SELECT o.trading_day, o.symbol, o.pnl_r,
       d.is_friday, d.atr_20, d.atr_vel_ratio, d.atr_vel_regime,
       EXTRACT(YEAR FROM o.trading_day) AS year
FROM orb_outcomes o
JOIN daily_features d ON o.trading_day = d.trading_day
  AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
WHERE o.orb_label = '1000' AND o.orb_minutes = 5
  AND o.entry_model = 'E0' AND o.confirm_bars = 1 AND o.rr_target = 2.0
  AND o.symbol IN ('MGC', 'MES', 'MNQ')
  AND d.atr_20 IS NOT NULL AND d.is_friday IS NOT NULL
"""

df_fri = con.execute(fri_sql).fetchdf()

report("\n--- ATR percentile threshold sensitivity (Friday vs Mon-Thu) ---")

fri_tests = []
for sym in ["MGC", "MES", "MNQ"]:
    sub = df_fri[df_fri["symbol"] == sym].copy()
    if len(sub) < 50:
        continue

    # Sort by date for expanding-window percentile (no lookahead)
    sub = sub.sort_values("trading_day").reset_index(drop=True)

    # Test percentile thresholds: 25th, 33rd, 50th, 67th, 75th
    for pct in [25, 33, 50, 67, 75]:
        # Expanding-window percentile — shift(1) = only prior days
        exp_thresh = sub["atr_20"].expanding(min_periods=20).quantile(pct / 100).shift(1)
        sub_clean = sub.assign(threshold=exp_thresh).dropna(subset=["threshold"])
        high_vol = sub_clean[sub_clean["atr_20"] > sub_clean["threshold"]]
        low_vol = sub_clean[sub_clean["atr_20"] <= sub_clean["threshold"]]
        latest_threshold = sub_clean["threshold"].iloc[-1] if len(sub_clean) > 0 else np.nan

        for vol_label, vol_sub in [("high_vol", high_vol), ("low_vol", low_vol)]:
            fri = clean(vol_sub[vol_sub["is_friday"] == True]["pnl_r"].values)
            not_fri = clean(vol_sub[vol_sub["is_friday"] == False]["pnl_r"].values)
            if len(fri) >= 10 and len(not_fri) >= 10:
                t, p, d = welch_t(fri, not_fri)
                delta = np.nanmean(fri) - np.nanmean(not_fri)
                fri_tests.append({
                    "sym": sym, "pct": pct, "vol": vol_label,
                    "N_fri": len(fri), "fri_avg": np.nanmean(fri),
                    "notfri_avg": np.nanmean(not_fri), "delta": delta,
                    "p": p, "d": d, "threshold": latest_threshold
                })

# Show results sorted by delta (most toxic Fridays first)
fri_tests.sort(key=lambda x: x["delta"])
report(f"\n  {'Sym':<5} {'Pct':<5} {'Vol':<10} {'N_fri':<7} {'Fri_R':>7} {'M-T_R':>7} "
       f"{'Delta':>7} {'p':>8} {'ATR_thresh':>10}")
report("  " + "-" * 80)
for ft in fri_tests[:15]:
    report(f"  {ft['sym']:<5} P{ft['pct']:<4} {ft['vol']:<10} {ft['N_fri']:<7} "
           f"{ft['fri_avg']:>+7.3f} {ft['notfri_avg']:>+7.3f} {ft['delta']:>+7.3f} "
           f"{fmt(ft['p']):>8} {ft['threshold']:>10.1f}")

# Year-by-year for the most toxic combo (using expanding threshold per day)
report("\n--- Year-by-year for top toxic Friday combos ---")
for ft in fri_tests[:3]:
    sym = ft["sym"]
    pct = ft["pct"]
    sub_yy = df_fri[df_fri["symbol"] == sym].copy().sort_values("trading_day").reset_index(drop=True)
    exp_t = sub_yy["atr_20"].expanding(min_periods=20).quantile(pct / 100).shift(1)
    sub_yy = sub_yy.assign(exp_threshold=exp_t).dropna(subset=["exp_threshold"])
    sub_yy = sub_yy[(sub_yy["atr_20"] > sub_yy["exp_threshold"]) & (sub_yy["is_friday"] == True)]
    report(f"\n  {sym} P{pct} high_vol Friday (latest ATR>{ft['threshold']:.1f}):")
    years = sorted(sub_yy["year"].unique())
    for y in years:
        yv = clean(sub_yy[sub_yy["year"] == y]["pnl_r"].values)
        if len(yv) >= 3:
            marker = "+" if np.nanmean(yv) > 0 else "-"
            report(f"    {int(y)}: N={len(yv):>3}, avgR={np.nanmean(yv):>+7.3f} [{marker}]")


# ============================================================
# PART 2b: CONTRASTING -- LOW-VOL FRIDAY (the positive side)
# ============================================================
report("\n--- Low-vol Friday: is there actually POSITIVE edge? ---")
for ft in sorted(fri_tests, key=lambda x: -x["delta"])[:5]:
    if ft["vol"] == "low_vol" and ft["delta"] > 0:
        sym = ft["sym"]
        pct = ft["pct"]
        sub_lv = df_fri[df_fri["symbol"] == sym].copy().sort_values("trading_day").reset_index(drop=True)
        exp_t = sub_lv["atr_20"].expanding(min_periods=20).quantile(pct / 100).shift(1)
        sub_lv = sub_lv.assign(exp_threshold=exp_t).dropna(subset=["exp_threshold"])
        sub_lv = sub_lv[(sub_lv["atr_20"] <= sub_lv["exp_threshold"]) & (sub_lv["is_friday"] == True)]
        vals = clean(sub_lv["pnl_r"].values)
        t_stat, p_1s = one_sample_t(vals)
        report(f"  {sym} P{pct} low_vol Friday: N={len(vals)}, avgR={np.nanmean(vals):+.3f}, "
               f"p_vs_0={fmt(p_1s)}, {sample_label(len(vals))}")


# ============================================================
# PART 3: CONCORDANCE AT OTHER SESSIONS (0900, 1800)
# ============================================================
report("\n" + "=" * 70)
report("PART 3: CONCORDANCE AT OTHER SESSIONS (0900, 1800)")
report("=" * 70)
report("  NOTE: 0900 is DST-contaminated (us_dst), 1800 is DST-contaminated (uk_dst)")
report("  These tests are NOT DST-split. Treat as exploratory only.")

part3_tests = []  # Collect for BH pooling
for session in ["0900", "1800"]:
    for sym in ["MGC", "MES", "MNQ"]:
        sess_sql = f"""
        SELECT o.trading_day, o.pnl_r,
               EXTRACT(YEAR FROM o.trading_day) AS year
        FROM orb_outcomes o
        WHERE o.orb_label = '{session}' AND o.orb_minutes = 5
          AND o.symbol = '{sym}' AND o.entry_model = 'E0'
          AND o.confirm_bars = 1 AND o.rr_target = 2.0
        """
        df_sess = con.execute(sess_sql).fetchdf()
        df_sess_conc = df_sess.merge(df_conc[["trading_day", "concordance"]], on="trading_day")

        for conc_type in ["all_wide", "all_narrow"]:
            vals = clean(df_sess_conc[df_sess_conc["concordance"] == conc_type]["pnl_r"].values)
            other = clean(df_sess_conc[df_sess_conc["concordance"] != conc_type]["pnl_r"].values)
            if len(vals) >= 10 and len(other) >= 10:
                t, p, d = welch_t(vals, other)
                report(f"  {sym} {session} {conc_type}: N={len(vals)}, avgR={np.nanmean(vals):+.3f}, "
                       f"baseline={np.nanmean(other):+.3f}, p={fmt(p)}, {sample_label(len(vals))}")
                if not np.isnan(p):
                    part3_tests.append({"label": f"SESS_{sym}_{session}_{conc_type}", "p": p})


# ============================================================
# PART 4: IS CONCORDANCE JUST A PROXY FOR SINGLE-INSTRUMENT G4+?
# ============================================================
report("\n" + "=" * 70)
report("PART 4: IS CONCORDANCE JUST G4+ IN DISGUISE?")
report("=" * 70)

# Check: on all_wide days, what % also pass G4 for the target instrument?
part4_tests = []  # Collect for BH pooling
for sym in ["MGC", "MES", "MNQ"]:
    check_sql = f"""
    SELECT d.trading_day, d.orb_1000_size,
           CASE WHEN d.orb_1000_size >= 4 THEN 1 ELSE 0 END AS passes_g4
    FROM daily_features d
    WHERE d.symbol = '{sym}' AND d.orb_minutes = 5
      AND d.orb_1000_size IS NOT NULL
    """
    df_check = con.execute(check_sql).fetchdf()
    df_check_conc = df_check.merge(df_conc[["trading_day", "concordance"]], on="trading_day")

    for conc_type in ["all_wide", "all_narrow", "mixed"]:
        sub = df_check_conc[df_check_conc["concordance"] == conc_type]
        if len(sub) > 0:
            pct_g4 = 100 * sub["passes_g4"].mean()
            avg_size = sub["orb_1000_size"].mean()
            report(f"  {sym} {conc_type}: {len(sub)} days, "
                   f"{pct_g4:.0f}% pass G4, avg orb_size={avg_size:.1f}")

    # KEY TEST: among G4+ days, does concordance STILL matter?
    g4_only = df_check_conc[df_check_conc["passes_g4"] == 1]
    g4_wide = g4_only[g4_only["concordance"] == "all_wide"]
    g4_not_wide = g4_only[g4_only["concordance"] != "all_wide"]
    report(f"  {sym} WITHIN G4+: all_wide={len(g4_wide)}, not_wide={len(g4_not_wide)}")

    # Get actual outcomes for G4+ all_wide vs G4+ not_wide
    g4_out_sql = f"""
    SELECT o.trading_day, o.pnl_r
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{sym}' AND o.orb_label = '1000' AND o.orb_minutes = 5
      AND o.entry_model = 'E0' AND o.confirm_bars = 1 AND o.rr_target = 2.0
      AND d.orb_1000_size >= 4
    """
    df_g4_out = con.execute(g4_out_sql).fetchdf()
    df_g4_out_conc = df_g4_out.merge(df_conc[["trading_day", "concordance"]], on="trading_day")

    wide_r = clean(df_g4_out_conc[df_g4_out_conc["concordance"] == "all_wide"]["pnl_r"].values)
    not_wide_r = clean(df_g4_out_conc[df_g4_out_conc["concordance"] != "all_wide"]["pnl_r"].values)
    if len(wide_r) >= 10 and len(not_wide_r) >= 10:
        t, p, d = welch_t(wide_r, not_wide_r)
        report(f"  {sym} G4+ all_wide: N={len(wide_r)}, avgR={np.nanmean(wide_r):+.3f} "
               f"vs G4+ not_wide: N={len(not_wide_r)}, avgR={np.nanmean(not_wide_r):+.3f}, "
               f"p={fmt(p)}, d={d:+.3f}")
        if not np.isnan(p):
            part4_tests.append({"label": f"G4WITHIN_{sym}_wide_vs_not", "p": p})
    report("")


# ============================================================
# BH FDR on all Part 1-4 tests
# ============================================================
report("\n" + "=" * 70)
report("BH FDR CORRECTION (Parts 1-4 pooled)")
report("=" * 70)

# Collect all p-values from ALL parts (1, 1b, 2, 3, 4)
all_p = [t["p"] for t in all_tests if not np.isnan(t["p"])]
all_labels_conc = [f"{t['sym']}_{t['em']}_RR{t['rr']}_{t['conc']}" for t in all_tests if not np.isnan(t["p"])]

# Add Part 1b (G4+ concordance one-sample t-tests)
for pt in part1b_tests:
    all_p.append(pt["p"])
    all_labels_conc.append(pt["label"])

# Add Part 2 (friday tests)
for ft in fri_tests:
    if not np.isnan(ft["p"]):
        all_p.append(ft["p"])
        all_labels_conc.append(f"FRI_{ft['sym']}_P{ft['pct']}_{ft['vol']}")

# Add Part 3 (cross-session concordance)
for pt in part3_tests:
    all_p.append(pt["p"])
    all_labels_conc.append(pt["label"])

# Add Part 4 (G4+ within concordance)
for pt in part4_tests:
    all_p.append(pt["p"])
    all_labels_conc.append(pt["label"])

adjusted = bh_fdr(all_p, q=0.10)
survivors = [(l, p, adj) for l, p, adj in zip(all_labels_conc, all_p, adjusted) if adj < 0.10]

report(f"\n  Total tests: {len(all_p)}")
report(f"  BH survivors (q=0.10): {len(survivors)}")
for label, raw_p, adj_p in sorted(survivors, key=lambda x: x[2])[:20]:
    report(f"    {label}: raw_p={fmt(raw_p)}, p_bh={fmt(adj_p)}")

# Also show: how many concordance tests survive at different year thresholds
report("\n--- Concordance tests with year-consistency filter ---")
for min_pos_pct in [0.5, 0.6, 0.67, 0.75]:
    consistent = [t for t in all_tests
                  if t["total_years"] >= 3
                  and t["pos_years"] / t["total_years"] >= min_pos_pct
                  and not np.isnan(t["p"]) and t["p"] < 0.05]
    report(f"  {int(min_pos_pct*100)}%+ years positive AND raw p<0.05: {len(consistent)} tests")
    for t in consistent[:5]:
        report(f"    {t['sym']} {t['em']} RR{t['rr']} {t['conc']}: N={t['N']}, "
               f"avgR={t['avg_r']:+.3f}, {t['pos_years']}/{t['total_years']} yrs+, p={fmt(t['p'])}")


# ============================================================
# WRITE REPORT
# ============================================================
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    f.write("# Cross-Session Deep Dive Findings\n")
    f.write(f"**Date:** 2026-02-21\n")
    f.write(f"**Script:** research/research_cross_session_deep.py\n\n")
    for line in report_lines:
        f.write(line + "\n")

    f.write("\n## HONEST SUMMARY\n\n")
    f.write("### SURVIVED\n")
    f.write("- [List BH survivors with consistent year-by-year]\n\n")
    f.write("### DID NOT SURVIVE\n")
    f.write("- [List findings that failed year-by-year or BH]\n\n")
    f.write("### CAVEATS\n")
    f.write("- Concordance medians and ATR thresholds use expanding window with shift(1) -- NO lookahead\n")
    f.write("- First 20 days dropped (insufficient lookback for stable expanding median)\n")
    f.write("- MNQ only has 2 years of data -- any MNQ finding is REGIME at best\n")
    f.write("- 0900 session tests NOT us_dst-split (applies to Part 3)\n")
    f.write("- 1800 session tests NOT uk_dst-split (applies to Part 3)\n\n")
    f.write("### NEXT STEPS\n")
    f.write("- [Based on what survives]\n")

report(f"\nFindings written to {OUTPUT}")
con.close()

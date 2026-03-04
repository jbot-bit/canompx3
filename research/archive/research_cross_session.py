"""
Cross-Session & Cross-Instrument Diamond Hunting
=================================================
Fresh-eyes analysis looking for non-obvious patterns in the existing data.
5 queries with statistical tests, BH FDR correction, year-by-year splits.

LOOK-AHEAD RULES:
- ORB size: known at ORB close (e.g., 0905 for 0900 session). SAFE.
- break_dir: known at break time. SAFE if break_ts < target session start.
- outcome (win/loss): NOT known until session end. UNSAFE for same-day prediction.
- gap_open_points, atr_20, atr_vel_ratio: known at day start. SAFE.

Output: research/output/cross_session_findings.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
from scipy import stats
from pipeline.paths import GOLD_DB_PATH

DB_PATH = str(GOLD_DB_PATH)
OUTPUT = Path(__file__).resolve().parent / "output" / "cross_session_findings.md"

# BH FDR correction
def bh_fdr(p_values, q=0.10):
    """Benjamini-Hochberg FDR at q=0.10. Returns adjusted p-values."""
    n = len(p_values)
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


def clean(arr):
    """Remove NaN from numpy array."""
    a = np.array(arr, dtype=float)
    return a[~np.isnan(a)]


def welch_t(group_a, group_b):
    """Welch's t-test, returns (t_stat, p_value, cohens_d)."""
    a, b = clean(group_a), clean(group_b)
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan, np.nan)
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.nanmean(a) - np.nanmean(b)) / pooled_std if pooled_std > 0 else 0
    return (t_stat, p_val, d)


def year_split(df, val_col="pnl_r", year_col="year"):
    """Year-by-year summary."""
    years = sorted(df[year_col].unique())
    rows = []
    for y in years:
        sub = df[df[year_col] == y]
        vals = sub[val_col].values
        rows.append({
            "year": y, "N": len(vals),
            "avg_r": np.nanmean(vals) if len(vals) > 0 else np.nan,
            "positive": np.nanmean(vals) > 0 if len(vals) >= 5 else None
        })
    return rows


def fmt_pval(p):
    if np.isnan(p): return "N/A"
    if p < 0.001: return f"{p:.4f}"
    return f"{p:.3f}"


def sample_label(n):
    if n < 30: return "INVALID"
    if n < 100: return "REGIME"
    if n < 200: return "PRELIMINARY"
    if n < 500: return "CORE"
    return "HIGH-CONFIDENCE"


findings = []  # Collect all findings for report

con = duckdb.connect(DB_PATH, read_only=True)

# ============================================================
# QUERY 1: MGC 0900 break -> MES/MNQ 1000 quality
# ============================================================
print("=" * 60)
print("Q1: MGC 0900 break direction -> MES/MNQ 1000 quality")
print("=" * 60)

q1_sql = """
WITH mgc_0900 AS (
    SELECT trading_day,
           orb_0900_break_dir AS mgc_dir,
           orb_0900_break_ts AS mgc_break_ts,
           orb_0900_size AS mgc_orb_size
    FROM daily_features
    WHERE symbol = 'MGC'
      AND orb_minutes = 5
      AND orb_0900_break_dir IS NOT NULL
),
target_outcomes AS (
    SELECT o.trading_day, o.symbol, o.orb_label, o.pnl_r,
           o.entry_model, o.confirm_bars, o.rr_target,
           EXTRACT(YEAR FROM o.trading_day) AS year
    FROM orb_outcomes o
    WHERE o.orb_label = '1000'
      AND o.orb_minutes = 5
      AND o.symbol IN ('MES', 'MNQ')
      AND o.entry_model = 'E0'
      AND o.confirm_bars = 1
      AND o.rr_target = 2.0
)
SELECT t.*, m.mgc_dir, m.mgc_orb_size
FROM target_outcomes t
JOIN mgc_0900 m ON t.trading_day = m.trading_day
"""

df_q1 = con.execute(q1_sql).fetchdf()
print(f"  Rows: {len(df_q1)}")

q1_results = []
for sym in ["MES", "MNQ"]:
    sub = df_q1[df_q1["symbol"] == sym]
    for direction in ["LONG", "SHORT"]:
        grp = sub[sub["mgc_dir"] == direction]["pnl_r"].values
        baseline = sub["pnl_r"].values
        if len(grp) >= 10:
            t, p, d = welch_t(grp, baseline)
            avg = np.nanmean(grp)
            print(f"  {sym} 1000 when MGC 0900={direction}: N={len(grp)}, avgR={avg:+.3f}, "
                  f"baseline={np.nanmean(baseline):+.3f}, p={fmt_pval(p)}, d={d:+.3f}, "
                  f"{sample_label(len(grp))}")
            q1_results.append({"label": f"{sym}_1000_mgc0900_{direction}",
                              "N": len(grp), "avg_r": avg, "p": p, "d": d})

    # Also test: MGC 0900 break direction ALIGNS with target trade direction
    # When MGC breaks LONG and MES also breaks LONG, vs misaligned
    aligned_sql = f"""
    WITH mgc_0900 AS (
        SELECT trading_day, orb_0900_break_dir AS mgc_dir
        FROM daily_features
        WHERE symbol = 'MGC' AND orb_minutes = 5
          AND orb_0900_break_dir IS NOT NULL
    ),
    target AS (
        SELECT o.trading_day, o.pnl_r,
               d.orb_1000_break_dir AS target_dir,
               EXTRACT(YEAR FROM o.trading_day) AS year
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day = d.trading_day
          AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{sym}' AND o.orb_label = '1000'
          AND o.orb_minutes = 5 AND o.entry_model = 'E0'
          AND o.confirm_bars = 1 AND o.rr_target = 2.0
          AND d.orb_1000_break_dir IS NOT NULL
    )
    SELECT t.*, m.mgc_dir,
           CASE WHEN t.target_dir = m.mgc_dir THEN 'aligned' ELSE 'opposed' END AS alignment
    FROM target t
    JOIN mgc_0900 m ON t.trading_day = m.trading_day
    """
    df_align = con.execute(aligned_sql).fetchdf()
    for align_type in ["aligned", "opposed"]:
        grp = df_align[df_align["alignment"] == align_type]["pnl_r"].values
        other = df_align[df_align["alignment"] != align_type]["pnl_r"].values
        if len(grp) >= 10 and len(other) >= 10:
            t, p, d = welch_t(grp, other)
            avg = np.nanmean(grp)
            print(f"  {sym} 1000 {align_type} with MGC 0900: N={len(grp)}, avgR={avg:+.3f}, "
                  f"vs opposite={np.nanmean(other):+.3f}, p={fmt_pval(p)}, d={d:+.3f}, "
                  f"{sample_label(len(grp))}")
            q1_results.append({"label": f"{sym}_1000_mgc_align_{align_type}",
                              "N": len(grp), "avg_r": avg, "p": p, "d": d})

findings.append(("Q1: MGC 0900 -> MES/MNQ 1000", q1_results))

# ============================================================
# QUERY 2: Multi-instrument ORB size concordance at 1000
# ============================================================
print("\n" + "=" * 60)
print("Q2: Multi-instrument ORB size concordance at 1000")
print("=" * 60)

# Step 1: Pull pivoted ORB sizes (no median — computed via expanding window below)
q2_sizes_sql = """
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
df_pivot = con.execute(q2_sizes_sql).fetchdf().sort_values("trading_day").reset_index(drop=True)

# Step 2: Expanding-window median — shift(1) means each day only sees PRIOR days
for col in ["mgc_size", "mes_size", "mnq_size"]:
    df_pivot[f"{col}_med"] = df_pivot[col].expanding(min_periods=20).median().shift(1)
df_pivot = df_pivot.dropna(subset=["mgc_size_med", "mes_size_med", "mnq_size_med"])

# Step 3: Classify concordance (no lookahead — medians are strictly backward-looking)
all_wide = ((df_pivot["mgc_size"] > df_pivot["mgc_size_med"]) &
            (df_pivot["mes_size"] > df_pivot["mes_size_med"]) &
            (df_pivot["mnq_size"] > df_pivot["mnq_size_med"]))
all_narrow = ((df_pivot["mgc_size"] < df_pivot["mgc_size_med"]) &
              (df_pivot["mes_size"] < df_pivot["mes_size_med"]) &
              (df_pivot["mnq_size"] < df_pivot["mnq_size_med"]))
df_pivot["concordance"] = "mixed"
df_pivot.loc[all_wide, "concordance"] = "all_wide"
df_pivot.loc[all_narrow, "concordance"] = "all_narrow"

print(f"  Days with expanding concordance: {len(df_pivot)}")
for c in ["all_wide", "all_narrow", "mixed"]:
    n = len(df_pivot[df_pivot["concordance"] == c])
    print(f"    {c}: {n} ({100*n/len(df_pivot):.1f}%)")

# Step 4: Pull outcomes and merge with concordance
q2_outcomes_sql = """
SELECT o.trading_day, o.symbol, o.pnl_r,
       EXTRACT(YEAR FROM o.trading_day) AS year
FROM orb_outcomes o
WHERE o.orb_label = '1000' AND o.orb_minutes = 5
  AND o.entry_model = 'E0' AND o.confirm_bars = 1
  AND o.rr_target = 2.0
  AND o.symbol IN ('MGC', 'MES', 'MNQ')
"""
df_q2 = con.execute(q2_outcomes_sql).fetchdf()
df_q2 = df_q2.merge(df_pivot[["trading_day", "concordance"]], on="trading_day")
print(f"  Outcome rows with concordance: {len(df_q2)}")

q2_results = []
for sym in ["MGC", "MES", "MNQ"]:
    sub = df_q2[df_q2["symbol"] == sym]
    for conc in ["all_wide", "all_narrow", "mixed"]:
        grp = sub[sub["concordance"] == conc]["pnl_r"].values
        other = sub[sub["concordance"] != conc]["pnl_r"].values
        if len(grp) >= 10:
            t, p, d = welch_t(grp, other)
            avg = np.nanmean(grp)
            print(f"  {sym} 1000 {conc}: N={len(grp)}, avgR={avg:+.3f}, "
                  f"baseline={np.nanmean(other):+.3f}, p={fmt_pval(p)}, d={d:+.3f}, "
                  f"{sample_label(len(grp))}")
            q2_results.append({"label": f"{sym}_1000_{conc}", "N": len(grp),
                              "avg_r": avg, "p": p, "d": d})

    # Year-by-year for best group
    for conc in ["all_wide", "all_narrow"]:
        grp_df = sub[sub["concordance"] == conc]
        if len(grp_df) >= 20:
            yrs = year_split(grp_df)
            pos_years = sum(1 for y in yrs if y["positive"] is True)
            total_years = sum(1 for y in yrs if y["positive"] is not None)
            print(f"    {sym} {conc} year-by-year: {pos_years}/{total_years} positive")

findings.append(("Q2: ORB size concordance at 1000", q2_results))

# ============================================================
# QUERY 3: Gap direction × break direction interaction
# ============================================================
print("\n" + "=" * 60)
print("Q3: Overnight gap × ORB break direction interaction")
print("=" * 60)

q3_sql = """
SELECT o.trading_day, o.symbol, o.orb_label, o.pnl_r,
       d.gap_open_points, d.gap_type,
       d.orb_1000_break_dir AS break_dir,
       CASE
           WHEN d.gap_open_points > 0 AND d.orb_1000_break_dir = 'LONG' THEN 'gap_aligned'
           WHEN d.gap_open_points < 0 AND d.orb_1000_break_dir = 'SHORT' THEN 'gap_aligned'
           WHEN d.gap_open_points > 0 AND d.orb_1000_break_dir = 'SHORT' THEN 'gap_opposed'
           WHEN d.gap_open_points < 0 AND d.orb_1000_break_dir = 'LONG' THEN 'gap_opposed'
           ELSE 'flat_gap'
       END AS gap_alignment,
       EXTRACT(YEAR FROM o.trading_day) AS year
FROM orb_outcomes o
JOIN daily_features d ON o.trading_day = d.trading_day
  AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
WHERE o.orb_label = '1000' AND o.orb_minutes = 5
  AND o.entry_model = 'E0' AND o.confirm_bars = 1 AND o.rr_target = 2.0
  AND o.symbol IN ('MGC', 'MES', 'MNQ')
  AND d.gap_open_points IS NOT NULL
  AND d.orb_1000_break_dir IS NOT NULL
"""

df_q3 = con.execute(q3_sql).fetchdf()
print(f"  Rows: {len(df_q3)}")

q3_results = []
for sym in ["MGC", "MES", "MNQ"]:
    sub = df_q3[df_q3["symbol"] == sym]
    aligned = sub[sub["gap_alignment"] == "gap_aligned"]["pnl_r"].values
    opposed = sub[sub["gap_alignment"] == "gap_opposed"]["pnl_r"].values
    if len(aligned) >= 10 and len(opposed) >= 10:
        t, p, d = welch_t(aligned, opposed)
        print(f"  {sym} 1000 gap_aligned: N={len(aligned)}, avgR={np.nanmean(aligned):+.3f}, {sample_label(len(aligned))}")
        print(f"  {sym} 1000 gap_opposed: N={len(opposed)}, avgR={np.nanmean(opposed):+.3f}, {sample_label(len(opposed))}")
        print(f"    delta={np.nanmean(aligned)-np.nanmean(opposed):+.3f}, p={fmt_pval(p)}, d={d:+.3f}")
        q3_results.append({"label": f"{sym}_1000_gap_aligned",
                          "N": len(aligned), "avg_r": np.nanmean(aligned), "p": p, "d": d})
        q3_results.append({"label": f"{sym}_1000_gap_opposed",
                          "N": len(opposed), "avg_r": np.nanmean(opposed), "p": p, "d": d})

# Also test gap SIZE interaction (large gap vs small gap)
for sym in ["MGC", "MES", "MNQ"]:
    sub = df_q3[df_q3["symbol"] == sym].copy()
    if len(sub) < 30:
        continue
    sub = sub.sort_values("trading_day").reset_index(drop=True)
    sub["exp_med_gap"] = sub["gap_open_points"].abs().expanding(min_periods=20).median().shift(1)
    sub = sub.dropna(subset=["exp_med_gap"])
    big_gap = sub[sub["gap_open_points"].abs() > sub["exp_med_gap"]]["pnl_r"].values
    small_gap = sub[sub["gap_open_points"].abs() <= sub["exp_med_gap"]]["pnl_r"].values
    med_gap = sub["exp_med_gap"].iloc[-1]  # latest threshold for reporting
    if len(big_gap) >= 10 and len(small_gap) >= 10:
        t, p, d = welch_t(big_gap, small_gap)
        print(f"  {sym} 1000 big_gap(>{med_gap:.1f}pts): N={len(big_gap)}, avgR={np.nanmean(big_gap):+.3f}, {sample_label(len(big_gap))} "
              f"vs small: N={len(small_gap)}, avgR={np.nanmean(small_gap):+.3f}, p={fmt_pval(p)}")
        q3_results.append({"label": f"{sym}_1000_big_gap_vs_small",
                          "N": len(big_gap), "avg_r": np.nanmean(big_gap), "p": p, "d": d})

findings.append(("Q3: Gap × break direction", q3_results))

# ============================================================
# QUERY 4: Same-instrument cross-session cascading
# ============================================================
print("\n" + "=" * 60)
print("Q4: Same-instrument session cascading (0900 break -> 1000 quality)")
print("=" * 60)

q4_sql = """
WITH early_session AS (
    SELECT trading_day, symbol,
           orb_0900_break_dir AS early_dir,
           orb_0900_size AS early_size,
           CASE WHEN orb_0900_break_dir IS NOT NULL THEN 'break' ELSE 'no_break' END AS early_status
    FROM daily_features
    WHERE orb_minutes = 5
),
outcomes AS (
    SELECT o.trading_day, o.symbol, o.pnl_r, o.orb_label,
           EXTRACT(YEAR FROM o.trading_day) AS year
    FROM orb_outcomes o
    WHERE o.orb_label = '1000' AND o.orb_minutes = 5
      AND o.entry_model = 'E0' AND o.confirm_bars = 1
      AND o.rr_target = 2.0
)
SELECT oc.*, es.early_dir, es.early_size, es.early_status
FROM outcomes oc
JOIN early_session es ON oc.trading_day = es.trading_day AND oc.symbol = es.symbol
"""

df_q4 = con.execute(q4_sql).fetchdf()
print(f"  Rows: {len(df_q4)}")

q4_results = []
for sym in ["MGC", "MES", "MNQ"]:
    sub = df_q4[df_q4["symbol"] == sym]
    # Test: 0900 had a break vs no break
    brk = sub[sub["early_status"] == "break"]["pnl_r"].values
    no_brk = sub[sub["early_status"] == "no_break"]["pnl_r"].values
    if len(brk) >= 10 and len(no_brk) >= 10:
        t, p, d = welch_t(brk, no_brk)
        print(f"  {sym}: 0900 break exists -> 1000: N={len(brk)}, avgR={np.nanmean(brk):+.3f}, {sample_label(len(brk))}")
        print(f"  {sym}: 0900 no break   -> 1000: N={len(no_brk)}, avgR={np.nanmean(no_brk):+.3f}, {sample_label(len(no_brk))}")
        print(f"    delta={np.nanmean(brk)-np.nanmean(no_brk):+.3f}, p={fmt_pval(p)}")
        q4_results.append({"label": f"{sym}_0900brk_to_1000",
                          "N": len(brk), "avg_r": np.nanmean(brk), "p": p, "d": d})

    # Test: 0900 break direction ALIGNS with 1000 break direction
    sub_with_dir = sub[sub["early_dir"].notna()].copy()
    if len(sub_with_dir) >= 20:
        aligned_q4_sql = f"""
        WITH early AS (
            SELECT trading_day, orb_0900_break_dir AS e_dir
            FROM daily_features WHERE symbol = '{sym}' AND orb_minutes = 5
              AND orb_0900_break_dir IS NOT NULL
        ),
        late AS (
            SELECT o.trading_day, o.pnl_r,
                   d.orb_1000_break_dir AS l_dir
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day = d.trading_day
              AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = '{sym}' AND o.orb_label = '1000'
              AND o.orb_minutes = 5 AND o.entry_model = 'E0'
              AND o.confirm_bars = 1 AND o.rr_target = 2.0
              AND d.orb_1000_break_dir IS NOT NULL
        )
        SELECT l.*, e.e_dir,
               CASE WHEN l.l_dir = e.e_dir THEN 'same' ELSE 'flip' END AS cascade
        FROM late l JOIN early e ON l.trading_day = e.trading_day
        """
        df_casc = con.execute(aligned_q4_sql).fetchdf()
        for cas_type in ["same", "flip"]:
            grp = df_casc[df_casc["cascade"] == cas_type]["pnl_r"].values
            other = df_casc[df_casc["cascade"] != cas_type]["pnl_r"].values
            if len(grp) >= 10 and len(other) >= 10:
                t, p, d = welch_t(grp, other)
                print(f"  {sym} 0900->1000 {cas_type}: N={len(grp)}, avgR={np.nanmean(grp):+.3f}, p={fmt_pval(p)}, {sample_label(len(grp))}")
                q4_results.append({"label": f"{sym}_cascade_{cas_type}",
                                  "N": len(grp), "avg_r": np.nanmean(grp), "p": p, "d": d})

findings.append(("Q4: Same-instrument session cascading", q4_results))

# ============================================================
# QUERY 5: Friday toxicity × vol regime (MES 1000)
# ============================================================
print("\n" + "=" * 60)
print("Q5: Friday toxicity × vol regime (MES 1000)")
print("=" * 60)

q5_sql = """
SELECT o.trading_day, o.pnl_r,
       d.is_friday, d.day_of_week,
       d.atr_vel_ratio, d.atr_vel_regime, d.atr_20,
       EXTRACT(YEAR FROM o.trading_day) AS year
FROM orb_outcomes o
JOIN daily_features d ON o.trading_day = d.trading_day
  AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
WHERE o.symbol = 'MES' AND o.orb_label = '1000' AND o.orb_minutes = 5
  AND o.entry_model = 'E0' AND o.confirm_bars = 1 AND o.rr_target = 2.0
  AND d.atr_vel_ratio IS NOT NULL
"""

df_q5 = con.execute(q5_sql).fetchdf()
print(f"  Rows: {len(df_q5)}")

q5_results = []
# Split vol regime using expanding-window median (no lookahead)
df_q5 = df_q5.sort_values("trading_day").reset_index(drop=True)
df_q5["exp_med_atr"] = df_q5["atr_20"].expanding(min_periods=20).median().shift(1)
df_q5 = df_q5.dropna(subset=["exp_med_atr"])
df_q5["vol_regime"] = np.where(df_q5["atr_20"] > df_q5["exp_med_atr"], "high_vol", "low_vol")

for vol in ["high_vol", "low_vol"]:
    sub = df_q5[df_q5["vol_regime"] == vol]
    fri = sub[sub["is_friday"] == True]["pnl_r"].values
    not_fri = sub[sub["is_friday"] == False]["pnl_r"].values
    if len(fri) >= 10 and len(not_fri) >= 10:
        t, p, d = welch_t(fri, not_fri)
        print(f"  {vol} Friday: N={len(fri)}, avgR={np.nanmean(fri):+.3f}, {sample_label(len(fri))}")
        print(f"  {vol} Mon-Thu: N={len(not_fri)}, avgR={np.nanmean(not_fri):+.3f}, {sample_label(len(not_fri))}")
        print(f"    delta={np.nanmean(fri)-np.nanmean(not_fri):+.3f}, p={fmt_pval(p)}")
        q5_results.append({"label": f"MES_1000_fri_{vol}", "N": len(fri),
                          "avg_r": np.nanmean(fri), "p": p, "d": d})

# Also: expanding vs contracting ATR on Friday specifically
fri_only = df_q5[df_q5["is_friday"] == True]
for regime in ["expanding", "contracting"]:
    grp = fri_only[fri_only["atr_vel_regime"] == regime]["pnl_r"].values
    other = fri_only[fri_only["atr_vel_regime"] != regime]["pnl_r"].values
    if len(grp) >= 10:
        t, p, d = welch_t(grp, other)
        print(f"  Friday {regime}: N={len(grp)}, avgR={np.nanmean(grp):+.3f}, p={fmt_pval(p)}, {sample_label(len(grp))}")
        q5_results.append({"label": f"MES_1000_fri_{regime}",
                          "N": len(grp), "avg_r": np.nanmean(grp), "p": p, "d": d})

# BONUS: test across ALL instruments, not just MES
for sym in ["MGC", "MNQ"]:
    q5b_sql = f"""
    SELECT o.trading_day, o.pnl_r, d.is_friday, d.atr_20
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{sym}' AND o.orb_label = '1000' AND o.orb_minutes = 5
      AND o.entry_model = 'E0' AND o.confirm_bars = 1 AND o.rr_target = 2.0
      AND d.atr_20 IS NOT NULL
    """
    df_5b = con.execute(q5b_sql).fetchdf()
    if len(df_5b) >= 30:
        df_5b = df_5b.sort_values("trading_day").reset_index(drop=True)
        df_5b["exp_med"] = df_5b["atr_20"].expanding(min_periods=20).median().shift(1)
        df_5b = df_5b.dropna(subset=["exp_med"])
        for vol_label, mask in [("high_vol", df_5b["atr_20"] > df_5b["exp_med"]),
                                ("low_vol", df_5b["atr_20"] <= df_5b["exp_med"])]:
            sub = df_5b[mask]
            fri = sub[sub["is_friday"] == True]["pnl_r"].values
            not_fri = sub[sub["is_friday"] == False]["pnl_r"].values
            if len(fri) >= 10 and len(not_fri) >= 10:
                t, p, d = welch_t(fri, not_fri)
                print(f"  {sym} {vol_label} Friday: N={len(fri)}, avgR={np.nanmean(fri):+.3f}, {sample_label(len(fri))} "
                      f"vs Mon-Thu: avgR={np.nanmean(not_fri):+.3f}, p={fmt_pval(p)}")
                q5_results.append({"label": f"{sym}_1000_fri_{vol_label}",
                                  "N": len(fri), "avg_r": np.nanmean(fri), "p": p, "d": d})

findings.append(("Q5: Friday × vol regime", q5_results))

# ============================================================
# BH FDR CORRECTION (all tests pooled)
# ============================================================
print("\n" + "=" * 60)
print("BH FDR CORRECTION")
print("=" * 60)

all_p = []
all_labels = []
for section, results in findings:
    for r in results:
        if not np.isnan(r["p"]):
            all_p.append(r["p"])
            all_labels.append(f"{section}: {r['label']}")

adjusted = bh_fdr(all_p, q=0.10)
survivors = [(l, p, adj) for l, p, adj in zip(all_labels, all_p, adjusted) if adj < 0.10]

print(f"\n  Total tests: {len(all_p)}")
print(f"  BH survivors (q=0.10): {len(survivors)}")
for label, raw_p, adj_p in sorted(survivors, key=lambda x: x[2]):
    print(f"    {label}: raw_p={fmt_pval(raw_p)}, p_bh={fmt_pval(adj_p)}")

# Also show top-5 by raw p-value even if not BH-significant
print("\n  Top 5 by raw p-value:")
ranked = sorted(zip(all_labels, all_p, adjusted), key=lambda x: x[1])[:5]
for label, raw_p, adj_p in ranked:
    print(f"    {label}: raw_p={fmt_pval(raw_p)}, p_bh={fmt_pval(adj_p)}")

# ============================================================
# WRITE FINDINGS REPORT
# ============================================================
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    f.write("# Cross-Session & Cross-Instrument Research Findings\n")
    f.write(f"**Date:** 2026-02-21\n")
    f.write(f"**Script:** research/research_cross_session.py\n")
    f.write(f"**Total tests:** {len(all_p)}\n")
    f.write(f"**BH survivors (q=0.10):** {len(survivors)}\n\n")

    if survivors:
        f.write("## BH Survivors\n")
        for label, raw_p, adj_p in sorted(survivors, key=lambda x: x[2]):
            f.write(f"- **{label}**: raw_p={fmt_pval(raw_p)}, p_bh={fmt_pval(adj_p)}\n")
        f.write("\n")

    f.write("## Top 5 by Raw P-Value\n")
    for label, raw_p, adj_p in ranked:
        f.write(f"- **{label}**: raw_p={fmt_pval(raw_p)}, p_bh={fmt_pval(adj_p)}\n")
    f.write("\n")

    for section, results in findings:
        f.write(f"## {section}\n")
        for r in results:
            sig = " **BH-SIG**" if any(l == f"{section}: {r['label']}" for l, _, a in survivors) else ""
            f.write(f"- {r['label']}: N={r['N']}, avgR={r['avg_r']:+.3f}, "
                    f"p={fmt_pval(r['p'])}, d={r.get('d', 0):+.3f}{sig}\n")
        f.write("\n")

    f.write("## Methodology\n")
    f.write("- All tests: Welch's t-test (unequal variance)\n")
    f.write("- FDR: Benjamini-Hochberg at q=0.10\n")
    f.write("- Anchor: E0 CB1 RR2.0 at 1000 session (standardized comparison)\n")
    f.write("- Look-ahead: break_dir known at entry time; gap/ATR known at day start\n")
    f.write("- orb_minutes=5 filter on all daily_features joins\n")
    f.write("- Sample labels: INVALID(<30) / REGIME(30-99) / PRELIMINARY(100-199) / CORE(200-499) / HIGH-CONFIDENCE(500+)\n")
    f.write("\n## Caveats\n")
    f.write("- Concordance (Q2), gap size (Q3), ATR (Q5) medians use expanding window with shift(1) — NO lookahead\n")
    f.write("- First 20 days dropped (insufficient lookback for stable expanding median)\n")
    f.write("- Q1/Q4: 0900 session used as predictor WITHOUT us_dst split — 0900 is DST-contaminated\n")
    f.write("- MNQ only has ~2 years of data — any MNQ finding is REGIME at best\n")

print(f"\nFindings written to {OUTPUT}")

con.close()

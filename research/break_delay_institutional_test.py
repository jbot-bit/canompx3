"""
Break Speed & Late Entry Institutional Test Battery
====================================================
Tests break_delay effect across ALL active instruments x ALL sessions.
Uses ONLY canonical layers (orb_outcomes, daily_features).
Computes entry_delay from timestamps, not derived columns.

5 Tests:
  T1: Break speed (FAST/MEDIUM/SLOW) per instrument x session
  T2: Late entry (EARLY/MID/LATE) per instrument x session
  T3: Interaction (break speed x entry delay)
  T4: Robustness (year-by-year, ORB size, DST season)
  T5: Confirm bars interaction
"""
import sys
import warnings
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH

warnings.filterwarnings("ignore", category=RuntimeWarning)
BRISBANE = ZoneInfo("Australia/Brisbane")
INSTRUMENTS = ["MNQ", "MGC", "MES"]


# -- Phase 1: Pre-compute orb_close_ts --------------------------------

def resolve_orb_close(trading_day, session_name, orb_minutes):
    """Compute ORB close time (UTC-aware) from session catalog."""
    resolver = SESSION_CATALOG[session_name]["resolver"]
    hour, minute = resolver(trading_day)

    # Trading day: 09:00 Brisbane -> next 09:00 Brisbane
    # Sessions with hour < 9 are on calendar day AFTER trading_day
    cal_day = trading_day + timedelta(days=1) if hour < 9 else trading_day

    session_start = datetime(
        cal_day.year, cal_day.month, cal_day.day, hour, minute, tzinfo=BRISBANE
    )
    return session_start + timedelta(minutes=orb_minutes)


# -- Phase 2: Load data -----------------------------------------------

print("Loading data from canonical layers...")
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

# Load orb_outcomes for active instruments -- only rows with entry_ts (actual trades)
df = con.sql("""
    SELECT trading_day, symbol, orb_label, orb_minutes, rr_target,
           confirm_bars, entry_model, entry_ts, outcome, pnl_r,
           risk_dollars
    FROM orb_outcomes
    WHERE symbol IN ('MNQ', 'MGC', 'MES')
      AND entry_ts IS NOT NULL
      AND outcome != 'no_break'
""").df()
print(f"  orb_outcomes: {len(df):,} rows with entries")

# Load ORB size from daily_features for robustness checks
# We'll build a lookup for orb_size per (trading_day, symbol, session, orb_minutes)
sessions = df["orb_label"].unique()
size_frames = []
for sess in sessions:
    size_col = f"orb_{sess}_size"
    try:
        sf = con.sql(f"""
            SELECT trading_day, symbol, orb_minutes,
                   "{size_col}" as orb_size,
                   '{sess}' as orb_label
            FROM daily_features
            WHERE symbol IN ('MNQ', 'MGC', 'MES')
              AND "{size_col}" IS NOT NULL
        """).df()
        size_frames.append(sf)
    except Exception:
        pass  # Column doesn't exist for this session

if size_frames:
    sizes_df = pd.concat(size_frames, ignore_index=True)
else:
    sizes_df = pd.DataFrame()

con.close()
print(f"  ORB sizes: {len(sizes_df):,} rows")


# -- Phase 3: Compute entry_delay_min from timestamps -----------------

print("Computing entry_delay_min from timestamps...")

# Pre-compute orb_close for all unique (trading_day, session, orb_minutes)
unique_combos = df[["trading_day", "orb_label", "orb_minutes"]].drop_duplicates()
print(f"  Unique (day, session, aperture) combos: {len(unique_combos):,}")

orb_close_cache = {}
errors = 0
for _, row in unique_combos.iterrows():
    key = (row["trading_day"], row["orb_label"], row["orb_minutes"])
    try:
        orb_close_cache[key] = resolve_orb_close(*key)
    except Exception:
        errors += 1

print(f"  Resolved: {len(orb_close_cache):,} session times ({errors} errors)")

# Build orb_close lookup DataFrame for vectorized merge
orb_close_rows = [
    {"trading_day": k[0], "orb_label": k[1], "orb_minutes": k[2], "orb_close_ts": v}
    for k, v in orb_close_cache.items()
]
orb_close_df = pd.DataFrame(orb_close_rows)
orb_close_df["orb_close_ts"] = pd.to_datetime(orb_close_df["orb_close_ts"], utc=True)

# Merge and compute vectorized
df = df.merge(orb_close_df, on=["trading_day", "orb_label", "orb_minutes"], how="inner")
df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
df["entry_delay_min"] = (df["entry_ts"] - df["orb_close_ts"]).dt.total_seconds() / 60.0
df = df.dropna(subset=["entry_delay_min"])

# Sanity: distribution
print(f"  Rows with valid entry_delay: {len(df):,}")
print(f"  entry_delay_min: min={df['entry_delay_min'].min():.1f}, "
      f"median={df['entry_delay_min'].median():.1f}, "
      f"max={df['entry_delay_min'].max():.1f}")

# Negative delays = entry before ORB close (shouldn't happen for valid trades)
neg = (df["entry_delay_min"] < 0).sum()
if neg > 0:
    print(f"  WARNING: {neg} rows with negative entry_delay (entry before ORB close)")
    # Keep them but flag
    df["entry_delay_min"] = df["entry_delay_min"].clip(lower=0)


# -- Helper: statistical functions -------------------------------------

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return np.nan
    return (g1.mean() - g2.mean()) / pooled_std


def ttest_vs_zero(arr):
    """Two-sided t-test H0: mean = 0."""
    arr = arr.dropna()
    if len(arr) < 2:
        return np.nan, np.nan, len(arr)
    t, p = stats.ttest_1samp(arr, 0)
    return t, p, len(arr)


def ttest_two_groups(g1, g2):
    """Two-sample t-test (Welch's)."""
    g1, g2 = g1.dropna(), g2.dropna()
    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan, len(g1) + len(g2) - 2
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    df_approx = len(g1) + len(g2) - 2  # approximate
    return t, p, df_approx


# -- TEST 1: Break Speed Effect ---------------------------------------

print("\n" + "=" * 80)
print("TEST 1: BREAK SPEED EFFECT -- all instruments x all sessions")
print("  Buckets: FAST (<=5m), MEDIUM (5-15m), SLOW (>15m)")
print("  Using entry_delay_min as proxy for break speed")
print("  (For E2/CB1, entry_delay ~= break_delay)")
print("=" * 80)

# Bucket
df["speed_bucket"] = pd.cut(
    df["entry_delay_min"],
    bins=[-0.001, 5, 15, 999999],
    labels=["FAST", "MEDIUM", "SLOW"],
)

results_t1 = []
for (inst, sess, om), grp in df.groupby(["symbol", "orb_label", "orb_minutes"]):
    for bucket in ["FAST", "MEDIUM", "SLOW"]:
        sub = grp[grp["speed_bucket"] == bucket]["pnl_r"]
        n = len(sub)
        if n < 5:
            continue
        mean_pnl = sub.mean()
        median_pnl = sub.median()
        wr = (sub > 0).mean()
        std_pnl = sub.std(ddof=1)
        t_stat, p_val, _ = ttest_vs_zero(sub)

        results_t1.append({
            "instrument": inst, "session": sess, "orb_minutes": om,
            "bucket": bucket, "N": n, "mean_pnl_r": mean_pnl,
            "median_pnl_r": median_pnl, "win_rate": wr,
            "std": std_pnl, "ttest_p": p_val,
        })

t1_df = pd.DataFrame(results_t1)

# BH FDR correction
if len(t1_df) > 0 and t1_df["ttest_p"].notna().sum() > 0:
    valid_mask = t1_df["ttest_p"].notna()
    reject, adj_p, _, _ = multipletests(
        t1_df.loc[valid_mask, "ttest_p"], alpha=0.05, method="fdr_bh"
    )
    t1_df.loc[valid_mask, "bh_adjusted_p"] = adj_p
    t1_df.loc[valid_mask, "bh_significant"] = reject
    K_t1 = valid_mask.sum()
else:
    K_t1 = 0

# FAST vs SLOW comparison per (instrument, session, orb_minutes)
fast_slow_results = []
for (inst, sess, om), grp in df.groupby(["symbol", "orb_label", "orb_minutes"]):
    fast = grp[grp["speed_bucket"] == "FAST"]["pnl_r"]
    slow = grp[grp["speed_bucket"] == "SLOW"]["pnl_r"]
    if len(fast) < 10 or len(slow) < 10:
        continue
    t_stat, p_val, dof = ttest_two_groups(fast, slow)
    d = cohens_d(fast, slow)
    fast_slow_results.append({
        "instrument": inst, "session": sess, "orb_minutes": om,
        "fast_N": len(fast), "slow_N": len(slow),
        "fast_mean": fast.mean(), "slow_mean": slow.mean(),
        "fast_vs_slow_p": p_val, "cohens_d": d, "dof": dof,
    })

fs_df = pd.DataFrame(fast_slow_results)

# BH FDR on FAST vs SLOW tests
if len(fs_df) > 0 and fs_df["fast_vs_slow_p"].notna().sum() > 0:
    valid_mask = fs_df["fast_vs_slow_p"].notna()
    reject, adj_p, _, _ = multipletests(
        fs_df.loc[valid_mask, "fast_vs_slow_p"], alpha=0.05, method="fdr_bh"
    )
    fs_df.loc[valid_mask, "bh_adjusted_p"] = adj_p
    fs_df.loc[valid_mask, "bh_significant"] = reject
    K_fs = valid_mask.sum()
else:
    K_fs = 0

# Print Test 1 results
print(f"\nK (per-bucket tests) = {K_t1}")
print(f"K (FAST vs SLOW tests) = {K_fs}")

print("\n--- Per-Bucket Results (sorted by session, instrument) ---")
if len(t1_df) > 0:
    for (sess, inst), grp in t1_df.groupby(["session", "instrument"]):
        print(f"\n  {inst} {sess}:")
        for _, row in grp.sort_values("bucket").iterrows():
            sig = "*" if row.get("bh_significant", False) else ""
            bh_p = row.get("bh_adjusted_p", np.nan)
            bh_str = f"BH_p={bh_p:.4f}" if not np.isnan(bh_p) else "BH_p=N/A"
            flag = " [N<30]" if row["N"] < 30 else ""
            print(
                f"    O{row['orb_minutes']:<3d} {row['bucket']:<7s} "
                f"N={row['N']:<5d} mean={row['mean_pnl_r']:+.4f} "
                f"med={row['median_pnl_r']:+.4f} WR={row['win_rate']:.1%} "
                f"std={row['std']:.4f} p={row['ttest_p']:.4f} "
                f"{bh_str}{sig}{flag}"
            )

print("\n--- FAST vs SLOW Comparison ---")
print(f"{'Inst':<5s} {'Session':<20s} OM  {'Fast_N':>6s} {'Slow_N':>6s} "
      f"{'Fast_mean':>10s} {'Slow_mean':>10s} {'p':>8s} {'BH_p':>8s} "
      f"{'d':>6s} {'Sig':>4s}")
print("-" * 110)
if len(fs_df) > 0:
    for _, row in fs_df.sort_values(["session", "instrument"]).iterrows():
        sig = "YES" if row.get("bh_significant", False) else "no"
        bh_p = row.get("bh_adjusted_p", np.nan)
        bh_str = f"{bh_p:.4f}" if not np.isnan(bh_p) else "N/A"
        d_str = f"{row['cohens_d']:.3f}" if not np.isnan(row["cohens_d"]) else "N/A"
        print(
            f"{row['instrument']:<5s} {row['session']:<20s} {row['orb_minutes']:<3d} "
            f"{row['fast_N']:>6d} {row['slow_N']:>6d} "
            f"{row['fast_mean']:>+10.4f} {row['slow_mean']:>+10.4f} "
            f"{row['fast_vs_slow_p']:>8.4f} {bh_str:>8s} "
            f"{d_str:>6s} {sig:>4s}"
        )

# Identify surviving sessions for Tests 3-5
surviving_t1 = set()
if len(fs_df) > 0:
    for _, row in fs_df.iterrows():
        if row.get("bh_significant", False) and abs(row.get("cohens_d", 0)) >= 0.2:
            surviving_t1.add((row["instrument"], row["session"], row["orb_minutes"]))

print(f"\nSurvivors (BH sig + |d|>=0.2): {len(surviving_t1)}")
for s in sorted(surviving_t1):
    print(f"  {s}")


# -- TEST 2: Late Entry Effect ----------------------------------------

print("\n" + "=" * 80)
print("TEST 2: LATE ENTRY EFFECT -- all instruments x all sessions")
print("  Buckets: EARLY (<=10m), MID (10-30m), LATE (>30m)")
print("=" * 80)

df["entry_bucket"] = pd.cut(
    df["entry_delay_min"],
    bins=[-0.001, 10, 30, 999999],
    labels=["EARLY", "MID", "LATE"],
)

results_t2 = []
for (inst, sess, om), grp in df.groupby(["symbol", "orb_label", "orb_minutes"]):
    for bucket in ["EARLY", "MID", "LATE"]:
        sub = grp[grp["entry_bucket"] == bucket]["pnl_r"]
        n = len(sub)
        if n < 5:
            continue
        mean_pnl = sub.mean()
        median_pnl = sub.median()
        wr = (sub > 0).mean()
        std_pnl = sub.std(ddof=1)
        t_stat, p_val, _ = ttest_vs_zero(sub)
        results_t2.append({
            "instrument": inst, "session": sess, "orb_minutes": om,
            "bucket": bucket, "N": n, "mean_pnl_r": mean_pnl,
            "median_pnl_r": median_pnl, "win_rate": wr,
            "std": std_pnl, "ttest_p": p_val,
        })

t2_df = pd.DataFrame(results_t2)

if len(t2_df) > 0 and t2_df["ttest_p"].notna().sum() > 0:
    valid_mask = t2_df["ttest_p"].notna()
    reject, adj_p, _, _ = multipletests(
        t2_df.loc[valid_mask, "ttest_p"], alpha=0.05, method="fdr_bh"
    )
    t2_df.loc[valid_mask, "bh_adjusted_p"] = adj_p
    t2_df.loc[valid_mask, "bh_significant"] = reject
    K_t2 = valid_mask.sum()
else:
    K_t2 = 0

# EARLY vs LATE comparison
early_late_results = []
for (inst, sess, om), grp in df.groupby(["symbol", "orb_label", "orb_minutes"]):
    early = grp[grp["entry_bucket"] == "EARLY"]["pnl_r"]
    late = grp[grp["entry_bucket"] == "LATE"]["pnl_r"]
    if len(early) < 10 or len(late) < 10:
        continue
    t_stat, p_val, dof = ttest_two_groups(early, late)
    d = cohens_d(early, late)
    early_late_results.append({
        "instrument": inst, "session": sess, "orb_minutes": om,
        "early_N": len(early), "late_N": len(late),
        "early_mean": early.mean(), "late_mean": late.mean(),
        "early_vs_late_p": p_val, "cohens_d": d, "dof": dof,
    })

el_df = pd.DataFrame(early_late_results)

if len(el_df) > 0 and el_df["early_vs_late_p"].notna().sum() > 0:
    valid_mask = el_df["early_vs_late_p"].notna()
    reject, adj_p, _, _ = multipletests(
        el_df.loc[valid_mask, "early_vs_late_p"], alpha=0.05, method="fdr_bh"
    )
    el_df.loc[valid_mask, "bh_adjusted_p"] = adj_p
    el_df.loc[valid_mask, "bh_significant"] = reject
    K_el = valid_mask.sum()
else:
    K_el = 0

print(f"\nK (per-bucket tests) = {K_t2}")
print(f"K (EARLY vs LATE tests) = {K_el}")

print("\n--- Per-Bucket Results ---")
if len(t2_df) > 0:
    for (sess, inst), grp in t2_df.groupby(["session", "instrument"]):
        print(f"\n  {inst} {sess}:")
        for _, row in grp.sort_values("bucket").iterrows():
            sig = "*" if row.get("bh_significant", False) else ""
            bh_p = row.get("bh_adjusted_p", np.nan)
            bh_str = f"BH_p={bh_p:.4f}" if not np.isnan(bh_p) else "BH_p=N/A"
            flag = " [N<30]" if row["N"] < 30 else ""
            print(
                f"    O{row['orb_minutes']:<3d} {row['bucket']:<6s} "
                f"N={row['N']:<5d} mean={row['mean_pnl_r']:+.4f} "
                f"med={row['median_pnl_r']:+.4f} WR={row['win_rate']:.1%} "
                f"std={row['std']:.4f} p={row['ttest_p']:.4f} "
                f"{bh_str}{sig}{flag}"
            )

print("\n--- EARLY vs LATE Comparison ---")
print(f"{'Inst':<5s} {'Session':<20s} OM  {'Early_N':>7s} {'Late_N':>7s} "
      f"{'Early_mean':>11s} {'Late_mean':>10s} {'p':>8s} {'BH_p':>8s} "
      f"{'d':>6s} {'Sig':>4s}")
print("-" * 110)
if len(el_df) > 0:
    for _, row in el_df.sort_values(["session", "instrument"]).iterrows():
        sig = "YES" if row.get("bh_significant", False) else "no"
        bh_p = row.get("bh_adjusted_p", np.nan)
        bh_str = f"{bh_p:.4f}" if not np.isnan(bh_p) else "N/A"
        d_str = f"{row['cohens_d']:.3f}" if not np.isnan(row["cohens_d"]) else "N/A"
        print(
            f"{row['instrument']:<5s} {row['session']:<20s} {row['orb_minutes']:<3d} "
            f"{row['early_N']:>7d} {row['late_N']:>7d} "
            f"{row['early_mean']:>+11.4f} {row['late_mean']:>+10.4f} "
            f"{row['early_vs_late_p']:>8.4f} {bh_str:>8s} "
            f"{d_str:>6s} {sig:>4s}"
        )

surviving_t2 = set()
if len(el_df) > 0:
    for _, row in el_df.iterrows():
        if row.get("bh_significant", False) and abs(row.get("cohens_d", 0)) >= 0.2:
            surviving_t2.add((row["instrument"], row["session"], row["orb_minutes"]))

print(f"\nSurvivors (BH sig + |d|>=0.2): {len(surviving_t2)}")
for s in sorted(surviving_t2):
    print(f"  {s}")


# -- TEST 3: Interaction ----------------------------------------------

both_surviving = surviving_t1 & surviving_t2

print("\n" + "=" * 80)
print("TEST 3: INTERACTION -- break speed x entry delay")
print(f"  Sessions with BOTH effects: {len(both_surviving)}")
print("=" * 80)

if both_surviving:
    for inst, sess, om in sorted(both_surviving):
        grp = df[(df["symbol"] == inst) & (df["orb_label"] == sess) & (df["orb_minutes"] == om)]
        print(f"\n  {inst} {sess} O{om}:")

        # Cross-tabulation
        print(f"  {'':>12s} {'EARLY':>12s} {'MID':>12s} {'LATE':>12s}")
        for speed in ["FAST", "MEDIUM", "SLOW"]:
            row_parts = []
            for entry in ["EARLY", "MID", "LATE"]:
                cell = grp[(grp["speed_bucket"] == speed) & (grp["entry_bucket"] == entry)]["pnl_r"]
                if len(cell) >= 5:
                    row_parts.append(f"N={len(cell):>3d} {cell.mean():+.3f}")
                else:
                    row_parts.append(f"N={len(cell):>3d}   ---")
            print(f"  {speed:>12s} {row_parts[0]:>12s} {row_parts[1]:>12s} {row_parts[2]:>12s}")

        # FAST_EARLY vs alternatives
        fast_early = grp[(grp["speed_bucket"] == "FAST") & (grp["entry_bucket"] == "EARLY")]["pnl_r"]
        fast_all = grp[grp["speed_bucket"] == "FAST"]["pnl_r"]
        early_all = grp[grp["entry_bucket"] == "EARLY"]["pnl_r"]
        unfiltered = grp["pnl_r"]

        print(f"\n  Comparison:")
        print(f"    FAST_EARLY:  N={len(fast_early):>5d} mean={fast_early.mean():+.4f} WR={((fast_early>0).mean()):.1%}")
        print(f"    FAST_ALL:    N={len(fast_all):>5d} mean={fast_all.mean():+.4f} WR={((fast_all>0).mean()):.1%}")
        print(f"    EARLY_ALL:   N={len(early_all):>5d} mean={early_all.mean():+.4f} WR={((early_all>0).mean()):.1%}")
        print(f"    UNFILTERED:  N={len(unfiltered):>5d} mean={unfiltered.mean():+.4f} WR={((unfiltered>0).mean()):.1%}")

        # FAST_EARLY vs FAST_ALL
        fast_not_early = grp[(grp["speed_bucket"] == "FAST") & (grp["entry_bucket"] != "EARLY")]["pnl_r"]
        if len(fast_early) >= 10 and len(fast_not_early) >= 10:
            t, p, dof = ttest_two_groups(fast_early, fast_not_early)
            d = cohens_d(fast_early, fast_not_early)
            print(f"\n  FAST_EARLY vs FAST_NOT_EARLY: p={p:.4f}, d={d:.3f}")
            if p > 0.10:
                print("  -> Interaction adds NOTHING over break speed alone")
            else:
                print(f"  -> Interaction IS additive (p={p:.4f})")
        else:
            print("  -> Insufficient N for interaction test")
else:
    print("  No sessions have both effects surviving. Test 3 = N/A.")


# -- TEST 4: Robustness ----------------------------------------------

print("\n" + "=" * 80)
print("TEST 4: ROBUSTNESS CHECKS")
print("=" * 80)

# Only test sessions where break speed survived T1
for inst, sess, om in sorted(surviving_t1):
    grp = df[(df["symbol"] == inst) & (df["orb_label"] == sess) & (df["orb_minutes"] == om)]
    fast = grp[grp["speed_bucket"] == "FAST"]["pnl_r"]
    slow = grp[grp["speed_bucket"] == "SLOW"]["pnl_r"]

    print(f"\n  --- {inst} {sess} O{om} ---")

    # 4a: Year-by-year stability
    print("  4a. Year-by-year (FAST mean - SLOW mean):")
    grp_copy = grp.copy()
    grp_copy["year"] = grp_copy["trading_day"].apply(lambda d: d.year)
    years = sorted(grp_copy["year"].unique())
    fast_wins = 0
    total_years = 0
    for yr in years:
        yr_grp = grp_copy[grp_copy["year"] == yr]
        yr_fast = yr_grp[yr_grp["speed_bucket"] == "FAST"]["pnl_r"]
        yr_slow = yr_grp[yr_grp["speed_bucket"] == "SLOW"]["pnl_r"]
        if len(yr_fast) >= 5 and len(yr_slow) >= 5:
            delta = yr_fast.mean() - yr_slow.mean()
            total_years += 1
            if delta > 0:
                fast_wins += 1
            print(f"    {yr}: FAST={yr_fast.mean():+.4f} (N={len(yr_fast)}), "
                  f"SLOW={yr_slow.mean():+.4f} (N={len(yr_slow)}), "
                  f"delta={delta:+.4f} {'Y' if delta > 0 else 'N'}")
        else:
            print(f"    {yr}: N too small (FAST={len(yr_fast)}, SLOW={len(yr_slow)})")

    if total_years > 0:
        pct = fast_wins / total_years
        stable = "STABLE" if pct >= 0.6 else "FRAGILE"
        print(f"  -> FAST > SLOW in {fast_wins}/{total_years} years ({pct:.0%}) = {stable}")
    else:
        print("  -> Insufficient years")

    # 4b: ORB size interaction
    print("  4b. ORB size interaction:")
    if len(sizes_df) > 0:
        merged = grp.merge(
            sizes_df,
            on=["trading_day", "symbol", "orb_label", "orb_minutes"],
            how="inner",
        )
        if len(merged) > 0 and "orb_size" in merged.columns:
            median_size = merged["orb_size"].median()
            for label, subset in [("small", merged[merged["orb_size"] <= median_size]),
                                  ("large", merged[merged["orb_size"] > median_size])]:
                s_fast = subset[subset["speed_bucket"] == "FAST"]["pnl_r"]
                s_slow = subset[subset["speed_bucket"] == "SLOW"]["pnl_r"]
                if len(s_fast) >= 10 and len(s_slow) >= 10:
                    t, p, _ = ttest_two_groups(s_fast, s_slow)
                    print(f"    {label} ORB (<=>{median_size:.1f}): "
                          f"FAST={s_fast.mean():+.4f} (N={len(s_fast)}), "
                          f"SLOW={s_slow.mean():+.4f} (N={len(s_slow)}), p={p:.4f}")
                else:
                    print(f"    {label} ORB: N too small (FAST={len(s_fast)}, SLOW={len(s_slow)})")
        else:
            print("    No ORB size data after merge")
    else:
        print("    No ORB size data available")

    # 4c: DST/season check (approximate: Nov-Mar = winter, Apr-Oct = summer for US)
    print("  4c. DST/season:")
    grp_copy = grp.copy()
    grp_copy["month"] = grp_copy["trading_day"].apply(lambda d: d.month)
    grp_copy["season"] = grp_copy["month"].apply(
        lambda m: "WINTER" if m in [11, 12, 1, 2, 3] else "SUMMER"
    )
    for season in ["WINTER", "SUMMER"]:
        s_grp = grp_copy[grp_copy["season"] == season]
        s_fast = s_grp[s_grp["speed_bucket"] == "FAST"]["pnl_r"]
        s_slow = s_grp[s_grp["speed_bucket"] == "SLOW"]["pnl_r"]
        if len(s_fast) >= 10 and len(s_slow) >= 10:
            t, p, _ = ttest_two_groups(s_fast, s_slow)
            print(f"    {season}: FAST={s_fast.mean():+.4f} (N={len(s_fast)}), "
                  f"SLOW={s_slow.mean():+.4f} (N={len(s_slow)}), p={p:.4f}")
        else:
            print(f"    {season}: N too small (FAST={len(s_fast)}, SLOW={len(s_slow)})")

    # 4d: Sample size adequacy
    print("  4d. Sample size:")
    for bucket in ["FAST", "MEDIUM", "SLOW"]:
        n = len(grp[grp["speed_bucket"] == bucket])
        flag = " [INADEQUATE]" if n < 50 else ""
        print(f"    {bucket}: N={n}{flag}")

if not surviving_t1:
    print("  No survivors from Test 1. Test 4 = N/A.")


# -- TEST 5: Confirm Bars Interaction ---------------------------------

print("\n" + "=" * 80)
print("TEST 5: CONFIRM BARS x BREAK SPEED INTERACTION")
print("=" * 80)

for inst, sess, om in sorted(surviving_t1):
    grp = df[(df["symbol"] == inst) & (df["orb_label"] == sess) & (df["orb_minutes"] == om)]
    print(f"\n  --- {inst} {sess} O{om} ---")
    print(f"  {'CB':>4s} {'Fast_N':>7s} {'Fast_mean':>10s} {'Slow_N':>7s} "
          f"{'Slow_mean':>10s} {'Delta':>8s} {'p':>8s}")
    print("  " + "-" * 65)

    cb_effect_at_1_only = True
    for cb in sorted(grp["confirm_bars"].unique()):
        cb_grp = grp[grp["confirm_bars"] == cb]
        cb_fast = cb_grp[cb_grp["speed_bucket"] == "FAST"]["pnl_r"]
        cb_slow = cb_grp[cb_grp["speed_bucket"] == "SLOW"]["pnl_r"]
        if len(cb_fast) >= 10 and len(cb_slow) >= 10:
            t, p, _ = ttest_two_groups(cb_fast, cb_slow)
            delta = cb_fast.mean() - cb_slow.mean()
            sig = "*" if p < 0.05 else ""
            print(f"  CB{cb:>2d} {len(cb_fast):>7d} {cb_fast.mean():>+10.4f} "
                  f"{len(cb_slow):>7d} {cb_slow.mean():>+10.4f} "
                  f"{delta:>+8.4f} {p:>8.4f}{sig}")
            if cb > 1 and p < 0.10:
                cb_effect_at_1_only = False
        else:
            print(f"  CB{cb:>2d} {len(cb_fast):>7d}        --- "
                  f"{len(cb_slow):>7d}        --- "
                  f"      ---       --- [N too small]")

    if cb_effect_at_1_only:
        print("  -> Effect ONLY at CB=1: may be confirmation timing, not break speed")
    else:
        print("  -> Effect persists at CB>=2: genuine break speed signal")

if not surviving_t1:
    print("  No survivors from Test 1. Test 5 = N/A.")


# -- SUMMARY ----------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Total rows analyzed: {len(df):,}")
print(f"  Test 1 survivors (break speed, BH + |d|>=0.2): {len(surviving_t1)}")
print(f"  Test 2 survivors (late entry, BH + |d|>=0.2): {len(surviving_t2)}")
print(f"  Test 3 interaction sessions: {len(both_surviving)}")
print(f"  BH FDR K values: T1_buckets={K_t1}, T1_fast_slow={K_fs}, "
      f"T2_buckets={K_t2}, T2_early_late={K_el}")

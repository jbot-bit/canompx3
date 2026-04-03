"""Research: Prior-day close position effect on ORB breakout outcomes.

Hypothesis: ORB breakout outcomes differ on days where the prior day closed
near its extreme (exhaustion signal), AFTER conditioning on direction.

Tests A-D are promotion-eligible (K=6 for BH FDR).
Quintile scan and permutation test are sanity checks only.

@research-source: prev_close_position_test.py
@data-source: orb_outcomes JOIN daily_features (canonical layers)
@look-ahead-safe: prev_day_* uses rows[i-1] in build_daily_features.py:1271-1281
"""

import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

# Load all O5 data with prev_close_position, conditioned on direction
df = con.execute("""
    SELECT
        o.symbol AS instrument,
        o.trading_day,
        o.orb_label,
        o.direction,
        o.pnl_r,
        d.prev_day_direction,
        (d.prev_day_close - d.prev_day_low) / d.prev_day_range AS prev_close_pos
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_minutes = 5
      AND o.symbol IN ('MGC', 'MNQ', 'MES')
      AND d.prev_day_range > 0
      AND d.prev_day_direction IS NOT NULL
      AND o.pnl_r IS NOT NULL
""").fetchdf()

print(f"Total rows: {len(df)}")
print(
    f"Bear days: {(df['prev_day_direction'] == 'bear').sum()}, Bull days: {(df['prev_day_direction'] == 'bull').sum()}"
)
print()

bear = df[df["prev_day_direction"] == "bear"]
bull = df[df["prev_day_direction"] == "bull"]

all_p = []  # Collect for BH FDR


def report_comparison(label, group1, group2, label1="Extreme", label2="Rest"):
    """Run t-test and print results. Returns (label, p_value)."""
    t, p = stats.ttest_ind(group1, group2)
    n1, n2 = len(group1), len(group2)
    m1, m2 = group1.mean(), group2.mean()
    wr1 = (group1 > 0).sum() / n1
    wr2 = (group2 > 0).sum() / n2
    print(f"  {label1:12s}: N={n1:5d}  mean_R={m1:+.5f}  WR={wr1:.1%}  std={group1.std():.3f}")
    print(f"  {label2:12s}: N={n2:5d}  mean_R={m2:+.5f}  WR={wr2:.1%}  std={group2.std():.3f}")
    print(f"  Delta mean_R: {m1 - m2:+.5f}")
    print(f"  t={t:.3f}, p={p:.6f}")
    return (label, p)


# =====================================================================
# TEST A: Bear days — extreme low close vs rest
# =====================================================================
print("=" * 80)
print("TEST A: BEAR DAYS — extreme low close (<0.2) vs rest (>=0.2)")
print("=" * 80)
bear_extreme = bear[bear["prev_close_pos"] < 0.2]["pnl_r"]
bear_rest = bear[bear["prev_close_pos"] >= 0.2]["pnl_r"]
result = report_comparison("A: Bear ext vs rest", bear_extreme, bear_rest)
all_p.append(result)

print("  Per instrument:")
for inst in sorted(df["instrument"].unique()):
    sub_ext = bear[(bear["prev_close_pos"] < 0.2) & (bear["instrument"] == inst)]["pnl_r"]
    sub_rest = bear[(bear["prev_close_pos"] >= 0.2) & (bear["instrument"] == inst)]["pnl_r"]
    if len(sub_ext) > 10 and len(sub_rest) > 10:
        t_i, p_i = stats.ttest_ind(sub_ext, sub_rest)
        print(
            f"    {inst}: ext N={len(sub_ext)}, mean={sub_ext.mean():+.5f} | "
            f"rest N={len(sub_rest)}, mean={sub_rest.mean():+.5f} | p={p_i:.4f}"
        )
print()

# =====================================================================
# TEST B: Bear days — directional split (long vs short separately)
# =====================================================================
print("=" * 80)
print("TEST B: BEAR DAYS — extreme low, LONG vs SHORT outcomes separately")
print("=" * 80)
for direction in ["long", "short"]:
    bear_dir = bear[bear["direction"] == direction]
    ext = bear_dir[bear_dir["prev_close_pos"] < 0.2]["pnl_r"]
    rest = bear_dir[bear_dir["prev_close_pos"] >= 0.2]["pnl_r"]
    if len(ext) > 10 and len(rest) > 10:
        result = report_comparison(f"B: Bear ext {direction}", ext, rest, f"Ext {direction}", f"Rest {direction}")
        all_p.append(result)
        print()

# =====================================================================
# TEST C: Bull days — extreme high close vs rest
# =====================================================================
print("=" * 80)
print("TEST C: BULL DAYS — extreme high close (>0.8) vs rest (<=0.8)")
print("=" * 80)
bull_extreme = bull[bull["prev_close_pos"] > 0.8]["pnl_r"]
bull_rest = bull[bull["prev_close_pos"] <= 0.8]["pnl_r"]
result = report_comparison("C: Bull ext vs rest", bull_extreme, bull_rest)
all_p.append(result)

print("  Per instrument:")
for inst in sorted(df["instrument"].unique()):
    sub_ext = bull[(bull["prev_close_pos"] > 0.8) & (bull["instrument"] == inst)]["pnl_r"]
    sub_rest = bull[(bull["prev_close_pos"] <= 0.8) & (bull["instrument"] == inst)]["pnl_r"]
    if len(sub_ext) > 10 and len(sub_rest) > 10:
        t_i, p_i = stats.ttest_ind(sub_ext, sub_rest)
        print(
            f"    {inst}: ext N={len(sub_ext)}, mean={sub_ext.mean():+.5f} | "
            f"rest N={len(sub_rest)}, mean={sub_rest.mean():+.5f} | p={p_i:.4f}"
        )
print()

# =====================================================================
# TEST D: Bull days — directional split
# =====================================================================
print("=" * 80)
print("TEST D: BULL DAYS — extreme high, LONG vs SHORT outcomes separately")
print("=" * 80)
for direction in ["long", "short"]:
    bull_dir = bull[bull["direction"] == direction]
    ext = bull_dir[bull_dir["prev_close_pos"] > 0.8]["pnl_r"]
    rest = bull_dir[bull_dir["prev_close_pos"] <= 0.8]["pnl_r"]
    if len(ext) > 10 and len(rest) > 10:
        result = report_comparison(f"D: Bull ext {direction}", ext, rest, f"Ext {direction}", f"Rest {direction}")
        all_p.append(result)
        print()

# =====================================================================
# BH FDR CORRECTION
# =====================================================================
print("=" * 80)
print("BH FDR CORRECTION")
print("=" * 80)

K = len(all_p)
print(f"K = {K} tests")
sorted_p = sorted(all_p, key=lambda x: x[1])
any_survive = False
for rank, (name, p) in enumerate(sorted_p, 1):
    bh_threshold = 0.05 * rank / K
    survives = "SURVIVES" if p <= bh_threshold else "KILLED"
    if p <= bh_threshold:
        any_survive = True
    print(f"  Rank {rank}: {name:30s} p={p:.6f}  BH_thresh={bh_threshold:.4f}  {survives}")

print()
if not any_survive:
    print(">>> NO TESTS SURVIVE BH FDR. Feature has no significant effect.")
else:
    print(">>> At least one test survives BH FDR. Check per-instrument and year stability.")

# =====================================================================
# SANITY CHECK 1: NTILE(5) quintile means (within-direction)
# =====================================================================
print()
print("=" * 80)
print("SANITY CHECK: Within-direction quintile means (not promotion-eligible)")
print("=" * 80)

for dir_label, subset in [("BEAR", bear), ("BULL", bull)]:
    subset = subset.copy()
    subset["quintile"] = pd.qcut(subset["prev_close_pos"], 5, labels=False, duplicates="drop") + 1
    print(f"\n  {dir_label} days:")
    for q in sorted(subset["quintile"].unique()):
        qdata = subset[subset["quintile"] == q]["pnl_r"]
        print(f"    Q{q}: N={len(qdata):5d}  mean_R={qdata.mean():+.5f}  WR={((qdata > 0).sum() / len(qdata)):.1%}")

# =====================================================================
# SANITY CHECK 2: Permutation test (1000 shuffles)
# =====================================================================
print()
print("=" * 80)
print("SANITY CHECK: Permutation test — is extreme vs rest delta real?")
print("=" * 80)

rng = np.random.default_rng(42)
N_PERMS = 1000

# Test the largest effect: bear extreme vs rest
observed_delta = bear_extreme.mean() - bear_rest.mean()
n_ext = len(bear_extreme)
combined = bear["pnl_r"].values
perm_deltas = np.zeros(N_PERMS)

for i in range(N_PERMS):
    shuffled = rng.permutation(combined)
    perm_deltas[i] = shuffled[:n_ext].mean() - shuffled[n_ext:].mean()

perm_p = (np.abs(perm_deltas) >= np.abs(observed_delta)).mean()
print(f"  Bear extreme vs rest: observed delta = {observed_delta:+.5f}")
print(f"  Permutation p-value (1000 shuffles): {perm_p:.4f}")
print(f"  Observed delta rank: {(np.abs(perm_deltas) >= np.abs(observed_delta)).sum()}/{N_PERMS}")

# Same for bull
observed_delta_bull = bull_extreme.mean() - bull_rest.mean()
n_ext_bull = len(bull_extreme)
combined_bull = bull["pnl_r"].values
perm_deltas_bull = np.zeros(N_PERMS)

for i in range(N_PERMS):
    shuffled = rng.permutation(combined_bull)
    perm_deltas_bull[i] = shuffled[:n_ext_bull].mean() - shuffled[n_ext_bull:].mean()

perm_p_bull = (np.abs(perm_deltas_bull) >= np.abs(observed_delta_bull)).mean()
print(f"\n  Bull extreme vs rest: observed delta = {observed_delta_bull:+.5f}")
print(f"  Permutation p-value (1000 shuffles): {perm_p_bull:.4f}")
print(f"  Observed delta rank: {(np.abs(perm_deltas_bull) >= np.abs(observed_delta_bull)).sum()}/{N_PERMS}")

# =====================================================================
# YEAR-BY-YEAR STABILITY (for any test that looked promising)
# =====================================================================
print()
print("=" * 80)
print("YEAR-BY-YEAR STABILITY: Bear extreme vs rest")
print("=" * 80)

bear_copy = bear.copy()
bear_copy["year"] = pd.to_datetime(bear_copy["trading_day"]).dt.year
for year in sorted(bear_copy["year"].unique()):
    yr_data = bear_copy[bear_copy["year"] == year]
    ext = yr_data[yr_data["prev_close_pos"] < 0.2]["pnl_r"]
    rest = yr_data[yr_data["prev_close_pos"] >= 0.2]["pnl_r"]
    if len(ext) > 5 and len(rest) > 5:
        delta = ext.mean() - rest.mean()
        print(
            f"  {year}: ext N={len(ext):4d} mean={ext.mean():+.5f} | rest N={len(rest):4d} mean={rest.mean():+.5f} | delta={delta:+.5f}"
        )

print()
print("=" * 80)
print("YEAR-BY-YEAR STABILITY: Bull extreme vs rest")
print("=" * 80)

bull_copy = bull.copy()
bull_copy["year"] = pd.to_datetime(bull_copy["trading_day"]).dt.year
for year in sorted(bull_copy["year"].unique()):
    yr_data = bull_copy[bull_copy["year"] == year]
    ext = yr_data[yr_data["prev_close_pos"] > 0.8]["pnl_r"]
    rest = yr_data[yr_data["prev_close_pos"] <= 0.8]["pnl_r"]
    if len(ext) > 5 and len(rest) > 5:
        delta = ext.mean() - rest.mean()
        print(
            f"  {year}: ext N={len(ext):4d} mean={ext.mean():+.5f} | rest N={len(rest):4d} mean={rest.mean():+.5f} | delta={delta:+.5f}"
        )

con.close()
print()
print("=" * 80)
print("DONE")
print("=" * 80)

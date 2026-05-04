"""Skeptical re-audit of OVNRNG_50_FAST10 finding from PR #44.

Tests whether the OVNRNG_50 (absolute-points threshold) is a scale/regime
artifact — MNQ tripled from ~7000 to ~20000 over 2019-2026, so a 50-point
absolute threshold is 0.7% of price in 2019 but 0.25% in 2026.

Canonical truth only: orb_outcomes JOIN daily_features. No derived layers.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")

print("=" * 80)
print("SKEPTICAL RE-AUDIT — OVNRNG_50_FAST10 at MNQ NYSE_OPEN 5m E2 RR1.0 CB1")
print(f"Ran at: {pd.Timestamp.now('UTC')}")
print("=" * 80)

q = """
SELECT o.trading_day, o.pnl_r,
       CASE WHEN o.pnl_r > 0 THEN 1 ELSE 0 END AS win,
       o.entry_price,
       d.*
FROM orb_outcomes o
JOIN daily_features d
  ON o.trading_day = d.trading_day
 AND o.symbol = d.symbol
 AND o.orb_minutes = d.orb_minutes
WHERE o.symbol = 'MNQ'
  AND o.orb_label = 'NYSE_OPEN'
  AND o.orb_minutes = 5
  AND o.entry_model = 'E2'
  AND o.rr_target = 1.0
  AND o.confirm_bars = 1
  AND o.pnl_r IS NOT NULL
ORDER BY o.trading_day
"""
df = DB.execute(q).fetchdf()
df["year"] = pd.to_datetime(df["trading_day"]).dt.year
print(f"\nUniverse: MNQ NYSE_OPEN 5m E2 RR1.0 CB1, n={len(df)}, "
      f"{df.trading_day.min()} → {df.trading_day.max()}")

# Step 1: price / overnight-range distribution by year
print("\n" + "-" * 80)
print("STEP 1 — PRICE LEVEL AND OVERNIGHT_RANGE DISTRIBUTION BY YEAR")
print("-" * 80)
print(f"  {'Year':6s} {'N':>5s} {'price':>9s} {'ovn_mean':>9s} "
      f"{'ovn%_price':>11s} {'ovn_p50':>8s} {'ovn_p90':>8s}")
for y, grp in df.groupby("year"):
    price = grp["entry_price"].mean()
    ovn = grp["overnight_range"].mean()
    p50 = grp["overnight_range"].quantile(0.50)
    p90 = grp["overnight_range"].quantile(0.90)
    ovn_pct = ovn / price * 100
    print(f"  {y:6d} {len(grp):>5d} {price:>9.1f} {ovn:>9.1f} "
          f"{ovn_pct:>10.3f}% {p50:>8.1f} {p90:>8.1f}")

# Step 2: fire rate year-over-year
print("\n" + "-" * 80)
print("STEP 2 — OVNRNG_50_FAST10 FIRE RATE BY YEAR (scale-artifact check)")
print("-" * 80)
sig_combined = filter_signal(df, "OVNRNG_50_FAST10", "NYSE_OPEN")
sig_ovn = filter_signal(df, "OVNRNG_50", "NYSE_OPEN")
df["fire_combined"] = sig_combined
df["fire_ovn"] = sig_ovn
print(f"  {'Year':6s} {'N':>5s} {'OVNRNG_50 fire':>16s} {'OVN+FAST10':>12s}")
for y, grp in df.groupby("year"):
    print(f"  {y:6d} {len(grp):>5d} {grp['fire_ovn'].mean():>16.3f} "
          f"{grp['fire_combined'].mean():>12.3f}")

# Step 3: lift decomposition
print("\n" + "-" * 80)
print("STEP 3 — PER-YEAR LIFT DECOMPOSITION")
print("-" * 80)
print(f"  {'Year':6s} {'N_unf':>6s} {'ExpR_unf':>10s} {'N_filt':>7s} "
      f"{'ExpR_filt':>11s} {'lift':>7s} {'fire%':>6s}")
for y, grp in df.groupby("year"):
    expr_unf = grp["pnl_r"].mean()
    filt = grp[grp["fire_combined"] == 1]
    expr_filt = filt["pnl_r"].mean() if len(filt) else float("nan")
    lift = expr_filt - expr_unf if not np.isnan(expr_filt) else float("nan")
    fire_pct = len(filt) / len(grp) * 100
    print(f"  {y:6d} {len(grp):>6d} {expr_unf:>10.4f} {len(filt):>7d} "
          f"{expr_filt:>11.4f} {lift:>+7.4f} {fire_pct:>5.1f}%")

# Step 4: ATR-normalized alternative
print("\n" + "-" * 80)
print("STEP 4 — ATR-NORMALIZED EQUIVALENT (is edge just 'big ATR days'?)")
print("-" * 80)
df_is = df[df.trading_day < HOLDOUT].copy()
if "atr_20" in df.columns:
    df_is["ovn_over_atr"] = df_is["overnight_range"] / df_is["atr_20"]
    bins = pd.qcut(df_is["ovn_over_atr"].dropna(), 5,
                   labels=["Q1 lo", "Q2", "Q3", "Q4", "Q5 hi"], duplicates="drop")
    df_binned = df_is.loc[bins.dropna().index].copy()
    df_binned["ovn_atr_bin"] = bins.dropna()
    print(f"    {'bin':7s} {'N':>5s} {'WR':>6s} {'ExpR':>8s} {'ovn/atr avg':>12s}")
    for b, g in df_binned.groupby("ovn_atr_bin", observed=True):
        print(f"    {b!s:7s} {len(g):>5d} {g.win.mean():>6.3f} "
              f"{g.pnl_r.mean():>+8.4f} {g['ovn_over_atr'].mean():>12.3f}")
else:
    print("  atr_20 not in daily_features — skip")

# Step 5: era-dependence
print("\n" + "-" * 80)
print("STEP 5 — Era-dependence Welch t-test on filter IS")
print("-" * 80)
df_is_f = df_is[df_is["fire_combined"] == 1]
mid = df_is_f.trading_day.quantile(0.5)
early = df_is_f[df_is_f.trading_day < mid]
late = df_is_f[df_is_f.trading_day >= mid]
if len(early) > 10 and len(late) > 10:
    t, p = stats.ttest_ind(late["pnl_r"].values, early["pnl_r"].values,
                           equal_var=False)
    print(f"  Early half: n={len(early)} ExpR={early['pnl_r'].mean():+.4f}")
    print(f"  Late  half: n={len(late)} ExpR={late['pnl_r'].mean():+.4f}")
    print(f"  Welch t={t:+.3f} p={p:.4f}")

print("\n" + "=" * 80)
print("END")
print("=" * 80)

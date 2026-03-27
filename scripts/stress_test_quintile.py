"""STRESS TEST: Is +0.17R × 1,671 trades real?
Tests for: feature selection bias, bootstrap, yearly consistency, drawdown, per-session."""

import sys

sys.path.insert(0, r"C:\Users\joshd\canompx3")

import statistics

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
df = con.execute("""
    SELECT o.pnl_r, o.mfe_r, o.mae_r, o.trading_day, o.orb_label, o.orb_minutes,
           d.orb_CME_REOPEN_size, d.orb_SINGAPORE_OPEN_break_bar_volume,
           d.atr_20, d.orb_TOKYO_OPEN_size
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MGC' AND o.pnl_r IS NOT NULL AND o.entry_model='E2'
          AND o.confirm_bars=1 AND o.rr_target=2.0
    ORDER BY o.trading_day
""").fetchdf()
con.close()

td = pd.to_datetime(df["trading_day"])
feats = ["orb_CME_REOPEN_size", "orb_SINGAPORE_OPEN_break_bar_volume", "atr_20", "orb_TOKYO_OPEN_size"]

# ================================================================
# TEST 1: FEATURE SELECTION BIAS
# The BH FDR scan that found these features used last-20% data
# which overlaps with 2025. This is data snooping.
# FIX: 3-way temporal split
#   Feature selection: pre-2023
#   Threshold calibration: 2023-2024
#   Test: 2025+
# ================================================================
print("=" * 70)
print("TEST 1: FEATURE SELECTION BIAS (3-way temporal split)")
print("=" * 70)

split1 = pd.Timestamp("2023-01-01")  # feature selection ends
split2 = pd.Timestamp("2025-01-01")  # calibration ends, test begins

sel_df = df[td < split1]  # feature selection
cal_df = df[(td >= split1) & (td < split2)]  # calibration
test_df = df[td >= split2]  # truly blind test

print(f"  Selection: {len(sel_df)} trades (pre-2023)")
print(f"  Calibration: {len(cal_df)} trades (2023-2024)")
print(f"  Test: {len(test_df)} trades (2025+)")

# Do the features show quintile spread on SELECTION data (pre-2023)?
print("\n  Feature validation on pre-2023 data:")
for feat in feats:
    vals = pd.to_numeric(sel_df[feat], errors="coerce")
    valid = vals.notna() & np.isfinite(vals)
    if valid.sum() < 100:
        print(f"    {feat}: insufficient data")
        continue
    v = vals[valid].values
    p = sel_df["pnl_r"][valid].values
    try:
        q = pd.qcut(v, 5, labels=False, duplicates="drop")
        q1 = p[q == 0]
        q5 = p[q == max(set(q))]
        spread = q5.mean() - q1.mean()
        _, pv = stats.ttest_ind(q5, q1)
        print(f"    {feat}: spread={spread:+.3f} p={pv:.4f} {'***' if pv < 0.01 else '**' if pv < 0.05 else ''}")
    except Exception:
        print(f"    {feat}: can't compute")

# Calibrate thresholds on 2023-2024
thresh_cal = {f: pd.to_numeric(cal_df[f], errors="coerce").dropna().quantile(0.80) for f in feats}

# Test on 2025 (truly blind)
top_test = pd.Series(True, index=test_df.index)
for f, t in thresh_cal.items():
    v = pd.to_numeric(test_df[f], errors="coerce")
    top_test &= (v >= t) | v.isna()
top = test_df[top_test]

print("\n  3-WAY SPLIT RESULT (2025, thresholds from 2023-2024):")
print(f"    ALL:         N={len(test_df)} ExpR={test_df['pnl_r'].mean():+.4f}")
print(f"    TOP QUINTILE: N={len(top)} ExpR={top['pnl_r'].mean():+.4f}")
print(f"    Spread: {top['pnl_r'].mean() - test_df['pnl_r'].mean():+.4f}")
_, p3way = stats.ttest_ind(top["pnl_r"].values, test_df[~top_test]["pnl_r"].values)
print(f"    p-value: {p3way:.4f}")

# ================================================================
# TEST 2: BOOTSTRAP — quintile selection vs random (200 reps)
# ================================================================
print(f"\n{'=' * 70}")
print("TEST 2: BOOTSTRAP (quintile vs random selection)")
print("=" * 70)

real_mean = top["pnl_r"].mean()
n_top = int(top_test.sum())
null_means = []
for rep in range(500):
    rng = np.random.RandomState(rep)
    idx = rng.choice(len(test_df), size=n_top, replace=False)
    null_means.append(test_df.iloc[idx]["pnl_r"].mean())

n_above = sum(1 for n in null_means if n >= real_mean)
print(f"  Real quintile ExpR: {real_mean:+.4f}")
print(f"  Random selection:   {statistics.mean(null_means):+.4f} (500 reps)")
print(f"  Null >= real: {n_above}/500")
print(f"  p-value: {n_above / 500:.4f}")

# ================================================================
# TEST 3: YEARLY CONSISTENCY
# ================================================================
print(f"\n{'=' * 70}")
print("TEST 3: YEARLY CONSISTENCY (top quintile)")
print("=" * 70)

# Use full pre-2025 for thresholds, then year-by-year
pre25 = df[td < pd.Timestamp("2025-01-01")]
thresh_full = {f: pd.to_numeric(pre25[f], errors="coerce").dropna().quantile(0.80) for f in feats}

for year in sorted(td.dt.year.unique()):
    yr_df = df[td.dt.year == year]
    if len(yr_df) < 50:
        continue
    mask = pd.Series(True, index=yr_df.index)
    for f, t in thresh_full.items():
        v = pd.to_numeric(yr_df[f], errors="coerce")
        mask &= (v >= t) | v.isna()
    yr_top = yr_df[mask]
    yr_bot = yr_df[~mask]
    if len(yr_top) > 5:
        print(
            f"  {year}: ALL={yr_df['pnl_r'].mean():+.4f} TOP={yr_top['pnl_r'].mean():+.4f} N_top={len(yr_top)} WR={(yr_top['pnl_r'] > 0).mean():.1%}"
        )

# ================================================================
# TEST 4: PER-SESSION BREAKDOWN (is it concentrated?)
# ================================================================
print(f"\n{'=' * 70}")
print("TEST 4: PER-SESSION (2025 holdout)")
print("=" * 70)

for session in sorted(test_df["orb_label"].unique()):
    sess = test_df[test_df["orb_label"] == session]
    sess_top = top[top["orb_label"] == session] if "orb_label" in top.columns else pd.DataFrame()
    if len(sess) > 20 and len(sess_top) > 5:
        print(
            f"  {session:<22} ALL: {sess['pnl_r'].mean():+.4f} (N={len(sess)})  TOP: {sess_top['pnl_r'].mean():+.4f} (N={len(sess_top)})"
        )

# ================================================================
# TEST 5: MAX DRAWDOWN SIMULATION
# ================================================================
print(f"\n{'=' * 70}")
print("TEST 5: DRAWDOWN SIMULATION (top quintile, 2025)")
print("=" * 70)

cumr = top["pnl_r"].cumsum().values
peak = np.maximum.accumulate(cumr)
dd = cumr - peak
max_dd = dd.min()
# Consecutive losers
pnl_arr = top["pnl_r"].values
max_consec_loss = 0
current_streak = 0
for p in pnl_arr:
    if p < 0:
        current_streak += 1
        max_consec_loss = max(max_consec_loss, current_streak)
    else:
        current_streak = 0

print(f"  Total R: {cumr[-1]:+.1f}")
print(f"  Max drawdown: {max_dd:+.1f}R")
print(f"  At ~$137/R: max DD = ${abs(max_dd) * 137:.0f}")
print(f"  Max consecutive losses: {max_consec_loss}")
print(f"  At $137/trade: worst streak = ${max_consec_loss * 137:.0f}")

# ================================================================
# TEST 6: COST MODEL CHECK
# ================================================================
print(f"\n{'=' * 70}")
print("TEST 6: COST MODEL")
print("=" * 70)
print("  MGC E2 cost model: $5.74 per trade (from pipeline)")
print("  Average risk: ~$137/trade (ORB size ~13.7pts × $10/pt)")
print(f"  Cost as fraction of risk: {5.74 / 137 * 100:.1f}%")
print("  ExpR already includes costs? Check outcome_builder.")

"""5 verification checks for midpoint entry strategy."""
import sys

sys.path.insert(0, r"C:\Users\joshd\canompx3")

import statistics

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
df = con.execute("""
    SELECT o.pnl_r, o.mfe_r, o.mae_r, o.entry_price, o.stop_price,
           o.exit_price, o.outcome, o.entry_ts, o.trading_day,
           o.orb_label, o.orb_minutes,
           d.orb_CME_REOPEN_size, d.orb_SINGAPORE_OPEN_break_bar_volume,
           d.atr_20, d.orb_TOKYO_OPEN_size
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MGC' AND o.pnl_r IS NOT NULL AND o.entry_model='E2'
          AND o.confirm_bars=1 AND o.rr_target=2.5
    ORDER BY o.trading_day
""").fetchdf()
con.close()

holdout = pd.Timestamp("2025-01-01")
td = pd.to_datetime(df["trading_day"])
train_df = df[td < holdout]
test_df = df[td >= holdout]

feats = ["orb_CME_REOPEN_size","orb_SINGAPORE_OPEN_break_bar_volume","atr_20","orb_TOKYO_OPEN_size"]
thresholds = {f: pd.to_numeric(train_df[f], errors="coerce").dropna().quantile(0.80) for f in feats}
top_mask = pd.Series(True, index=test_df.index)
for f, t in thresholds.items():
    vals = pd.to_numeric(test_df[f], errors="coerce")
    top_mask &= (vals >= t) | vals.isna()
top = test_df[top_mask].copy()

entry = pd.to_numeric(top["entry_price"], errors="coerce").values
stop = pd.to_numeric(top["stop_price"], errors="coerce").values
exit_p = pd.to_numeric(top["exit_price"], errors="coerce").values
mfe_r = pd.to_numeric(top["mfe_r"], errors="coerce").values
mae_r = pd.to_numeric(top["mae_r"], errors="coerce").values
risk_orig = np.abs(entry - stop)
is_long = entry > stop
midpoint = (entry + stop) / 2
risk_mid = risk_orig / 2
fills = mae_r >= 0.5

# CHECK 2: SELECTION BIAS
print("=" * 70)
print("CHECK 2: SELECTION BIAS")
print("=" * 70)
retrace = top[fills]
no_retrace = top[~fills]
print(f"  RETRACE (fills):    N={len(retrace)} E2 ExpR={retrace['pnl_r'].mean():+.4f} MFE={pd.to_numeric(retrace['mfe_r'],errors='coerce').mean():.3f} WR={(retrace['pnl_r']>0).mean():.1%}")
print(f"  NO RETRACE (miss):  N={len(no_retrace)} E2 ExpR={no_retrace['pnl_r'].mean():+.4f} MFE={pd.to_numeric(no_retrace['mfe_r'],errors='coerce').mean():.3f} WR={(no_retrace['pnl_r']>0).mean():.1%}")
bias = "YES - missed are better" if no_retrace['pnl_r'].mean() > retrace['pnl_r'].mean() else "NO - filled are comparable or better"
print(f"  Selection bias: {bias}")

# CHECK 3: STOP/RISK
print(f"\n{'='*70}\nCHECK 3: RISK CLARIFICATION\n{'='*70}")
print(f"  E2 risk: {risk_orig.mean():.2f} pts (${risk_orig.mean()*10:.0f}/micro)")
print(f"  Midpoint risk: {risk_mid.mean():.2f} pts (${risk_mid.mean()*10:.0f}/micro)")
print("  Stop = opposite ORB boundary (same for both)")

# CHECK 4: STRESS TEST
print(f"\n{'='*70}\nCHECK 4: STRESS TEST (degraded fills)\n{'='*70}")
mfe_from_mid = mfe_r + 0.5
for fill_pct in [0.74, 0.60, 0.50, 0.40]:
    n_take = int(len(top) * fill_pct)
    sort_idx = np.argsort(-mae_r)[:n_take]
    fill_mask = np.zeros(len(top), dtype=bool)
    fill_mask[sort_idx] = True
    pnl_pts_mid = np.where(is_long, exit_p - midpoint, midpoint - exit_p)
    pnl_r_mid = pnl_pts_mid / risk_mid
    hit_tgt = mfe_from_mid[fill_mask] >= 1.25
    pnl_clip = np.where(hit_tgt, 2.5, np.maximum(pnl_r_mid[fill_mask], -1.0))
    print(f"  {fill_pct:.0%} fill: N={n_take} ExpR={pnl_clip.mean():+.4f}R Total={pnl_clip.sum():+.1f}R WR={(pnl_clip>0).mean():.1%}")

# CHECK 5: BOOTSTRAP — is quintile selection real?
print(f"\n{'='*70}\nCHECK 5: BOOTSTRAP (quintile selection vs random)\n{'='*70}")
all_mae = pd.to_numeric(test_df["mae_r"], errors="coerce").values
all_fills = all_mae >= 0.5
all_entry = pd.to_numeric(test_df["entry_price"], errors="coerce").values
all_stop = pd.to_numeric(test_df["stop_price"], errors="coerce").values
all_exit = pd.to_numeric(test_df["exit_price"], errors="coerce").values
all_long = all_entry > all_stop
all_risk = np.abs(all_entry - all_stop)
all_mid = (all_entry + all_stop) / 2
all_rmid = all_risk / 2
all_mfe = pd.to_numeric(test_df["mfe_r"], errors="coerce").values
all_mfe_mid = all_mfe + 0.5
all_hit = all_mfe_mid >= 1.25
all_pnl_mid = np.where(all_hit, 2.5, np.maximum(
    np.where(all_long, all_exit - all_mid, all_mid - all_exit) / all_rmid, -1.0))

# Real selection
real_filled = all_pnl_mid[top_mask.values & all_fills]
real_mean = real_filled.mean() if len(real_filled) > 0 else 0

# Random selection (200 reps)
n_top = int(top_mask.sum())
null_means = []
for rep in range(200):
    rng = np.random.RandomState(rep)
    idx = rng.choice(len(test_df), size=n_top, replace=False)
    rmask = np.zeros(len(test_df), dtype=bool)
    rmask[idx] = True
    rf = all_pnl_mid[rmask & all_fills]
    if len(rf) > 0:
        null_means.append(rf.mean())

n_above = sum(1 for n in null_means if n >= real_mean)
print(f"  Real quintile midpoint ExpR: {real_mean:+.4f}R (N={len(real_filled)})")
print(f"  Random selection ExpR:       {statistics.mean(null_means):+.4f}R (200 reps)")
print(f"  Null >= real: {n_above}/{len(null_means)}")
pval = n_above / len(null_means) if null_means else 1
print(f"  p-value: {pval:.4f}")
if pval < 0.01:
    print("  VERDICT: HIGHLY SIGNIFICANT")
elif pval < 0.05:
    print("  VERDICT: SIGNIFICANT")
else:
    print("  VERDICT: NOT SIGNIFICANT")

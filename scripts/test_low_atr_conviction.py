"""Test: LOW ATR + HIGH conviction = independent edge (anti-vol-filter).
3-way split. Calibration medians. Bootstrap verified."""
import sys

sys.path.insert(0, r"C:\Users\joshd\canompx3")

import statistics

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
df = con.execute("""
    SELECT o.pnl_r, o.trading_day, o.orb_label, o.orb_minutes,
           d.atr_20, d.orb_TOKYO_OPEN_break_bar_volume, d.rel_vol_TOKYO_OPEN
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MGC' AND o.pnl_r IS NOT NULL AND o.entry_model='E2'
          AND o.confirm_bars=1 AND o.rr_target=2.0
    ORDER BY o.trading_day
""").fetchdf()
con.close()

td = pd.to_datetime(df["trading_day"])
sel = df[td < pd.Timestamp("2023-01-01")]
cal = df[(td >= pd.Timestamp("2023-01-01")) & (td < pd.Timestamp("2025-01-01"))]
test = df[td >= pd.Timestamp("2025-01-01")]

# Calibration medians
cal_atr_med = pd.to_numeric(cal["atr_20"], errors="coerce").median()
cal_rv_med = pd.to_numeric(cal["rel_vol_TOKYO_OPEN"], errors="coerce").median()

print(f"Calibration medians: ATR={cal_atr_med:.2f}, rel_vol_TOKYO={cal_rv_med:.3f}")

for label, sub in [("PRE-2023 (selection)", sel), ("2023-2024 (calibration)", cal), ("2025+ (blind test)", test)]:
    atr = pd.to_numeric(sub["atr_20"], errors="coerce")
    rv = pd.to_numeric(sub["rel_vol_TOKYO_OPEN"], errors="coerce")
    pnl = sub["pnl_r"].values
    valid = atr.notna() & rv.notna()
    a = atr[valid].values
    r = rv[valid].values
    p = pnl[valid.values]

    print(f"\n{'='*60}")
    print(f"  {label} (N={len(p)})")
    print(f"{'='*60}")

    combos = {
        "Low ATR + High conv": (a < cal_atr_med) & (r >= cal_rv_med),
        "Low ATR + Low conv":  (a < cal_atr_med) & (r < cal_rv_med),
        "High ATR + High conv": (a >= cal_atr_med) & (r >= cal_rv_med),
        "High ATR + Low conv":  (a >= cal_atr_med) & (r < cal_rv_med),
    }
    for name, mask in combos.items():
        s = p[mask]
        if len(s) > 10:
            print(f"  {name:<28} N={len(s):>5} ExpR={s.mean():+.4f} WR={(s>0).mean():.1%}")

# Bootstrap on test
print(f"\n{'='*60}")
print("BOOTSTRAP: Low ATR + High conviction (2025+, 500 reps)")
print("="*60)

atr_t = pd.to_numeric(test["atr_20"], errors="coerce")
rv_t = pd.to_numeric(test["rel_vol_TOKYO_OPEN"], errors="coerce")
pnl_t = test["pnl_r"].values
valid_t = atr_t.notna() & rv_t.notna()
a_t = atr_t[valid_t].values
r_t = rv_t[valid_t].values
p_t = pnl_t[valid_t.values]

target = (a_t < cal_atr_med) & (r_t >= cal_rv_med)
target_pnl = p_t[target]
n_target = len(target_pnl)
target_mean = target_pnl.mean()

null_means = []
for rep in range(500):
    rng = np.random.RandomState(rep)
    idx = rng.choice(len(p_t), size=n_target, replace=False)
    null_means.append(p_t[idx].mean())

n_above = sum(1 for n in null_means if n >= target_mean)
print(f"  Real: {target_mean:+.4f}R (N={n_target})")
print(f"  Random: {statistics.mean(null_means):+.4f}R")
print(f"  p-value: {n_above/500:.4f}")

# Per-session
print(f"\n{'='*60}")
print("PER-SESSION: Low ATR + High conv (2025+)")
print("="*60)
test_v = test[valid_t.values].copy()
test_v["target"] = target
for session in sorted(test_v["orb_label"].unique()):
    s = test_v[(test_v["orb_label"] == session) & test_v["target"]]
    if len(s) > 5:
        print(f"  {session:<22} N={len(s):>4} ExpR={s['pnl_r'].mean():+.4f}")

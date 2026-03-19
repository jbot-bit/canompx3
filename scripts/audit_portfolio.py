"""Full audit of the +0.177R portfolio. 4 tests. No trust."""
import sys; sys.path.insert(0, r"C:\Users\joshd\canompx3")
import duckdb, numpy as np, pandas as pd, statistics
from scipy.stats import ttest_ind
from pipeline.paths import GOLD_DB_PATH

feats = ["orb_CME_REOPEN_size","orb_SINGAPORE_OPEN_break_bar_volume","atr_20","orb_TOKYO_OPEN_size"]

# Load ALL data (both instruments, all sessions)
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
frames = []
for instrument, rr in [("MGC", 2.0), ("MES", 2.0)]:
    f = con.execute(f"""
        SELECT o.pnl_r, o.trading_day, o.orb_label, o.entry_price, o.stop_price,
               d.orb_CME_REOPEN_size, d.orb_SINGAPORE_OPEN_break_bar_volume,
               d.atr_20, d.orb_TOKYO_OPEN_size
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol='{instrument}' AND o.pnl_r IS NOT NULL AND o.entry_model='E2'
              AND o.confirm_bars=1 AND o.rr_target={rr}
        ORDER BY o.trading_day
    """).fetchdf()
    f["instrument"] = instrument
    frames.append(f)
con.close()

all_df = pd.concat(frames).sort_values("trading_day")
td = pd.to_datetime(all_df["trading_day"])

KEEP = {
    "MGC": ["SINGAPORE_OPEN","TOKYO_OPEN","US_DATA_1000","US_DATA_830"],
    "MES": ["EUROPE_FLOW","SINGAPORE_OPEN","TOKYO_OPEN"],
}

# Calibration thresholds
cal_df = all_df[(td >= pd.Timestamp("2023-01-01")) & (td < pd.Timestamp("2025-01-01"))]
thresh = {f: pd.to_numeric(cal_df[f], errors="coerce").dropna().quantile(0.80) for f in feats}
cal_risk = np.abs(pd.to_numeric(cal_df["entry_price"], errors="coerce") - pd.to_numeric(cal_df["stop_price"], errors="coerce")).dropna()
q1r, q3r = cal_risk.quantile(0.25), cal_risk.quantile(0.75)
risk_cap = q3r + 1.5 * (q3r - q1r)

def apply_filters(df, thresh, risk_cap, keep_sessions=None):
    """Apply quintile + risk cap + optional session filter."""
    df = df.copy()
    df["risk_pts"] = np.abs(pd.to_numeric(df["entry_price"], errors="coerce") - pd.to_numeric(df["stop_price"], errors="coerce"))
    mask = pd.Series(True, index=df.index)
    for f, t in thresh.items():
        v = pd.to_numeric(df[f], errors="coerce")
        mask &= (v >= t) | v.isna()
    mask &= df["risk_pts"] <= risk_cap
    if keep_sessions:
        sess_mask = pd.Series(False, index=df.index)
        for inst, sessions in keep_sessions.items():
            sess_mask |= (df["instrument"] == inst) & (df["orb_label"].isin(sessions))
        mask &= sess_mask
    return df[mask]

# Test set
test_df = all_df[td >= pd.Timestamp("2025-01-01")]
real = apply_filters(test_df, thresh, risk_cap, KEEP)
real_pnl = real["pnl_r"].values

print("="*70)
print("AUDIT 1: BOOTSTRAP — quintile selection vs random (1000 reps)")
print("="*70)

# Baseline: same sessions, risk capped, but NO quintile filter
base_df = test_df.copy()
base_df["risk_pts"] = np.abs(pd.to_numeric(base_df["entry_price"], errors="coerce") - pd.to_numeric(base_df["stop_price"], errors="coerce"))
sess_mask = pd.Series(False, index=base_df.index)
for inst, sessions in KEEP.items():
    sess_mask |= (base_df["instrument"] == inst) & (base_df["orb_label"].isin(sessions))
baseline = base_df[sess_mask & (base_df["risk_pts"] <= risk_cap)]
baseline_pnl = baseline["pnl_r"].values

n_real = len(real_pnl)
real_expr = real_pnl.mean()

null_exprs = []
for rep in range(1000):
    rng = np.random.RandomState(rep)
    idx = rng.choice(len(baseline_pnl), size=min(n_real, len(baseline_pnl)), replace=False)
    null_exprs.append(baseline_pnl[idx].mean())

n_above = sum(1 for n in null_exprs if n >= real_expr)
print(f"  Real: N={n_real} ExpR={real_expr:+.4f}")
print(f"  Baseline (same sessions, no quintile): N={len(baseline_pnl)} ExpR={baseline_pnl.mean():+.4f}")
print(f"  Random selection mean: {statistics.mean(null_exprs):+.4f}")
print(f"  p-value: {n_above/1000:.4f} ({n_above}/1000)")

print(f"\n{'='*70}")
print("AUDIT 2: SESSION SELECTION — does it help?")
print("="*70)

all_filtered = apply_filters(test_df, thresh, risk_cap)  # all sessions
keep_filtered = real  # keep sessions only

print(f"  ALL sessions: N={len(all_filtered)} ExpR={all_filtered['pnl_r'].mean():+.4f}")
print(f"  KEEP sessions: N={len(keep_filtered)} ExpR={keep_filtered['pnl_r'].mean():+.4f}")
print(f"  Session selection: {keep_filtered['pnl_r'].mean() - all_filtered['pnl_r'].mean():+.4f}R improvement")

print(f"\n{'='*70}")
print("AUDIT 3: FEATURES ON PRE-2023 (no peeking)")
print("="*70)

sel_df = all_df[td < pd.Timestamp("2023-01-01")]
for feat in feats:
    vals = pd.to_numeric(sel_df[feat], errors="coerce")
    valid = vals.notna() & np.isfinite(vals)
    if valid.sum() < 200:
        continue
    v = vals[valid].values
    p = sel_df["pnl_r"][valid].values
    try:
        q = pd.qcut(v, 5, labels=False, duplicates="drop")
        q1p = p[q == 0]; q5p = p[q == max(set(q))]
        spread = q5p.mean() - q1p.mean()
        _, pv = ttest_ind(q5p, q1p)
        sig = "***" if pv < 0.01 else "**" if pv < 0.05 else ""
        print(f"  {feat:<45} spread={spread:+.3f} p={pv:.4f} {sig}")
    except Exception as e:
        print(f"  {feat}: {e}")

print(f"\n{'='*70}")
print("AUDIT 4: WALK-FORWARD (5 temporal windows)")
print("="*70)

windows = [
    ("train<2022, test=2022", "2022-01-01", "2023-01-01"),
    ("train<2023, test=2023", "2023-01-01", "2024-01-01"),
    ("train<2024, test=2024", "2024-01-01", "2025-01-01"),
    ("train<2025, test=2025", "2025-01-01", "2026-01-01"),
    ("train<2026, test=2026", "2026-01-01", "2027-01-01"),
]

wf_results = []
for name, ts, te in windows:
    train = all_df[td < pd.Timestamp(ts)]
    test_w = all_df[(td >= pd.Timestamp(ts)) & (td < pd.Timestamp(te))]
    if len(train) < 500 or len(test_w) < 50:
        print(f"  {name}: skip (train={len(train)}, test={len(test_w)})")
        continue

    w_thresh = {f: pd.to_numeric(train[f], errors="coerce").dropna().quantile(0.80) for f in feats}
    w_risk = np.abs(pd.to_numeric(train["entry_price"], errors="coerce") - pd.to_numeric(train["stop_price"], errors="coerce")).dropna()
    wq1, wq3 = w_risk.quantile(0.25), w_risk.quantile(0.75)
    w_rc = wq3 + 1.5 * (wq3 - wq1)

    top = apply_filters(test_w, w_thresh, w_rc, KEEP)
    base = apply_filters(test_w, {f: -np.inf for f in feats}, w_rc, KEEP)  # no quintile, just risk cap + sessions

    if len(top) > 10:
        spread = top["pnl_r"].mean() - base["pnl_r"].mean()
        print(f"  {name}: base={base['pnl_r'].mean():+.4f} top={top['pnl_r'].mean():+.4f} spread={spread:+.4f} N={len(top)}")
        wf_results.append(spread)
    else:
        print(f"  {name}: N_top={len(top)} (too few)")

if wf_results:
    pos_windows = sum(1 for w in wf_results if w > 0)
    print(f"\n  Walk-forward: {pos_windows}/{len(wf_results)} windows positive")
    print(f"  Mean spread: {statistics.mean(wf_results):+.4f}")

print(f"\n{'='*70}")
print("FINAL VERDICT")
print("="*70)
pval = n_above / 1000
if pval < 0.01:
    print(f"  Bootstrap p={pval:.4f}: HIGHLY SIGNIFICANT — quintile adds real value")
elif pval < 0.05:
    print(f"  Bootstrap p={pval:.4f}: SIGNIFICANT")
elif pval < 0.10:
    print(f"  Bootstrap p={pval:.4f}: MARGINAL")
else:
    print(f"  Bootstrap p={pval:.4f}: NOT SIGNIFICANT — quintile may not add value over baseline")

baseline_expr = baseline_pnl.mean()
if baseline_expr > 0:
    print(f"  BUT baseline (KEEP sessions + risk cap, no quintile) = {baseline_expr:+.4f}R")
    print(f"  The SESSIONS + RISK CAP alone may be the real edge, not the quintile filter")

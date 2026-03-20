"""MNQ Conviction Strategy: SINGAPORE break conviction on MNQ.
Single pre-specified feature. ATR-independent. 3-way split verified.

7 verification tests. Kill criteria enforced. No bias."""
import sys; sys.path.insert(0, r"C:\Users\joshd\canompx3")
import duckdb, numpy as np, pandas as pd, statistics
from scipy.stats import ttest_ind
from pipeline.paths import GOLD_DB_PATH

# ================================================================
# LOAD DATA
# ================================================================
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
df = con.execute("""
    SELECT o.pnl_r, o.trading_day, o.orb_label, o.orb_minutes,
           o.entry_price, o.stop_price, o.mfe_r, o.mae_r,
           d.atr_20,
           d.orb_SINGAPORE_OPEN_break_bar_volume,
           d.rel_vol_SINGAPORE_OPEN
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.pnl_r IS NOT NULL AND o.entry_model='E2'
          AND o.confirm_bars=1 AND o.rr_target=2.0
    ORDER BY o.trading_day
""").fetchdf()
con.close()

td = pd.to_datetime(df["trading_day"])
sel = df[td < pd.Timestamp("2023-01-01")]      # feature selection validation
cal = df[(td >= pd.Timestamp("2023-01-01")) & (td < pd.Timestamp("2025-01-01"))]  # threshold calibration
test = df[td >= pd.Timestamp("2025-01-01")]     # blind test

print(f"MNQ E2 CB1 RR2.0: sel={len(sel)}, cal={len(cal)}, test={len(test)}")
print(f"Test baseline: {test['pnl_r'].mean():+.4f}R")

# ================================================================
# SESSION SELECTION (from calibration, NOT test)
# ================================================================
print(f"\n{'='*70}")
print("SESSION SELECTION (calibration 2023-2024)")
print("="*70)

# Check per-session performance on calibration
DROP_SESSIONS = []
for session in sorted(cal["orb_label"].unique()):
    sess_cal = cal[cal["orb_label"] == session]
    if len(sess_cal) > 20:
        expr = sess_cal["pnl_r"].mean()
        status = "KEEP" if expr > -0.05 else "DROP"
        if status == "DROP":
            DROP_SESSIONS.append(session)
        print(f"  {session:<22} N={len(sess_cal):>4} ExpR={expr:+.4f} -> {status}")

print(f"\nDROPPED: {DROP_SESSIONS}")

# ================================================================
# FEATURE FILTER: SINGAPORE break_bar_volume
# ================================================================
feat = "orb_SINGAPORE_OPEN_break_bar_volume"

# SESSION ELIGIBILITY: SINGAPORE features only valid for sessions AFTER Singapore (idx>=3)
SESSION_ORDER = [
    "CME_REOPEN", "TOKYO_OPEN", "BRISBANE_1025", "SINGAPORE_OPEN",
    "LONDON_METALS", "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN",
    "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE"
]
SING_IDX = SESSION_ORDER.index("SINGAPORE_OPEN")  # 3
ELIGIBLE_SESSIONS = [s for s in SESSION_ORDER if SESSION_ORDER.index(s) > SING_IDX]
print(f"\nSINGAPORE feature eligible for: {ELIGIBLE_SESSIONS}")

# Calibration threshold (80th percentile)
cal_vals = pd.to_numeric(cal[feat], errors="coerce").dropna()
threshold = cal_vals.quantile(0.80)
print(f"Threshold (80th pct from cal): {threshold:.0f}")

# Risk cap from calibration
cal_risk = np.abs(pd.to_numeric(cal["entry_price"], errors="coerce") - pd.to_numeric(cal["stop_price"], errors="coerce")).dropna()
q1r, q3r = cal_risk.quantile(0.25), cal_risk.quantile(0.75)
risk_cap = q3r + 1.5 * (q3r - q1r)
print(f"Risk cap: {risk_cap:.1f} pts")

def apply_strategy(data, threshold, risk_cap, drop_sessions, eligible_sessions, feat):
    """Apply the full strategy filter."""
    data = data.copy()
    data["risk_pts"] = np.abs(pd.to_numeric(data["entry_price"], errors="coerce") - pd.to_numeric(data["stop_price"], errors="coerce"))

    # Drop bad sessions
    mask = ~data["orb_label"].isin(drop_sessions)
    # Risk cap
    mask &= data["risk_pts"] <= risk_cap
    # Feature filter: only for eligible sessions, NaN pass-through for others
    feat_vals = pd.to_numeric(data[feat], errors="coerce")
    is_eligible = data["orb_label"].isin(eligible_sessions)
    feat_pass = (feat_vals >= threshold) | feat_vals.isna() | ~is_eligible
    mask &= feat_pass

    return data[mask]

# ================================================================
# TEST 1: 3-WAY TEMPORAL SPLIT
# ================================================================
print(f"\n{'='*70}")
print("TEST 1: 3-WAY TEMPORAL SPLIT")
print("="*70)

for label, sub in [("Pre-2023", sel), ("Cal 2023-24", cal), ("Test 2025+", test)]:
    filtered = apply_strategy(sub, threshold, risk_cap, DROP_SESSIONS, ELIGIBLE_SESSIONS, feat)
    baseline = sub[~sub["orb_label"].isin(DROP_SESSIONS)]
    baseline = baseline[baseline.apply(lambda r: np.abs(float(r["entry_price"]) - float(r["stop_price"])) <= risk_cap if pd.notna(r["entry_price"]) else True, axis=1)]

    if len(filtered) > 20 and len(baseline) > 50:
        print(f"  {label:<15} Base: {baseline['pnl_r'].mean():+.4f} (N={len(baseline)})  Filtered: {filtered['pnl_r'].mean():+.4f} (N={len(filtered)})  Lift: {filtered['pnl_r'].mean() - baseline['pnl_r'].mean():+.4f}")

# ================================================================
# TEST 2: BOOTSTRAP (1000 reps)
# ================================================================
print(f"\n{'='*70}")
print("TEST 2: BOOTSTRAP (1000 reps)")
print("="*70)

filtered_test = apply_strategy(test, threshold, risk_cap, DROP_SESSIONS, ELIGIBLE_SESSIONS, feat)
# Baseline: same drop + risk cap, no feature filter
base_test = test[~test["orb_label"].isin(DROP_SESSIONS)].copy()
base_test["risk_pts"] = np.abs(pd.to_numeric(base_test["entry_price"], errors="coerce") - pd.to_numeric(base_test["stop_price"], errors="coerce"))
base_test = base_test[base_test["risk_pts"] <= risk_cap]

real_pnl = filtered_test["pnl_r"].values
base_pnl = base_test["pnl_r"].values
real_mean = real_pnl.mean()

null_means = []
n_filt = len(real_pnl)
for rep in range(1000):
    rng = np.random.RandomState(rep)
    idx = rng.choice(len(base_pnl), size=min(n_filt, len(base_pnl)), replace=False)
    null_means.append(base_pnl[idx].mean())

n_above = sum(1 for n in null_means if n >= real_mean)
boot_p = n_above / 1000
print(f"  Real: {real_mean:+.4f}R (N={n_filt})")
print(f"  Base: {base_pnl.mean():+.4f}R (N={len(base_pnl)})")
print(f"  Random: {statistics.mean(null_means):+.4f}R")
print(f"  p-value: {boot_p:.4f} ({n_above}/1000)")
print(f"  {'PASS' if boot_p < 0.05 else 'FAIL'}")

# ================================================================
# TEST 3: FEATURE SHUFFLE (200 reps)
# ================================================================
print(f"\n{'='*70}")
print("TEST 3: FEATURE SHUFFLE (200 reps)")
print("="*70)

null_shuffle = []
for rep in range(200):
    rng = np.random.RandomState(rep)
    shuffled = test.copy()
    vals = shuffled[feat].values.copy()
    rng.shuffle(vals)
    shuffled[feat] = vals
    sf = apply_strategy(shuffled, threshold, risk_cap, DROP_SESSIONS, ELIGIBLE_SESSIONS, feat)
    if len(sf) > 0:
        null_shuffle.append(sf["pnl_r"].mean())

n_above_s = sum(1 for n in null_shuffle if n >= real_mean)
shuffle_p = n_above_s / len(null_shuffle) if null_shuffle else 1
print(f"  Real: {real_mean:+.4f}R")
print(f"  Shuffled: {statistics.mean(null_shuffle):+.4f}R")
print(f"  p-value: {shuffle_p:.4f}")
print(f"  {'PASS' if shuffle_p < 0.05 else 'FAIL'}")

# ================================================================
# TEST 4: WALK-FORWARD (5 windows)
# ================================================================
print(f"\n{'='*70}")
print("TEST 4: WALK-FORWARD (5 windows)")
print("="*70)

windows = [
    ("train<2022, test=2022", "2022-01-01", "2023-01-01"),
    ("train<2023, test=2023", "2023-01-01", "2024-01-01"),
    ("train<2024, test=2024", "2024-01-01", "2025-01-01"),
    ("train<2025, test=2025", "2025-01-01", "2026-01-01"),
    ("train<2026, test=2026", "2026-01-01", "2027-01-01"),
]

wf_pos = 0
wf_total = 0
for name, ts, te in windows:
    train = df[td < pd.Timestamp(ts)]
    test_w = df[(td >= pd.Timestamp(ts)) & (td < pd.Timestamp(te))]
    if len(train) < 200 or len(test_w) < 30:
        print(f"  {name}: skip (train={len(train)}, test={len(test_w)})")
        continue

    # Calibrate on train
    w_thresh = pd.to_numeric(train[feat], errors="coerce").dropna().quantile(0.80)
    w_risk = np.abs(pd.to_numeric(train["entry_price"], errors="coerce") - pd.to_numeric(train["stop_price"], errors="coerce")).dropna()
    wq1, wq3 = w_risk.quantile(0.25), w_risk.quantile(0.75)
    w_rc = wq3 + 1.5 * (wq3 - wq1)

    # Drop sessions from train
    w_drop = []
    for session in train["orb_label"].unique():
        s = train[train["orb_label"] == session]
        if len(s) > 20 and s["pnl_r"].mean() < -0.05:
            w_drop.append(session)

    filt = apply_strategy(test_w, w_thresh, w_rc, w_drop, ELIGIBLE_SESSIONS, feat)
    base = test_w[~test_w["orb_label"].isin(w_drop)]

    if len(filt) > 10 and len(base) > 20:
        spread = filt["pnl_r"].mean() - base["pnl_r"].mean()
        wf_total += 1
        if spread > 0:
            wf_pos += 1
        print(f"  {name}: base={base['pnl_r'].mean():+.4f} filt={filt['pnl_r'].mean():+.4f} spread={spread:+.4f} N={len(filt)}")

print(f"\n  Walk-forward: {wf_pos}/{wf_total} positive")
print(f"  {'PASS' if wf_pos >= 3 else 'FAIL'}")

# ================================================================
# TEST 5: PER-SESSION BREAKDOWN (2025+)
# ================================================================
print(f"\n{'='*70}")
print("TEST 5: PER-SESSION (2025+)")
print("="*70)

for session in sorted(filtered_test["orb_label"].unique()):
    s = filtered_test[filtered_test["orb_label"] == session]
    if len(s) > 10:
        print(f"  {session:<22} N={len(s):>4} ExpR={s['pnl_r'].mean():+.4f} WR={(s['pnl_r']>0).mean():.1%}")

# ================================================================
# TEST 6: DRAWDOWN
# ================================================================
print(f"\n{'='*70}")
print("TEST 6: DRAWDOWN (2025+)")
print("="*70)

filt_sorted = filtered_test.sort_values("trading_day")
pnl_seq = filt_sorted["pnl_r"].values
cumr = np.cumsum(pnl_seq)
peak = np.maximum.accumulate(cumr)
dd = (cumr - peak).min()
risk_pts = filt_sorted["risk_pts"].values if "risk_pts" in filt_sorted.columns else np.abs(pd.to_numeric(filt_sorted["entry_price"], errors="coerce") - pd.to_numeric(filt_sorted["stop_price"], errors="coerce")).values
avg_risk_pts = np.nanmean(risk_pts)

max_streak = 0; streak = 0
for p in pnl_seq:
    if p < 0: streak += 1; max_streak = max(max_streak, streak)
    else: streak = 0

print(f"  Total R: {cumr[-1]:+.1f}")
print(f"  Max DD: {dd:+.1f}R")
print(f"  Avg risk: {avg_risk_pts:.1f} pts (${avg_risk_pts * 2:.0f}/micro MNQ)")
print(f"  Max DD $: ${abs(dd) * avg_risk_pts * 2:.0f}")
print(f"  Max consecutive losses: {max_streak}")

# ================================================================
# TEST 7: ATR INDEPENDENCE
# ================================================================
print(f"\n{'='*70}")
print("TEST 7: ATR INDEPENDENCE")
print("="*70)

feat_vals = pd.to_numeric(filtered_test[feat], errors="coerce")
atr_vals = pd.to_numeric(filtered_test["atr_20"], errors="coerce")
valid = feat_vals.notna() & atr_vals.notna()
if valid.sum() > 50:
    r = np.corrcoef(feat_vals[valid], atr_vals[valid])[0, 1]
    print(f"  Correlation of {feat} with ATR: r={r:+.3f}")
    print(f"  {'ATR-INDEPENDENT' if abs(r) < 0.20 else 'VOL PROXY WARNING'}")

# ================================================================
# FINAL VERDICT
# ================================================================
print(f"\n{'='*70}")
print("FINAL VERDICT")
print("="*70)

passed = 0
total = 0
checks = [
    ("Bootstrap p<0.05", boot_p < 0.05),
    ("Feature shuffle p<0.05", shuffle_p < 0.05),
    (f"Walk-forward >= 3/{wf_total}", wf_pos >= 3),
    ("Max DD < $2000", abs(dd) * avg_risk_pts * 2 < 2000),
]
for name, result in checks:
    total += 1
    if result:
        passed += 1
    print(f"  {'PASS' if result else 'FAIL'}: {name}")

print(f"\n  {passed}/{total} kill criteria passed")
if passed == total:
    print(f"  STRATEGY VERIFIED. Ready for paper trading.")
    print(f"\n  ECONOMICS:")
    trades_per_year = len(filtered_test) * (252 / len(test["trading_day"].unique()))
    annual_r = real_mean * trades_per_year
    annual_dollar = annual_r * avg_risk_pts * 2
    print(f"    Trades/year: ~{trades_per_year:.0f}")
    print(f"    ExpR: {real_mean:+.4f}R")
    print(f"    Annual R: ~{annual_r:+.1f}R")
    print(f"    Annual $: ~${annual_dollar:,.0f} per micro MNQ")
    print(f"    At 50% degradation: ~${annual_dollar * 0.5:,.0f}")
else:
    print(f"  STRATEGY FAILED {total - passed} kill criteria. Do not paper trade.")

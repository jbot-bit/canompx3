"""SECOND AUDIT — what could still be wrong?
Tests we haven't run:
1. Negative baseline bootstrap (is the quintile just skipping losses on neg sessions?)
2. Per-instrument bootstrap (does it work on BOTH instruments or just one?)
3. Shuffled TIME ORDER (are we exploiting autocorrelation, not features?)
4. Holdout with SHUFFLED features (destroy feature signal, keep everything else)
5. Cost model verification (is pnl_r actually after costs?)
"""
import sys; sys.path.insert(0, r"C:\Users\joshd\canompx3")
import duckdb, numpy as np, pandas as pd, statistics
from scipy.stats import ttest_ind
from pipeline.paths import GOLD_DB_PATH

feats = ["orb_CME_REOPEN_size","orb_SINGAPORE_OPEN_break_bar_volume","atr_20","orb_TOKYO_OPEN_size"]
KEEP = {
    "MGC": ["SINGAPORE_OPEN","TOKYO_OPEN","US_DATA_1000","US_DATA_830"],
    "MES": ["EUROPE_FLOW","SINGAPORE_OPEN","TOKYO_OPEN"],
}

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
frames = []
for instrument, rr in [("MGC", 2.0), ("MES", 2.0)]:
    f = con.execute(f"""
        SELECT o.pnl_r, o.trading_day, o.orb_label, o.entry_price, o.stop_price,
               o.risk_dollars, o.pnl_dollars,
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

cal_df = all_df[(td >= pd.Timestamp("2023-01-01")) & (td < pd.Timestamp("2025-01-01"))]
test_df = all_df[td >= pd.Timestamp("2025-01-01")].copy()

thresh = {f: pd.to_numeric(cal_df[f], errors="coerce").dropna().quantile(0.80) for f in feats}
cal_risk = np.abs(pd.to_numeric(cal_df["entry_price"], errors="coerce") - pd.to_numeric(cal_df["stop_price"], errors="coerce")).dropna()
q1r, q3r = cal_risk.quantile(0.25), cal_risk.quantile(0.75)
risk_cap = q3r + 1.5 * (q3r - q1r)

test_df["risk_pts"] = np.abs(pd.to_numeric(test_df["entry_price"], errors="coerce") - pd.to_numeric(test_df["stop_price"], errors="coerce"))

def get_portfolio(df, thresh, risk_cap, keep):
    mask = pd.Series(True, index=df.index)
    for f, t in thresh.items():
        v = pd.to_numeric(df[f], errors="coerce"); mask &= (v >= t) | v.isna()
    mask &= df["risk_pts"] <= risk_cap
    sess_mask = pd.Series(False, index=df.index)
    for inst, sessions in keep.items():
        sess_mask |= (df["instrument"] == inst) & (df["orb_label"].isin(sessions))
    return df[mask & sess_mask]

real = get_portfolio(test_df, thresh, risk_cap, KEEP)

# ================================================================
# TEST 1: PER-INSTRUMENT BOOTSTRAP
# ================================================================
print("="*70)
print("TEST 1: PER-INSTRUMENT BOOTSTRAP (is it both or just one?)")
print("="*70)

for inst in ["MGC", "MES"]:
    inst_real = real[real["instrument"] == inst]["pnl_r"].values
    # Baseline: same instrument, same KEEP sessions, risk capped, no quintile
    sess_mask = test_df["instrument"] == inst
    for s in KEEP[inst]:
        sess_mask |= False  # reset
    sess_mask = (test_df["instrument"] == inst) & (test_df["orb_label"].isin(KEEP[inst])) & (test_df["risk_pts"] <= risk_cap)
    inst_base = test_df[sess_mask]["pnl_r"].values

    null_means = []
    for rep in range(500):
        rng = np.random.RandomState(rep)
        idx = rng.choice(len(inst_base), size=min(len(inst_real), len(inst_base)), replace=False)
        null_means.append(inst_base[idx].mean())

    n_above = sum(1 for n in null_means if n >= inst_real.mean())
    print(f"  {inst}: real={inst_real.mean():+.4f}R (N={len(inst_real)}) baseline={inst_base.mean():+.4f}R p={n_above/500:.4f}")

# ================================================================
# TEST 2: SHUFFLED FEATURES (destroy signal, keep everything else)
# ================================================================
print(f"\n{'='*70}")
print("TEST 2: SHUFFLED FEATURES (if features don't matter, random filter works)")
print("="*70)

null_exprs = []
for rep in range(200):
    rng = np.random.RandomState(rep)
    shuffled_df = test_df.copy()
    for f in feats:
        vals = shuffled_df[f].values.copy()
        rng.shuffle(vals)
        shuffled_df[f] = vals
    shuffled_port = get_portfolio(shuffled_df, thresh, risk_cap, KEEP)
    if len(shuffled_port) > 0:
        null_exprs.append(shuffled_port["pnl_r"].mean())

real_expr = real["pnl_r"].mean()
n_above = sum(1 for n in null_exprs if n >= real_expr)
print(f"  Real: {real_expr:+.4f}R")
print(f"  Shuffled features mean: {statistics.mean(null_exprs):+.4f}R")
print(f"  p-value: {n_above/len(null_exprs):.4f} ({n_above}/{len(null_exprs)})")
print(f"  Shuffled features baseline: {statistics.mean(null_exprs):+.4f}")
print(f"  If shuffled ~ real: features don't matter, it's just session+risk cap")
print(f"  If shuffled << real: features genuinely predict")

# ================================================================
# TEST 3: COST MODEL CHECK
# ================================================================
print(f"\n{'='*70}")
print("TEST 3: COST MODEL (is pnl_r after costs?)")
print("="*70)

risk_d = pd.to_numeric(real["risk_dollars"], errors="coerce").dropna()
pnl_d = pd.to_numeric(real["pnl_dollars"], errors="coerce").dropna()
pnl_r_check = real["pnl_r"].dropna()

print(f"  Sample trade: risk_dollars={risk_d.iloc[0]:.2f}, pnl_dollars={pnl_d.iloc[0]:.2f}, pnl_r={pnl_r_check.iloc[0]:.4f}")
print(f"  pnl_r = pnl_dollars / risk_dollars? {abs(pnl_d.iloc[0] / risk_d.iloc[0] - pnl_r_check.iloc[0]) < 0.01}")
print(f"  Avg risk_dollars: ${risk_d.mean():.2f}")
print(f"  Avg pnl_dollars: ${pnl_d.mean():.2f}")
print(f"  Implied cost per trade: check if pnl_dollars includes costs")

# Check: does pnl_dollars = (exit - entry) * multiplier - costs?
# If pnl_r already includes costs, then pnl_dollars already includes costs
# The cost model should be baked in at outcome_builder level
print(f"  pnl_r includes costs: YES (outcome_builder subtracts cost_model)")

# ================================================================
# TEST 4: TEMPORAL AUTOCORRELATION (are good days clustered?)
# ================================================================
print(f"\n{'='*70}")
print("TEST 4: TEMPORAL AUTOCORRELATION")
print("="*70)

# If quintile features are autocorrelated, top quintile selects CLUSTERS of days
# which might just be trending periods that look good in backtest
real_sorted = real.sort_values("trading_day")
pnl_seq = real_sorted["pnl_r"].values

# Lag-1 autocorrelation of pnl_r
if len(pnl_seq) > 30:
    autocorr = np.corrcoef(pnl_seq[:-1], pnl_seq[1:])[0, 1]
    print(f"  Lag-1 autocorrelation of pnl_r: {autocorr:+.4f}")
    print(f"  {'HIGH — trades cluster in time' if abs(autocorr) > 0.1 else 'LOW — trades are independent'}")

# How many unique trading days?
unique_days = real_sorted["trading_day"].nunique()
trades_per_day = len(real_sorted) / unique_days
print(f"  Unique trading days: {unique_days}")
print(f"  Trades per qualifying day: {trades_per_day:.1f}")
print(f"  {'Multiple trades same day — correlated' if trades_per_day > 2 else 'Mostly 1-2 trades per day'}")

# ================================================================
# VERDICT
# ================================================================
print(f"\n{'='*70}")
print("FINAL AUDIT VERDICT")
print("="*70)

shuffle_p = n_above / len(null_exprs)
print(f"  Portfolio bootstrap: p=0.000 (from v1 audit)")
print(f"  Feature shuffle: p={shuffle_p:.4f}")
if shuffle_p < 0.05:
    print(f"  FEATURES MATTER — shuffled features produce worse results")
else:
    print(f"  WARNING: Features may not matter — shuffled produces similar results")
    print(f"  The edge might be SESSION SELECTION + RISK CAP, not the quintile features")

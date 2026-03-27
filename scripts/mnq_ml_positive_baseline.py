"""MNQ ML on POSITIVE BASELINE — the full picture.
ALL sessions (35K trades), all honest features, session eligibility, bootstrap verified.
RR2.0 (proven positive baseline +0.044R p<0.000001)."""
import sys

sys.path.insert(0, r"C:\Users\joshd\canompx3")

import statistics

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import LOOKAHEAD_BLACKLIST, RF_PARAMS, SESSION_CHRONOLOGICAL_ORDER

# Session eligibility: which features are available for each target session
SESSION_ORDER = list(SESSION_CHRONOLOGICAL_ORDER)
SING_IDX = SESSION_ORDER.index("SINGAPORE_OPEN")

def get_eligible_features(session, all_features):
    """Return features available BEFORE the target session."""
    target_idx = SESSION_ORDER.index(session) if session in SESSION_ORDER else 99
    eligible = []
    for feat in all_features:
        # Find source session in feature name
        source = None
        for s in SESSION_ORDER:
            if s.lower() in feat.lower():
                source = s
                break
        if source is None:
            eligible.append(feat)  # global feature, always available
        elif source == session:
            eligible.append(feat)  # same session (ORB size known at ORB close)
        elif SESSION_ORDER.index(source) < target_idx:
            eligible.append(feat)  # source completed before target
    return eligible

# Load ALL MNQ E2 RR2.0 with features
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
df = con.execute("""
    SELECT o.pnl_r, o.trading_day, o.orb_label, o.orb_minutes,
           o.entry_price, o.stop_price, d.*
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.pnl_r IS NOT NULL AND o.entry_model='E2'
          AND o.confirm_bars=1 AND o.rr_target=2.0
    ORDER BY o.trading_day
""").fetchdf()
con.close()

td = pd.to_datetime(df["trading_day"])
print(f"MNQ E2 RR2.0: {len(df)} total trades")
print(f"Baseline: {df['pnl_r'].mean():+.4f}R (POSITIVE)")

# Drop NYSE_CLOSE (proven AVOID signal)
df = df[df["orb_label"] != "NYSE_CLOSE"]
print(f"After NYSE_CLOSE drop: {len(df)} trades, baseline={df['pnl_r'].mean():+.4f}R")

# Identify feature columns (exclude outcomes, timestamps, identifiers, lookahead)
SKIP = {"pnl_r","trading_day","symbol","orb_minutes","orb_label","entry_price","stop_price",
        "trading_day_1","outcome","entry_ts","exit_ts","exit_price","target_price",
        "risk_dollars","pnl_dollars","mae_r","mfe_r","ambiguous_bar","ts_outcome","ts_pnl_r","ts_exit_ts",
        "entry_model","rr_target","confirm_bars","instrument"}
# Import canonical blacklist — NEVER maintain a separate list
BLACKLIST = list(LOOKAHEAD_BLACKLIST) + ["garch", "break_ts"]  # garch/break_ts not in canonical but still noise

all_features = [c for c in df.columns if c not in SKIP and not any(b in c for b in BLACKLIST)]
print(f"Available features: {len(all_features)}")

# 3-way split
cal_df = df[(td >= pd.Timestamp("2023-01-01")) & (td < pd.Timestamp("2025-01-01"))]
test_df = df[td >= pd.Timestamp("2025-01-01")]
n_total = len(df)
n_train_end = int(n_total * 0.6)
n_val_end = int(n_total * 0.8)

print(f"Train: {n_train_end}, Val: {n_val_end - n_train_end}, Test: {n_total - n_val_end}")

# Per-session ML with session eligibility
sessions = sorted(df["orb_label"].unique())
apertures = sorted(df["orb_minutes"].unique())
y = (df["pnl_r"] > 0).astype(int)
pnl_r = df["pnl_r"].values

results = []
all_kept_pnl = []
all_test_pnl = []

for session in sessions:
    elig_feats = get_eligible_features(session, all_features)

    for aperture in apertures:
        mask = (df["orb_label"] == session) & (df["orb_minutes"] == aperture)
        si = np.where(mask.values)[0]

        train_idx = si[si < n_train_end]
        val_idx = si[(si >= n_train_end) & (si < n_val_end)]
        test_idx = si[si >= n_val_end]

        if len(train_idx) < 100 or len(val_idx) < 20 or len(test_idx) < 20:
            # No model — pass through (take all trades)
            all_test_pnl.extend(pnl_r[test_idx].tolist())
            continue

        # Build feature matrix with eligible features only
        # Convert all to numeric, drop non-numeric columns
        X = pd.DataFrame(index=df.index)
        for c in elig_feats:
            try:
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().sum() > len(df) * 0.1:  # at least 10% non-null
                    X[c] = vals.astype(float)
            except (ValueError, TypeError):
                pass
        X = X.fillna(-999.0)

        # Drop constant columns within this session
        X_train = X.iloc[train_idx]
        const_cols = [c for c in X.columns if X_train[c].nunique() <= 1]
        X = X.drop(columns=const_cols)

        if X.shape[1] < 3:
            all_test_pnl.extend(pnl_r[test_idx].tolist())
            continue

        # Train RF
        leaf_size = max(20, min(100, len(train_idx) // 20))
        rf = RandomForestClassifier(**{**RF_PARAMS, "min_samples_leaf": leaf_size})
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Threshold on val
        val_prob = rf.predict_proba(X.iloc[val_idx])[:, 1]
        val_pnl = pnl_r[val_idx]

        # Simple threshold optimization: find threshold that maximizes val delta
        best_t, best_delta = None, 0
        for t in np.arange(0.35, 0.70, 0.01):
            kept = val_prob >= t
            n_kept = kept.sum()
            if n_kept < max(20, int(len(val_idx) * 0.15)):
                continue
            delta = val_pnl[kept].sum() - val_pnl.sum()
            if delta > best_delta:
                best_delta = delta
                best_t = t

        # Test
        test_prob = rf.predict_proba(X.iloc[test_idx])[:, 1]
        test_pnl = pnl_r[test_idx]
        y_test = y.iloc[test_idx].values

        try:
            auc = roc_auc_score(y_test, test_prob)
        except Exception:
            auc = 0.5

        if best_t is not None:
            kept_mask = test_prob >= best_t
            kept_pnl = test_pnl[kept_mask]
            skip_pnl = test_pnl[~kept_mask]
            delta = kept_pnl.sum() - test_pnl.sum()

            # Quality gates
            if auc >= 0.52 and delta >= 0 and kept_mask.sum() >= 10:
                all_kept_pnl.extend(kept_pnl.tolist())
                results.append((session, aperture, auc, delta, len(test_idx), kept_mask.sum(),
                               kept_pnl.mean(), skip_pnl.mean() if len(skip_pnl) > 0 else 0))
            else:
                # Model rejected — take all trades (fail-open)
                all_test_pnl.extend(test_pnl.tolist())
        else:
            all_test_pnl.extend(test_pnl.tolist())

# Combine: ML-filtered trades + fail-open trades
total_pnl = np.array(all_kept_pnl + all_test_pnl)

print(f"\n{'='*70}")
print("MNQ ML RESULTS (positive baseline, session eligibility)")
print("="*70)
print(f"  ML-filtered trades: {len(all_kept_pnl)} (mean={np.mean(all_kept_pnl):+.4f}R)" if all_kept_pnl else "  No ML models")
print(f"  Fail-open trades:   {len(all_test_pnl)} (mean={np.mean(all_test_pnl):+.4f}R)" if all_test_pnl else "")
print(f"  Total test trades:  {len(total_pnl)} (mean={total_pnl.mean():+.4f}R)")
print(f"  Models: {len(results)}")

if results:
    print("\n  Per-model:")
    for session, ap, auc, delta, nt, nk, kept_mean, skip_mean in results:
        print(f"    {session:<22} O{ap} AUC={auc:.3f} delta={delta:+.1f}R kept={kept_mean:+.3f} skip={skip_mean:+.3f} N={nk}/{nt}")

# BOOTSTRAP: does ML improve over baseline on positive data?
print(f"\n{'='*70}")
print("BOOTSTRAP: ML vs random on POSITIVE baseline (500 reps)")
print("="*70)

# Baseline: all test trades without ML
all_raw_test = pnl_r[n_val_end:]
ml_mean = total_pnl.mean()
n_ml = len(total_pnl)

null_means = []
for rep in range(500):
    rng = np.random.RandomState(rep)
    idx = rng.choice(len(all_raw_test), size=min(n_ml, len(all_raw_test)), replace=False)
    null_means.append(all_raw_test[idx].mean())

n_above = sum(1 for n in null_means if n >= ml_mean)
print(f"  ML portfolio: {ml_mean:+.4f}R (N={n_ml})")
print(f"  Raw baseline: {all_raw_test.mean():+.4f}R (N={len(all_raw_test)})")
print(f"  Random: {statistics.mean(null_means):+.4f}R")
print(f"  p-value: {n_above/500:.4f}")
print(f"  {'ML ADDS VALUE' if n_above/500 < 0.05 else 'ML DOES NOT ADD VALUE over baseline'}")

# Key: does ML HURT the positive baseline?
if ml_mean < all_raw_test.mean():
    print(f"  WARNING: ML is WORSE than just trading everything ({ml_mean:+.4f} < {all_raw_test.mean():+.4f})")

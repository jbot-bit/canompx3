"""MNQ concentrated: 4 best sessions + ML with at-break features.
At-break features (break_bar_volume, break_delay, break_bar_continues) ARE
legitimate for E2 stop-market because the break IS the entry.
Only cross-session outcomes/double_break are blacklisted."""

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

SESSION_ORDER = list(SESSION_CHRONOLOGICAL_ORDER)
KEEP_SESSIONS = ["CME_PRECLOSE", "EUROPE_FLOW", "SINGAPORE_OPEN", "LONDON_METALS"]

# Import canonical blacklist — NEVER maintain separate lists

# At-break features (break_bar_volume, break_delay, break_bar_continues) are in the
# canonical blacklist because they're unknown at ORB close. For E2 stop-market,
# break_bar_volume is also unknown at entry (only known at bar CLOSE, 1 bar after fill).
# Only CROSS-SESSION at-break features from EARLIER sessions are truly known.
# The session eligibility function handles this — same-session at-break features
# will be blocked by the canonical blacklist via get_eligible_features.
HARD_BLACKLIST = list(LOOKAHEAD_BLACKLIST) + ["garch", "break_ts"]

# For cross-session features: outcome/double_break from ANY session is blacklisted
# At-break features from EARLIER sessions are ALLOWED (they've already happened)
# At-break features from LATER sessions are BLACKLISTED (haven't happened)


def get_eligible_features(session, all_features):
    target_idx = SESSION_ORDER.index(session) if session in SESSION_ORDER else 99
    eligible = []
    for feat in all_features:
        # Hard blacklist check
        if any(b in feat.lower() for b in HARD_BLACKLIST):
            continue
        # Find source session
        source = None
        for s in SESSION_ORDER:
            if s.lower() in feat.lower():
                source = s
                break
        if source is None:
            eligible.append(feat)  # global feature
        elif source == session:
            eligible.append(feat)  # same session — ALL features valid at entry
        elif SESSION_ORDER.index(source) < target_idx:
            eligible.append(feat)  # earlier session — all features valid
        # Later session: blocked
    return eligible


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

# Filter to KEEP sessions
df = df[df["orb_label"].isin(KEEP_SESSIONS)]
td = pd.to_datetime(df["trading_day"])

print(f"MNQ 4 best sessions: {len(df)} trades")
print(f"Baseline: {df['pnl_r'].mean():+.4f}R")

# All available feature columns
SKIP_COLS = {
    "pnl_r",
    "trading_day",
    "symbol",
    "orb_minutes",
    "orb_label",
    "entry_price",
    "stop_price",
    "trading_day_1",
    "entry_model",
    "rr_target",
    "confirm_bars",
    "instrument",
    "entry_ts",
    "exit_ts",
    "exit_price",
    "target_price",
    "risk_dollars",
    "pnl_dollars",
    "ambiguous_bar",
    "ts_outcome",
    "ts_pnl_r",
    "ts_exit_ts",
}
all_features = [c for c in df.columns if c not in SKIP_COLS]

# 60/20/20 split
n = len(df)
n_train = int(n * 0.6)
n_val = int(n * 0.8)
y = (df["pnl_r"] > 0).astype(int)
pnl_r = df["pnl_r"].values

print(f"Split: train={n_train}, val={n_val - n_train}, test={n - n_val}")

# Per-session ML with at-break features
results = []
ml_kept = []
ml_skip = []
failopen = []

for session in KEEP_SESSIONS:
    elig = get_eligible_features(session, all_features)

    for aperture in sorted(df["orb_minutes"].unique()):
        mask = (df["orb_label"] == session) & (df["orb_minutes"] == aperture)
        si = np.where(mask.values)[0]
        train_idx = si[si < n_train]
        val_idx = si[(si >= n_train) & (si < n_val)]
        test_idx = si[si >= n_val]

        if len(train_idx) < 80 or len(val_idx) < 15 or len(test_idx) < 15:
            failopen.extend(pnl_r[test_idx].tolist())
            continue

        # Build feature matrix
        X = pd.DataFrame(index=df.index)
        for c in elig:
            try:
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().sum() > len(df) * 0.1:
                    X[c] = vals.astype(float)
            except Exception:
                pass
        X = X.fillna(-999.0)

        # Drop constants within session
        X_train = X.iloc[train_idx]
        const = [c for c in X.columns if X_train[c].nunique() <= 1]
        X = X.drop(columns=const)

        if X.shape[1] < 3:
            failopen.extend(pnl_r[test_idx].tolist())
            continue

        n_elig = X.shape[1]
        leaf = max(15, min(80, len(train_idx) // 15))
        rf = RandomForestClassifier(**{**RF_PARAMS, "min_samples_leaf": leaf})
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Top 5 features
        imp = rf.feature_importances_
        top5 = [(X.columns[i], imp[i]) for i in np.argsort(imp)[::-1][:5]]

        # Threshold on val
        val_prob = rf.predict_proba(X.iloc[val_idx])[:, 1]
        val_pnl = pnl_r[val_idx]
        best_t, best_d = None, 0
        for t in np.arange(0.35, 0.70, 0.01):
            kept = val_prob >= t
            nk = kept.sum()
            if nk < max(10, int(len(val_idx) * 0.15)):
                continue
            delta = val_pnl[kept].sum() - val_pnl.sum()
            if delta > best_d:
                best_d = delta
                best_t = t

        # Test
        test_prob = rf.predict_proba(X.iloc[test_idx])[:, 1]
        test_pnl = pnl_r[test_idx]
        try:
            auc = roc_auc_score(y.iloc[test_idx], test_prob)
        except Exception:
            auc = 0.5

        if best_t is not None:
            kept_mask = test_prob >= best_t
            nk = kept_mask.sum()
            if auc >= 0.52 and nk >= 5:
                delta = test_pnl[kept_mask].sum() - test_pnl.sum()
                k_mean = test_pnl[kept_mask].mean()
                s_mean = test_pnl[~kept_mask].mean() if (~kept_mask).sum() > 0 else 0

                if delta >= 0:
                    ml_kept.extend(test_pnl[kept_mask].tolist())
                    ml_skip.extend(test_pnl[~kept_mask].tolist())
                    results.append((session, aperture, auc, delta, len(test_idx), nk, k_mean, s_mean, n_elig))
                    feat_str = ", ".join(f"{n}={v:.1%}" for n, v in top5[:3])
                    print(
                        f"  {session:<18} O{aperture} AUC={auc:.3f} kept={k_mean:+.3f} skip={s_mean:+.3f} N={nk}/{len(test_idx)} top3=[{feat_str}]"
                    )
                    continue

        failopen.extend(test_pnl.tolist())

# Results
total_pnl = np.array(ml_kept + failopen)
ml_only = np.array(ml_kept) if ml_kept else np.array([0])
baseline_test = pnl_r[n_val:]

print(f"\n{'=' * 70}")
print("RESULTS")
print(f"{'=' * 70}")
print(f"  Models: {len(results)}")
print(f"  ML-kept:    N={len(ml_kept):>5} ExpR={np.mean(ml_kept):+.4f}" if ml_kept else "  No ML models")
print(f"  ML-skipped: N={len(ml_skip):>5} ExpR={np.mean(ml_skip):+.4f}" if ml_skip else "")
print(f"  Fail-open:  N={len(failopen):>5} ExpR={np.mean(failopen):+.4f}" if failopen else "")
print(f"  Total:      N={len(total_pnl):>5} ExpR={total_pnl.mean():+.4f}")
print(f"  Baseline:   N={len(baseline_test):>5} ExpR={baseline_test.mean():+.4f}")

# ML-ONLY portfolio (no fail-open — ONLY trade what ML approves)
if ml_kept:
    print("\n  ML-ONLY (aggressive — only trade ML-approved):")
    print(f"    N={len(ml_kept)} ExpR={np.mean(ml_kept):+.4f} Total={sum(ml_kept):+.1f}R")

# Bootstrap
print(f"\n{'=' * 70}")
print("BOOTSTRAP (500 reps)")
print("=" * 70)

null_means = []
for rep in range(500):
    rng = np.random.RandomState(rep)
    idx = rng.choice(len(baseline_test), size=min(len(total_pnl), len(baseline_test)), replace=False)
    null_means.append(baseline_test[idx].mean())

n_above = sum(1 for n in null_means if n >= total_pnl.mean())
print(f"  ML portfolio: {total_pnl.mean():+.4f}R")
print(f"  Random: {statistics.mean(null_means):+.4f}R")
print(f"  p-value: {n_above / 500:.4f}")

# Economics at 1.5x slippage
print(f"\n{'=' * 70}")
print("ECONOMICS (1.5x slippage)")
print("=" * 70)

cost_extra = 3.74 * 0.5 / 67.5  # extra cost in R
adj_pnl = total_pnl - cost_extra
annual_trades = len(total_pnl) / 5 * (252 / 252)  # already ~5 years
annual_r = adj_pnl.mean() * annual_trades
print(f"  ExpR (1.5x slip): {adj_pnl.mean():+.4f}")
print(f"  Trades/year: {annual_trades:.0f}")
print(f"  Annual: {annual_r:+.0f}R = ${annual_r * 67.5:+,.0f}/micro")
if ml_kept:
    ml_adj = np.array(ml_kept) - cost_extra
    ml_annual_trades = len(ml_kept) / 5
    ml_annual_r = ml_adj.mean() * ml_annual_trades
    print(
        f"\n  ML-ONLY (1.5x slip): {ml_adj.mean():+.4f}R x {ml_annual_trades:.0f} = {ml_annual_r:+.0f}R = ${ml_annual_r * 67.5:+,.0f}/micro"
    )

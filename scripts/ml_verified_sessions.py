"""ML on VERIFIED sessions only (NYSE_OPEN + COMEX_SETTLE RR1.0).
Positive baseline. New features (VWAP, velocity). Canonical blacklist.
Bootstrap verified. No 2026 data (sacred holdout)."""

import sys

sys.path.insert(0, r"C:\Users\joshd\canompx3")

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import LOOKAHEAD_BLACKLIST, RF_PARAMS, SESSION_CHRONOLOGICAL_ORDER

BLACKLIST = list(LOOKAHEAD_BLACKLIST) + ["garch", "break_ts"]
SKIP = {
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

SESSION_ORDER = list(SESSION_CHRONOLOGICAL_ORDER)


def get_eligible(session, all_feats):
    """Only features from sessions that completed BEFORE target."""
    target_idx = SESSION_ORDER.index(session) if session in SESSION_ORDER else 99
    eligible = []
    for feat in all_feats:
        if any(b in feat.lower() for b in BLACKLIST):
            continue
        source = None
        for s in SESSION_ORDER:
            if s.lower() in feat.lower():
                source = s
                break
        if source is None:
            eligible.append(feat)
        elif source == session:
            eligible.append(feat)  # same session ORB features known at entry
        elif SESSION_ORDER.index(source) < target_idx:
            eligible.append(feat)
    return eligible


con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

VERIFIED = [("NYSE_OPEN", 1.0), ("COMEX_SETTLE", 1.0)]

for session, rr in VERIFIED:
    df = con.execute(f"""
        SELECT o.pnl_r, o.trading_day, d.*
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol='MNQ' AND o.pnl_r IS NOT NULL AND o.entry_model='E2'
              AND o.confirm_bars=1 AND o.rr_target={rr} AND o.orb_label='{session}'
              AND o.trading_day < '2026-01-01'
        ORDER BY o.trading_day
    """).fetchdf()

    td = pd.to_datetime(df["trading_day"])
    all_feats = [c for c in df.columns if c not in SKIP]
    elig = get_eligible(session, all_feats)

    # Build X with eligible features
    X = pd.DataFrame(index=df.index)
    for c in elig:
        try:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().sum() > len(df) * 0.1:
                X[c] = v.astype(float)
        except Exception:
            pass
    X = X.fillna(-999.0)

    y = (df["pnl_r"] > 0).astype(int)
    pnl = df["pnl_r"].values

    # 60/20/20 split (all pre-2026)
    n = len(df)
    n_train = int(n * 0.6)
    n_val = int(n * 0.8)

    # Drop constants
    const = [c for c in X.columns if X.iloc[:n_train][c].nunique() <= 1]
    X = X.drop(columns=const)

    # Check which new features made it in
    new_feats = [c for c in X.columns if "vwap" in c.lower() or "velocity" in c.lower()]
    print(f"\n{'=' * 65}")
    print(f"  {session} RR{rr} | N={n} | Features={X.shape[1]} | New={len(new_feats)}")
    print(f"  New features: {new_feats[:5]}{'...' if len(new_feats) > 5 else ''}")
    print(f"  Baseline: {pnl.mean():+.4f}R | Test baseline: {pnl[n_val:].mean():+.4f}R")
    print(f"{'=' * 65}")

    if X.shape[1] < 5:
        print(f"  Too few features ({X.shape[1]})")
        continue

    # Train RF
    leaf = max(15, min(60, n_train // 15))
    rf = RandomForestClassifier(**{**RF_PARAMS, "min_samples_leaf": leaf})
    rf.fit(X.iloc[:n_train], y.iloc[:n_train])

    # Feature importance
    imp = rf.feature_importances_
    top10 = [(X.columns[i], imp[i]) for i in np.argsort(imp)[::-1][:10]]
    print("  Top 10 features:")
    for fname, fimp in top10:
        is_new = " [NEW]" if "vwap" in fname.lower() or "velocity" in fname.lower() else ""
        print(f"    {fname:<45} {fimp:.1%}{is_new}")

    # Threshold on val
    val_prob = rf.predict_proba(X.iloc[n_train:n_val])[:, 1]
    val_pnl = pnl[n_train:n_val]
    best_t, best_d = None, 0
    for t in np.arange(0.35, 0.75, 0.01):
        kept = val_prob >= t
        if kept.sum() < max(10, int(len(val_pnl) * 0.15)):
            continue
        delta = val_pnl[kept].sum() - val_pnl.sum()
        if delta > best_d:
            best_d = delta
            best_t = t

    # Test
    test_prob = rf.predict_proba(X.iloc[n_val:])[:, 1]
    test_pnl = pnl[n_val:]
    try:
        auc = roc_auc_score(y.iloc[n_val:], test_prob)
    except Exception:
        auc = 0.5

    if best_t is None:
        print(f"  No threshold beats baseline. AUC={auc:.3f}")
        print(f"  VERDICT: ML does not improve {session}. Trade raw baseline.")
        continue

    kept = test_prob >= best_t
    nk = kept.sum()
    k_pnl = test_pnl[kept]
    s_pnl = test_pnl[~kept]

    print("\n  Results:")
    print(f"    Baseline:  {test_pnl.mean():+.4f}R  N={len(test_pnl)}  WR={(test_pnl > 0).mean():.0%}")
    print(f"    ML kept:   {k_pnl.mean():+.4f}R  N={nk}  WR={(k_pnl > 0).mean():.0%}")
    print(f"    ML skip:   {s_pnl.mean():+.4f}R  N={len(s_pnl)}")
    print(f"    AUC={auc:.3f}  threshold={best_t:.2f}  skip={1 - nk / len(test_pnl):.0%}")

    # Bootstrap
    null_means = []
    for rep in range(500):
        rng = np.random.RandomState(rep)
        idx = rng.choice(len(test_pnl), size=nk, replace=False)
        null_means.append(test_pnl[idx].mean())
    n_above = sum(1 for n in null_means if n >= k_pnl.mean())
    bp = n_above / 500

    print(f"    Bootstrap: p={bp:.4f} ({'ML HELPS' if bp < 0.05 else 'ML DOES NOT HELP'})")
    print(f"    Lift: {k_pnl.mean() - test_pnl.mean():+.4f}R")

    if bp < 0.05:
        # Economics
        annual_trades = nk * (252 / (len(test_pnl) / (6900 / 5 / 12)))  # rough
        print(f"\n  ML IMPROVES {session}:")
        print(f"    From {test_pnl.mean():+.4f}R to {k_pnl.mean():+.4f}R")
        print(f"    Lift: {k_pnl.mean() - test_pnl.mean():+.4f}R per trade")

con.close()

"""Test ML with cross-session features (prior session taken).

Experiment: Does adding 'has prior session been taken today?' improve
the meta-label classifier? This is genuine pre-entry information that
traders use manually.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS
from trading_app.ml.config import RF_PARAMS
from trading_app.ml.features import transform_to_features

# Session chronological order (Brisbane time)
SESSION_ORDER = [
    "CME_REOPEN",
    "TOKYO_OPEN",
    "BRISBANE_1025",
    "SINGAPORE_OPEN",
    "EUROPE_FLOW",
    "LONDON_METALS",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
]


def build_cross_session_features(df):
    """Build features about prior sessions' ORB breaks.

    For each trade, looks at sessions EARLIER in the day to build:
    - prior_sessions_broken: count of prior sessions with ORB break
    - prior_sessions_long/short: directional counts
    - prior_orb_size_max/avg: prior session ORB sizes
    """
    n = len(df)
    broken = np.zeros(n)
    long_count = np.zeros(n)
    short_count = np.zeros(n)
    size_max = np.full(n, -999.0)
    size_avg = np.full(n, -999.0)

    for i in range(n):
        session = df.iloc[i]["orb_label"]
        if session not in SESSION_ORDER:
            continue

        session_idx = SESSION_ORDER.index(session)
        prior_sessions = SESSION_ORDER[:session_idx]

        b = 0
        lc = 0
        sc = 0
        sizes = []

        for ps in prior_sessions:
            break_col = f"orb_{ps}_break_dir"
            size_col = f"orb_{ps}_size"

            if break_col in df.columns:
                bd = df.iloc[i].get(break_col, None)
                if bd is not None and str(bd).lower() == "long":
                    b += 1
                    lc += 1
                elif bd is not None and str(bd).lower() == "short":
                    b += 1
                    sc += 1

            if size_col in df.columns:
                sz = df.iloc[i].get(size_col, None)
                if pd.notna(sz) and sz > 0:
                    sizes.append(float(sz))

        broken[i] = b
        long_count[i] = lc
        short_count[i] = sc
        if sizes:
            size_max[i] = max(sizes)
            size_avg[i] = np.mean(sizes)

    result = pd.DataFrame(
        {
            "prior_sessions_broken": broken,
            "prior_sessions_long": long_count,
            "prior_sessions_short": short_count,
            "prior_orb_size_max": size_max,
            "prior_orb_size_avg": size_avg,
        },
        index=df.index,
    )
    return result


def run_experiment(instrument="MNQ"):
    print(f"\n{'=' * 70}")
    print(f"  CROSS-SESSION FEATURE EXPERIMENT — {instrument}")
    print(f"{'=' * 70}")

    # Load raw data (need full df for cross-session columns)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    df = con.execute(
        """
        SELECT o.trading_day, o.symbol, o.orb_label, o.orb_minutes,
               o.entry_model, o.rr_target, o.confirm_bars, o.pnl_r, o.outcome,
               v.filter_type, d.*
        FROM orb_outcomes o
        JOIN validated_setups v
            ON o.symbol = v.instrument AND o.orb_label = v.orb_label
            AND o.entry_model = v.entry_model AND o.rr_target = v.rr_target
            AND o.confirm_bars = v.confirm_bars AND o.orb_minutes = v.orb_minutes
        JOIN daily_features d
            ON o.trading_day = d.trading_day AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = $instrument AND o.pnl_r IS NOT NULL AND v.status = 'active'
        ORDER BY o.trading_day
    """,
        {"instrument": instrument},
    ).fetchdf()
    con.close()

    # Apply filter eligibility + dedup
    keep_mask = np.zeros(len(df), dtype=bool)
    for idx in range(len(df)):
        row = df.iloc[idx]
        ft = row["filter_type"]
        filt = ALL_FILTERS.get(ft)
        if filt and filt.matches_row(row.to_dict(), row["orb_label"]):
            keep_mask[idx] = True
    df = df[keep_mask].reset_index(drop=True)

    dedup_cols = ["trading_day", "orb_label", "entry_model", "rr_target", "confirm_bars", "orb_minutes"]
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    print(f"Validated outcomes: {len(df):,}")

    # Build cross-session features
    print("Building cross-session features...")
    cross_feats = build_cross_session_features(df)
    print(
        f"  prior_sessions_broken: mean={cross_feats['prior_sessions_broken'].mean():.2f}, "
        f"max={cross_feats['prior_sessions_broken'].max():.0f}"
    )

    # Build standard features
    X = transform_to_features(df)

    # Remove orb_label one-hots and noise
    drop_cols = [c for c in X.columns if c.startswith("orb_label_")]
    drop_cols += [
        c
        for c in X.columns
        if any(c.startswith(p) for p in ["gap_type_", "atr_vel_regime_", "prev_day_direction_"])
        or c in ["confirm_bars", "orb_break_bar_continues", "orb_minutes"]
    ]
    X_clean = X.drop(columns=[c for c in drop_cols if c in X.columns])

    y = (df["pnl_r"] > 0).astype(int)
    pnl_r = df["pnl_r"].values
    meta_df = df[["trading_day", "orb_label", "pnl_r"]].copy()

    # === E3: Clean features (no cross-session) ===
    # === E5: Clean + cross-session features ===
    X_with_cross = X_clean.copy()
    for col in cross_feats.columns:
        X_with_cross[col] = cross_feats[col].values

    n_train = int(len(X_clean) * 0.8)

    for exp_name, X_exp in [("E3_clean", X_clean), ("E5_cross_session", X_with_cross)]:
        X_tr = X_exp.iloc[:n_train]
        X_te = X_exp.iloc[n_train:]
        y_tr = y.iloc[:n_train]
        y_te = y.iloc[n_train:]
        pnl_test = pnl_r[n_train:]

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)

        valid = ~np.isnan(pnl_test)
        baseline_total = pnl_test[valid].sum()
        baseline_avg = pnl_test[valid].mean()
        baseline_n = valid.sum()

        print(f"\n--- {exp_name} ({X_exp.shape[1]} features) ---")
        print(f"OOS AUC: {auc:.4f}")
        print(f"Baseline: N={baseline_n} avgR={baseline_avg:.4f} totalR={baseline_total:.2f}")
        print(f"{'Thresh':>7} {'Kept':>6} {'Skip%':>6} {'AvgR':>8} {'TotalR':>8} {'WR':>6} {'vs Base':>8}")

        for t in [0.45, 0.48, 0.50, 0.52, 0.55]:
            mask = (y_prob >= t) & valid
            n_kept = mask.sum()
            if n_kept < 100:
                continue
            kept_pnl = pnl_test[mask]
            avg_r = kept_pnl.mean()
            total_r = kept_pnl.sum()
            wr = (kept_pnl > 0).mean()
            skip = 1 - n_kept / valid.sum()
            delta = total_r - baseline_total
            print(f"{t:>7.2f} {n_kept:>6} {skip:>5.1%} {avg_r:>+8.4f} {total_r:>+8.2f} {wr:>5.1%} {delta:>+8.2f}")

        # Feature importance
        importances = rf.feature_importances_
        names = list(X_exp.columns)
        idx_sorted = np.argsort(importances)[::-1]
        print("\nTop 15 features:")
        for i in idx_sorted[:15]:
            marker = " <-- CROSS" if names[i].startswith("prior_") else ""
            print(f"  {names[i]:<35} {importances[i]:6.2%}{marker}")

        # Per-session breakdown at t=0.50
        meta_test = meta_df.iloc[n_train:].copy()
        meta_test["y_prob"] = y_prob
        meta_test["pnl_r_val"] = pnl_test

        print("\nPer-session breakdown at t=0.50:")
        for session in sorted(meta_test["orb_label"].unique()):
            smask = meta_test["orb_label"] == session
            if smask.sum() < 30:
                continue
            s_pnl = meta_test.loc[smask, "pnl_r_val"]
            s_prob = meta_test.loc[smask, "y_prob"]
            kept = s_prob >= 0.50
            if kept.sum() < 5:
                print(f"  SKIP:  {session:<20} N={smask.sum():>5} Kept={kept.sum():>5} (too few kept)")
                continue
            base_avg = s_pnl.mean()
            filt_avg = s_pnl[kept].mean()
            label = "HELPS" if filt_avg - base_avg > 0 else "HURTS"
            print(
                f"  {label}: {session:<20} N={smask.sum():>5} Kept={kept.sum():>5} "
                f"Skip={1 - kept.sum() / smask.sum():>5.1%} "
                f"Base={base_avg:>+.4f} Filt={filt_avg:>+.4f} "
                f"Lift={filt_avg - base_avg:>+.4f}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="MNQ")
    args = parser.parse_args()
    run_experiment(args.instrument)

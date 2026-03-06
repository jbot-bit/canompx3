"""Audit: Are cross-session features leaking session identity?

prior_sessions_broken is the #1 feature for MES at 12.2%.
If SINGAPORE_OPEN always has prior_sessions_broken ≈ 1.71 (avg),
while NYSE_OPEN always has ≈ 4.68, the model may be learning
"prior_broken < 2 → skip" which is just "skip SINGAPORE_OPEN"
with extra steps.

This audit checks whether cross-session features are correlated
with session identity enough to act as proxies.
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
from trading_app.ml.features import transform_to_features

SESSION_ORDER = [
    "CME_REOPEN",
    "TOKYO_OPEN",
    "BRISBANE_1025",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
]


def build_level_features(df):
    """Build prior-level proximity features."""
    n = len(df)
    nearest_to_high = np.full(n, -999.0)
    nearest_to_low = np.full(n, -999.0)
    levels_within_1r = np.zeros(n)
    levels_within_2r = np.zeros(n)
    is_nested = np.zeros(n)
    prior_size_ratio_max = np.full(n, -999.0)
    prior_broken_count = np.zeros(n)
    prior_long_count = np.zeros(n)
    prior_short_count = np.zeros(n)

    for i in range(n):
        row = df.iloc[i]
        session = row["orb_label"]
        if session not in SESSION_ORDER:
            continue

        current_high_col = f"orb_{session}_high"
        current_low_col = f"orb_{session}_low"
        current_size_col = f"orb_{session}_size"

        current_high = row.get(current_high_col)
        current_low = row.get(current_low_col)
        current_size = row.get(current_size_col)

        if pd.isna(current_high) or pd.isna(current_low) or pd.isna(current_size) or current_size <= 0:
            continue

        session_idx = SESSION_ORDER.index(session)
        prior_sessions = SESSION_ORDER[:session_idx]

        all_prior_levels = []
        prior_sizes = []

        for ps in prior_sessions:
            ps_high = row.get(f"orb_{ps}_high")
            ps_low = row.get(f"orb_{ps}_low")
            ps_size = row.get(f"orb_{ps}_size")
            ps_break = row.get(f"orb_{ps}_break_dir")

            if pd.notna(ps_high):
                all_prior_levels.append(float(ps_high))
            if pd.notna(ps_low):
                all_prior_levels.append(float(ps_low))

            if pd.notna(ps_size) and ps_size > 0:
                prior_sizes.append(float(ps_size))
                if pd.notna(ps_high) and pd.notna(ps_low):
                    if current_high <= ps_high and current_low >= ps_low:
                        is_nested[i] = 1

            if ps_break is not None and str(ps_break).lower() == "long":
                prior_broken_count[i] += 1
                prior_long_count[i] += 1
            elif ps_break is not None and str(ps_break).lower() == "short":
                prior_broken_count[i] += 1
                prior_short_count[i] += 1

        if not all_prior_levels:
            continue

        R = float(current_size)
        prior_arr = np.array(all_prior_levels)
        dist_from_high = np.abs(prior_arr - float(current_high)) / R
        dist_from_low = np.abs(prior_arr - float(current_low)) / R

        nearest_to_high[i] = dist_from_high.min()
        nearest_to_low[i] = dist_from_low.min()

        all_dists = np.minimum(dist_from_high, dist_from_low)
        levels_within_1r[i] = (all_dists <= 1.0).sum()
        levels_within_2r[i] = (all_dists <= 2.0).sum()

        if prior_sizes:
            prior_size_ratio_max[i] = max(prior_sizes) / R

    return pd.DataFrame(
        {
            "nearest_level_to_high_R": nearest_to_high,
            "nearest_level_to_low_R": nearest_to_low,
            "levels_within_1R": levels_within_1r,
            "levels_within_2R": levels_within_2r,
            "orb_nested_in_prior": is_nested,
            "prior_orb_size_ratio_max": prior_size_ratio_max,
            "prior_sessions_broken": prior_broken_count,
            "prior_sessions_long": prior_long_count,
            "prior_sessions_short": prior_short_count,
        },
        index=df.index,
    )


def run_audit(instrument="MES"):
    print(f"\n{'=' * 70}")
    print(f"  SESSION LEAKAGE AUDIT — {instrument}")
    print(f"{'=' * 70}")

    # Load data
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

    keep_mask = np.zeros(len(df), dtype=bool)
    for idx in range(len(df)):
        row = df.iloc[idx]
        filt = ALL_FILTERS.get(row["filter_type"])
        if filt and filt.matches_row(row.to_dict(), row["orb_label"]):
            keep_mask[idx] = True
    df = df[keep_mask].reset_index(drop=True)
    df = df.drop_duplicates(
        subset=["trading_day", "orb_label", "entry_model", "rr_target", "confirm_bars", "orb_minutes"], keep="first"
    ).reset_index(drop=True)

    level_feats = build_level_features(df)

    # AUDIT 1: Cross-session feature distributions per session
    print("\n--- AUDIT 1: Cross-session features by session ---")
    print("If distributions are tight and non-overlapping, features leak session identity.\n")
    cross_cols = [
        "prior_sessions_broken",
        "prior_sessions_long",
        "prior_sessions_short",
        "prior_orb_size_ratio_max",
        "levels_within_1R",
        "levels_within_2R",
    ]

    for col in cross_cols:
        print(f"\n  {col}:")
        for session in sorted(df["orb_label"].unique()):
            smask = df["orb_label"] == session
            vals = level_feats.loc[smask, col]
            valid_vals = vals[vals > -999] if col != "prior_sessions_broken" else vals
            if len(valid_vals) == 0:
                print(f"    {session:<20} N={smask.sum():>5} NO DATA")
                continue
            print(
                f"    {session:<20} N={smask.sum():>5} "
                f"mean={valid_vals.mean():>6.2f} std={valid_vals.std():>5.2f} "
                f"[{valid_vals.min():>5.1f}, {valid_vals.max():>5.1f}]"
            )

    # AUDIT 2: Can a classifier predict session from cross-session features alone?
    print("\n--- AUDIT 2: Session predictability from cross/level features ---")
    print("If accuracy >> random baseline, features ARE session proxies.\n")

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    session_labels = le.fit_transform(df["orb_label"])

    X_cross = level_feats.copy()
    X_cross = X_cross.fillna(-999)

    n_train = int(len(X_cross) * 0.8)
    X_tr = X_cross.iloc[:n_train]
    X_te = X_cross.iloc[n_train:]
    y_tr = session_labels[:n_train]
    y_te = session_labels[n_train:]

    from sklearn.ensemble import RandomForestClassifier as RFC

    clf = RFC(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    random_baseline = 1.0 / len(le.classes_)
    print(f"  Random baseline: {random_baseline:.1%}")
    print(f"  Cross/level features accuracy: {acc:.1%}")
    print(f"  Leakage ratio: {acc / random_baseline:.1f}x random")

    # Per-session classification accuracy
    for i, session in enumerate(le.classes_):
        smask = y_te == i
        if smask.sum() < 10:
            continue
        s_acc = (clf.predict(X_te.iloc[smask.nonzero()[0]]) == i).mean()
        print(f"    {session:<20} recall={s_acc:.1%} (N={smask.sum()})")

    # AUDIT 3: Test E6 WITHOUT cross-session features (levels only)
    print("\n--- AUDIT 3: E6 variants comparison ---")
    print("E6a = clean + levels only (no cross-session counts)")
    print("E6b = clean + cross-session counts only (no levels)")
    print("E6c = clean + ALL (current E6)")
    print("E6d = clean + cross-session NORMALIZED (broken/max_possible)\n")

    X_base = transform_to_features(df)
    orb_label_cols = [c for c in X_base.columns if c.startswith("orb_label_")]
    noise_cols = [
        c
        for c in X_base.columns
        if any(c.startswith(p) for p in ["gap_type_", "atr_vel_regime_", "prev_day_direction_"])
        or c in ["confirm_bars", "orb_break_bar_continues", "orb_minutes"]
    ]
    X_clean = X_base.drop(columns=[c for c in orb_label_cols + noise_cols if c in X_base.columns])

    y = (df["pnl_r"] > 0).astype(int)
    pnl_r = df["pnl_r"].values

    # Build normalized cross-session features
    # Normalize by SESSION_ORDER.index(session) to remove session identity
    norm_broken = np.zeros(len(df))
    norm_long = np.zeros(len(df))
    norm_short = np.zeros(len(df))
    for i in range(len(df)):
        session = df.iloc[i]["orb_label"]
        if session in SESSION_ORDER:
            idx = SESSION_ORDER.index(session)
            if idx > 0:
                norm_broken[i] = level_feats.iloc[i]["prior_sessions_broken"] / idx
                norm_long[i] = level_feats.iloc[i]["prior_sessions_long"] / idx
                norm_short[i] = level_feats.iloc[i]["prior_sessions_short"] / idx

    level_only_cols = [
        "nearest_level_to_high_R",
        "nearest_level_to_low_R",
        "levels_within_1R",
        "levels_within_2R",
        "orb_nested_in_prior",
        "prior_orb_size_ratio_max",
    ]
    cross_only_cols = ["prior_sessions_broken", "prior_sessions_long", "prior_sessions_short"]

    # E6a: levels only
    X_e6a = X_clean.copy()
    for col in level_only_cols:
        X_e6a[col] = level_feats[col].values

    # E6b: cross-session only
    X_e6b = X_clean.copy()
    for col in cross_only_cols:
        X_e6b[col] = level_feats[col].values

    # E6c: all (current)
    X_e6c = X_clean.copy()
    for col in level_feats.columns:
        X_e6c[col] = level_feats[col].values

    # E6d: normalized cross-session
    X_e6d = X_clean.copy()
    for col in level_only_cols:
        X_e6d[col] = level_feats[col].values
    X_e6d["prior_broken_pct"] = norm_broken
    X_e6d["prior_long_pct"] = norm_long
    X_e6d["prior_short_pct"] = norm_short

    n_train = int(len(X_clean) * 0.8)
    pnl_test = pnl_r[n_train:]
    valid = ~np.isnan(pnl_test)
    baseline_total = pnl_test[valid].sum()
    meta_test = df.iloc[n_train:][["trading_day", "orb_label", "pnl_r"]].copy()

    rf_params = dict(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=100,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    for name, X_exp in [
        ("E3_clean", X_clean),
        ("E6a_levels", X_e6a),
        ("E6b_cross", X_e6b),
        ("E6c_all", X_e6c),
        ("E6d_normalized", X_e6d),
    ]:
        X_tr = X_exp.iloc[:n_train]
        X_te = X_exp.iloc[n_train:]

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr, y.iloc[:n_train])
        y_prob = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y.iloc[n_train:], y_prob)

        print(f"  {name} ({X_exp.shape[1]} feat): AUC={auc:.4f}")

        # Show best threshold and per-session
        best_delta = -9999
        best_t = 0
        best_total = 0
        for t in np.arange(0.38, 0.55, 0.01):
            mask = (y_prob >= t) & valid
            n_kept = mask.sum()
            if n_kept < 100:
                continue
            total_r = pnl_test[mask].sum()
            delta = total_r - baseline_total
            if delta > best_delta:
                best_delta = delta
                best_t = t
                best_total = total_r

        skip = 1 - (y_prob >= best_t).sum() / valid.sum() if best_t > 0 else 0
        print(f"    Best: t={best_t:.2f} TotalR={best_total:+.1f} Delta={best_delta:+.1f} Skip={skip:.1%}")

        # Per-session at best threshold
        meta_test_copy = meta_test.copy()
        meta_test_copy["y_prob"] = y_prob
        meta_test_copy["pnl_r_val"] = pnl_test

        destroyed = []
        helped = []
        for session in sorted(meta_test_copy["orb_label"].unique()):
            smask = meta_test_copy["orb_label"] == session
            if smask.sum() < 30:
                continue
            s_pnl = meta_test_copy.loc[smask, "pnl_r_val"]
            s_prob = meta_test_copy.loc[smask, "y_prob"]
            kept = s_prob >= best_t
            if kept.sum() < 3:
                continue
            base_r = s_pnl.sum()
            filt_r = s_pnl[kept].sum()
            skip_pct = 1 - kept.sum() / smask.sum()
            delta = filt_r - base_r
            if skip_pct > 0.30 and delta < -10:
                destroyed.append(f"{session} (skip={skip_pct:.0%}, delta={delta:+.1f}R)")
            elif delta > 5:
                helped.append(f"{session} (delta={delta:+.1f}R)")

        if destroyed:
            print(f"    DESTROYED: {', '.join(destroyed)}")
        if helped:
            print(f"    HELPED: {', '.join(helped)}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="MES")
    args = parser.parse_args()
    run_audit(args.instrument)

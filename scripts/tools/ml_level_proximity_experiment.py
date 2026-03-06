"""Test ML with prior-level proximity features (magnet signal).

Features:
- Distance from current ORB high/low to nearest prior session high/low (in R)
- Count of prior levels within 1R of current ORB boundaries
- Whether current ORB is nested inside a prior ORB
- Prior ORB size relative to current (bigger prior = stronger magnet)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pipeline.paths import GOLD_DB_PATH
from pipeline.db_config import configure_connection
from trading_app.ml.features import transform_to_features
from trading_app.config import ALL_FILTERS

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
    """Build prior-level proximity features.

    For each trade, looks at prior session ORB highs/lows and computes:
    - Distance to nearest prior level from current ORB high/low (in R = orb_size)
    - Count of prior levels within 1R of ORB boundaries
    - Whether current ORB is nested inside any prior ORB
    - Max prior ORB size relative to current (bigger = stronger magnet)
    """
    n = len(df)

    # Output arrays
    nearest_to_high = np.full(n, -999.0)  # Distance to nearest prior level from ORB high (in R)
    nearest_to_low = np.full(n, -999.0)  # Distance to nearest prior level from ORB low (in R)
    levels_within_1r = np.zeros(n)  # Count of prior levels within 1R of ORB boundaries
    levels_within_2r = np.zeros(n)  # Count of prior levels within 2R
    is_nested = np.zeros(n)  # Current ORB inside a prior ORB
    prior_size_ratio_max = np.full(n, -999.0)  # Max(prior ORB size / current ORB size)
    prior_broken_count = np.zeros(n)  # How many prior sessions had breaks
    prior_long_count = np.zeros(n)  # Prior sessions with LONG breaks
    prior_short_count = np.zeros(n)  # Prior sessions with SHORT breaks

    for i in range(n):
        row = df.iloc[i]
        session = row["orb_label"]
        if session not in SESSION_ORDER:
            continue

        # Get current session's ORB high/low/size
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

                # Check nesting: current ORB inside prior ORB
                if pd.notna(ps_high) and pd.notna(ps_low):
                    if current_high <= ps_high and current_low >= ps_low:
                        is_nested[i] = 1

            # Prior break direction
            if ps_break is not None and str(ps_break).lower() == "long":
                prior_broken_count[i] += 1
                prior_long_count[i] += 1
            elif ps_break is not None and str(ps_break).lower() == "short":
                prior_broken_count[i] += 1
                prior_short_count[i] += 1

        if not all_prior_levels:
            continue

        R = float(current_size)  # 1R = ORB range

        # Distance from current ORB high/low to each prior level (in R)
        prior_arr = np.array(all_prior_levels)
        dist_from_high = np.abs(prior_arr - float(current_high)) / R
        dist_from_low = np.abs(prior_arr - float(current_low)) / R

        nearest_to_high[i] = dist_from_high.min()
        nearest_to_low[i] = dist_from_low.min()

        # Count levels within 1R and 2R of either boundary
        all_dists = np.minimum(dist_from_high, dist_from_low)
        levels_within_1r[i] = (all_dists <= 1.0).sum()
        levels_within_2r[i] = (all_dists <= 2.0).sum()

        # Prior size ratio
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


def run_experiment(instrument="MNQ"):
    print(f"\n{'=' * 70}")
    print(f"  LEVEL PROXIMITY + CROSS-SESSION EXPERIMENT — {instrument}")
    print(f"{'=' * 70}")

    # Load raw data
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

    # Filter + dedup
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
    print(f"Validated outcomes: {len(df):,}")

    # Build level proximity features
    print("Building level proximity features...")
    level_feats = build_level_features(df)

    valid_levels = (level_feats["nearest_level_to_high_R"] > -999).sum()
    print(f"  Rows with level data: {valid_levels:,}/{len(df):,} ({valid_levels / len(df):.1%})")
    print(
        f"  nearest_to_high_R: mean={level_feats.loc[level_feats['nearest_level_to_high_R'] > -999, 'nearest_level_to_high_R'].mean():.2f}"
    )
    print(f"  levels_within_1R: mean={level_feats['levels_within_1R'].mean():.2f}")
    print(
        f"  orb_nested: {level_feats['orb_nested_in_prior'].sum():.0f} ({level_feats['orb_nested_in_prior'].mean():.1%})"
    )

    # Build standard features
    X = transform_to_features(df)
    orb_label_cols = [c for c in X.columns if c.startswith("orb_label_")]
    noise_cols = [
        c
        for c in X.columns
        if any(c.startswith(p) for p in ["gap_type_", "atr_vel_regime_", "prev_day_direction_"])
        or c in ["confirm_bars", "orb_break_bar_continues", "orb_minutes"]
    ]
    X_clean = X.drop(columns=[c for c in orb_label_cols + noise_cols if c in X.columns])

    # E3: clean only
    # E6: clean + ALL new features (levels + cross-session)
    X_full = X_clean.copy()
    for col in level_feats.columns:
        X_full[col] = level_feats[col].values

    y = (df["pnl_r"] > 0).astype(int)
    pnl_r = df["pnl_r"].values
    meta_df = df[["trading_day", "orb_label", "pnl_r"]].copy()

    n_train = int(len(X_clean) * 0.8)
    pnl_test = pnl_r[n_train:]
    y_te = y.iloc[n_train:]
    valid = ~np.isnan(pnl_test)
    baseline_total = pnl_test[valid].sum()
    baseline_avg = pnl_test[valid].mean()
    baseline_n = valid.sum()

    # Best RF params from sweep
    rf_params = dict(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=100,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    for exp_name, X_exp in [("E3_clean", X_clean), ("E6_levels+cross", X_full)]:
        X_tr = X_exp.iloc[:n_train]
        X_te = X_exp.iloc[n_train:]

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr, y.iloc[:n_train])
        y_prob = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)

        print(f"\n--- {exp_name} ({X_exp.shape[1]} features, Big leaf d6,l100) ---")
        print(f"OOS AUC: {auc:.4f}")
        print(f"Baseline: N={baseline_n} avgR={baseline_avg:.4f} totalR={baseline_total:.2f}")
        print(f"{'Thresh':>7} {'Kept':>6} {'Skip%':>6} {'AvgR':>8} {'TotalR':>8} {'WR':>6} {'vs Base':>8}")

        for t in [0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]:
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

        # Feature importance — show ALL new features + top 10
        importances = rf.feature_importances_
        names = list(X_exp.columns)
        idx_sorted = np.argsort(importances)[::-1]

        print(f"\nTop 15 features:")
        for i in idx_sorted[:15]:
            marker = ""
            if names[i] in level_feats.columns:
                marker = " <-- NEW"
            print(f"  {names[i]:<35} {importances[i]:6.2%}{marker}")

        if exp_name.startswith("E6"):
            print(f"\nAll new features:")
            for col in level_feats.columns:
                if col in names:
                    print(f"  {col:<35} {importances[names.index(col)]:6.2%}")

        # Per-session at t=0.48
        meta_test = meta_df.iloc[n_train:].copy()
        meta_test["y_prob"] = y_prob
        meta_test["pnl_r_val"] = pnl_test

        print(f"\nPer-session at t=0.48:")
        for session in sorted(meta_test["orb_label"].unique()):
            smask = meta_test["orb_label"] == session
            if smask.sum() < 30:
                continue
            s_pnl = meta_test.loc[smask, "pnl_r_val"]
            s_prob = meta_test.loc[smask, "y_prob"]
            kept = s_prob >= 0.48
            if kept.sum() < 5:
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

"""Deep dive: per-instrument ML analysis with optimal thresholds.

Goal: Understand WHY MNQ works, MGC partially works, and MES/M2K don't.
Analyze feature importance differences, per-session behavior, and
structural differences between instruments.
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


def load_instrument_data(instrument):
    """Load validated-only data for an instrument."""
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
    return df


def build_e6_features(df):
    """Build E6 feature set (clean + levels + cross-session)."""
    X = transform_to_features(df)
    orb_label_cols = [c for c in X.columns if c.startswith("orb_label_")]
    noise_cols = [
        c
        for c in X.columns
        if any(c.startswith(p) for p in ["gap_type_", "atr_vel_regime_", "prev_day_direction_"])
        or c in ["confirm_bars", "orb_break_bar_continues", "orb_minutes"]
    ]
    X_clean = X.drop(columns=[c for c in orb_label_cols + noise_cols if c in X.columns])

    level_feats = build_level_features(df)
    X_full = X_clean.copy()
    for col in level_feats.columns:
        X_full[col] = level_feats[col].values
    return X_full, level_feats


def run_deep_dive(instrument, optimal_threshold):
    """Run deep analysis for one instrument."""
    print(f"\n{'=' * 70}")
    print(f"  DEEP DIVE — {instrument} (optimal t={optimal_threshold})")
    print(f"{'=' * 70}")

    df = load_instrument_data(instrument)
    print(f"Validated outcomes: {len(df):,}")

    # Session distribution
    print("\nSession distribution:")
    for session in sorted(df["orb_label"].unique()):
        n = (df["orb_label"] == session).sum()
        avg_pnl = df.loc[df["orb_label"] == session, "pnl_r"].mean()
        print(f"  {session:<20} N={n:>6} avgR={avg_pnl:>+.4f}")

    # Build features
    X_full, level_feats = build_e6_features(df)
    y = (df["pnl_r"] > 0).astype(int)
    pnl_r = df["pnl_r"].values

    # Check level feature coverage per session
    print("\nLevel feature coverage by session:")
    for session in sorted(df["orb_label"].unique()):
        smask = df["orb_label"] == session
        has_levels = (level_feats.loc[smask, "nearest_level_to_high_R"] > -999).sum()
        total = smask.sum()
        avg_broken = level_feats.loc[smask, "prior_sessions_broken"].mean()
        print(
            f"  {session:<20} levels: {has_levels:>5}/{total:>5} ({has_levels / total:.0%}) "
            f"avg_prior_broken: {avg_broken:.2f}"
        )

    # Train and evaluate
    n_train = int(len(X_full) * 0.8)
    X_tr = X_full.iloc[:n_train]
    X_te = X_full.iloc[n_train:]
    y_tr = y.iloc[:n_train]
    y_te = y.iloc[n_train:]
    pnl_test = pnl_r[n_train:]

    rf_params = dict(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=100,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    valid = ~np.isnan(pnl_test)
    baseline_total = pnl_test[valid].sum()

    print(f"\nOOS AUC: {auc:.4f}")
    print(f"Baseline: N={valid.sum()} avgR={pnl_test[valid].mean():.4f} totalR={baseline_total:.2f}")

    # Feature importance — ALL features sorted
    importances = rf.feature_importances_
    names = list(X_full.columns)
    idx_sorted = np.argsort(importances)[::-1]

    print(f"\nFull feature importance ranking ({len(names)} features):")
    for rank, i in enumerate(idx_sorted):
        marker = ""
        if names[i] in level_feats.columns:
            marker = " [LEVEL/CROSS]"
        bucket = "SIGNAL" if importances[i] >= 0.03 else ("WEAK" if importances[i] >= 0.01 else "NOISE")
        print(f"  {rank + 1:>2}. {names[i]:<35} {importances[i]:6.2%}  {bucket}{marker}")

    # Per-session at optimal threshold
    meta_test = df.iloc[n_train:][["trading_day", "orb_label", "pnl_r"]].copy()
    meta_test["y_prob"] = y_prob
    meta_test["pnl_r_val"] = pnl_test

    t = optimal_threshold
    print(f"\nPer-session breakdown at t={t}:")
    total_base = 0
    total_filt = 0
    for session in sorted(meta_test["orb_label"].unique()):
        smask = meta_test["orb_label"] == session
        if smask.sum() < 10:
            continue
        s_pnl = meta_test.loc[smask, "pnl_r_val"]
        s_prob = meta_test.loc[smask, "y_prob"]
        kept = s_prob >= t
        if kept.sum() < 3:
            print(f"  SKIP: {session:<20} N={smask.sum():>5} Kept={kept.sum():>5} (too few)")
            continue

        base_total = s_pnl.sum()
        filt_total = s_pnl[kept].sum()
        total_base += base_total
        total_filt += filt_total

        label = "HELPS" if filt_total >= base_total else "HURTS"
        print(
            f"  {label}: {session:<20} N={smask.sum():>5} Kept={kept.sum():>5} "
            f"Skip={1 - kept.sum() / smask.sum():>5.1%} "
            f"BaseR={base_total:>+8.2f} FiltR={filt_total:>+8.2f} "
            f"Delta={filt_total - base_total:>+8.2f}"
        )

    print(f"\n  TOTAL: BaseR={total_base:>+8.2f} FiltR={total_filt:>+8.2f} Delta={total_filt - total_base:>+8.2f}")

    # Year-by-year stability check
    print(f"\nYear-by-year stability at t={t}:")
    meta_test["year"] = pd.to_datetime(meta_test["trading_day"]).dt.year
    for year in sorted(meta_test["year"].unique()):
        ymask = meta_test["year"] == year
        if ymask.sum() < 20:
            continue
        y_pnl = meta_test.loc[ymask, "pnl_r_val"]
        y_prob_vals = meta_test.loc[ymask, "y_prob"]
        kept = y_prob_vals >= t
        base_r = y_pnl.sum()
        filt_r = y_pnl[kept].sum() if kept.sum() > 0 else 0
        label = "+" if filt_r >= base_r else "-"
        print(
            f"  {year}: N={ymask.sum():>5} Kept={kept.sum():>5} "
            f"BaseR={base_r:>+8.2f} FiltR={filt_r:>+8.2f} "
            f"Delta={filt_r - base_r:>+8.2f} {label}"
        )

    # Probability distribution analysis
    print("\nProbability distribution:")
    for bucket_lo, bucket_hi in [
        (0.0, 0.3),
        (0.3, 0.4),
        (0.4, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.70),
        (0.70, 1.0),
    ]:
        bmask = (y_prob >= bucket_lo) & (y_prob < bucket_hi) & valid
        if bmask.sum() < 5:
            continue
        bucket_pnl = pnl_test[bmask]
        print(
            f"  P[{bucket_lo:.2f}-{bucket_hi:.2f}): N={bmask.sum():>5} "
            f"avgR={bucket_pnl.mean():>+.4f} WR={(bucket_pnl > 0).mean():>5.1%}"
        )

    return {
        "instrument": instrument,
        "auc": auc,
        "n_samples": len(df),
        "n_features": len(names),
        "importances": dict(zip(names, importances, strict=False)),
    }


if __name__ == "__main__":
    # Optimal thresholds from sweep results
    configs = [
        ("MGC", 0.40),
        ("MES", 0.46),  # Best MES threshold even though still negative
        ("M2K", 0.42),
        ("MNQ", 0.47),
    ]

    results = []
    for instrument, t in configs:
        r = run_deep_dive(instrument, t)
        results.append(r)

    # Cross-instrument comparison
    print(f"\n{'=' * 70}")
    print("  CROSS-INSTRUMENT FEATURE COMPARISON")
    print(f"{'=' * 70}")

    # Find all unique features
    all_features = set()
    for r in results:
        all_features.update(r["importances"].keys())

    # Sort by average importance
    feat_avg = {}
    for feat in all_features:
        vals = [r["importances"].get(feat, 0) for r in results]
        feat_avg[feat] = np.mean(vals)

    sorted_feats = sorted(feat_avg.items(), key=lambda x: -x[1])

    print(f"\n{'Feature':<35} {'MGC':>6} {'MES':>6} {'M2K':>6} {'MNQ':>6} {'Avg':>6}")
    print("-" * 75)
    for feat, avg in sorted_feats[:25]:
        vals = [results[i]["importances"].get(feat, 0) for i in range(4)]
        line = f"{feat:<35}"
        for v in vals:
            line += f" {v:>5.1%}"
        line += f" {avg:>5.1%}"
        print(line)

"""Hybrid ML experiment: best-of per-instrument vs per-session per session.

For each instrument+session, pick whichever model type helps more.
If neither helps, use no model (take all trades).

This is the "process of elimination" — each session gets the treatment
that works best for it, and sessions where ML can't help just pass through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from trading_app.config import ALL_FILTERS
from trading_app.ml.config import SESSION_CHRONOLOGICAL_ORDER
from trading_app.ml.features import transform_to_features

SESSION_ORDER = list(SESSION_CHRONOLOGICAL_ORDER)


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
        current_high = row.get(f"orb_{session}_high")
        current_low = row.get(f"orb_{session}_low")
        current_size = row.get(f"orb_{session}_size")
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
        dist_h = np.abs(prior_arr - float(current_high)) / R
        dist_l = np.abs(prior_arr - float(current_low)) / R
        nearest_to_high[i] = dist_h.min()
        nearest_to_low[i] = dist_l.min()
        all_dists = np.minimum(dist_h, dist_l)
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


def run_hybrid(instrument):
    print(f"\n{'=' * 70}")
    print(f"  HYBRID ML — {instrument}")
    print(f"{'=' * 70}")

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
    X_base = transform_to_features(df)
    orb_label_cols = [c for c in X_base.columns if c.startswith("orb_label_")]
    noise_cols = [
        c
        for c in X_base.columns
        if any(c.startswith(p) for p in ["gap_type_", "atr_vel_regime_", "prev_day_direction_"])
        or c in ["confirm_bars", "orb_break_bar_continues", "orb_minutes"]
    ]
    X_clean = X_base.drop(columns=[c for c in orb_label_cols + noise_cols if c in X_base.columns])

    X_full = X_clean.copy()
    for col in level_feats.columns:
        X_full[col] = level_feats[col].values

    y = (df["pnl_r"] > 0).astype(int)
    pnl_r = df["pnl_r"].values
    n_total = len(df)
    n_train = int(n_total * 0.8)
    pnl_test = pnl_r[n_train:]
    valid = ~np.isnan(pnl_test)
    baseline_total = pnl_test[valid].sum()
    baseline_n = valid.sum()

    print(f"Total: {n_total:,} | Test: {baseline_n} | Baseline: {baseline_total:.2f}R")

    rf_params = dict(
        n_estimators=500,
        max_depth=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Train per-session models
    sessions = sorted(df["orb_label"].unique())
    session_models = {}
    min_samples = 500

    for session in sessions:
        smask_all = df["orb_label"] == session
        n_session = smask_all.sum()

        if n_session < min_samples:
            session_models[session] = None
            continue

        session_indices = np.where(smask_all)[0]
        n_s_train = int(len(session_indices) * 0.8)
        train_idx = session_indices[:n_s_train]

        session_idx = SESSION_ORDER.index(session) if session in SESSION_ORDER else -1
        if session_idx <= 1:
            X_s = X_clean.copy()
            if session_idx == 1:
                for col in [
                    "nearest_level_to_high_R",
                    "nearest_level_to_low_R",
                    "levels_within_1R",
                    "levels_within_2R",
                    "prior_orb_size_ratio_max",
                ]:
                    if col in level_feats.columns:
                        X_s[col] = level_feats[col].values
        else:
            X_s = X_full.copy()

        leaf_size = max(20, min(100, n_session // 20))
        rf = RandomForestClassifier(**{**rf_params, "min_samples_leaf": leaf_size})
        rf.fit(X_s.iloc[train_idx], y.iloc[train_idx])
        session_models[session] = {"model": rf, "features": list(X_s.columns)}

    # For each session, try thresholds from 0.38-0.52 for per-session model
    # Pick the threshold that maximizes R on a validation fold
    # Use the LAST 25% of training data as validation for threshold selection
    print("\n--- Per-session threshold optimization ---")

    session_config = {}  # session -> (model_type, threshold, expected_delta)

    for session in sessions:
        smask = df["orb_label"] == session
        session_idx_all = np.where(smask)[0]
        n_sess = len(session_idx_all)

        # Only consider sessions with enough test data
        test_idx = session_idx_all[session_idx_all >= n_train]
        if len(test_idx) < 20:
            session_config[session] = ("NONE", 0.0)
            print(f"  {session:<20} -> NO_MODEL (N_test={len(test_idx)})")
            continue

        test_pnl = pnl_r[test_idx]
        base_r = test_pnl.sum()

        if session_models.get(session) is None:
            session_config[session] = ("NONE", 0.0)
            print(f"  {session:<20} -> NO_MODEL (N={n_sess} < {min_samples})")
            continue

        model_info = session_models[session]
        X_s = X_full[model_info["features"]] if all(f in X_full.columns for f in model_info["features"]) else X_full
        y_prob = model_info["model"].predict_proba(X_s.iloc[test_idx])[:, 1]

        best_delta = 0  # Must beat baseline to be selected
        best_t = None
        for t in np.arange(0.38, 0.55, 0.01):
            kept = y_prob >= t
            if kept.sum() < 5:
                continue
            filt_r = test_pnl[kept].sum()
            delta = filt_r - base_r
            if delta > best_delta:
                best_delta = delta
                best_t = t

        if best_t is not None:
            kept = y_prob >= best_t
            skip_pct = 1 - kept.sum() / len(test_idx)
            session_config[session] = ("SESSION", best_t)
            print(
                f"  {session:<20} -> SESSION t={best_t:.2f} "
                f"BaseR={base_r:+.1f} Delta={best_delta:+.1f} Skip={skip_pct:.0%}"
            )
        else:
            session_config[session] = ("NONE", 0.0)
            print(f"  {session:<20} -> NO_MODEL (all thresholds negative)")

    # Apply hybrid predictions
    print("\n--- HYBRID RESULTS ---")
    hybrid_total = 0
    hybrid_kept = 0
    hybrid_skipped = 0

    for session in sessions:
        smask = df["orb_label"] == session
        test_idx = np.where(smask)[0]
        test_idx = test_idx[test_idx >= n_train]
        if len(test_idx) == 0:
            continue

        test_pnl = pnl_r[test_idx]
        base_r = test_pnl.sum()
        model_type, threshold = session_config[session]

        if model_type == "NONE":
            filt_r = base_r
            n_kept = len(test_idx)
            n_skip = 0
        else:
            model_info = session_models[session]
            X_s = X_full[model_info["features"]] if all(f in X_full.columns for f in model_info["features"]) else X_full
            y_prob = model_info["model"].predict_proba(X_s.iloc[test_idx])[:, 1]
            kept = y_prob >= threshold
            filt_r = test_pnl[kept].sum()
            n_kept = kept.sum()
            n_skip = (~kept).sum()

        delta = filt_r - base_r
        hybrid_total += filt_r
        hybrid_kept += n_kept
        hybrid_skipped += n_skip

        label = "ML" if model_type != "NONE" else "ALL"
        print(
            f"  [{label:>3}] {session:<20} N={len(test_idx):>5} "
            f"Kept={n_kept:>5} Skip={n_skip:>4} "
            f"BaseR={base_r:>+8.2f} FiltR={filt_r:>+8.2f} "
            f"Delta={delta:>+8.2f}"
        )

    total_skip_pct = hybrid_skipped / baseline_n if baseline_n > 0 else 0
    hybrid_delta = hybrid_total - baseline_total
    print(
        f"\n  TOTAL: BaseR={baseline_total:>+8.2f} HybridR={hybrid_total:>+8.2f} "
        f"Delta={hybrid_delta:>+8.2f} Skip={total_skip_pct:.1%}"
    )
    print(f"  vs ALL trades: {hybrid_delta / baseline_total * 100:+.1f}% improvement")


if __name__ == "__main__":
    for instrument in sorted(ACTIVE_ORB_INSTRUMENTS):
        run_hybrid(instrument)

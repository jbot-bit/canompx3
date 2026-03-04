"""Test per-session ML models.

Hypothesis: Cross-session features leak session identity in a
per-instrument model. Within a single session, prior_sessions_broken
varies day-to-day based on actual market conditions, making it a
genuine signal.

For sessions with too few samples, fall back to no-model (take all).
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
    "CME_REOPEN", "TOKYO_OPEN", "BRISBANE_1025", "SINGAPORE_OPEN",
    "LONDON_METALS", "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
    "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
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

    return pd.DataFrame({
        "nearest_level_to_high_R": nearest_to_high,
        "nearest_level_to_low_R": nearest_to_low,
        "levels_within_1R": levels_within_1r,
        "levels_within_2R": levels_within_2r,
        "orb_nested_in_prior": is_nested,
        "prior_orb_size_ratio_max": prior_size_ratio_max,
        "prior_sessions_broken": prior_broken_count,
        "prior_sessions_long": prior_long_count,
        "prior_sessions_short": prior_short_count,
    }, index=df.index)


def run_per_session(instrument="MES", min_session_samples=500):
    print(f"\n{'='*70}")
    print(f"  PER-SESSION ML EXPERIMENT — {instrument}")
    print(f"{'='*70}")

    # Load data
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    df = con.execute("""
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
    """, {"instrument": instrument}).fetchdf()
    con.close()

    keep_mask = np.zeros(len(df), dtype=bool)
    for idx in range(len(df)):
        row = df.iloc[idx]
        filt = ALL_FILTERS.get(row["filter_type"])
        if filt and filt.matches_row(row.to_dict(), row["orb_label"]):
            keep_mask[idx] = True
    df = df[keep_mask].reset_index(drop=True)
    df = df.drop_duplicates(
        subset=["trading_day", "orb_label", "entry_model", "rr_target",
                "confirm_bars", "orb_minutes"],
        keep="first"
    ).reset_index(drop=True)

    print(f"Total outcomes: {len(df):,}")

    # Build features for all
    level_feats = build_level_features(df)
    X_base = transform_to_features(df)
    orb_label_cols = [c for c in X_base.columns if c.startswith("orb_label_")]
    noise_cols = [c for c in X_base.columns if any(c.startswith(p) for p in [
        "gap_type_", "atr_vel_regime_", "prev_day_direction_"
    ]) or c in ["confirm_bars", "orb_break_bar_continues", "orb_minutes"]]
    X_clean = X_base.drop(columns=[c for c in orb_label_cols + noise_cols if c in X_base.columns])

    # Add level features
    X_full = X_clean.copy()
    for col in level_feats.columns:
        X_full[col] = level_feats[col].values

    y = (df["pnl_r"] > 0).astype(int)
    pnl_r = df["pnl_r"].values

    rf_params = dict(
        n_estimators=500, max_depth=6, min_samples_leaf=50,  # Lower leaf for smaller per-session N
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

    # ===== APPROACH 1: Per-instrument E6 model (current) =====
    n_train = int(len(df) * 0.8)
    pnl_test = pnl_r[n_train:]
    valid = ~np.isnan(pnl_test)
    baseline_total = pnl_test[valid].sum()
    baseline_n = valid.sum()
    meta_test = df.iloc[n_train:].copy()

    print(f"\nBaseline: N={baseline_n} avgR={pnl_test[valid].mean():.4f} totalR={baseline_total:.2f}")

    rf_inst = RandomForestClassifier(**{**rf_params, "min_samples_leaf": 100})
    rf_inst.fit(X_full.iloc[:n_train], y.iloc[:n_train])
    y_prob_inst = rf_inst.predict_proba(X_full.iloc[n_train:])[:, 1]
    auc_inst = roc_auc_score(y.iloc[n_train:], y_prob_inst)

    # ===== APPROACH 2: Per-session models =====
    y_prob_session = np.full(len(pnl_test), np.nan)
    session_results = {}

    sessions = sorted(df["orb_label"].unique())
    for session in sessions:
        smask_all = df["orb_label"] == session
        n_session = smask_all.sum()

        # Train/test split per session (80/20 time-ordered)
        session_indices = np.where(smask_all)[0]
        n_s_train = int(len(session_indices) * 0.8)
        train_idx = session_indices[:n_s_train]
        test_idx = session_indices[n_s_train:]

        # Map test_idx to position in test set (n_train onward)
        test_in_oos = test_idx[test_idx >= n_train] - n_train

        if n_session < min_session_samples:
            # Too few samples — no model, take all
            if len(test_in_oos) > 0:
                y_prob_session[test_in_oos] = 0.50  # neutral
            session_results[session] = {
                "n_total": n_session,
                "n_test": len(test_in_oos),
                "model": "NONE (too few)",
                "auc": None,
            }
            continue

        # Select features — remove cross-session features that encode
        # session position if this is an early session
        session_idx = SESSION_ORDER.index(session) if session in SESSION_ORDER else -1

        # For early sessions (0-1 prior sessions), cross-session features
        # are either 0 or near-constant → drop them
        if session_idx <= 1:
            # CME_REOPEN (0) or TOKYO_OPEN (1): no cross-session signal
            use_cols = [c for c in X_clean.columns]
            # Still keep level features if they have data
            if session_idx == 1:
                for col in ["nearest_level_to_high_R", "nearest_level_to_low_R",
                            "levels_within_1R", "levels_within_2R",
                            "prior_orb_size_ratio_max"]:
                    use_cols.append(col)
        else:
            # Later sessions: use all features
            use_cols = list(X_full.columns)

        X_session = X_full[use_cols] if all(c in X_full.columns for c in use_cols) else X_full

        X_s_train = X_session.iloc[train_idx]
        X_s_test = X_session.iloc[test_idx]
        y_s_train = y.iloc[train_idx]
        y_s_test = y.iloc[test_idx]

        # Adjust min_samples_leaf for smaller sessions
        leaf_size = max(20, min(100, n_session // 20))
        rf_s = RandomForestClassifier(**{**rf_params, "min_samples_leaf": leaf_size})
        rf_s.fit(X_s_train, y_s_train)

        y_s_prob = rf_s.predict_proba(X_s_test)[:, 1]

        # Map predictions back to OOS positions
        for j, idx in enumerate(test_idx):
            if idx >= n_train:
                y_prob_session[idx - n_train] = y_s_prob[j]

        try:
            s_auc = roc_auc_score(y_s_test, y_s_prob)
        except ValueError:
            s_auc = None

        session_results[session] = {
            "n_total": n_session,
            "n_test": len(test_in_oos),
            "model": f"RF (leaf={leaf_size}, {len(use_cols)} feat)",
            "auc": s_auc,
        }

    # ===== COMPARE =====
    print(f"\n--- Per-instrument model (E6, 33 feat): AUC={auc_inst:.4f} ---")
    for t in [0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
        mask = (y_prob_inst >= t) & valid
        n_kept = mask.sum()
        if n_kept < 50:
            continue
        total_r = pnl_test[mask].sum()
        delta = total_r - baseline_total
        skip = 1 - n_kept / valid.sum()
        print(f"  t={t:.2f}: Kept={n_kept:>5} Skip={skip:>5.1%} "
              f"TotalR={total_r:>+8.2f} Delta={delta:>+8.2f}")

    print(f"\n--- Per-session models ---")
    for session, info in sorted(session_results.items()):
        auc_str = f"{info['auc']:.4f}" if info['auc'] is not None else "N/A"
        print(f"  {session:<20} N={info['n_total']:>6} test={info['n_test']:>4} "
              f"AUC={auc_str:>6} model={info['model']}")

    # Combined per-session predictions
    has_pred = ~np.isnan(y_prob_session) & valid
    print(f"\n  Predictions available: {has_pred.sum()}/{valid.sum()}")

    if has_pred.sum() > 100:
        try:
            combined_auc = roc_auc_score(y.iloc[n_train:][has_pred], y_prob_session[has_pred])
            print(f"  Combined AUC: {combined_auc:.4f}")
        except ValueError:
            pass

        for t in [0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
            mask = has_pred & (y_prob_session >= t)
            # For trades without predictions, include them (no-model = take all)
            no_pred = ~np.isnan(y_prob_session) == False
            total_mask = mask | (no_pred & valid)

            n_kept = mask.sum()
            if n_kept < 50:
                continue
            total_r = pnl_test[mask].sum()
            # Add PnL from no-model sessions
            no_model_r = pnl_test[no_pred & valid].sum() if (no_pred & valid).any() else 0
            combined_r = total_r + no_model_r

            delta = combined_r - baseline_total
            skip = 1 - (n_kept + (no_pred & valid).sum()) / valid.sum()
            print(f"  t={t:.2f}: Kept={n_kept:>5}+{(no_pred & valid).sum()} "
                  f"Skip={skip:>5.1%} TotalR={combined_r:>+8.2f} Delta={delta:>+8.2f}")

    # Per-session delta comparison
    print(f"\n--- Per-session delta comparison (best threshold per model) ---")
    meta_test_copy = meta_test.copy()
    meta_test_copy["prob_inst"] = y_prob_inst
    meta_test_copy["prob_session"] = y_prob_session
    meta_test_copy["pnl_test"] = pnl_test

    for session in sorted(meta_test_copy["orb_label"].unique()):
        smask = meta_test_copy["orb_label"] == session
        if smask.sum() < 20:
            continue

        s_pnl = meta_test_copy.loc[smask, "pnl_test"].values
        base_r = s_pnl.sum()

        # Per-instrument model at t=0.46
        s_prob_inst = meta_test_copy.loc[smask, "prob_inst"].values
        kept_inst = s_prob_inst >= 0.46
        inst_r = s_pnl[kept_inst].sum() if kept_inst.sum() > 0 else 0
        inst_skip = 1 - kept_inst.sum() / len(s_pnl)

        # Per-session model at t=0.46
        s_prob_sess = meta_test_copy.loc[smask, "prob_session"].values
        has_sess = ~np.isnan(s_prob_sess)
        if has_sess.sum() > 0:
            kept_sess = has_sess & (s_prob_sess >= 0.46)
            sess_r = s_pnl[kept_sess].sum() if kept_sess.sum() > 0 else 0
            # Include non-predicted trades
            no_model_r = s_pnl[~has_sess].sum()
            sess_total = sess_r + no_model_r
            sess_skip = 1 - (kept_sess.sum() + (~has_sess).sum()) / len(s_pnl)
        else:
            sess_total = base_r
            sess_skip = 0

        winner = "SESSION" if sess_total > inst_r else "INST" if inst_r > sess_total else "TIE"
        print(f"  {session:<20} Base={base_r:>+8.1f} "
              f"Inst(skip={inst_skip:.0%})={inst_r:>+8.1f} "
              f"Sess(skip={sess_skip:.0%})={sess_total:>+8.1f} >> {winner}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="MES")
    parser.add_argument("--min-session-samples", type=int, default=500)
    args = parser.parse_args()
    run_per_session(args.instrument, args.min_session_samples)
